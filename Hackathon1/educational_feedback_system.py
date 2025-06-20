# AI-Powered Educational Feedback System
# This script implements an NLP-based system that analyzes student text responses, code submissions, and visual inputs (e.g., diagrams), 
# provides personalized feedback, identifies misconceptions, and generates adaptive practice questions using Hugging Face Transformers, 
# CodeBERT, and Vision Transformers.

# Install required libraries
# !pip install transformers torch sentence-transformers sentencepiece torchvision
# !pip install datasets pandas
# !pip install pylint

import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer, ViTForImageClassification, ViTImageProcessor
from sentence_transformers import SentenceTransformer, util
from torchvision import transforms
from PIL import Image
import numpy as np
import io
import ast
import re
from datasets import load_dataset
import pandas as pd
import requests
from pylint.lint import Run
from pylint.reporters.text import TextReporter
from io import StringIO

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device set to: {device}')

# Load models
t5_model = T5ForConditionalGeneration.from_pretrained('google/flan-t5-small').to(device)
t5_tokenizer = T5Tokenizer.from_pretrained('google/flan-t5-small')
sentence_model = SentenceTransformer('all-mpnet-base-v2').to(device)
codebert_model = SentenceTransformer('microsoft/codebert-base').to(device)
vit_processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
vit_model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224').to(device)

# Load datasets
# SQuAD 2.0 dataset
squad_dataset = load_dataset("squad_v2", split="validation[:100]")

# MCTest dataset from local files
#mctest_files = [
#   "mc160.dev.tsv",
#    "mc160.train.tsv",
#   "mc500.dev.tsv",
#    "mc500.train.tsv"
#]

#mctest_data = []
#or file_path in mctest_files:
#   with open(file_path, 'r', encoding='utf-8') as file:
#      lines = file.readlines()
#        for line in lines:
#            parts = line.strip().split('\t')
#            if len(parts) >= 6:
#                question = parts[1]
#                choices = parts[2:6]
#                correct_answer = choices[0]
#                print(f"Debug: MCTest data: {dict(zip(['question', 'correct_answer'], [question, correct_answer]))}")
#mctest_data.append({"question": question, "correct_answer": correct_answer})

#mctest_dataset = mctest_data[:50]

# Load EdNet KT1 dataset (sample via URL or local file)
ednet_url = "https://example.com/ednet_kt1_sample.csv"  # Placeholder URL
try:
    response = requests.get(ednet_url)
    ednet_data = pd.read_csv(io.StringIO(response.text))[:50]
except:
    # Fallback: Simulate EdNet data
    ednet_data = pd.DataFrame({
        "question": ["What is 2+2?", "Define a variable in Python."],
        "correct_answer": ["4", "A variable is a named storage location in memory."]
    })

# Function to generate realistic misconceptions using T5
def generate_synthetic_misconceptions(correct_answer):
    prompt = f"Given the correct answer '{correct_answer}', generate two concise misconceptions a student might have. Return them as a list of strings."
    input_ids = t5_tokenizer.encode(prompt, return_tensors='pt', max_length=512, truncation=True).to(device)
    outputs = t5_model.generate(
        input_ids,
        max_length=50,
        num_beams=5,
        early_stopping=True,
        no_repeat_ngram_size=2,
        num_return_sequences=2
    )
    misconceptions = [t5_tokenizer.decode(out, skip_special_tokens=True) for out in outputs]
    return [
        {"text": misconceptions[0], "correction": f"Correct is: {correct_answer}"},
        {"text": misconceptions[1], "correction": f"Correct is: {correct_answer}"}
    ]

# Create reference_data from SQuAD
reference_data = []
for item in squad_dataset:
    question = item["question"]
    answers = item["answers"]["text"]
    correct_answer = answers[0] if answers else "No answer provided"
    reference_data.append({
        "question": question,
        "correct_answer": correct_answer,
        "misconceptions": generate_synthetic_misconceptions(correct_answer),
        "type": "text",
        "source": "squad"
    })

# Add MCTest data to reference_data
#for item in mctest_dataset:
#    reference_data.append({
#        "question": item["question"],
#        "correct_answer": item["correct_answer"],
#        "misconceptions": generate_synthetic_misconceptions(item["correct_answer"]),
#        "type": "text",
#        "source": "mctest"
#    })

# Add EdNet data
for _, row in ednet_data.iterrows():
    reference_data.append({
        "question": row["question"],
        "correct_answer": row["correct_answer"],
        "misconceptions": generate_synthetic_misconceptions(row["correct_answer"]),
        "type": "text",
        "source": "ednet"
    })

# Add manual examples
reference_data.extend([
    {
        "question": "Write a Python function to calculate the factorial of a number.",
        "correct_answer": "def factorial(n):\n    if n == 0 or n == 1:\n        return 1\n    return n * factorial(n - 1)",
        "misconceptions": [
            {"text": "def factorial(n):\n    return n * factorial(n)", "correction": "This causes infinite recursion. You need a base case like if n == 0 or n == 1: return 1."},
            {"text": "def factorial(n):\n    return n * (n - 1)", "correction": "This only multiplies n by (n-1), not the full factorial."}
        ],
        "type": "code",
        "source": "manual"
    },
    {
        "question": "Identify the diagram of a plant cell.",
        "correct_answer": "An image of a plant cell with labeled parts: cell wall, chloroplasts, nucleus, and vacuole.",
        "misconceptions": [
            {"text": "Diagram shows an animal cell (no cell wall or chloroplasts)", "correction": "Plant cells have a cell wall and chloroplasts, unlike animal cells."},
            {"text": "Diagram lacks chloroplasts", "correction": "Chloroplasts are essential for photosynthesis in plant cells."}
        ],
        "type": "visual",
        "source": "manual"
    }
])

print(f"Loaded {len(reference_data)} questions, including {len(squad_dataset)} from SQuAD, and {len(ednet_data)} from EdNet.")

# Function to preprocess code (strip comments and normalize whitespace)
def preprocess_code(code_str):
    code_str = re.sub(r'#.*?\\n', '\\n', code_str)
    code_str = re.sub(r'\"\"\".*?\"\"\"', '', code_str, flags=re.DOTALL)
    code_str = ' '.join(code_str.split())
    return code_str

# Function to compute semantic similarity for text or code
def compute_similarity(response, reference, model_type='text'):
    model = sentence_model if model_type == 'text' else codebert_model
    if model_type == 'code':
        response = preprocess_code(response)
        reference = preprocess_code(reference)
    embeddings = model.encode([response, reference], convert_to_tensor=True, device=device)
    similarity = util.cos_sim(embeddings[0], embeddings[1]).item()
    return similarity

# Function to analyze code structure with pylint

def analyze_code_structure(code_str):
    try:
        print(f"Debug: Analyzing code structure for: {code_str}")
        tree = ast.parse(code_str)
        functions = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
        print(f"Debug: Found functions: {functions}")
        has_base_case = False
        for node in ast.walk(tree):
            if isinstance(node, ast.If):
                print(f"Debug: Found if statement: {ast.dump(node, indent=2)}")
                has_base_case = True
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id in functions:
                    print(f"Debug: Recursion detected for function {node.func.id}")
                    if not has_base_case:
                        return False, "Warning: Recursion detected. Add a base case!"
        if "factorial" in functions and not has_base_case:
            return False, "Warning: Factorial function missing base case (e.g., if n == 0 or n == 1: return 1)."
        print("Debug: Skipping pylint analysis due to version incompatibility")
        return True, "Code structure looks valid. No major issues found."
    except SyntaxError as e:
        return False, f"Syntax error: {e}"
    except Exception as e:
        return False, f"Error analyzing code: {str(e)}"

# Function to process visual inputs
def process_visual_input(image, question_data):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    inputs = vit_processor(images=image_tensor, return_tensors="pt").to(device)
    outputs = vit_model(**inputs)
    predicted_class = torch.argmax(outputs.logits, dim=1).item()
    
    labels = {0: "plant cell", 1: "animal cell"}
    predicted_label = labels.get(predicted_class, "unknown")
    is_correct = predicted_label == "plant cell" and question_data['question'].lower().startswith("identify the diagram of a plant cell")
    
    feedback = "Correct! Diagram shows a plant cell with cell wall and chloroplasts." if is_correct else "Incorrect. Expected a plant cell with cell wall, chloroplasts, nucleus, and vacuole."
    
    misconceptions = []
    if not is_correct:
        feedback += f" Detected: {predicted_label}."
        for mc in question_data['misconceptions']:
            misconceptions.append(f"Misconception: {mc['text']}. Fix: {mc['correction']}")
        if predicted_label == "animal cell":
            feedback += " Missing features: cell wall, chloroplasts."
    
    return feedback, misconceptions

# [Previous code remains unchanged until the function definitions]

# Function to identify misconceptions
def identify_misconceptions(response, question_data, input_type='text'):
    # Ensure feedback is always a list
    feedback = []
    if input_type == 'visual':
        return feedback
    model_type = 'code' if input_type == 'code' else 'text'
    try:
        response_embedding = (sentence_model if model_type == 'text' else codebert_model).encode(response, convert_to_tensor=True, device=device)
        tree = ast.parse(response)
        is_recursive = any(isinstance(node, ast.Call) and isinstance(node.func, ast.Name) for node in ast.walk(tree))
    except (SyntaxError, ValueError, AttributeError):
        is_recursive = False  # Fallback if parsing fails (e.g., for non-code input)

    if not hasattr(question_data, 'get') or 'misconceptions' not in question_data:
        return feedback  # Return empty list if question_data is invalid

    for misconception in question_data['misconceptions']:
        if not isinstance(misconception, dict) or 'text' not in misconception:
            continue  # Skip invalid misconception entries
        try:
            mc_embedding = (sentence_model if model_type == 'text' else codebert_model).encode(misconception['text'], convert_to_tensor=True, device=device)
            similarity = util.cos_sim(response_embedding, mc_embedding).item()
            if similarity > 0.55 and any(word in response.lower() for word in misconception['text'].lower().split()[:3]):
                if misconception['text'] == "def factorial(n): return n * factorial(n)" and not is_recursive:
                    continue  # Skip if not recursive
                if misconception['text'] not in [f.get('text', '') for f in feedback]:  # Avoid duplicates
                    feedback.append(f"Misconception detected: {misconception['text']}. Correction: {misconception['correction']}")
        except Exception:
            continue  # Skip if embedding fails
    return feedback

# Function to generate feedback
def generate_feedback(response, question_data, input_type='text'):
    if input_type == 'visual' and vit_model is None:
        return "Error: Visual model not loaded. Please check dependencies."
    if input_type == 'visual':
        feedback, misconceptions = process_visual_input(response, question_data)
        if misconceptions and isinstance(misconceptions, list):
            feedback += "\n" + "\n".join(misconceptions)
        return feedback

    if sentence_model is None or codebert_model is None:
        return f"Error: NLP models not loaded. Correct version: {question_data.get('correct_answer', 'N/A')}"

    model_type = 'code' if input_type == 'code' else 'text'
    try:
        similarity = compute_similarity(response, question_data.get('correct_answer', ''), model_type)
    except Exception:
        similarity = 0.0  # Fallback if similarity computation fails

    if input_type == 'code':
        is_valid, structure_feedback = analyze_code_structure(response)
        feedback = structure_feedback if not is_valid else "Code structure looks valid. "
        misconceptions = identify_misconceptions(response, question_data, input_type)
        if misconceptions and isinstance(misconceptions, list):
            feedback += "\n" + "\n".join(misconceptions)
        if not is_valid or similarity < 0.65:
            feedback += f"\nNeeds improvement. Correct version: \n{question_data.get('correct_answer', '')}"
        elif similarity > 0.85:
            feedback = "Excellent! Your answer is very accurate." + (f"\n{misconceptions}" if misconceptions and isinstance(misconceptions, list) else "")
        else:
            feedback += f"\nGood effort! Your answer is partially correct, but consider: \n{question_data.get('correct_answer', '')}"
    else:
        if similarity > 0.85:
            return "Excellent! Your answer is very accurate."
        elif similarity > 0.65:
            return f"Good effort! Your answer is partially correct, but consider: {question_data.get('correct_answer', '')}"
        else:
            return f"Needs improvement. Correct version: {question_data.get('correct_answer', '')}"
    return feedback

# Function to generate adaptive practice question
def generate_practice_question(question_data, student_response=None):
    difficulty = "medium"
    if student_response and isinstance(student_response, str):
        similarity = compute_similarity(student_response, question_data.get('correct_answer', ''), 'text' if question_data.get('type', 'text') == 'text' else 'code')
        difficulty = "hard" if similarity < 0.65 else "easy" if similarity > 0.85 else "medium"
    
    prompt = f"Given the question '{question_data.get('question', '')}' and correct answer '{question_data.get('correct_answer', '')}', generate a {difficulty}-difficulty follow-up question that tests a related concept or clarifies a misunderstanding. Keep it educational and clear."
    if student_response and isinstance(student_response, str):
        prompt += f" Student's incorrect response: '{student_response}'. Address this specific mistake."
    
    try:
        input_ids = t5_tokenizer.encode(prompt, return_tensors='pt', max_length=512, truncation=True).to(device)
        outputs = t5_model.generate(
            input_ids,
            max_length=60,
            num_beams=4,
            temperature=0.8,
            no_repeat_ngram_size=2,
            top_k=50
        )
        return t5_tokenizer.decode(outputs[0], skip_special_tokens=True)
    except Exception:
        return "Error generating practice question. Please try again."

# Main function to process student response
def process_student_response(question, student_response):
    question_data = next((item for item in reference_data if item['question'].lower() == question.lower()), None)
    if not question_data:
        return "Question not found in database."

    input_type = question_data.get('type', 'text')  # Default to 'text' if type is missing
    feedback = generate_feedback(student_response, question_data, input_type)

    if input_type != 'visual':
        misconceptions = identify_misconceptions(student_response, question_data, input_type)
        if misconceptions and isinstance(misconceptions, list):
            feedback += "\n" + "\n".join(misconceptions)

    practice_question = generate_practice_question(question_data, student_response if input_type != 'visual' else None)

    return {
        'feedback': feedback,
        'practice_question': practice_question
    }

# [Rest of the code, including example usage, remains unchanged]
# Example usage
if __name__ == "__main__":
    # Text-based examples
    print("Testing SQuAD Example:")
    question = reference_data[0]['question']
    student_response = "In the early 2000s."
    result = process_student_response(question, student_response)
    if isinstance(result, str):
        print("Error:", result)
    else:
        print("Feedback:", result['feedback'])
        print("Practice Question:", result['practice_question'])

#    print("\nTesting MCTest Example:")
#    question = mctest_dataset[0]['question']
#    student_response = mctest_dataset[0]['correct_answer']
#   result = process_student_response(question, student_response)
#    if isinstance(result, str):
#        print("Error:", result)
#    else:
#        print("Feedback:", result['feedback'])
#        print("Practice Question:", result['practice_question'])

    print("\nTesting EdNet Example:")
    question = ednet_data.iloc[0]['question']
    student_response = "5"
    result = process_student_response(question, student_response)
    if isinstance(result, str):
        print("Error:", result)
    else:
        print("Feedback:", result['feedback'])
        print("Practice Question:", result['practice_question'])

    # Text-based examples
    print("\nText-based Example 1:")
    question = reference_data[0]['question']
    student_response = "In the early 2000s."
    result = process_student_response(question, student_response)
    if isinstance(result, str):
        print("Error:", result)
    else:
        print("Feedback:", result['feedback'])
        print("Practice Question:", result['practice_question'])

    print("\nText-based Example 2:")
    question = reference_data[1]['question']
    student_response = "Spice Girls."
    result = process_student_response(question, student_response)
    if isinstance(result, str):
        print("Error:", result)
    else:
        print("Feedback:", result['feedback'])
        print("Practice Question:", result['practice_question'])

    # Code and visual examples
    print("\nCode-based Example:")
    question = "Write a Python function to calculate the factorial of a number."
    student_response = "def factorial(n):\n    return n * (n - 1)"
    result = process_student_response(question, student_response)
    print("Feedback:", result['feedback'])
    print("Practice Question:", result['practice_question'])

    print("\nVisual-based Example:")
    question = "Identify the diagram of a plant cell."
    student_response = Image.new('RGB', (224, 224), color='green')
    result = process_student_response(question, student_response)
    print("Feedback:", result['feedback'])
    print("Practice Question:", result['practice_question'])