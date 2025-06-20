import streamlit as st
import pandas as pd
from PIL import Image
from educational_feedback_system import process_student_response, reference_data
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
from datetime import datetime

# Initialize Firebase only if no app exists
if not firebase_admin._apps:
    cred = credentials.Certificate('ai-educational-feedback-98570-firebase-adminsdk-fbsvc-a9b12948a6.json')
    default_app = firebase_admin.initialize_app(cred, {
        'databaseURL': 'https://ai-educational-feedback-98570-default-rtdb.firebaseio.com/'
    })
ref = db.reference('/')

# Streamlit app configuration
st.set_page_config(page_title="AI-Powered Educational Feedback System", layout="wide")

# Title and description
st.title("AI-Powered Educational Feedback System")
st.markdown("""
This system analyzes student responses (text, code, or images) and provides personalized feedback,
identifies misconceptions, and generates adaptive practice questions.
""")

# Sidebar for question selection
st.sidebar.header("Select a Question")
question_types = ["All", "Text", "Code", "Visual"]
selected_type = st.sidebar.selectbox("Filter by question type", question_types)

# Filter questions based on type
if selected_type == "All":
    filtered_questions = [item["question"] for item in reference_data]
else:
    filtered_questions = [item["question"] for item in reference_data if item["type"].lower() == selected_type.lower()]

selected_question = st.sidebar.selectbox("Choose a question", filtered_questions)

# Main content area
st.header("Submit Your Response")

# Input method selection
input_method = st.radio("Select input type", ["Text", "Code", "Image"], index=0)

# Response input
student_response = None
if input_method == "Text":
    student_response = st.text_area("Enter your text response", height=100)
elif input_method == "Code":
    student_response = st.text_area("Enter your code response", height=200, placeholder="Write your code here...")
elif input_method == "Image":
    uploaded_image = st.file_uploader("Upload an image (e.g., diagram)", type=["png", "jpg", "jpeg"])
    if uploaded_image:
        student_response = Image.open(uploaded_image)
        st.image(student_response, caption="Uploaded Image", use_column_width=True)

# Function to process and store student input
def process_student_input(student_id, response, feedback, practice_question):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M')
    student_ref = ref.child('students').child(student_id)
    student_ref.set({
        'response': str(response) if not isinstance(response, Image.Image) else 'Image uploaded',
        'feedback': feedback,
        'practice_question': practice_question,
        'timestamp': timestamp,
        'question': selected_question
    })
    print(f"Stored for {student_id} at {timestamp}: {feedback}")

# Submit button
if st.button("Submit Response"):
    if student_response:
        with st.spinner("Processing your response..."):
            result = process_student_response(selected_question, student_response)
            if isinstance(result, str):
                st.error(result)
            else:
                # Display feedback
                st.subheader("Feedback")
                st.write(result["feedback"])

                # Display practice question
                st.subheader("Practice Question")
                st.write(result["practice_question"])

                # Store in Firebase with a unique student ID (e.g., timestamp-based)
                student_id = f"student_{int(datetime.now().timestamp())}"
                process_student_input(student_id, student_response, result["feedback"], result["practice_question"])
    else:
        st.warning("Please provide a response before submitting.")

# Display sample questions and their correct answers
st.header("Sample Questions")
question_df = pd.DataFrame([
    {
        "Question": item["question"],
        "Type": item["type"],
        "Correct Answer": item["correct_answer"],
        "Source": item["source"]
    } for item in reference_data
])
st.dataframe(question_df, use_container_width=True)