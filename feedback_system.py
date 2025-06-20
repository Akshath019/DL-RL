import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
from transformers import pipeline
from datetime import datetime

# Initialize Firebase
cred = credentials.Certificate('ai-educational-feedback-98570-firebase-adminsdk-fbsvc-a9b12948a6.json')
default_app = firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://ai-educational-feedback-98570-default-rtdb.firebaseio.com/'
})
ref = db.reference('/')

# Set up sentiment analysis
sentiment_analyzer = pipeline("sentiment-analysis")

# Custom analysis function (replace with your project’s logic)
def analyze_response(response):
    # Use sentiment analysis as a base
    result = sentiment_analyzer(response)[0]
    feedback = f"Feedback: Sentiment is {result['label']} with score {result['score']:.2f}"
    # Add your specific checks (e.g., math errors or code validation)
    if "2 + 2" in response.lower() and "5" in response.lower():
        feedback += " Incorrect! 2 + 2 = 4."
    return feedback

# Function to process and store student input
def process_student_input(student_id, response):
    feedback = analyze_response(response)
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M')
    student_ref = ref.child('students').child(student_id)
    student_ref.set({
        'response': response,
        'feedback': feedback,
        'timestamp': timestamp
    })
    print(f"Stored for {student_id} at {timestamp}: {feedback}")

# Example integration with your project (replace with your input method)
if __name__ == "__main__":
    try:
        # Simulate student inputs (replace with your data source, e.g., Streamlit or API)
        student_inputs = [
            ('student1', 'I think 2 + 2 = 5'),
            ('student2', 'Code: def add(a, b): return a + b')
        ]
        for student_id, response in student_inputs:
            process_student_input(student_id, response)
        # Retrieve and print all data
        all_students = ref.child('students').get()
        print("All student data:", all_students)
    except Exception as e:
        print(f"Error: {e}")