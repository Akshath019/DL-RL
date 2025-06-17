import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import plotly.express as px
import smtplib
from email.message import EmailMessage

# Configure pandas display limits
pd.set_option("styler.render.max_elements", 10_000_000)

# Page config MUST be first
st.set_page_config(page_title="💳 Fraud Detector", layout="wide")

# Load models
@st.cache_resource
def load_models():
    try:
        model = load_model('fraud_dnn.h5')
        scaler = joblib.load('scaler.pkl')
        return model, scaler
    except Exception as e:
        st.error(f"Error loading model or scaler: {str(e)}")
        return None, None

model, scaler = load_models()

# Stop app if model fails
if model is None or scaler is None:
    st.stop()

# --- UI ---
st.title("Real-time Fraud Detection System")
uploaded_file = st.file_uploader("Upload transaction data (CSV)", type="csv")

# Email sending function
def send_email_alert(to_email, fraud_df):
    if fraud_df.empty:
        return

    msg = EmailMessage()
    msg['Subject'] = '🚨 Fraud Alert: Suspicious Transactions Detected'
    msg['From'] = 'compsciinikhil@gmail.com'
    msg['To'] = to_email

    msg.set_content(
        f"Hi,\n\n{len(fraud_df)} fraudulent transactions were detected.\n"
        "Please find the attached CSV file with details.\n\nRegards,\nFraud Detection App"
    )

    csv_data = fraud_df.to_csv(index=False)
    msg.add_attachment(
        csv_data.encode('utf-8'),
        maintype='text',
        subtype='csv',
        filename='fraud_transactions.csv'
    )

    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
        smtp.login('compsciinikhil@gmail.com', 'fdve yjgx wzqg dvqm')  # Use your App password
        smtp.send_message(msg)

# --- Main App Logic ---
if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)

        # Preprocess
        X = df.copy()
        if 'Class' in X.columns:
            X = X.drop('Class', axis=1)
        if 'Time' not in X.columns or 'Amount' not in X.columns:
            st.error("CSV must contain 'Time' and 'Amount' columns.")
            st.stop()
        X[['Time', 'Amount']] = scaler.transform(X[['Time', 'Amount']])

        # Predict
        probabilities = model.predict(X)
        predictions = (probabilities > 0.5).astype(int)
        df['Fraud Prediction'] = ['✅ Genuine' if x == 0 else '🚨 Fraud' for x in predictions]
        df['Fraud Probability'] = probabilities

        # Amount Filter
        st.subheader("🔍 Filter Transactions by Amount")
        amount_range = st.slider(
            "Select Amount Range ($)",
            float(df['Amount'].min()),
            float(df['Amount'].max()),
            (float(df['Amount'].min()), float(df['Amount'].max()))
        )
        filtered_df = df[(df['Amount'] >= amount_range[0]) & (df['Amount'] <= amount_range[1])]
        fraud_df = filtered_df[filtered_df['Fraud Prediction'] == '🚨 Fraud']

        # Metrics
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Transactions (Filtered)", len(filtered_df))
        with col2:
            fraud_count = len(fraud_df)
            st.metric("Fraud Detected (Filtered)", f"{fraud_count} ({fraud_count/len(filtered_df):.1%})")

        # Email input
        if not fraud_df.empty:
            st.subheader("📧 Send Fraud Alert via Email")
            to_email = st.text_input("Enter recipient email address")
            if st.button("Send Email Alert"):
                if to_email:
                    send_email_alert(to_email, fraud_df)
                    st.success(f"Email sent to {to_email} with fraud transaction details!")
                else:
                    st.warning("Please enter a valid email address.")

        # Fraud Data
        if not fraud_df.empty:
            st.subheader("🚨 Fraudulent Transactions")
            st.dataframe(fraud_df[['Time', 'Amount', 'Fraud Prediction', 'Fraud Probability']].head(1000).style.format({'Fraud Probability': '{:.2%}'}))
        else:
            st.success("✅ No fraudulent transactions detected in the selected range!")

        # Charts
        st.subheader("Fraud Distribution")
        fraud_counts = filtered_df['Fraud Prediction'].value_counts()
        fig = px.pie(
            values=fraud_counts.values,
            names=fraud_counts.index,
            title="Fraud vs Genuine Transactions",
            color_discrete_map={'✅ Genuine': '#00CC96', '🚨 Fraud': '#FF4B4B'},
        )
        fig.update_traces(textinfo='percent+label')
        st.plotly_chart(fig)

        st.subheader("💰 Amount Distribution")
        fig = px.histogram(
            filtered_df, x="Amount", color="Fraud Prediction",
            nbins=50, barmode="overlay",
            color_discrete_map={'✅ Genuine': '#00CC96', '🚨 Fraud': '#FF4B4B'},
        )
        st.plotly_chart(fig)

        st.subheader("⏰ Transaction Time Analysis")
        fig = px.scatter(
            filtered_df, x='Time', y='Amount', color='Fraud Prediction',
            opacity=0.6,
            color_discrete_map={'✅ Genuine': '#00CC96', '🚨 Fraud': '#FF4B4B'},
        )
        st.plotly_chart(fig)

        st.subheader("📊 Amount Distribution Statistics")
        fig = px.box(
            filtered_df, x='Fraud Prediction', y='Amount',
            color='Fraud Prediction',
            color_discrete_map={'✅ Genuine': '#00CC96', '🚨 Fraud': '#FF4B4B'},
        )
        st.plotly_chart(fig)

        if not fraud_df.empty:
            st.subheader("🎯 Fraud Probability Distribution")
            fig = px.histogram(
                fraud_df, x='Fraud Probability', nbins=50,
                color_discrete_sequence=['#FF4B4B']
            )
            st.plotly_chart(fig)

    except Exception as e:
        st.error(f"Error processing CSV: {str(e)}")