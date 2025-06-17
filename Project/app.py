import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import plotly.express as px

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

# Check if models loaded successfully
if model is None or scaler is None:
    st.stop()

# --- UI ---
st.title("Real-time Fraud Detection System")

uploaded_file = st.file_uploader("Upload transaction data (CSV)", type="csv")

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
        
        # Predictions with probabilities
        probabilities = model.predict(X)
        predictions = (probabilities > 0.5).astype(int)
        df['Fraud Prediction'] = ['✅ Genuine' if x == 0 else '🚨 Fraud' for x in predictions]
        df['Fraud Probability'] = probabilities
        
        # Filter by amount range
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
        
        # Show fraud cases
        if not fraud_df.empty:
            st.subheader("🚨 Fraudulent Transactions")
            st.dataframe(fraud_df[['Time', 'Amount', 'Fraud Prediction', 'Fraud Probability']].head(1000).style.format({'Fraud Probability': '{:.2%}'}))
        else:
            st.success("✅ No fraudulent transactions detected in the selected range!")
        
        # Visualization 1: Fraud Distribution (Pie Chart)
        st.subheader("Fraud Distribution")
        fraud_counts = filtered_df['Fraud Prediction'].value_counts()
        fig = px.pie(
            values=fraud_counts.values,
            names=fraud_counts.index,
            title="Fraud vs Genuine Transactions",
            color_discrete_map={'✅ Genuine': '#00CC96', '🚨 Fraud': '#FF4B4B'},
            labels={'value': 'Count', 'name': 'Transaction Type'}
        )
        fig.update_traces(textinfo='percent+label', hoverinfo='label+percent+value')
        fig.update_layout(width=500, height=400)
        st.plotly_chart(fig)
        
        # Visualization 2: Transaction Amount Distribution (Histogram)
        st.subheader("💰 Amount Distribution")
        fig = px.histogram(
            filtered_df, x="Amount", color="Fraud Prediction",
            title="Transaction Amount Distribution",
            nbins=50, barmode="overlay",
            color_discrete_map={'✅ Genuine': '#00CC96', '🚨 Fraud': '#FF4B4B'},
            labels={'Amount': 'Transaction Amount ($)'}
        )
        fig.update_layout(xaxis_title="Amount ($)", yaxis_title="Count")
        st.plotly_chart(fig)
        
        # Visualization 3: Time vs Amount Scatter Plot
        st.subheader("⏰ Transaction Time Analysis")
        fig = px.scatter(
            filtered_df, x='Time', y='Amount', color='Fraud Prediction',
            title="Transaction Amount vs Time",
            color_discrete_map={'✅ Genuine': '#00CC96', '🚨 Fraud': '#FF4B4B'},
            opacity=0.6,
            labels={'Time': 'Time (seconds)', 'Amount': 'Amount ($)'}
        )
        fig.update_layout(xaxis_title="Time (seconds)", yaxis_title="Amount ($)")
        st.plotly_chart(fig)
        
        # Visualization 4: Box Plot for Amount by Fraud Status
        st.subheader("📊 Amount Distribution Statistics")
        fig = px.box(
            filtered_df, x='Fraud Prediction', y='Amount',
            title="Transaction Amount by Fraud Status",
            color='Fraud Prediction',
            color_discrete_map={'✅ Genuine': '#00CC96', '🚨 Fraud': '#FF4B4B'},
            labels={'Amount': 'Transaction Amount ($)'}
        )
        fig.update_layout(xaxis_title="Transaction Type", yaxis_title="Amount ($)")
        st.plotly_chart(fig)
        
        # Visualization 5: Fraud Probability Distribution (for fraud cases)
        if not fraud_df.empty:
            st.subheader("🎯 Fraud Probability Distribution")
            fig = px.histogram(
                fraud_df, x='Fraud Probability', 
                title="Distribution of Fraud Prediction Probabilities",
                nbins=50, color_discrete_sequence=['#FF4B4B'],
                labels={'Fraud Probability': 'Fraud Probability'}
            )
            fig.update_layout(xaxis_title="Fraud Probability", yaxis_title="Count")
            st.plotly_chart(fig)
    
    except Exception as e:
        st.error(f"Error processing CSV: {str(e)}")