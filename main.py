import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os
import sklearn
from pathlib import Path

# Model loading with proper error handling
@st.cache_resource
def load_model(model_path):
    try:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
        
        # Add version checking before loading
        import sklearn
        current_version = sklearn.__version__
        st.info(f"Current scikit-learn version: {current_version}")
        
        model = joblib.load(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

# Configuration
MODEL_PATH = "churn_xgb_pipeline.joblib"
PROBABILITY_THRESHOLD = 0.45

# Load model
model = load_model(MODEL_PATH)

if model is None:
    st.error("Failed to load the model. Please check if the model file exists.")
    st.stop()

# Title and description
st.title("Customer Churn Predictor")
st.write("Enter customer details to predict the likelihood of churn.")

# Input form
with st.form(key='churn_form'):
    col1, col2 = st.columns(2)
    
    with col1:
        tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=1)
        monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=200.0, value=50.0)
        total_charges = st.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0, value=100.0)
        num_services = st.number_input("Number of services", min_value=0, max_value=9, value=1)
    
    with col2:
        gender = st.selectbox("Gender", ["Male", "Female"])
        tenure_bucket = st.selectbox("Tenure Bucket", ["1-12", "13-24", "25-36", "37-48", "49-60", "61-72"])
        partner_status = st.selectbox("Partner Status", ["Yes", "No"])
        contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])

    # Service related inputs
    st.subheader("Service Information")
    col3, col4 = st.columns(2)
    
    with col3:
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
        tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
        phone_service = st.selectbox("Phone Service", ["Yes", "No"])
    
    with col4:
        payment_method = st.selectbox("Payment Method", 
            ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
        multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
        online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
        device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])

    # Additional features
    st.subheader("Additional Information")
    col5, col6 = st.columns(2)
    
    with col5:
        streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
        streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
    
    with col6:
        senior_citizen = st.selectbox("Senior Citizen", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["Yes", "No"])
        paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])

    submit_button = st.form_submit_button(label='Predict Churn')

def preprocess_input_data(df):
    """Preprocess input data to match model requirements"""
    # Convert Yes/No to 1/0 for binary fields
    binary_columns = ['Partner', 'PhoneService', 'PaperlessBilling', 'Dependents']
    for col in binary_columns:
        df[col] = df[col].map({'Yes': 1, 'No': 0})
    
    # Convert SeniorCitizen to numeric
    df['SeniorCitizen'] = df['SeniorCitizen'].map({'Yes': 1, 'No': 0})
    
    # Handle special cases for internet-related services
    internet_cols = ['OnlineSecurity', 'TechSupport', 'OnlineBackup', 
                    'DeviceProtection', 'StreamingTV', 'StreamingMovies']
    for col in internet_cols:
        df[col] = df[col].map({
            'Yes': 1, 
            'No': 0, 
            'No internet service': 0
        })
    
    # Handle MultipleLines
    df['MultipleLines'] = df['MultipleLines'].map({
        'Yes': 1, 
        'No': 0, 
        'No phone service': 0
    })
    
    return df

# Prediction logic
if submit_button:
    try:
        # Create input DataFrame
        input_data = pd.DataFrame({
            'tenure': [tenure],
            'MonthlyCharges': [monthly_charges],
            'TotalCharges': [total_charges],
            'gender': [gender],
            'tenure_bucket': [tenure_bucket],
            'Partner': [partner_status],
            'num_services': [num_services],
            'Contract': [contract],
            'InternetService': [internet_service],
            'PaymentMethod': [payment_method],
            'OnlineSecurity': [online_security],
            'TechSupport': [tech_support],
            'PhoneService': [phone_service],
            'MultipleLines': [multiple_lines],
            'OnlineBackup': [online_backup],
            'DeviceProtection': [device_protection],
            'StreamingTV': [streaming_tv],
            'StreamingMovies': [streaming_movies],
            'SeniorCitizen': [senior_citizen],
            'Dependents': [dependents],
            'PaperlessBilling': [paperless_billing]
        })

        # Preprocess the input data
        input_data = preprocess_input_data(input_data)
        
        # Make prediction
        probability = model.predict_proba(input_data)[:, 1][0]
        prediction = "Yes" if probability >= PROBABILITY_THRESHOLD else "No"
        
        # Display results
        st.success("Prediction completed successfully!")
        col7, col8 = st.columns(2)
        with col7:
            st.metric(label="Churn Probability", value=f"{probability:.2%}")
        with col8:
            st.metric(label="Predicted Churn", value=prediction)
            
        # Additional insights
        if prediction == "Yes":
            st.warning("⚠️ This customer is at risk of churning!")
            st.write("Consider implementing retention strategies.")
        else:
            st.info("✅ This customer is likely to stay!")
            
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        st.write("Please check if all input values are correct and try again.")