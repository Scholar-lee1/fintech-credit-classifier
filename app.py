import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Load model
try:
    with open("logistic_model.pkl", "rb") as f:
        model = pickle.load(f)
except FileNotFoundError:
    st.error("'logistic_model.pkl' not found!, make sure model is saved")

# Page layout
st.set_page_config(page_title="Credit Risk Evaluator", layout="centered")
st.title(" Smart Fintech Credit Risk Evaluator")
st.write("Enter the applicant's details below to evaluate loan eligibility")

# Input field
st.subheader("Applicant Information")

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    married = st.selectbox("Marital Status", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
    education = st.selectbox("Education", ["Graduate", "Not Graduate"])
    self_employed = st.selectbox("Self employed?", ["Yes", "No"])
    
with col2:
    applicant_income = st.number_input("Applicant Monthly Income ($)", min_value=0, value=5000, step=500)
    coapplicant_income = st.number_input("Co-applicant Monthly Income ($)", min_value=0, value=0, step=500)
    loan_amount = st.number_input("Loan Amount (in thousands, e.g., 150 for $ 150,000)", min_value=0, value=150, step=10)
    loan_term = st.selectbox("Loan Term (in Days)", [360, 180, 120, 84, 60])
    credit_history = st.selectbox("Credit History Guidelines?", ["Yes", "No"])

gender_mapped = 1 if gender == "Male" else 0
married_mapped = 1 if married == "Yes" else 0
education_mapped = 1 if education == "Graduate" else 0
self_employed_mapped = 1 if self_employed == "Yes" else 0
credit_history_mapped = 1 if credit_history == "Yes"else 0

if dependents == "3+":
    dependents_mapped = 3
else:
    dependents_mapped = int(dependents)

# Predict Button


if st.button("Evaluate Credit Application", use_container_width=True):
# Creating a dictionary that match the original pre-encoded CSV columns
    user_input = {
        'Gender': gender,
        'Married': married,
        'Dependents': dependents,
        'Education': education,
        'Self_Employed': self_employed,
        'ApplicantIncome': applicant_income,
        'CoapplicantIncome': coapplicant_income,
        'LoanAmount': loan_amount,
        'Loan_Amount_Term': loan_term,
        'Credit_History': credit_history_mapped
    }
    
    
    input_df = pd.DataFrame([user_input])
    
# Load data-set
    try:
        df_blueprint = pd.read_csv("train.csv")
        df_blueprint = df_blueprint.drop(columns=['Loan_ID', 'Loan_Status'], errors='ignore')
    except FileNotFoundError:
        st.error("'train.csv' not found! Please ensure it is in your project directory.")
        st.stop()
        
    # Combining, so all possible categories are known(panas get_dummies)
    combined = pd.concat([df_blueprint, input_df], ignore_index=True)
    combined_encoded = pd.get_dummies(combined)
    final_features = combined_encoded.iloc[[-1]].copy()
    
# FORCE alignment with model features in the exact order
    # filling thy missing value with zero
    expected_features = model.feature_names_in_
    final_features = final_features.reindex(columns=expected_features, fill_value=0)
    
# Predict
    prediction = model.predict(final_features)
    confidence = model.predict_proba(final_features)[0][1] * 100
    
# Display results
    st.subheader("Evaluation Result")
    if prediction[0] == 1:
        st.success("Loan Approved!!")
        st.write(f"The applicant is classified as Low Risk with a confidence score of {confidence:.1f}%")
        st.balloons()
    else:
        st.error("Loan Denied.")
        st.write(f"The applicant is classified as High Risk (Approval probability is only {confidence:.1f}%)")