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
