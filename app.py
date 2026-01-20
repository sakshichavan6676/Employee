import streamlit as st
import pickle
import pandas as pd
import numpy as np

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Employee Attrition Prediction", layout="centered")

st.title("👨‍💼 Employee Attrition Prediction")
st.write("Predict whether an employee will leave the company")

# ---------------- LOAD MODEL & ENCODER ----------------
with open("employee_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("nominal_encoder.pkl", "rb") as f:
    ohe = pickle.load(f)

# Nominal columns (must match training)
nominal_cols = ['Education', 'City', 'Gender', 'BenchEver']

# ---------------- USER INPUT ----------------
st.subheader("Employee Details")

education = st.selectbox(
    "Education",
    ["Below College", "College", "Bachelor", "Master", "Doctor"]
)

city = st.selectbox(
    "City",
    ["Bangalore", "Pune", "Delhi", "Mumbai", "Chennai"]
)

gender = st.selectbox(
    "Gender",
    ["Male", "Female"]
)

bench = st.selectbox(
    "BenchEver",
    ["Yes", "No"]
)

joining_year = st.number_input("Joining Year", min_value=2000, max_value=2030, step=1)
payment = st.number_input("Payment (Salary)", min_value=0)
age = st.number_input("Age", min_value=18, max_value=65)
experience = st.number_input("Experience (Years)", min_value=0, max_value=40)

# ---------------- CREATE INPUT DATAFRAME ----------------
input_df = pd.DataFrame({
    'Education': [education],
    'City': [city],
    'Gender': [gender],
    'BenchEver': [bench],
    'JoiningYear': [joining_year],
    'Payment': [payment],
    'Age': [age],
    'Experience': [experience]
})

# ---------------- ENCODING ----------------
if st.button("🚀 Predict Leave Status"):

    # Encode nominal features
    X_nominal = ohe.transform(input_df[nominal_cols])

    # Numeric features (same order as training)
    X_numeric = input_df[['JoiningYear', 'Payment', 'Age', 'Experience']].values

    # Combine features
    X_final = np.hstack([X_nominal, X_numeric])

    # ---------------- PREDICTION ----------------
    prediction = model.predict(X_final)

    # Classification output
    result = "Yes (Employee will leave)" if prediction[0] == 1 else "No (Employee will stay)"

    st.success(f"📌 Prediction: **{result}**")

    # Probability (if model supports it)
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(X_final)[0][1]
        st.info(f"📊 Probability of Leaving: **{prob:.2%}**")
