import streamlit as st
import pickle
import pandas as pd
import numpy as np

st.set_page_config(page_title="Employee Attrition Prediction")

st.title("👨‍💼 Employee Attrition Prediction")

# Load model
with open("employee_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load encoder
with open("nominal_encoder.pkl", "rb") as f:
    ohe = pickle.load(f)

# ---------- USER INPUT ----------
education = st.selectbox("Education", ["Below College", "College", "Bachelor", "Master", "Doctor"])
city = st.selectbox("City", ["Bangalore", "Pune", "Delhi", "Mumbai", "Chennai"])
gender = st.selectbox("Gender", ["Male", "Female"])
bench = st.selectbox("Ever Benched", ["Yes", "No"])

joining_year = st.number_input("Joining Year", 2000, 2030)
payment = st.number_input("Payment", 0)
age = st.number_input("Age", 18, 65)
experience = st.number_input("Experience", 0, 40)

# 🔥 COLUMN NAMES MUST MATCH TRAINING DATA EXACTLY
input_df = pd.DataFrame({
    "Education": [education],
    "City": [city],
    "Gender": [gender],
    "EverBenched": [bench],   # ← adjust if encoder expects different name
    "JoiningYear": [joining_year],
    "Payment": [payment],
    "Age": [age],
    "Experience": [experience]
})

if st.button("Predict"):

    # Nominal columns EXACTLY as encoder expects
    nominal_cols = list(ohe.feature_names_in_)
    X_nominal = ohe.transform(input_df[nominal_cols])

    # Numeric columns
    X_numeric = input_df[["JoiningYear", "Payment", "Age", "Experience"]].values

    X_final = np.hstack((X_nominal, X_numeric))

    prediction = model.predict(X_final)

    if prediction[0] == 1:
        st.error("❌ Employee will LEAVE")
    else:
        st.success("✅ Employee will STAY")
