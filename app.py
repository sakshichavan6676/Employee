import streamlit as st
import pickle
import pandas as pd
import numpy as np

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Employee Prediction App",
    layout="centered"
)

st.title("👨‍💼 Employee Prediction System")
st.write("Fill the employee details to get prediction")

# ---------------- LOAD MODEL & ENCODER ----------------
with open("employee_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("nominal_encoder.pkl", "rb") as f:
    ohe = pickle.load(f)

# Nominal columns (MUST match training)
nominal_cols = ['Education', 'City', 'Gender', 'BenchEver']

# ---------------- USER INPUT ----------------
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

# ---------------- CREATE INPUT DATAFRAME ----------------
input_df = pd.DataFrame({
    'Education': [education],
    'City': [city],
    'Gender': [gender],
    'BenchEver': [bench]
})

# ---------------- PREDICTION ----------------
if st.button("🚀 Predict"):

    # Encode nominal features
    X_encoded = ohe.transform(input_df[nominal_cols])

    # Predict
    prediction = model.predict(X_encoded)

    # If classification model
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(X_encoded)[0][1]
        st.success(f"✅ Prediction: **{prediction[0]}**")
        st.info(f"📊 Probability: **{prob:.2f}**")
    else:
        st.success(f"✅ Prediction Result: **{prediction[0]}**")
