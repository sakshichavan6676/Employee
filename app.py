import streamlit as st
import pickle
import pandas as pd
import numpy as np

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Employee Attrition Prediction",
    layout="centered"
)

st.title("👨‍💼 Employee Attrition Prediction")
st.write("Predict whether an employee will leave the organization")

# ---------------- LOAD MODEL & ENCODER ----------------
with open("employee_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("nominal_encoder.pkl", "rb") as f:
    ohe = pickle.load(f)

# Safety check (prevents silent crashes)
if not hasattr(ohe, "transform"):
    st.error("❌ Loaded encoder is not a valid OneHotEncoder")
    st.stop()

# ---------------- USER INPUT UI ----------------
education = st.selectbox(
    "Education Level",
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

joining_year = st.number_input(
    "Joining Year",
    min_value=2000,
    max_value=2030,
    step=1
)

payment = st.number_input(
    "Payment",
    min_value=0
)

age = st.number_input(
    "Age",
    min_value=18,
    max_value=65
)

experience = st.number_input(
    "Experience (Years)",
    min_value=0,
    max_value=40
)

# ---------------- CREATE INPUT DATAFRAME ----------------
input_df = pd.DataFrame({
    "Education": [education],
    "City": [city],
    "Gender": [gender],
    "BenchEver": [bench],
    "JoiningYear": [joining_year],
    "Payment": [payment],
    "Age": [age],
    "Experience": [experience]
})

# ---------------- PREDICTION ----------------
if st.button("🔮 Predict"):

    # ✅ FIX: enforce same column order as training
    nominal_cols = list(ohe.feature_names_in_)
    input_nominal = input_df[nominal_cols]

    # Encode nominal features
    X_nominal = ohe.transform(input_nominal)

    # Numeric features
    numeric_cols = ["JoiningYear", "Payment", "Age", "Experience"]
    X_numeric = input_df[numeric_cols].values

    # Final input
    X_final = np.hstack((X_nominal, X_numeric))

    # Predict
    prediction = model.predict(X_final)

    # Output
    if prediction[0] == 1:
        st.error("❌ Employee is likely to LEAVE")
    else:
        st.success("✅ Employee is likely to STAY")
