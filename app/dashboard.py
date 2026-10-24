import streamlit as st
import pandas as pd
from pathlib import Path
import joblib

st.set_page_config(
    page_title="AI E-Health",
    page_icon="🩺",
    layout="centered"
)

st.title("🩺 AI E-Health")
st.subheader("Diabetes Risk Prediction")
st.write(
    "Educational demonstration using a machine-learning model."
)

st.warning(
    "This application is for learning only. "
    "It does not diagnose diabetes or replace a healthcare professional."
)

st.header("Patient Information")

age = st.number_input("Age (years)", min_value=1, max_value=120, value=30)
pregnancies = st.number_input(
    "Number of pregnancies",
    min_value=0,
    max_value=20,
    value=0
)
glucose = st.number_input(
    "Glucose level",
    min_value=0,
    max_value=300,
    value=100
)
blood_pressure = st.number_input(
    "Blood pressure",
    min_value=0,
    max_value=200,
    value=70
)
skin_thickness = st.number_input(
    "Skin thickness",
    min_value=0,
    max_value=100,
    value=20
)
insulin = st.number_input(
    "Insulin",
    min_value=0,
    max_value=1000,
    value=80
)
bmi = st.number_input(
    "BMI",
    min_value=0.0,
    max_value=100.0,
    value=25.0
)
pedigree = st.number_input(
    "Diabetes pedigree function",
    min_value=0.0,
    max_value=3.0,
    value=0.5
)

# Load the trained model
ROOT_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT_DIR / "models" / "diabetes_model.joblib"

if MODEL_PATH.exists():
    model = joblib.load(MODEL_PATH)
else:
    model = None
    st.error("Trained model not found. Please run app/main.py first.")

if st.button("Estimate risk", type="primary"):
    if model is not None:
        # Keep the same feature names and order used during training
        patient_data = pd.DataFrame([{
            "Pregnancies": pregnancies,
            "Glucose": glucose,
            "BloodPressure": blood_pressure,
            "SkinThickness": skin_thickness,
            "Insulin": insulin,
            "BMI": bmi,
            "DiabetesPedigreeFunction": pedigree,
            "Age": age
        }])

        prediction = model.predict(patient_data)[0]
        probabilities = model.predict_proba(patient_data)[0]
        risk_probability = probabilities[1]

        st.subheader("Prediction result")

        if prediction == 1:
            st.warning("The model identified a pattern associated with diabetes.")
        else:
            st.success("The model did not identify a diabetes-associated pattern.")

        st.write(f"Model-estimated probability for class 1: {risk_probability:.1%}")
        st.caption(
            "Educational output only — this result is not a medical diagnosis "
            "and should not be used for healthcare decisions."
        )
