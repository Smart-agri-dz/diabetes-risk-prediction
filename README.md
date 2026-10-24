
# AI E-Health — Diabetes Risk Prediction

## About the Project
This project is an educational machine-learning application
that predicts diabetes-related patterns using patient data.

It uses a Random Forest classifier and a Streamlit interface.

## Features
- Train a machine-learning model
- Estimate predictions from patient input
- Display results in a simple web interface

## Technologies
- Python
- Pandas
- Scikit-learn
- Streamlit
- Joblib

## Dataset
The project uses a diabetes dataset containing patient features
such as glucose, blood pressure, BMI, age, and an outcome label.

## Installation

Install the required packages:

```bash
pip install -r requirements.txt
```

## Run the Application

Start the Streamlit dashboard:

```bash
python -m streamlit run app/dashboard.py
```

## Train the Model

To train and save the model:

```bash
python app/main.py
```

## Disclaimer
This application is for educational purposes only.
It does not diagnose diabetes or replace a healthcare professional.
Do not use its predictions for medical decisions.
