import pandas as pd
import joblib

from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Project paths
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT_DIR / "data" / "diabetes.csv"
MODEL_DIR = ROOT_DIR / "models"
MODEL_PATH = MODEL_DIR / "diabetes_model.joblib"

# 1. Load the dataset
data = pd.read_csv(DATA_PATH)

# 2. Separate features and target
X = data.drop(columns=["Outcome"])
y = data["Outcome"]

# 3. Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 4. Create the model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight="balanced"
)

# 5. Train the model
model.fit(X_train, y_train)

# 6. Evaluate the model
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("=== Diabetes Risk Prediction ===")
print("Model training completed!")
print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")
print(f"Test accuracy: {accuracy:.2%}")

print("\n=== Classification Report ===")
print(classification_report(y_test, predictions))

# 7. Save the model
MODEL_DIR.mkdir(parents=True, exist_ok=True)
joblib.dump(model, MODEL_PATH)

print(f"\nModel saved successfully: {MODEL_PATH}")
print("Educational use only - not for medical diagnosis.")