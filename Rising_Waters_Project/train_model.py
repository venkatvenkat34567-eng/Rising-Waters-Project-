import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

# Load dataset
df = pd.read_excel("flood dataset.xlsx")

# Features and target
X = df[[
    "Temp",
    "Humidity",
    "Cloud Cover",
    "ANNUAL",
    "Jan-Feb",
    "Mar-May",
    "Jun-Sep",
    "Oct-Dec",
    "avgjune",
    "sub"
]]

y = df["flood"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# Scale data
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train model
model = XGBClassifier(
    random_state=42,
    eval_metric="logloss"
)

model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print(f"Model Accuracy: {accuracy*100:.2f}%")

# Create model folder
os.makedirs("model", exist_ok=True)

# Save model and scaler
joblib.dump(model, "model/flood_model.pkl")
joblib.dump(scaler, "model/transform.save")

print("Model saved successfully.")
print("Scaler saved successfully.")