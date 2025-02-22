import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Simulated dataset (age, cholesterol, blood pressure, glucose, risk_label)
data = {
    'Age': [45, 60, 50, 30, 70, 55, 40, 65, 75, 35],
    'Cholesterol': [220, 250, 230, 180, 270, 240, 210, 260, 280, 190],
    'BloodPressure': [130, 140, 135, 120, 150, 138, 125, 145, 155, 118],
    'Glucose': [90, 110, 105, 85, 130, 115, 95, 125, 140, 80],
    'Risk': [1, 1, 1, 0, 1, 1, 0, 1, 1, 0]  # 1 = High Risk, 0 = Low Risk
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Split data into features and labels
X = df[['Age', 'Cholesterol', 'BloodPressure', 'Glucose']]
y = df['Risk']

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy)
print("Classification Report:\n", classification_report(y_test, y_pred))

# Predict risk for a new patient
new_patient = np.array([[50, 225, 128, 100]])  # Age, Cholesterol, BP, Glucose
new_patient_scaled = scaler.transform(new_patient)
risk_prediction = model.predict(new_patient_scaled)
print("Predicted Risk:", "High" if risk_prediction[0] == 1 else "Low")