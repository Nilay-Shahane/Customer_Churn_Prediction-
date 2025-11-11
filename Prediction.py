import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load dataset
df_churn = pd.read_csv("Churn_Data.csv")

# Drop unnecessary columns
df_churn.drop(['Surname', 'Gender', 'RowNumber', 'CustomerId'], axis=1, inplace=True)

# Convert categorical columns to dummy variables
df_churn = pd.get_dummies(df_churn, drop_first=True)  # 'Geography' → dummy columns

# Split features and target
X = df_churn.drop("Exited", axis=1)
y = df_churn['Exited']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)

# Scale data
sc = StandardScaler()
X_rftrain = sc.fit_transform(X_train)
X_rftest = sc.transform(X_test)

# Train Random Forest model
rfc = RandomForestClassifier(
    criterion='gini',
    n_estimators=100,
    random_state=10,
    max_features='sqrt'
)
rfc.fit(X_rftrain, y_train)

# Predictions and evaluation
rfc_pred_test = rfc.predict(X_rftest)
print("Confusion Matrix:\n", confusion_matrix(y_test, rfc_pred_test))
accuracy_rf = accuracy_score(y_test, rfc_pred_test) * 100
print(f"The accuracy score of the Random Forest model is: {accuracy_rf:.2f}%")
print(classification_report(y_test, rfc_pred_test))

# 🔹 Refit scaler and model on full data for deployment
scaler_final = StandardScaler()
X_scaled = scaler_final.fit_transform(X)

model_final = RandomForestClassifier(
    criterion='gini',
    n_estimators=100,
    random_state=10,
    max_features='sqrt'
)
model_final.fit(X_scaled, y)

# 🔹 Save both model and scaler
with open('customer_churn_model.pkl', 'wb') as model_file:
    pickle.dump(model_final, model_file)

with open('scaler.pkl', 'wb') as scaler_file:
    pickle.dump(scaler_final, scaler_file)

print("✅ Model and Scaler saved successfully!")
