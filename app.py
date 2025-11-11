from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)


with open('customer_churn_model.pkl', 'rb') as file:
    model = pickle.load(file)
with open('scaler.pkl', 'rb') as scaler_file:
    sc = pickle.load(scaler_file)

feature_order = [
    'CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts',
    'HasCrCard', 'IsActiveMember', 'EstimatedSalary',
    'Geography_Germany', 'Geography_Spain'
]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predictions():
    try:
        # Get basic numeric fields
        CreditScore = float(request.form['CreditScore'])
        Age = float(request.form['Age'])
        Tenure = float(request.form['Tenure'])
        Balance = float(request.form['Balance'])
        NumOfProducts = float(request.form['NumOfProducts'])
        HasCrCard = float(request.form['HasCrCard'])
        IsActiveMember = float(request.form['IsActiveMember'])
        EstimatedSalary = float(request.form['EstimatedSalary'])

        # Handle Geography select box (values are comma separated like "1,0")
        geo_value = request.form['Geography']
        geo_germany, geo_spain = map(float, geo_value.split(','))

        # Combine all features in the correct order
        data = [[
            CreditScore, Age, Tenure, Balance, NumOfProducts,
            HasCrCard, IsActiveMember, EstimatedSalary,
            geo_germany, geo_spain
        ]]

        input_df = pd.DataFrame(data, columns=feature_order)
        scaled_input = sc.transform(input_df)

        prediction = model.predict(scaled_input)[0]
        probability = model.predict_proba(scaled_input)[0][1]

        if prediction == 1:
            result = f"Customer is likely to CHURN (probability: {probability:.2f})"
        else:
            result = f"Customer is NOT likely to churn (probability: {probability:.2f})"

        return render_template('index.html', prediction_text=result)

    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {e}")

if __name__ == '__main__':
    app.run(debug=True)
