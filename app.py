import os
# Suppress TensorFlow logs if they exist
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

from flask import Flask, request, render_template, jsonify
import numpy as np
import pandas as pd
import pickle

app = Flask(__name__)

# Load model and scaler using relative paths for Render
try:
    with open('model/model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('model/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    print("Model and Scaler loaded successfully!")
except Exception as e:
    model = None
    scaler = None
    print(f"CRITICAL ERROR: Could not load models. Error: {e}")

MODEL_COLUMNS = [
    'Tenure Months', 'Monthly Charges', 'CLTV',
    'Gender_Female', 'Gender_Male', 
    'Senior Citizen_No', 'Senior Citizen_Yes', 
    'Partner_No', 'Partner_Yes', 
    'Dependents_No', 'Dependents_Yes', 
    'Phone Service_No', 'Phone Service_Yes', 
    'Multiple Lines_No', 'Multiple Lines_No phone service', 'Multiple Lines_Yes', 
    'Internet Service_DSL', 'Internet Service_Fiber optic', 'Internet Service_No', 
    'Online Security_No', 'Online Security_No internet service', 'Online Security_Yes', 
    'Online Backup_No', 'Online Backup_No internet service', 'Online Backup_Yes', 
    'Device Protection_No', 'Device Protection_No internet service', 'Device Protection_Yes', 
    'Tech Support_No', 'Tech Support_No internet service', 'Tech Support_Yes', 
    'Streaming TV_No', 'Streaming TV_No internet service', 'Streaming TV_Yes', 
    'Streaming Movies_No', 'Streaming Movies_No internet service', 'Streaming Movies_Yes', 
    'Contract_Month-to-month', 'Contract_One year', 'Contract_Two year', 
    'Paperless Billing_No', 'Paperless Billing_Yes', 
    'Payment Method_Bank transfer (automatic)', 'Payment Method_Credit card (automatic)', 
    'Payment Method_Electronic check', 'Payment Method_Mailed check'
]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None or scaler is None:
        return jsonify({'error': 'Model files not found on server.'}), 500
    
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data received'}), 400

        # Create DataFrame and Process
        df = pd.DataFrame([data])
        df_encoded = pd.get_dummies(df)
        df_encoded = df_encoded.reindex(columns=MODEL_COLUMNS, fill_value=0)
        
        # Ensure all columns are numeric
        df_encoded = df_encoded.apply(pd.to_numeric)
        
        input_scaled = scaler.transform(df_encoded)
        prediction = int(model.predict(input_scaled)[0])

        return jsonify({'prediction': prediction})

    except Exception as e:
        print(f"Prediction Error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)