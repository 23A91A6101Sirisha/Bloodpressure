from flask import Flask, request, render_template
import pandas as pd
import pickle
import os
from pathlib import Path

app = Flask(__name__)

# Load model
try:
    model_path = Path(r"D:\Flask\model.pkl")  # Updated path
    model = pickle.load(open(model_path, 'rb'))
    print("✓ Model loaded successfully")
except Exception as e:
    model = None
    print(f"✗ Error loading model: {str(e)}")

# Prediction mapping
RESULT_MAP = {
    0: "NORMAL",
    1: "HYPERTENSION (Stage-1)",
    2: "HYPERTENSION (Stage-2)",
    3: "HYPERTENSIVE CRISIS"
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict')  # Changed from predict_form to match your file name
def predict_form():
    return render_template('predict.html')  # Changed to predict.html

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return render_template('result.html', 
                            prediction_text="Model not loaded. Please check server logs.")
    
    try:
        # Get form data
        form_data = {
            'Gender': float(request.form['Gender']),
            'Age': float(request.form['Age']),
            'History': float(request.form.get('History', 0)),
            'Patient': float(request.form['Patient']),
            'TakeMedication': float(request.form.get('TakeMedication', 0)),
            'Severity': float(request.form['Severity']),
            'BreathShortness': float(request.form['BreathShortness']),
            'VisualChanges': float(request.form['VisualChanges']),
            'NoseBleeding': float(request.form['NoseBleeding']),
            'Whendiagnoused': float(request.form['Whendiagnoused']),
            'Systolic': float(request.form['Systolic']),
            'Diastolic': float(request.form['Diastolic']),
            'ControlledDiet': float(request.form['ControlledDiet'])
        }
        
        # Create DataFrame with correct feature order
        features = ['Gender', 'Age', 'History', 'Patient', 'TakeMedication',
                   'Severity', 'BreathShortness', 'VisualChanges',
                   'NoseBleeding', 'Whendiagnoused', 'Systolic',
                   'Diastolic', 'ControlledDiet']
        
        input_df = pd.DataFrame([form_data], columns=features)
        
        # Make prediction
        prediction = model.predict(input_df)[0]
        result = RESULT_MAP.get(prediction, "Unknown stage")
        
        return render_template('result.html', prediction_text=result)
    
    except Exception as e:
        return render_template('result.html', 
                            prediction_text=f"Error: {str(e)}")

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
