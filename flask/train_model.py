# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pickle
import os
from pathlib import Path

def load_and_preprocess_data(data_path):
    """Load and preprocess the patient data"""
    try:
        df = pd.read_csv(data_path)
        df.rename(columns={'C': 'Gender'}, inplace=True)
        
        # Columns to encode
        columns = ['Gender', 'Age', 'History', 'Patient', 'TakeMedication',
                  'Severity', 'BreathShortness', 'VisualChanges',
                  'NoseBleeding', 'Whendiagnoused', 'Systolic',
                  'Diastolic', 'ControlledDiet', 'Stages']
        
        # Label encoding
        label_encoder = LabelEncoder()
        for col in columns:
            df[col] = label_encoder.fit_transform(df[col])
        
        return df
    except Exception as e:
        print(f"Error in preprocessing: {str(e)}")
        return None

def train_model(df):
    """Train and evaluate the Random Forest model"""
    X = df.drop('Stages', axis=1)
    y = df['Stages']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    print(f"Model Accuracy: {acc:.2f}")
    print("Classification Report:")
    print(report)
    
    return model

def save_model(model, filename='model.pkl'):
    """Save the trained model to a file"""
    try:
        with open(filename, 'wb') as f:
            pickle.dump(model, f)
        print(f"Model saved successfully as {filename}")
    except Exception as e:
        print(f"Error saving model: {str(e)}")

if __name__ == "__main__":
    # Get the path to the data file
    BASE_DIR = Path(__file__).parent
    data_path = BASE_DIR / "patient_data.csv"
    
    print(f"Loading data from: {data_path}")
    
    if not data_path.exists():
        print("Error: Data file not found!")
        print("Current directory contents:")
        for item in BASE_DIR.iterdir():
            print(f" - {item.name}")
    else:
        # Load and preprocess data
        df = load_and_preprocess_data(data_path)
        
        if df is not None:
            # Train model
            model = train_model(df)
            
            # Save model
            save_model(model)
            
           
