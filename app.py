import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load the trained model and preprocessing objects
model = joblib.load('LR_Model.joblib')
label_enc = LabelEncoder()
scaler = StandardScaler()

# Define a function to preprocess input data
def preprocess_input(age, sleep_duration, quality_of_sleep, physical_activity_level,
                     stress_level, heart_rate, daily_steps, gender, occupation, bmi_category, blood_pressure):
    
    # Encode categorical features
    gender_encoded = label_enc.fit_transform([gender])[0]
    occupation_encoded = label_enc.fit_transform([occupation])[0]
    bmi_category_encoded = label_enc.fit_transform([bmi_category])[0]
    blood_pressure_encoded = label_enc.fit_transform([blood_pressure])[0]
    
    # Scale numeric features
    numeric_data = np.array([[age, sleep_duration, quality_of_sleep, physical_activity_level,
                              stress_level, heart_rate, daily_steps]])
    numeric_scaled = scaler.fit_transform(numeric_data)
    
    # Combine encoded categorical and scaled numeric features
    processed_data = np.concatenate((numeric_scaled, [[gender_encoded, occupation_encoded,
                                                       bmi_category_encoded, blood_pressure_encoded]]), axis=1)
    
    return processed_data

@app.route('/predict_sleep_disorder', methods=['POST'])
def predict_sleep_disorder():
    try:
        # Retrieve input data from the JSON request
        data = request.get_json()

        age = float(data['age'])
        sleep_duration = float(data['sleep_duration'])
        quality_of_sleep = float(data['quality_of_sleep'])
        physical_activity_level = float(data['physical_activity_level'])
        stress_level = float(data['stress_level'])
        heart_rate = float(data['heart_rate'])
        daily_steps = float(data['daily_steps'])
        gender = data['gender']
        occupation = data['occupation']
        bmi_category = data['bmi_category']
        blood_pressure = data['blood_pressure']
        
        # Preprocess input data
        processed_data = preprocess_input(age, sleep_duration, quality_of_sleep, physical_activity_level,
                                          stress_level, heart_rate, daily_steps, gender, occupation,
                                          bmi_category, blood_pressure)
        
        print("Input Data (Scaled and Encoded):")
        print(processed_data)
        
        # Predict sleep disorder using the loaded model
        predicted_disorder = model.predict(processed_data)

        # Prepare the response
        response = {'sleep_disorder': int(predicted_disorder[0])}
        return jsonify(response)
    
    except Exception as e:
        # Handle any errors that occur
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
