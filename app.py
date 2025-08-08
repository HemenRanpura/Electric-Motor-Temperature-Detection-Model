from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import sys

app = Flask(__name__)

# Load trained model and scaler with error handling
try:
    model = joblib.load('pm_temp_random_forest_model.pkl')
    scaler = joblib.load('pm_temp_scaler.pkl')
except FileNotFoundError:
    print("Error: Model or scaler file not found. Please run the training script first.")
    sys.exit(1)
except Exception as e:
    print(f"Error loading model or scaler: {e}")
    sys.exit(1)

# Home route to serve HTML
@app.route('/')
def index():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        features = ['u_q', 'u_d', 'i_q', 'i_d', 'motor_speed', 'torque', 
                    'coolant', 'ambient', 'stator_winding', 'stator_yoke', 'stator_tooth']
        if not all(feature in data for feature in features):
            missing = [f for f in features if f not in data]
            return jsonify({'error': f'Missing features: {", ".join(missing)}'}), 400
        values = [float(data[feature]) for feature in features]
        scaled_input = scaler.transform([values])
        prediction = model.predict(scaled_input)[0]
        return jsonify({'predicted_pm_temp': round(prediction, 2)})
    except ValueError as e:
        return jsonify({'error': 'Invalid input: All features must be numbers'}), 400
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True)