from flask import Flask, request, jsonify
import cv2
import numpy as np
import pandas as pd
import joblib

# Flask app
app = Flask(__name__)

# Load CSV
disease_info = pd.read_csv("plant_diseases.csv", encoding="ISO-8859-1")

@app.route('/predict', methods=['POST'])
def predict():
    # Get uploaded image
    file = request.files.get('image')
    if not file:
        return jsonify({"error": "No file uploaded"})

    # Preprocess the image
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    img = cv2.resize(img, (128, 128))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    features = hog(gray, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm='L2-Hys').reshape(1, -1)

    # Load selected model
    model_name = request.form.get('model', 'svm').lower()
    model_path = f"{model_name}_model.pkl"
    model = joblib.load(model_path)

    # Predict
    prediction = model.predict(features)[0]
    disease_name = class_names[prediction]

    # Fetch disease info
    disease_details = disease_info[disease_info['Disease Name'] == disease_name]
    if disease_details.empty:
        return jsonify({"error": "Disease information not found"})

    result = {
        "disease_name": disease_name,
        "description": disease_details['Description'].values[0],
        "prevention": disease_details['Prevention Steps'].values[0]
    }
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True)
