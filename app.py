from flask import Flask, request, render_template
import os
import cv2
import joblib
import pandas as pd
from skimage.feature import hog
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}
MODEL_PATH = 'model.pkl'
PCA_PATH = 'pca_transformer.pkl'
CSV_PATH = 'plant_diseases.csv'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load CSV data
def load_disease_info(csv_file):
    try:
        df = pd.read_csv(csv_file, encoding='ISO-8859-1')
        return {
            row['Diseases name']: {
                "description": row['description'].strip(),
                "prevention": row['prevention'].strip()
            }
            for _, row in df.iterrows()
        }
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        return {}

disease_info = load_disease_info(CSV_PATH)

# Validate file type
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Predict function
def test_single_image(image_path, model_path, pca_path, class_names):
    try:
        model = joblib.load(model_path)
        pca = joblib.load(pca_path)

        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (64, 64))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        features = hog(gray, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm='L2-Hys')

        features_pca = pca.transform([features])
        predicted_label = model.predict(features_pca)
        return class_names[predicted_label[0]]
    except Exception as e:
        print(f"Prediction error: {e}")
        return "Unknown"

@app.route('/')
def upload_file():
    return render_template('upload.html')

@app.route('/upload', methods=['POST'])
def handle_upload():
    if 'file' not in request.files:
        return render_template('upload.html', error="No file part in the request.")

    file = request.files['file']

    if file.filename == '':
        return render_template('upload.html', error="No file selected.")

    if file and allowed_file(file.filename):
        try:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            class_names = [
                "Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust", "Apple___healthy",
                "Blueberry___healthy", "Cherry_(including_sour)___healthy", "Cherry_(including_sour)___Powdery_mildew",
                "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot", "Corn_(maize)___Common_rust_", "Corn_(maize)___healthy",
                "Corn_(maize)___Northern_Leaf_Blight", "Grape___Black_rot", "Grape___Esca_(Black_Measles)", "Grape___healthy",
                "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)", "Orange___Haunglongbing_(Citrus_greening)",
                "Peach___Bacterial_spot", "Peach___healthy", "Pepper,_bell___Bacterial_spot", "Pepper,_bell___healthy",
                "Potato___Early_blight", "Potato___healthy", "Potato___Late_blight", "Raspberry___healthy",
                "Soybean___healthy", "Squash___Powdery_mildew", "Strawberry___healthy", "Strawberry___Leaf_scorch",
                "Tomato___Bacterial_spot", "Tomato___Early_blight", "Tomato___healthy", "Tomato___Late_blight",
                "Tomato___Leaf_Mold", "Tomato___Septoria_leaf_spot", "Tomato___Spider_mites Two-spotted_spider_mite",
                "Tomato___Target_Spot", "Tomato___Tomato_mosaic_virus", "Tomato___Tomato_Yellow_Leaf_Curl_Virus"
            ]

            prediction = test_single_image(file_path, MODEL_PATH, PCA_PATH, class_names)

            description = disease_info.get(prediction, {}).get('description', 'No description available.')
            prevention = disease_info.get(prediction, {}).get('prevention', 'No prevention advice available.')

            return render_template('upload.html', prediction=prediction, description=description, prevention=prevention, file_path=f"/{UPLOAD_FOLDER}/{filename}")
        except Exception as e:
            print(f"Error during file upload or prediction: {e}")
            return render_template('upload.html', error="An error occurred during prediction.")
    else:
        return render_template('upload.html', error="File type not allowed.")

if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(debug=True)