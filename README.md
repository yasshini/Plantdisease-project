# Plant Disease Detection

This project is a Flask web application that lets users upload an image of a plant leaf and predict the disease using a trained machine learning model. The app also displays a short description of the disease and prevention guidance.

## Features
- Upload JPG, JPEG, or PNG plant leaf images
- Predict plant disease using a saved ML model
- Show disease description and prevention steps
- Simple and lightweight web interface

## Project Structure
- `app.py` — Flask application and prediction logic
- `templates/upload.html` — web page for file upload and result display
- `static/styles/style.css` — styling for the UI
- `static/uploads/` — uploaded image folder created automatically at runtime
- `model.pkl` — trained ML model
- `pca_transformer.pkl` — PCA transformer used during prediction
- `plant_diseases.csv` — disease descriptions and prevention text
- `training.ipynb` — training notebook for the model

## Requirements
- Python 3.10 or newer
- pip
- A virtual environment is recommended

## Install
1. Open a terminal in the project folder.
2. Create a virtual environment:

   Windows:
   ```bash
   python -m venv .venv
   .\.venv\Scripts\activate
   ```

   macOS/Linux:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```
4. Download datset : https://drive.google.com/file/d/1BlVARo-EFWEhMvV9bOwdvswDcqM1LXbG/view?usp=sharing
## Run the App
From the project folder, run:

```bash
python app.py
```

Then open this URL in your browser:

```text
http://127.0.0.1:5000/
```

## Notes
- The app automatically creates the `static/uploads` folder when started.
- Supported file types are `jpg`, `jpeg`, and `png`.
- If you are using a different Python installation, replace `python` with `py` on Windows or `python3` on Linux/macOS.

## Troubleshooting
- If `ModuleNotFoundError` appears, reactivate the virtual environment and reinstall the dependencies.
- If the app does not open on the browser, confirm the Flask server is running without errors in the terminal.
- If the upload fails, check that the image is a valid JPG/PNG/JPEG file.
