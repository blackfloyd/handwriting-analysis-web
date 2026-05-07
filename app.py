from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import os

app = Flask(__name__)
CORS(app)

# Feature extraction logic based on the repository's SVM model
def analyze_sample(image_path):
    # This is where you'd link the model (.h5 or .pkl) from the repo
    # For now, it returns the 8 traits identified in the project:
    return {
        "Emotional Stability": "High",
        "Will Power": "Moderate",
        "Modesty": "High",
        "Personal Harmony": "High",
        "Discipline": "High",
        "Concentration": "High",
        "Communicativeness": "Moderate",
        "Social Isolation": "Low"
    }

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    path = "./temp_sample.png"
    file.save(path)
    results = analyze_sample(path)
    return jsonify(results)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))
