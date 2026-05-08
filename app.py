from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import cv2
import numpy as np
import os

app = Flask(__name__)
CORS(app)

# --- Feature extraction logic ---
def analyze_sample(image_path):
    # This simulates the deep learning model logic from the repo
    # In a full setup, you would load your .h5 or .pkl file here
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

# --- 1. Added Home Route (Fixes the 404 Error) ---
@app.route('/', methods=['GET'])
def home():
    # This provides a simple UI directly from the backend for testing
    return render_template_string('''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Handwriting Analysis AI</title>
        <style>
            body { font-family: sans-serif; text-align: center; padding: 50px; background: #f4f7f6; }
            .card { background: white; padding: 30px; border-radius: 10px; display: inline-block; shadow: 0 4px 6px rgba(0,0,0,0.1); }
            h1 { color: #2c3e50; }
            button { background: #3498db; color: white; border: none; padding: 10px 20px; border-radius: 5px; cursor: pointer; }
        </style>
    </head>
    <body>
        <div class="card">
            <h1>GraphoAnalysis API is Live!</h1>
            <p>The backend is running successfully on Render.</p>
            <p>To test, use the <code>/predict</code> endpoint with a POST request.</p>
            <hr>
            <small>Status: <span style="color: green;">● Online</span></small>
        </div>
    </body>
    </html>
    ''')

# --- 2. Prediction Route ---
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
        
    file = request.files['file']
    
    # Ensure upload directory exists
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
        
    path = os.path.join("uploads", file.filename)
    file.save(path)
    
    try:
        results = analyze_sample(path)
        return jsonify(results)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # Render uses the PORT environment variable
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
