from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import cv2
import numpy as np
import os
import tensorflow as tf
import gdown

app = Flask(__name__)
CORS(app)

# --- MODEL LOADING LOGIC ---
def load_project_model():
    model_path = 'models.h5'
    
    # If the model isn't there, download it from Google Drive
    if not os.path.exists(model_path):
        print("Model not found. Downloading from Google Drive...")
        # Replace the string below with your 52MB file's ID
        # Example ID looks like: 1A2B3C4D5E6F7G8H9I0J
        file_id = 'YOUR_FILE_ID_HERE' 
        url = f'https://drive.google.com/uc?id={file_id}'
        gdown.download(url, model_path, quiet=False)
    
    # Load with compile=False to save RAM on Render's free tier
    return tf.keras.models.load_model(model_path, compile=False)

# Load the model once when the server starts
model = load_project_model()

# --- PRE-PROCESSING (From griffith.ipynb) ---
def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (128, 128)) # Ensure this matches your training size
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=(0, -1))
    return img

# --- ROUTES ---

@app.route('/', methods=['GET'])
def home():
    # This serves the UI with the "Analyze Now" button
    return render_template_string('''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Handwriting Analysis AI</title>
        <script src="https://cdn.tailwindcss.com"></script>
    </head>
    <body class="bg-gray-100 min-h-screen flex items-center justify-center p-6">
        <div class="max-w-md w-full bg-white rounded-xl shadow-lg p-8">
            <h1 class="text-2xl font-bold text-gray-800 text-center mb-6">GraphoAnalysis AI</h1>
            <input type="file" id="fileInput" class="mb-4 block w-full text-sm text-gray-500 file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:bg-blue-50 file:text-blue-700 hover:file:bg-blue-100"/>
            <button onclick="analyze()" id="btn" class="w-full bg-blue-600 text-white py-2 rounded-lg font-semibold hover:bg-blue-700 transition">Analyze Now</button>
            <div id="results" class="mt-6 hidden">
                <h3 class="font-bold text-gray-700 border-b pb-2 mb-2">Traits Detected:</h3>
                <ul id="list" class="space-y-1 text-sm text-gray-600"></ul>
            </div>
        </div>
        <script>
            async function analyze() {
                const file = document.getElementById('fileInput').files[0];
                if(!file) return alert("Select a file!");
                const btn = document.getElementById('btn');
                btn.innerText = "Processing...";
                const formData = new FormData();
                formData.append('file', file);
                const res = await fetch('/predict', { method: 'POST', body: formData });
                const data = await res.json();
                const list = document.getElementById('list');
                list.innerHTML = "";
                for(let [k,v] of Object.entries(data)) {
                    list.innerHTML += `<li class="flex justify-between"><b>${k}:</b> <span>${v}</span></li>`;
                }
                document.getElementById('results').classList.remove('hidden');
                btn.innerText = "Analyze Now";
            }
        </script>
    </body>
    </html>
    ''')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['file']
    if not os.path.exists('uploads'): os.makedirs('uploads')
    path = os.path.join("uploads", file.filename)
    file.save(path)

    try:
        processed_img = preprocess_image(path)
        prediction = model.predict(processed_img)
        
        # MAPPING: Change these based on your model's 8 traits
        results = {
            "Emotional Stability": "High" if prediction[0][0] > 0.5 else "Moderate",
            "Will Power": "Strong" if prediction[0][1] > 0.5 else "Average",
            "Social Isolation": "Low" if prediction[0][2] < 0.5 else "High",
            "Discipline": "High" if prediction[0][3] > 0.5 else "Standard"
        }
        return jsonify(results)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))
