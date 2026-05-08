import gdown
import os

def load_actual_model():
    model_path = 'models.h5'
    if not os.path.exists(model_path):
        # Replace with your File ID from the Google Drive share link
        file_id = 'YOUR_GOOGLE_DRIVE_FILE_ID'
        url = f'https://drive.google.com/uc?id={file_id}'
        gdown.download(url, model_path, quiet=False)
    
    return tf.keras.models.load_model(model_path)

model = load_actual_model()
from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import cv2
import numpy as np
import os

app = Flask(__name__)
CORS(app)

# --- Logic Mockup (Replace with your actual model logic later) ---
def analyze_sample(image_path):
    return {
        "Emotional Stability": "High",
        "Will Power": "Moderate",
        "Modesty": "High",
        "Discipline": "High",
        "Concentration": "High",
        "Social Isolation": "Low"
    }

# --- The Visual Website UI ---
@app.route('/', methods=['GET'])
def home():
    return render_template_string('''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Handwriting Analysis AI</title>
        <script src="https://cdn.tailwindcss.com"></script>
    </head>
    <body class="bg-gray-100 min-h-screen flex items-center justify-center p-6">
        <div class="max-w-md w-full bg-white rounded-xl shadow-lg p-8">
            <h1 class="text-2xl font-bold text-gray-800 text-center mb-2">GraphoAnalysis AI</h1>
            <p class="text-gray-500 text-center mb-8">Upload your handwriting to analyze behavior.</p>
            
            <div class="space-y-4">
                <input type="file" id="handwritingFile" class="block w-full text-sm text-gray-500 file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-blue-50 file:text-blue-700 hover:file:bg-blue-100 cursor-pointer"/>
                <button onclick="uploadImage()" id="btn" class="w-full bg-blue-600 text-white py-2 rounded-lg font-semibold hover:bg-blue-700 transition">Analyze Now</button>
            </div>

            <div id="results" class="mt-8 hidden">
                <h3 class="font-bold text-gray-700 border-b pb-2 mb-4">Analysis Results:</h3>
                <ul id="traitList" class="space-y-2"></ul>
            </div>
        </div>

        <script>
            async function uploadImage() {
                const fileInput = document.getElementById('handwritingFile');
                const btn = document.getElementById('btn');
                if (!fileInput.files[0]) return alert("Please select a file first!");

                btn.innerText = "Analyzing...";
                btn.disabled = true;

                const formData = new FormData();
                formData.append('file', fileInput.files[0]);

                try {
                    const response = await fetch('/predict', { method: 'POST', body: formData });
                    const data = await response.json();
                    
                    const list = document.getElementById('traitList');
                    list.innerHTML = "";
                    for (const [trait, value] of Object.entries(data)) {
                        list.innerHTML += `<li class="flex justify-between bg-gray-50 p-2 rounded"><strong>${trait}:</strong> <span class="text-blue-600">${value}</span></li>`;
                    }
                    document.getElementById('results').classList.remove('hidden');
                } catch (e) {
                    alert("Error: " + e);
                } finally {
                    btn.innerText = "Analyze Now";
                    btn.disabled = false;
                }
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
    return jsonify(analyze_sample(path))

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))
