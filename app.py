from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import cv2
import numpy as np
import os
import tensorflow as tf

app = Flask(__name__)
CORS(app)

# 1. Load your optimized 52MB model
# We use compile=False to save even more RAM
try:
    model = tf.keras.models.load_model('models.h5', compile=False)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error: {e}")

def preprocess_image(image_path):
    # This must match your griffith.ipynb training dimensions
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (128, 128)) 
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=(0, -1))
    return img

@app.route('/', methods=['GET'])
def home():
    # Returns the UI we built
    return render_template_string(open('index.html').read())

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
        
        # Adjust these keys based on your specific 8 traits
        results = {
            "Emotional Stability": "High" if prediction[0][0] > 0.5 else "Moderate",
            "Mental Energy": "High" if prediction[0][1] > 0.5 else "Average",
            "Social Orientation": "Extrovert" if prediction[0][2] > 0.5 else "Introvert"
        }
        return jsonify(results)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))
