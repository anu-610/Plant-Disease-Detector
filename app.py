import os
import tensorflow as tf
from flask import Flask, request, render_template, jsonify
from PIL import Image
import numpy as np
import google.generativeai as genai # <-- Added Gemini library

# Force CPU usage to ensure stability
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# --- 1. CONFIGURE GEMINI API ---
# IMPORTANT: Replace with your actual key. Do not commit this to a public GitHub repo.
GEMINI_API_KEY = "AIzaSyDEj6-DQczZxrxLRjH3J0gocb2N7H7PVRs" 
genai.configure(api_key=GEMINI_API_KEY)
# Initialize the Gemini model (gemini-1.5-flash is fast and free)
gemini_model = genai.GenerativeModel('gemini-2.5-flash')
print("✅ Gemini model initialized.")
# -----------------------------

# Check for GPU/CPU (will now always report CPU)
gpu_devices = tf.config.list_physical_devices('GPU')
if gpu_devices:
    print("✅ TensorFlow is using the GPU.")
else:
    print("💡 TensorFlow is intentionally using the CPU.")

# Initialize the Flask app
app = Flask(__name__)

# --- LOAD THE TRAINED MODEL ---
print("Loading the image classification model...")
model = tf.keras.models.load_model('model.h5')
print("Model loaded successfully!")

class_names = [
    'Pepper__bell___Bacterial_spot', 'Pepper__bell___healthy',
    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight',
    'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites_Two-spotted_spider_mite', 'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]

# --- PREDICTION FUNCTION ---
def predict_disease(image_path):
    img = Image.open(image_path).resize((224, 224))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions[0])
    predicted_class_name = class_names[predicted_class_index]
    confidence = np.max(predictions[0])
    return predicted_class_name, confidence

# --- DEFINE THE ROUTES ---
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    file = request.files.get('file')
    if not file or file.filename == '':
        return jsonify({'error': 'No selected file'})

    try:
        # Save the file temporarily
        filepath = os.path.join('uploads', file.filename)
        os.makedirs('uploads', exist_ok=True)
        file.save(filepath)

        # Make prediction with the image model
        disease_name_raw, confidence = predict_disease(filepath)
        
        # Clean up the uploaded file
        os.remove(filepath)

        disease_name_clean = disease_name_raw.replace('___', ' - ').replace('__', ' ').replace('_', ' ')

        # --- 2. CALL GEMINI FOR A SOLUTION ---
        solution = "No solution generated." # Default message
        # We don't need a solution if the plant is healthy
        if "healthy" not in disease_name_raw:
            try:
                # Create a specific prompt for the Gemini model
                prompt = f"You are an expert botanist advising a farmer. The plant has been diagnosed with '{disease_name_clean}'. Provide a clear, concise, and actionable solution in 2-3 short sentences. Use simple language."
                response = gemini_model.generate_content(prompt)
                solution = response.text
                print(solution)
            except Exception as e:
                print(f"Error calling Gemini API: {e}")
                solution = "Could not generate a real-time solution. Please consult an agricultural expert."
        else:
            solution = "The plant appears to be healthy. Continue to monitor and provide good care."
        # ------------------------------------

        # Return the enhanced result as JSON
        return jsonify({
            'disease': disease_name_clean,
            'confidence': f"{confidence:.2%}",
            'solution': solution
        })
    except Exception as e:
        print(f"An error occurred during prediction: {e}")
        return jsonify({'error': 'Failed to process the image.'}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5001)