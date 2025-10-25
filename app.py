import os
import tensorflow as tf
from flask import Flask, request, render_template, jsonify
from PIL import Image
import numpy as np
import requests
import google.generativeai as genai

# --- 1. CONFIGURATION ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyDEj6-DQczZxrxLRjH3J0gocb2N7H7PVRs")
genai.configure(api_key=GEMINI_API_KEY)
generation_config = { "temperature": 0.7, "top_p": 1, "top_k": 1, "max_output_tokens": 2048 }
safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
]
gemini_model = genai.GenerativeModel(
    model_name="gemini-2.5-flash",
    generation_config=generation_config,
    safety_settings=safety_settings
)

# --- 2. LOAD VISION MODEL & CLASS NAMES ---
MODEL_URL = "https://github.com/anu-610/Plant-Disease-Detector/blob/main/model.h5"
MODEL_PATH = "model.h5"

if MODEL_URL != "YOUR_DIRECT_DOWNLOAD_LINK_HERE" and not os.path.exists(MODEL_PATH):
    print(f"Downloading model from {MODEL_URL}...")
    try:
        response = requests.get(MODEL_URL, stream=True)
        response.raise_for_status()
        with open(MODEL_PATH, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print("Model downloaded successfully!")
    except Exception as e:
        print(f"Error downloading model: {e}")
else:
    if not os.path.exists(MODEL_PATH):
        print(f"Warning: Model file not found at {MODEL_PATH}")

print("Loading the vision model...")
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Vision model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}. Ensure the model file is correct.")
    model = None

class_names = [
    'Pepper__bell___Bacterial_spot', 'Pepper__bell___healthy', 'Potato___Early_blight',
    'Potato___Late_blight', 'Potato___healthy', 'Tomato___Bacterial_spot',
    'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites_Two-spotted_spider_mite',
    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
]

# --- 3. HELPER FUNCTIONS ---

def predict_disease_from_image(image_path):
    if model is None:
        return "Error: Vision model is not loaded.", 0.0
    try:
        img = Image.open(image_path).resize((224, 224))
        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0)
        predictions = model.predict(img_array)
        predicted_class_index = np.argmax(predictions[0])
        predicted_class_name_raw = class_names[predicted_class_index]
        confidence = np.max(predictions[0])
        disease_name_clean = predicted_class_name_raw.replace('___', ' - ').replace('__', ' ').replace('_', ' ')

        print("The disease: ", disease_name_clean)
        return disease_name_clean, confidence
    except Exception as e:
        print(f"Error during prediction: {e}")
        return f"Error processing image: {e}", 0.0

def get_gemini_response(prompt):
    try:
        convo = gemini_model.start_chat(history=[])
        convo.send_message(prompt)
        return convo.last.text
    except Exception as e:
        print(f"Error communicating with Gemini: {e}")
        return f"Error from AI: {e}"

def analyze_multimodal_query(image_disease, text_query):
    system_prompt = (
        "You are an expert agricultural assistant named Crop Protector. "
        "Your job is to help farmers diagnose and solve plant problems. "
        "You must follow these rules:\n"
        "1.  Be concise, empathetic, and provide actionable advice.\n"
        "2.  If you are given a disease name from my vision model, use it as the primary diagnosis.\n"
        "3.  If you are given both a vision model diagnosis AND a user's text query, analyze both.\n"
        "4.  **Conflict Handling:** If the user's text (e.g., 'my potato') and the vision model's diagnosis (e.g., 'Tomato - ...') are about *different plants*, you MUST address both. First, answer the user's text query. Second, state that the provided image was diagnosed as [disease_name] and provide a solution for it.\n"
        "5.  **Validation:** If the user's text query seems like a random paragraph or is not related to farming, politely state that you can only answer questions about agriculture and plant diseases. Then, if an image was provided, proceed to analyze the image."
    )
    final_prompt = f"{system_prompt}\n\nHere is the situation:\n"
    if image_disease:
        final_prompt += f"- My vision model has analyzed the user's image and identified: **{image_disease}**.\n"
    if text_query:
        final_prompt += f"- The user has also provided this text query: \"**{text_query}**\"\n"
    final_prompt += "\nPlease provide a complete and helpful response to the user."
    return get_gemini_response(final_prompt)

# --- 4. FLASK APP & ROUTES ---

app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'The vision model is not loaded. Please check the server logs.'}), 500

    image_file = request.files.get('file')
    text_query = request.form.get('text_query', '').strip()
    
    # This dictionary will hold all our results
    result = {
        "disease": None,
        "confidence": None,
        "solution": None
    }
    
    if not image_file and not text_query:
        return jsonify({'error': 'Please upload an image or ask a question.'}), 400

    # --- 1. Process Image (if provided) ---
    if image_file:
        try:
            filepath = os.path.join('uploads', image_file.filename)
            os.makedirs('uploads', exist_ok=True)
            image_file.save(filepath)
            
            disease_name, confidence = predict_disease_from_image(filepath)
            os.remove(filepath)
            
            if "Error" in disease_name:
                return jsonify({'error': disease_name}), 500
            
            # Save the vision model's direct output
            result["disease"] = disease_name
            result["confidence"] = f"{confidence:.2%}"
                
        except Exception as e:
            return jsonify({'error': f'Error saving file: {e}'}), 500

    # --- 2. Get Gemini's Response (The Solution) ---
    if not result["disease"] and text_query:
        # Case 1: Only Text is provided
        prompt = (
            "You are an expert agricultural assistant. A user has asked a question. "
            "First, validate if the question is about farming or plant diseases. "
            "If it is, answer it directly. If it is not, politely decline. "
            f"The user's question is: \"{text_query}\""
        )
        result["solution"] = get_gemini_response(prompt)
    
    elif result["disease"] and not text_query:
        # Case 2: Only Image is provided
        prompt = (
            "You are an expert agricultural assistant. My vision model identified a disease. "
            "Please provide a concise diagnosis and actionable solution. "
            f"The disease is: **{result['disease']}**"
        )
        result["solution"] = get_gemini_response(prompt)
    
    elif result["disease"] and text_query:
        # Case 3: Both Image and Text are provided
        result["solution"] = analyze_multimodal_query(result["disease"], text_query)
        
    else:
         return jsonify({'error': 'An unexpected error occurred.'}), 500

    # --- 3. Return the full, structured response ---
    return jsonify(result)

# --- 5. RUN THE APP ---
if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=5001)

