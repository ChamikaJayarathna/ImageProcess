from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from pymongo import MongoClient
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

app = Flask(__name__)
CORS(app)
load_dotenv()

# Load Trained Model and Class Labels
model = load_model("model.keras")
labels = np.load("class_labels.npy", allow_pickle=True).item()

# MongoDB Atlas Connection
client = MongoClient(os.getenv("DATABASE_URL"))
db = client["estate"]
rental_collection = db["rentals"]

# Image Preprocessing
def preprocess_image(image_path, target_size=(224, 224)):
    img = load_img(image_path, target_size=target_size)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

# AI-Based Search Endpoint
@app.route('/predict', methods=['POST'])
def ai_search():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    filename = secure_filename(file.filename)
    file_path = os.path.join("uploads", filename)
    file.save(file_path)

    # Preprocess and Predict
    img_array = preprocess_image(file_path)
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_label = labels.get(predicted_class_index, "Unknown")

    # Query MongoDB
    matching_rentals = list(rental_collection.find({"features": {"$regex": predicted_label, "$options": "i"}}))

    # Convert ObjectId to String
    for rental in matching_rentals:
        rental["_id"] = str(rental["_id"])

    # Clean up the file
    os.remove(file_path)

    return jsonify({
        "predicted_label": predicted_label,
        "matching_rentals": matching_rentals
    })

if __name__ == '__main__':
    if not os.path.exists("uploads"):
        os.makedirs("uploads")
    app.run(debug=True)
