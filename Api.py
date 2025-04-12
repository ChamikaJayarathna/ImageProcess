from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from pymongo import MongoClient
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
import requests
from PIL import Image
import io
import json
from bson import ObjectId
from datetime import datetime

app = Flask(__name__)
CORS(app)
load_dotenv()

# Load Trained Models
try:
    classification_model = load_model("model.keras")
    feature_extraction_model = load_model("feature_model.keras")
    print("✅ Model loaded successfully.")
except Exception as e:
    print(f"❌ Error loading model: {e}")

# MongoDB Connection
try:
    client = MongoClient(os.getenv("DATABASE_URL"))
    db = client["property_estate"]
    rental_collection = db["properties"]
    print("✅ Database connection successful.")
except Exception as e:
    print(f"❌ Error connecting to database: {e}")

# Custom JSON Encoder for ObjectId
class JSONEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, ObjectId):
            return str(o)
        if isinstance(o, datetime):
            return o.isoformat()
        return super().default(o)

# Image Preprocessing
def preprocess_image(image, target_size=(224, 224)):
    image = image.resize(target_size)
    image_array = img_to_array(image)
    image_array = np.expand_dims(image_array, axis=0) / 255.0
    return image_array

# Extract Features Using Feature Model
def extract_features(image_array):
    features = feature_extraction_model.predict(image_array)[0]
    return features / np.linalg.norm(features)

# Download and Process Image from URL
def get_image_from_url(url):
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            image = Image.open(io.BytesIO(response.content)).convert("RGB")
            return preprocess_image(image)
        else:
            print(f"Failed to fetch image: {url}, Status Code: {response.status_code}")
    except Exception as e:
        print(f"Error loading image from {url}: {e}")
    return None

# Convert MongoDB document to JSON-serializable format
def serialize_document(doc):
    return json.loads(json.dumps(doc, cls=JSONEncoder))

# Precompute and Store Features in MongoDB
def update_property_features():
    all_properties = list(rental_collection.find({"images": {"$exists": True, "$ne": []}}))
    
    for prop in all_properties:
        property_id = prop["_id"]
        print("Processing Property ID:", property_id)
        features_list = []

        for image_url in prop.get("images", []):
            img_array = get_image_from_url(image_url)
            if img_array is not None:
                features = extract_features(img_array).tolist()
                features_list.append(features)

        rental_collection.update_one(
            {"_id": property_id},
            {"$set": {"features": features_list}}
        )
    print("Feature extraction and storage completed.")
    
@app.route('/', methods=['GET'])
def home():
    return jsonify({"message": "API is running"}), 200


# AI-Based Image Search Endpoint
@app.route('/predict', methods=['POST'])
def ai_search():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    # Save and preprocess uploaded image
    filename = secure_filename(file.filename)
    file_path = os.path.join("uploads", filename)
    file.save(file_path)

    image = Image.open(file_path).convert("RGB")
    img_array = preprocess_image(image)
    uploaded_features = extract_features(img_array)

    # Fetch stored property features
    all_properties = rental_collection.find({"features": {"$exists": True, "$ne": []}})
    
    matched_properties = []
    for prop in all_properties:
        for stored_features in prop.get("features", []):
            stored_features = np.array(stored_features)
            
            # Compute Cosine Similarity
            similarity = np.dot(uploaded_features, stored_features) / (
                np.linalg.norm(uploaded_features) * np.linalg.norm(stored_features)
            )

            if similarity > 0.5:
                matched_properties.append(serialize_document(prop))
                break 

    os.remove(file_path)
    return json.dumps({"matched_properties": matched_properties}, cls=JSONEncoder), 200, {'Content-Type': 'application/json'}

# Run Server
if __name__ == '__main__':
    if not os.path.exists("uploads"):
        os.makedirs("uploads")
    if os.getenv("WERKZEUG_RUN_MAIN") == "true":
        update_property_features()
    app.run(host="0.0.0.0", port=5000)
