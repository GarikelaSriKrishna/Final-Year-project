from flask import Flask, request, jsonify, render_template
import pickle
import os
import numpy as np
import json
from flask_cors import CORS

app = Flask(__name__, template_folder='templates')
CORS(app)  # Enable Cross-Origin Resource Sharing

# Paths to models and scalers
MODELS_PATH = {
    "instagram": "models/instagram/hybrid_model.h5",
    "facebook": "models/facebook/hybrid_model.h5"
}
SCALERS_PATH = {
    "instagram": "models/instagram/scaler.pkl",
    "facebook": "models/facebook/scaler.pkl"
}

# Folder for storing search history
HISTORY_FOLDER = "history"
os.makedirs(HISTORY_FOLDER, exist_ok=True)

# Function to load models dynamically
def load_model(platform):
    with open(MODELS_PATH[platform], 'rb') as f:
        model = pickle.load(f)
    with open(SCALERS_PATH[platform], 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler

# Placeholder function to fetch user data (replace with actual API calls)
def fetch_profile_data(platform, username):
    # Example: Implement API calls for real-time data fetching
    return {
        "followers": np.random.randint(100, 10000),
        "following": np.random.randint(50, 5000),
        "posts": np.random.randint(10, 1000),
        "profile_picture": np.random.choice([0, 1])  # 1 if profile pic exists, else 0
    }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    username = data.get('username')
    platform = data.get('platform')

    if not username or platform not in MODELS_PATH:
        return jsonify({"error": "Invalid username or platform"}), 400

    # Load the corresponding model and scaler
    model, scaler = load_model(platform)

    # Fetch profile data
    profile_data = fetch_profile_data(platform, username)

    # Convert data into model input format
    features = np.array([[profile_data["followers"], profile_data["following"],
                          profile_data["posts"], profile_data["profile_picture"]]])

    # Scale features
    scaled_features = scaler.transform(features)

    # Predict
    prediction = model.predict(scaled_features)
    result = "Fake Profile" if prediction[0] == 1 else "Legitimate Profile"

    # Save search history
    history_entry = {
        "username": username,
        "platform": platform,
        "prediction": result
    }
    with open(os.path.join(HISTORY_FOLDER, "search_history.json"), "a") as f:
        f.write(json.dumps(history_entry) + "\n")

    return jsonify({"username": username, "platform": platform, "prediction": result})

if __name__ == '__main__':
    app.run(debug=True, port=5500)
