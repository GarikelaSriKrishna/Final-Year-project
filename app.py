from flask import Flask, request, jsonify, render_template
import pickle
import os
import numpy as np
import json
import requests
from flask_cors import CORS
import tweepy

app = Flask(__name__, template_folder='templates')
CORS(app)

# API Credentials (Replace with your actual keys)
INSTAGRAM_ACCESS_TOKEN = "your_instagram_access_token"
TWITTER_API_KEY = "your_twitter_api_key"
TWITTER_API_SECRET = "your_twitter_api_secret"
TWITTER_BEARER_TOKEN = "your_twitter_bearer_token"
FACEBOOK_ACCESS_TOKEN = "your_facebook_access_token"

# Twitter API Authentication
auth = tweepy.OAuthHandler(TWITTER_API_KEY, TWITTER_API_SECRET)
twitter_api = tweepy.API(auth)

# Paths to models and scalers
MODELS_PATH = {
    "instagram": "models/instagram/hybrid_model.h5",
    "twitter": "models/twitter/hybrid_model.h5",
    "facebook": "models/facebook/hybrid_model.h5"
}
SCALERS_PATH = {
    "instagram": "models/instagram/scaler.pkl",
    "twitter": "models/twitter/scaler.pkl",
    "facebook": "models/facebook/scaler.pkl"
}

HISTORY_FOLDER = "history"
os.makedirs(HISTORY_FOLDER, exist_ok=True)

def load_model(platform):
    with open(MODELS_PATH[platform], 'rb') as f:
        model = pickle.load(f)
    with open(SCALERS_PATH[platform], 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler

def fetch_instagram_data(username):
    url = f"https://graph.instagram.com/{username}?fields=id,username,followers_count,follows_count,media_count,profile_picture_url&access_token={INSTAGRAM_ACCESS_TOKEN}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        return {
            "followers": data.get("followers_count", 0),
            "following": data.get("follows_count", 0),
            "posts": data.get("media_count", 0),
            "profile_picture": 1 if data.get("profile_picture_url") else 0
        }
    return None

def fetch_twitter_data(username):
    try:
        user = twitter_api.get_user(screen_name=username)
        return {
            "followers": user.followers_count,
            "following": user.friends_count,
            "posts": user.statuses_count,
            "profile_picture": 1 if user.profile_image_url else 0
        }
    except:
        return None

def fetch_facebook_data(username):
    url = f"https://graph.facebook.com/{username}?fields=id,name,followers_count,friends&access_token={FACEBOOK_ACCESS_TOKEN}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        return {
            "followers": data.get("followers_count", 0),
            "following": len(data.get("friends", {}).get("data", [])),
            "posts": 100,  # Placeholder (requires advanced permissions)
            "profile_picture": 1  # Placeholder (requires advanced permissions)
        }
    return None

def fetch_profile_data(platform, username):
    if platform == "instagram":
        return fetch_instagram_data(username)
    elif platform == "twitter":
        return fetch_twitter_data(username)
    elif platform == "facebook":
        return fetch_facebook_data(username)
    return None

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

    model, scaler = load_model(platform)
    profile_data = fetch_profile_data(platform, username)
    
    if profile_data is None:
        return jsonify({"error": "Could not retrieve user data"}), 400

    features = np.array([[profile_data["followers"], profile_data["following"],
                          profile_data["posts"], profile_data["profile_picture"]]])

    scaled_features = scaler.transform(features)
    prediction = model.predict(scaled_features)
    result = "Fake Profile" if prediction[0] == 1 else "Legitimate Profile"

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