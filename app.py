import os
import re
import string
import pickle
from pathlib import Path

import numpy as np
import tensorflow as tf
from flask import Flask, jsonify, render_template, request
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"
NLTK_DATA = BASE_DIR / "nltk_data"
NLTK_DATA.mkdir(exist_ok=True)
nltk.data.path.append(str(NLTK_DATA))

def ensure_nltk_data():
    resources = [
        ("corpora/stopwords", "stopwords"),
        ("corpora/wordnet", "wordnet"),
        ("corpora/omw-1.4", "omw-1.4"),
    ]
    for resource_path, package in resources:
        try:
            nltk.data.find(resource_path)
        except LookupError:
            nltk.download(package, download_dir=str(NLTK_DATA), quiet=True)

ensure_nltk_data()

# Load the exact artifacts produced by the notebook.
model = tf.keras.models.load_model(MODEL_DIR / "news_classifier.keras")
with open(MODEL_DIR / "tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)
with open(MODEL_DIR / "label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

CATEGORIES = label_encoder.classes_.tolist()

MAX_LEN = 150
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text: str) -> str:
    """Match the notebook preprocessing pipeline."""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\d+", "", text)

    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return " ".join(tokens)

def predict_text(text: str):
    cleaned = preprocess_text(text)
    sequence = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(
        sequence,
        maxlen=MAX_LEN,
        padding="post",
        truncating="post",
    )

    probabilities = model.predict(padded, verbose=0)[0]
    top_indices = np.argsort(probabilities)[-3:][::-1]

    predictions = []
    for index in top_indices:
        predictions.append({
            "category": str(label_encoder.inverse_transform([int(index)])[0]),
            "confidence": round(float(probabilities[index]) * 100, 2),
        })

    return predictions

app = Flask(__name__)

@app.get("/")
def index():
    return render_template(
        "index.html",
        categories=CATEGORIES
    )

@app.post("/predict")
def predict():
    data = request.get_json(silent=True) or request.form
    text = (data.get("text") or "").strip()

    if not text:
        return jsonify({"error": "Enter text."}), 400

    if len(text) < 10:
        return jsonify({"error": "Enter more text."}), 400

    try:
        predictions = predict_text(text)
        return jsonify({
            "top_prediction": predictions[0],
            "predictions": predictions,
        })
    except Exception as exc:
        app.logger.exception("Prediction failed")
        return jsonify({"error": "Prediction failed. Check the server logs."}), 500

@app.get("/health")
def health():
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=False)
