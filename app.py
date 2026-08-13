import os
import re
import string
import pickle
from pathlib import Path
from threading import Lock

import numpy as np

from flask import Flask, jsonify, render_template, request

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

from ai_edge_litert.interpreter import Interpreter


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"

NLTK_DATA = BASE_DIR / "nltk_data"
nltk.data.path.append(str(NLTK_DATA))


def check_nltk_data():
    required_resources = [
        "corpora/stopwords",
        "corpora/wordnet",
        "corpora/omw-1.4",
    ]

    for resource in required_resources:
        try:
            nltk.data.find(resource)
        except LookupError as exc:
            raise RuntimeError(
                f"Missing NLTK resource: {resource}. "
                "Make sure the nltk_data folder is included in the repository."
            ) from exc


check_nltk_data()

# ============================================================
# LOAD TFLITE / LITERT MODEL
# ============================================================

MODEL_PATH = MODEL_DIR / "news_classifier.tflite"

interpreter = Interpreter(
    model_path=str(MODEL_PATH)
)

interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

INPUT_INDEX = input_details[0]["index"]
OUTPUT_INDEX = output_details[0]["index"]

INPUT_DTYPE = input_details[0]["dtype"]

# Your converted model has fixed shape [1, 150]
MAX_LEN = int(input_details[0]["shape"][1])

# Protect interpreter from concurrent access
interpreter_lock = Lock()


# ============================================================
# LOAD TOKENIZER + LABEL ENCODER
# ============================================================

with open(MODEL_DIR / "tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

with open(MODEL_DIR / "label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)


CATEGORIES = label_encoder.classes_.tolist()


# ============================================================
# TEXT PREPROCESSING
# ============================================================

stop_words = set(
    stopwords.words("english")
)

lemmatizer = WordNetLemmatizer()


def preprocess_text(text: str) -> str:
    """Match the original notebook preprocessing."""

    text = str(text).lower()

    text = re.sub(
        r"http\S+|www\S+|https\S+",
        "",
        text
    )

    text = re.sub(
        r"<.*?>",
        "",
        text
    )

    text = text.translate(
        str.maketrans(
            "",
            "",
            string.punctuation
        )
    )

    text = re.sub(
        r"\d+",
        "",
        text
    )

    tokens = text.split()

    tokens = [
        word
        for word in tokens
        if word not in stop_words
    ]

    tokens = [
        lemmatizer.lemmatize(word)
        for word in tokens
    ]

    return " ".join(tokens)


# ============================================================
# TOKENIZER + PADDING
# ============================================================

def make_padded_input(text: str) -> np.ndarray:

    cleaned = preprocess_text(text)

    sequence = tokenizer.texts_to_sequences(
        [cleaned]
    )

    padded = np.zeros(
        (1, MAX_LEN),
        dtype=np.int32
    )

    if sequence and sequence[0]:

        seq = sequence[0][:MAX_LEN]

        padded[0, :len(seq)] = seq

    # TFLite model expects float32
    return padded.astype(INPUT_DTYPE)


# ============================================================
# PREDICTION
# ============================================================

def predict_text(text: str):

    padded = make_padded_input(text)

    with interpreter_lock:

        interpreter.set_tensor(
            INPUT_INDEX,
            padded
        )

        interpreter.invoke()

        probabilities = interpreter.get_tensor(
            OUTPUT_INDEX
        )[0].copy()

    top_indices = np.argsort(
        probabilities
    )[-3:][::-1]

    predictions = []

    for index in top_indices:

        category = label_encoder.inverse_transform(
            [int(index)]
        )[0]

        predictions.append({
            "category": str(category),
            "confidence": round(
                float(probabilities[index]) * 100,
                2
            )
        })

    return predictions


# ============================================================
# FLASK
# ============================================================

app = Flask(__name__)


@app.get("/")
def index():

    return render_template(
        "index.html",
        categories=CATEGORIES
    )


@app.post("/predict")
def predict():

    data = (
        request.get_json(silent=True)
        or request.form
    )

    text = (
        data.get("text") or ""
    ).strip()

    if not text:

        return jsonify({
            "error": "Enter text."
        }), 400

    if len(text) < 10:

        return jsonify({
            "error": "Enter more text."
        }), 400

    try:

        predictions = predict_text(text)

        return jsonify({
            "top_prediction": predictions[0],
            "predictions": predictions
        })

    except Exception:

        app.logger.exception(
            "Prediction failed"
        )

        return jsonify({
            "error":
                "Prediction failed. Check server logs."
        }), 500


@app.get("/health")
def health():

    return jsonify({
        "status": "ok"
    })


if __name__ == "__main__":

    port = int(
        os.environ.get(
            "PORT",
            10000
        )
    )

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )