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


# ============================================================
# CHECK NLTK RESOURCES
# ============================================================

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
# LOAD TFLITE MODEL
# ============================================================

MODEL_PATH = MODEL_DIR / "news_classifier.tflite"

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model file not found: {MODEL_PATH}"
    )

interpreter = Interpreter(
    model_path=str(MODEL_PATH)
)

interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

INPUT_INDEX = input_details[0]["index"]
OUTPUT_INDEX = output_details[0]["index"]

INPUT_DTYPE = input_details[0]["dtype"]

# Your converted model is [1, 150]
MAX_LEN = int(input_details[0]["shape"][1])

interpreter_lock = Lock()


# ============================================================
# LOAD TOKENIZER DATA
# ============================================================

TOKENIZER_PATH = MODEL_DIR / "tokenizer_data.pkl"

if not TOKENIZER_PATH.exists():
    raise FileNotFoundError(
        f"Tokenizer file not found: {TOKENIZER_PATH}"
    )

with open(TOKENIZER_PATH, "rb") as f:
    tokenizer_data = pickle.load(f)


word_index = tokenizer_data["word_index"]
num_words = tokenizer_data.get("num_words")
oov_token = tokenizer_data.get("oov_token")

oov_index = None

if oov_token:
    oov_index = word_index.get(oov_token)


# ============================================================
# LOAD LABEL ENCODER
# ============================================================

LABEL_ENCODER_PATH = MODEL_DIR / "label_encoder.pkl"

if not LABEL_ENCODER_PATH.exists():
    raise FileNotFoundError(
        f"Label encoder not found: {LABEL_ENCODER_PATH}"
    )

with open(LABEL_ENCODER_PATH, "rb") as f:
    label_encoder = pickle.load(f)


CATEGORIES = label_encoder.classes_.tolist()


# ============================================================
# NLTK PREPROCESSING
# ============================================================

stop_words = set(
    stopwords.words("english")
)

lemmatizer = WordNetLemmatizer()


def preprocess_text(text: str) -> str:
    """
    Match the preprocessing used during model training.
    """

    text = str(text).lower()

    # Remove URLs
    text = re.sub(
        r"http\S+|www\S+|https\S+",
        "",
        text
    )

    # Remove HTML
    text = re.sub(
        r"<.*?>",
        "",
        text
    )

    # Remove punctuation
    text = text.translate(
        str.maketrans(
            "",
            "",
            string.punctuation
        )
    )

    # Remove numbers
    text = re.sub(
        r"\d+",
        "",
        text
    )

    # Tokenize by whitespace
    tokens = text.split()

    # Remove stopwords
    tokens = [
        word
        for word in tokens
        if word not in stop_words
    ]

    # Lemmatize
    tokens = [
        lemmatizer.lemmatize(word)
        for word in tokens
    ]

    return " ".join(tokens)


# ============================================================
# TOKENIZER REPLACEMENT
# ============================================================

def text_to_sequence(text: str):
    """
    Equivalent to Keras tokenizer.texts_to_sequences()
    using the saved word_index dictionary.
    """

    tokens = text.split()

    sequence = []

    for word in tokens:

        index = word_index.get(word)

        # Unknown word
        if index is None:

            if oov_index is not None:
                index = oov_index
            else:
                continue

        # Respect tokenizer num_words limit
        if num_words is not None and index >= num_words:

            if oov_index is not None:
                index = oov_index
            else:
                continue

        sequence.append(index)

    return sequence


# ============================================================
# CREATE MODEL INPUT
# ============================================================

def make_padded_input(text: str) -> np.ndarray:

    cleaned = preprocess_text(text)

    sequence = text_to_sequence(cleaned)

    # Fixed input shape: [1, 150]
    padded = np.zeros(
        (1, MAX_LEN),
        dtype=np.int32
    )

    sequence = sequence[:MAX_LEN]

    if sequence:
        padded[0, :len(sequence)] = sequence

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

    # Top 3 classes
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
# FLASK APP
# ============================================================

app = Flask(__name__)


# ============================================================
# HOME
# ============================================================

@app.get("/")
def index():

    return render_template(
        "index.html",
        categories=CATEGORIES
    )


# ============================================================
# PREDICTION API
# ============================================================

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
            "error": "Prediction failed. Check server logs."
        }), 500


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health():

    return jsonify({
        "status": "ok"
    })


# ============================================================
# LOCAL DEVELOPMENT
# ============================================================

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