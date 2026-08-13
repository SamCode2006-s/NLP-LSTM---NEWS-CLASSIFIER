# NLP — LSTM Flask App

A Render-ready Flask frontend/backend for the uploaded news classification model.

## Included model assets

- `model/news_classifier.keras`
- `model/tokenizer.pkl`
- `model/label_encoder.pkl`

The inference pipeline follows the uploaded notebook:
- lowercase
- URL/HTML/punctuation/number removal
- English stop-word removal
- lemmatization
- tokenizer sequences
- post-padding/truncation to 150 tokens
- model prediction
- top 3 categories

## Local run

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
# source .venv/bin/activate

pip install -r requirements.txt
python app.py
```

Open `http://localhost:10000`.

## Render

Render's Flask deployment flow uses `pip install -r requirements.txt` as the build command and Gunicorn as the production server. The included `render.yaml` is ready for a Web Service. 

For Render:
1. Push this folder to GitHub.
2. Create a Render Web Service from the repository.
3. Use the included `render.yaml`, or set:
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --threads 2 --timeout 120`

The first startup downloads the required NLTK data into the local `nltk_data` directory if it is not already present.
