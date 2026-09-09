import os
import torch
import xgboost as xgb
from transformers import AutoTokenizer, AutoModel
from src.preprocess import clean_text

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "offline_hybrid_model")

# Load Offline Models
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModel.from_pretrained(MODEL_PATH)
xgb_clf = xgb.XGBClassifier()
xgb_clf.load_model(os.path.join(MODEL_PATH, "xgboost_spam.json"))

def predict_spam(email_text: str) -> dict:
    if not email_text.strip():
        return {"label": "Normal", "confidence": 0.0}

    # 1. Preprocess
    masked_text = clean_text(email_text)

    # 2. Extract Embeddings
    inputs = tokenizer(masked_text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        embedding = model(**inputs).last_hidden_state[:, 0, :].numpy()

    # 3. XGBoost Prediction
    prediction = int(xgb_clf.predict(embedding)[0])
    confidence = float(xgb_clf.predict_proba(embedding)[0][prediction])

    label_map = {0: "Normal", 1: "Spam"}
    return {
        "label": label_map[prediction],
        "confidence": round(confidence * 100, 2)
    }
