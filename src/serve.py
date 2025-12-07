# src/serve.py
from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel
from joblib import load
from datetime import datetime
from pathlib import Path
import csv

from .config import MODEL_PATH, BASE_DIR

app = FastAPI(
    title="Sentiment Analysis API",
    description="API para classificação de sentimento de reviews da Amazon.",
    version="1.0.0",
)


class TextInput(BaseModel):
    text: str


class FeedbackInput(BaseModel):
    text: str
    user_sentiment: int  # 0 = negativo, 1 = positivo


# ====== Configuração de LOG simples ======
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

PREDICTIONS_LOG = LOG_DIR / "predictions_log.csv"
FEEDBACK_LOG = LOG_DIR / "feedback_log.csv"

# Cabeçalho predictions_log.csv
if not PREDICTIONS_LOG.exists():
    with PREDICTIONS_LOG.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "text_length", "sentiment", "confidence"])

# Cabeçalho feedback_log.csv
if not FEEDBACK_LOG.exists():
    with FEEDBACK_LOG.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "timestamp",
            "text_length",
            "model_sentiment",
            "model_confidence",
            "user_sentiment",
            "is_correct",
        ])
# =========================================


print(f"🔁 Carregando modelo a partir de: {MODEL_PATH}")
model = load(MODEL_PATH)
print("✅ Modelo carregado com sucesso!")


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/predict")
def predict_sentiment(input_data: TextInput):
    """
    Recebe um texto e retorna o sentimento previsto pelo modelo.
    Também registra a previsão em um log CSV para monitoramento.
    """
    review_text = input_data.text

    pred = model.predict([review_text])[0]
    proba = model.predict_proba([review_text])[0]

    sentiment = int(pred)
    label = "positivo" if sentiment == 1 else "negativo"
    confidence = float(proba[sentiment])

    # Log de previsão
    timestamp = datetime.utcnow().isoformat()
    text_length = len(review_text)

    with PREDICTIONS_LOG.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([timestamp, text_length, sentiment, confidence])

    return {
        "sentiment": sentiment,
        "label": label,
        "confidence": confidence,
    }


@app.post("/feedback")
def send_feedback(feedback: FeedbackInput):
    """
    Endpoint para o usuário enviar o sentimento que ele considera correto.
    Usamos isso para monitorar a qualidade do modelo em produção.
    """
    review_text = feedback.text
    user_sentiment = int(feedback.user_sentiment)

    # Rodamos o modelo novamente nesse texto
    pred = model.predict([review_text])[0]
    proba = model.predict_proba([review_text])[0]

    model_sentiment = int(pred)
    model_confidence = float(proba[model_sentiment])

    is_correct = int(model_sentiment == user_sentiment)

    timestamp = datetime.utcnow().isoformat()
    text_length = len(review_text)

    with FEEDBACK_LOG.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            timestamp,
            text_length,
            model_sentiment,
            model_confidence,
            user_sentiment,
            is_correct,
        ])

    return {
        "model_sentiment": model_sentiment,
        "model_label": "positivo" if model_sentiment == 1 else "negativo",
        "model_confidence": model_confidence,
        "user_sentiment": user_sentiment,
        "is_correct": bool(is_correct),
        "message": "Feedback registrado com sucesso.",
    }
