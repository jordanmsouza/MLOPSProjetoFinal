# src/serve.py
from __future__ import annotations

from datetime import datetime
from pathlib import Path
import csv

from fastapi import FastAPI
from pydantic import BaseModel

from .config import (
    BASE_DIR,
    MODEL_PATH,
    MLFLOW_TRACKING_URI,
    MLFLOW_MODEL_NAME,
)

import mlflow
import mlflow.sklearn
from joblib import load as joblib_load

# ==========================
#  Configuração de logs
# ==========================

LOGS_DIR = BASE_DIR / "logs"
PREDICTIONS_LOG = LOGS_DIR / "predictions_log.csv"
FEEDBACK_LOG = LOGS_DIR / "feedback_log.csv"

LOGS_DIR.mkdir(parents=True, exist_ok=True)


def _init_csv(path: Path, header: list[str]) -> None:
    """Cria o arquivo CSV com header se ainda não existir."""
    if not path.exists():
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(header)


_init_csv(
    PREDICTIONS_LOG,
    ["timestamp", "text", "text_length", "sentiment", "label", "confidence"],
)

_init_csv(
    FEEDBACK_LOG,
    [
        "timestamp",
        "text",
        "text_length",
        "model_sentiment",
        "model_confidence",
        "user_sentiment",
        "is_correct",
    ],
)

# ==========================
#  Carregamento do modelo
# ==========================

model = None
MODEL_SOURCE = None

try:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    model_uri = f"models:/{MLFLOW_MODEL_NAME}/latest"
    print(f"🔁 Tentando carregar modelo do MLflow Registry: {model_uri}")
    model = mlflow.sklearn.load_model(model_uri)
    MODEL_SOURCE = f"mlflow:{model_uri}"
    print("✅ Modelo carregado do MLflow com sucesso!")
except Exception as e:
    print(f"⚠️ Falha ao carregar do MLflow ({e}). Tentando via joblib...")
    print(f"🔁 Carregando modelo local de: {MODEL_PATH}")
    model = joblib_load(MODEL_PATH)
    MODEL_SOURCE = f"joblib:{MODEL_PATH}"
    print("✅ Modelo carregado via joblib com sucesso!")


# ==========================
#  API
# ==========================

app = FastAPI(
    title="Sentiment Analysis API",
    description="API para classificação de sentimento de reviews da Amazon.",
    version="1.1.0",
)


class PredictRequest(BaseModel):
    text: str


class FeedbackRequest(BaseModel):
    text: str
    user_sentiment: int  # 1 = positivo, 0 = negativo


def _predict_internal(text: str) -> tuple[int, float]:
    """Executa a previsão usando o modelo carregado."""
    proba = model.predict_proba([text])[0]
    # assumindo ordem [negativo, positivo]
    confidence_positive = float(proba[1])
    sentiment = 1 if confidence_positive >= 0.5 else 0
    confidence = confidence_positive if sentiment == 1 else float(proba[0])
    return sentiment, confidence


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_source": str(MODEL_SOURCE),
    }


@app.post("/predict")
def predict(request: PredictRequest):
    text = request.text.strip()

    if not text:
        return {
            "error": "Texto vazio não é permitido.",
        }

    sentiment, confidence = _predict_internal(text)

    label = "positivo" if sentiment == 1 else "negativo"
    text_length = len(text)
    timestamp = datetime.utcnow().isoformat()

    # Log da predição
    with PREDICTIONS_LOG.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [timestamp, text, text_length, sentiment, label, confidence]
        )

    return {
        "sentiment": sentiment,
        "label": label,
        "confidence": confidence,
        "text_length": text_length,
        "timestamp": timestamp,
    }


@app.post("/feedback")
def feedback(request: FeedbackRequest):
    text = request.text.strip()
    user_sentiment = int(request.user_sentiment)

    # Recalcula a previsão do modelo para registrar junto com o feedback
    model_sentiment, model_confidence = _predict_internal(text)

    is_correct = int(model_sentiment == user_sentiment)
    text_length = len(text)
    timestamp = datetime.utcnow().isoformat()

    with FEEDBACK_LOG.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                timestamp,
                text,
                text_length,
                model_sentiment,
                model_confidence,
                user_sentiment,
                is_correct,
            ]
        )

    return {
        "model_sentiment": model_sentiment,
        "model_label": "positivo" if model_sentiment == 1 else "negativo",
        "model_confidence": model_confidence,
        "user_sentiment": user_sentiment,
        "is_correct": bool(is_correct),
        "message": "Feedback registrado com sucesso.",
    }
