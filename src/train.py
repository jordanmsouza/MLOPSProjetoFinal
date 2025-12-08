# src/train.py
from __future__ import annotations

import mlflow
import mlflow.sklearn
import pandas as pd
from joblib import dump
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.pipeline import Pipeline

from .config import (
    DATA_PROCESSED_TRAIN,
    DATA_PROCESSED_TEST,
    MODEL_PATH,
    MLFLOW_TRACKING_URI,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_MODEL_NAME,
)


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Carrega dados processados de treino e teste."""
    print("📥 Lendo dados processados...")
    train_df = pd.read_csv(DATA_PROCESSED_TRAIN)
    test_df = pd.read_csv(DATA_PROCESSED_TEST)

    print(f"Train: {train_df.shape}, Test: {test_df.shape}")
    return train_df, test_df


def build_pipeline() -> Pipeline:
    """Cria o pipeline de TF-IDF + Regressão Logística."""
    pipeline = Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=40_000,
                    ngram_range=(1, 2),
                    stop_words="english",
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    max_iter=1_000,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    return pipeline


def train_model(train_df: pd.DataFrame, test_df: pd.DataFrame):
    X_train = train_df["text"]
    y_train = train_df["sentiment"]

    X_test = test_df["text"]
    y_test = test_df["sentiment"]

    pipeline = build_pipeline()

    # Configura MLflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    with mlflow.start_run():
        print("\n🚀 Treinando modelo...")
        pipeline.fit(X_train, y_train)

        print("\n📊 Avaliando modelo...")
        y_pred = pipeline.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        print(f"Accuracy: {acc:.4f}")
        print(f"F1-score: {f1:.4f}")
        print("\nClassification report:")
        print(classification_report(y_test, y_pred))

        # Log de parâmetros
        mlflow.log_param("tfidf_max_features", 40_000)
        mlflow.log_param("tfidf_ngram_range", "1-2")
        mlflow.log_param("clf_max_iter", 1_000)

        # Log de métricas
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1", f1)

        # Log do modelo no MLflow (Model Registry)
        mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path="model",
            registered_model_name=MLFLOW_MODEL_NAME,
        )

        # Também salva localmente para servir via joblib (como já fazemos hoje)
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        dump(pipeline, MODEL_PATH)
        print(f"💾 Modelo salvo em: {MODEL_PATH}")

    return pipeline, acc, f1


def main():
    train_df, test_df = load_data()
    train_model(train_df, test_df)


if __name__ == "__main__":
    main()
