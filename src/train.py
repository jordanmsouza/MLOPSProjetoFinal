# src/train.py
from __future__ import annotations

import pandas as pd
from joblib import dump
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, classification_report

from .config import (
    DATA_PROCESSED_TRAIN,
    DATA_PROCESSED_TEST,
    MODEL_PATH,
)


def load_data():
    if not Path(DATA_PROCESSED_TRAIN).exists():
        raise FileNotFoundError(f"Train processado não encontrado: {DATA_PROCESSED_TRAIN}")

    if not Path(DATA_PROCESSED_TEST).exists():
        raise FileNotFoundError(f"Test processado não encontrado: {DATA_PROCESSED_TEST}")

    print(f"📥 Lendo train: {DATA_PROCESSED_TRAIN}")
    train_df = pd.read_csv(DATA_PROCESSED_TRAIN)

    print(f"📥 Lendo test: {DATA_PROCESSED_TEST}")
    test_df = pd.read_csv(DATA_PROCESSED_TEST)

    return train_df, test_df


def train_model(train_df: pd.DataFrame, test_df: pd.DataFrame):
    X_train = train_df["text"]
    y_train = train_df["sentiment"]

    X_test = test_df["text"]
    y_test = test_df["sentiment"]

    # Pipeline TF-IDF → Regressão Logística
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=40000,
            ngram_range=(1, 2),
            stop_words="english"
        )),
        ("clf", LogisticRegression(
            max_iter=1000,
            n_jobs=-1
        )),
    ])

    print("\n🚀 Treinando modelo...")
    pipeline.fit(X_train, y_train)

    print("\n📊 Avaliando modelo...")
    y_pred = pipeline.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"\n🔎 Accuracy: {acc:.4f}")
    print(f"🔎 F1-Score: {f1:.4f}")

    print("\n📄 Classification Report:")
    print(classification_report(y_test, y_pred))

    # Salvar modelo
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    dump(pipeline, MODEL_PATH)

    print(f"\n💾 Modelo salvo em: {MODEL_PATH}")

    return pipeline, acc, f1


def main():
    train_df, test_df = load_data()
    train_model(train_df, test_df)


if __name__ == "__main__":
    main()
