# src/data_prep.py
from __future__ import annotations

import pandas as pd
from pathlib import Path
from .config import (
    RAW_TRAIN_SAMPLE,
    RAW_TEST_SAMPLE,
    DATA_PROCESSED_TRAIN,
    DATA_PROCESSED_TEST,
)


def map_label_to_sentiment(label: int | float):
    """
    Converte a coluna 'label' do dataset binário em 'sentiment':
      - label == 2 -> 1 (positivo)
      - label == 1 -> 0 (negativo)
      - qualquer outra coisa -> None (descartado)
    """
    try:
        l = int(label)
    except (TypeError, ValueError):
        return None

    if l == 2:
        return 1  # positivo
    elif l == 1:
        return 0  # negativo
    else:
        return None


def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Recebe um DF com colunas [label, title, text] e devolve
    um DF com [text, sentiment] pronto para o modelo.
    """
    expected_cols = {"label", "title", "text"}
    if not expected_cols.issubset(df.columns):
        raise ValueError(
            f"Colunas esperadas {expected_cols}, mas o DF tem {df.columns.tolist()}"
        )

    # Mantém só label e text, remove nulos
    df = df[["label", "text"]].dropna(subset=["label", "text"])

    # Converte label para numérico
    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    df = df.dropna(subset=["label"])

    # Mapeia para sentiment binário
    df["sentiment"] = df["label"].apply(map_label_to_sentiment)

    # Remove qualquer linha que não seja 1 ou 2
    df = df.dropna(subset=["sentiment"])
    df["sentiment"] = df["sentiment"].astype(int)

    df_final = df[["text", "sentiment"]].reset_index(drop=True)

    print("📊 Distribuição de classes (sentiment):")
    print(df_final["sentiment"].value_counts(normalize=True).rename("proporção"))
    print(df_final["sentiment"].value_counts().rename("contagem"))

    return df_final


def main() -> None:
    # 1. Ler arquivos crus reduzidos
    if not Path(RAW_TRAIN_SAMPLE).exists():
        raise FileNotFoundError(f"RAW_TRAIN_SAMPLE não encontrado: {RAW_TRAIN_SAMPLE}")
    if not Path(RAW_TEST_SAMPLE).exists():
        raise FileNotFoundError(f"RAW_TEST_SAMPLE não encontrado: {RAW_TEST_SAMPLE}")

    print(f"📥 Lendo train bruto de: {RAW_TRAIN_SAMPLE}")
    train_raw = pd.read_csv(RAW_TRAIN_SAMPLE)

    print(f"📥 Lendo test bruto de: {RAW_TEST_SAMPLE}")
    test_raw = pd.read_csv(RAW_TEST_SAMPLE)

    # 2. Preparar ambos
    print("\n🔧 Preparando TRAIN...")
    train_prepared = prepare_dataframe(train_raw)

    print("\n🔧 Preparando TEST...")
    test_prepared = prepare_dataframe(test_raw)

    # 3. Salvar em data/processed
    DATA_PROCESSED_TRAIN.parent.mkdir(parents=True, exist_ok=True)
    train_prepared.to_csv(DATA_PROCESSED_TRAIN, index=False)
    print(f"\n💾 Train processado salvo em: {DATA_PROCESSED_TRAIN}")

    DATA_PROCESSED_TEST.parent.mkdir(parents=True, exist_ok=True)
    test_prepared.to_csv(DATA_PROCESSED_TEST, index=False)
    print(f"💾 Test processado salvo em: {DATA_PROCESSED_TEST}")

    print("\n✅ Pré-processamento concluído!")


if __name__ == "__main__":
    main()
