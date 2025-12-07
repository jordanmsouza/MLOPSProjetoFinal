from __future__ import annotations

import kagglehub
import pandas as pd
from pathlib import Path


# Diretórios base (você pode mover isso para um config.py se preferir)
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"

RAW_TRAIN_SAMPLE = RAW_DIR / "amazon_reviews_train_sample.csv"
RAW_TEST_SAMPLE = RAW_DIR / "amazon_reviews_test_sample.csv"

KAGGLE_DATASET_ID = "kritanjalijain/amazon-reviews"


def download_dataset(dataset_id: str = KAGGLE_DATASET_ID) -> Path:
    """
    Faz o download do dataset do Kaggle via kagglehub
    e retorna o diretório onde os arquivos foram salvos.
    """
    print("🔽 Baixando dataset do Kaggle...")
    path_str = kagglehub.dataset_download(dataset_id)
    path = Path(path_str)
    print("📁 Path dos arquivos:", path)
    return path


def reduce_dataset(
    input_file: Path,
    output_file: Path,
    target_size: int = 200_000,
    chunk_size: int = 50_000,
) -> pd.DataFrame:
    """
    Reduz o dataset mantendo todas as colunas.
    Adiciona nomes corretos às colunas conforme README:
        label, title, text
    """

    print(f"\n🔍 Reduzindo dataset: {input_file}")

    if not input_file.exists():
        raise FileNotFoundError(f"Arquivo de entrada não encontrado: {input_file}")

    data_chunks = []
    rows_loaded = 0

    # Leitor em chunks (já aproveitamos o primeiro chunk para descobrir nº de colunas)
    reader = pd.read_csv(input_file, chunksize=chunk_size, header=None)

    try:
        first_chunk = next(reader)
    except StopIteration:
        raise ValueError(f"Arquivo {input_file} está vazio.")

    n_cols = first_chunk.shape[1]
    text_col_index = n_cols - 1

    print(f"📌 Número de colunas: {n_cols}")
    print(f"📌 Índice da coluna de texto (para limpeza mínima): {text_col_index}")

    # Processa primeiro chunk
    first_chunk = first_chunk.dropna(subset=[text_col_index])
    data_chunks.append(first_chunk)
    rows_loaded += len(first_chunk)
    print(f"✔️ Linhas carregadas (primeiro chunk): {rows_loaded}")

    # Processa os demais chunks
    for chunk in reader:
        chunk = chunk.dropna(subset=[text_col_index])
        data_chunks.append(chunk)
        rows_loaded += len(chunk)

        print(f"✔️ Linhas carregadas: {rows_loaded}")

        if rows_loaded >= target_size:
            break

    df = pd.concat(data_chunks, ignore_index=True)

    # Garante o tamanho desejado (ou menor, se não houver linhas suficientes)
    if len(df) > target_size:
        df = df.head(target_size)

    # =============== NOMEANDO COLUNAS ==================
    column_names = ["label", "title", "text"]
    if len(column_names) != df.shape[1]:
        raise Exception(
            f"O dataset tem {df.shape[1]} colunas, mas esperávamos 3. "
            f"Verifique a formatação de {input_file}."
        )

    df.columns = column_names
    # ===================================================

    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, index=False)

    print(f"\n💾 Dataset reduzido salvo em: {output_file}")
    print(f"📊 Formato final: {df.shape}")
    print(f"📌 Colunas: {df.columns.tolist()}")

    return df


def main() -> None:
    # 1. Download
    path = download_dataset()

    # 2. Paths dos arquivos reais
    train_path = path / "train.csv"
    test_path = path / "test.csv"

    if not train_path.exists():
        raise FileNotFoundError(f"train.csv não encontrado em {train_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"test.csv não encontrado em {test_path}")

    # 3. Reduzir TRAIN (maior)
    reduce_dataset(
        input_file=train_path,
        output_file=RAW_TRAIN_SAMPLE,
        target_size=200_000,
        chunk_size=50_000,
    )

    # 4. Reduzir TEST (pode ser menor)
    reduce_dataset(
        input_file=test_path,
        output_file=RAW_TEST_SAMPLE,
        target_size=50_000,
        chunk_size=50_000,
    )


if __name__ == "__main__":
    main()
