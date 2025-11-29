import kagglehub
import pandas as pd
import os


def download_dataset(dataset_id: str) -> str:
    print("🔽 Baixando dataset do Kaggle...")
    path = kagglehub.dataset_download(dataset_id)
    print("📁 Path dos arquivos:", path)
    return path


def reduce_dataset(input_file: str, output_file: str, target_size=200_000, chunk_size=50_000):
    """
    Reduz o dataset mantendo todas as colunas.
    Adiciona nomes corretos às colunas conforme README.
    """

    print(f"\n🔍 Reduzindo dataset: {input_file}")

    data_chunks = []
    rows_loaded = 0

    # Detectar número de colunas
    sample = pd.read_csv(input_file, nrows=5, header=None)
    n_cols = sample.shape[1]
    text_col_index = n_cols - 1

    print(f"📌 Número de colunas: {n_cols}")
    print(f"📌 Índice da coluna de texto (para limpeza mínima): {text_col_index}")

    for chunk in pd.read_csv(input_file, chunksize=chunk_size, header=None):

        # Remover linhas sem texto
        chunk = chunk.dropna(subset=[text_col_index])

        data_chunks.append(chunk)
        rows_loaded += len(chunk)

        print(f"✔️ Linhas carregadas: {rows_loaded}")

        if rows_loaded >= target_size:
            break

    df = pd.concat(data_chunks, ignore_index=True)
    df = df.head(target_size)

    # =============== AQUI NOMEAMOS AS COLUNAS ===============
    column_names = ["label", "title", "text"]
    if len(column_names) != df.shape[1]:
        raise Exception(f"O dataset tem {df.shape[1]} colunas, mas esperávamos 3. Verifique formatação.")

    df.columns = column_names
    # ========================================================

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df.to_csv(output_file, index=False)

    print(f"\n💾 Dataset reduzido salvo em: {output_file}")
    print(f"📊 Formato final: {df.shape}")
    print(f"📌 Colunas: {df.columns.tolist()}")

    return df


def main():
    dataset_id = "kritanjalijain/amazon-reviews"

    # 1. Download
    path = download_dataset(dataset_id)

    # 2. Paths dos arquivos reais
    train_path = os.path.join(path, "train.csv")
    test_path = os.path.join(path, "test.csv")

    if not os.path.exists(train_path):
        raise FileNotFoundError("train.csv não encontrado!")
    if not os.path.exists(test_path):
        raise FileNotFoundError("test.csv não encontrado!")

    # 3. Reduzir TRAIN (maior)
    reduce_dataset(
        input_file=train_path,
        output_file="data/amazon_reviews_train_sample.csv",
        target_size=200_000,
        chunk_size=50_000
    )

    # 4. Reduzir TEST (pode ser menor)
    reduce_dataset(
        input_file=test_path,
        output_file="data/amazon_reviews_test_sample.csv",
        target_size=50_000,
        chunk_size=50_000
    )


if __name__ == "__main__":
    main()
