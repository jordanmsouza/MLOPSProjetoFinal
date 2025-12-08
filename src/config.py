from pathlib import Path

# Diretório base do projeto
BASE_DIR = Path(__file__).resolve().parent.parent

# Dados
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

RAW_TRAIN_SAMPLE = RAW_DIR / "amazon_reviews_train_sample.csv"
RAW_TEST_SAMPLE = RAW_DIR / "amazon_reviews_test_sample.csv"

DATA_PROCESSED_TRAIN = PROCESSED_DIR / "train.csv"
DATA_PROCESSED_TEST = PROCESSED_DIR / "test.csv"

# Modelo local
MODEL_PATH = BASE_DIR / "models" / "sentiment_model.joblib"

# MLflow
MLFLOW_TRACKING_DIR = BASE_DIR / "mlruns"
MLFLOW_TRACKING_URI = "file:mlruns"   # <-- chave para funcionar em qualquer lugar
MLFLOW_EXPERIMENT_NAME = "sentiment_analysis_amazon_reviews"
MLFLOW_MODEL_NAME = "sentiment-logreg-tfidf"

RANDOM_STATE = 42
