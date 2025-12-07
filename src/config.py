from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

RAW_TRAIN_SAMPLE = RAW_DIR / "amazon_reviews_train_sample.csv"
RAW_TEST_SAMPLE = RAW_DIR / "amazon_reviews_test_sample.csv"

DATA_PROCESSED_TRAIN = PROCESSED_DIR / "train.csv"
DATA_PROCESSED_TEST = PROCESSED_DIR / "test.csv"

MODEL_PATH = BASE_DIR / "models" / "sentiment_model.joblib"

RANDOM_STATE = 42