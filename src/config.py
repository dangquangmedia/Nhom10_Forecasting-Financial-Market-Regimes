from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODEL_DIR = DATA_DIR / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

RAW_OHLCV_PATH = RAW_DATA_DIR / "sp500_ohlcv.csv"
RAW_SECTORS_PATH = RAW_DATA_DIR / "sp500_companies.csv"
RAW_VIX_PATH = RAW_DATA_DIR / "vixcls.csv"

CLEAN_MARKET_INPUTS_PATH = PROCESSED_DATA_DIR / "sp500_clean.csv"
STOCK_FEATURES_PATH = PROCESSED_DATA_DIR / "stock_features.csv"
MARKET_FEATURES_PATH = PROCESSED_DATA_DIR / "market_features.csv"
FINAL_DATASET_PATH = PROCESSED_DATA_DIR / "final_dataset.csv"
MODEL_RESULTS_PATH = REPORTS_DIR / "model_results.csv"
XGBOOST_TEST_PREDICTIONS_PATH = REPORTS_DIR / "xgboost_test_predictions.csv"
XGBOOST_CONFUSION_MATRIX_PATH = REPORTS_DIR / "xgboost_confusion_matrix.csv"
XGBOOST_CLASSIFICATION_REPORT_PATH = REPORTS_DIR / "xgboost_classification_report.csv"
XGBOOST_FEATURE_IMPORTANCE_PATH = REPORTS_DIR / "xgboost_feature_importance.csv"
LOGISTIC_REGRESSION_MODEL_PATH = MODEL_DIR / "logistic_regression.joblib"
RANDOM_FOREST_MODEL_PATH = MODEL_DIR / "random_forest.joblib"
XGBOOST_MODEL_PATH = MODEL_DIR / "xgboost.joblib"

PRIMARY_MODEL_NAME = "xgboost"
RANDOM_STATE = 42
