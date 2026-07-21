from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "student_performance.csv"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
MODEL_PATH = ARTIFACT_DIR / "student_performance_pipeline.joblib"
METRICS_PATH = ARTIFACT_DIR / "model_metrics.json"

TARGET = "math_score"
NUMERIC_FEATURES = ["reading_score", "writing_score"]
CATEGORICAL_FEATURES = [
    "parental_level_of_education",
    "lunch",
    "test_preparation_course",
]
MODEL_FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES
