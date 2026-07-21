from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import (
    CATEGORICAL_FEATURES,
    DATA_PATH,
    METRICS_PATH,
    MODEL_FEATURES,
    MODEL_PATH,
    NUMERIC_FEATURES,
    TARGET,
)
from .data import load_dataset

RANDOM_STATE = 42


def build_preprocessor() -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("one_hot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )


def candidate_models() -> dict[str, object]:
    return {
        "linear_regression": LinearRegression(),
        "gradient_boosting": GradientBoostingRegressor(random_state=RANDOM_STATE),
        "random_forest": RandomForestRegressor(
            n_estimators=200,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }


def train_and_save(
    data_path: Path = DATA_PATH,
    model_path: Path = MODEL_PATH,
    metrics_path: Path = METRICS_PATH,
) -> dict[str, object]:
    frame = load_dataset(data_path)
    features = frame[MODEL_FEATURES]
    target = frame[TARGET]
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=RANDOM_STATE,
    )

    evaluations: dict[str, dict[str, float]] = {}
    fitted_pipelines: dict[str, Pipeline] = {}

    for name, estimator in candidate_models().items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                ("model", estimator),
            ]
        )
        pipeline.fit(x_train, y_train)
        predictions = pipeline.predict(x_test)
        evaluations[name] = {
            "mae": round(float(mean_absolute_error(y_test, predictions)), 4),
            "rmse": round(float(mean_squared_error(y_test, predictions) ** 0.5), 4),
            "r2": round(float(r2_score(y_test, predictions)), 4),
        }
        fitted_pipelines[name] = pipeline

    best_model_name = max(evaluations, key=lambda name: evaluations[name]["r2"])
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(fitted_pipelines[best_model_name], model_path)

    report: dict[str, object] = {
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_rows": len(frame),
        "train_rows": len(x_train),
        "test_rows": len(x_test),
        "target": TARGET,
        "features": MODEL_FEATURES,
        "excluded_sensitive_features": ["gender", "race_ethnicity"],
        "best_model": best_model_name,
        "models": evaluations,
    }
    metrics_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the student performance model.")
    parser.add_argument("--data", type=Path, default=DATA_PATH, help="Path to the input CSV")
    parser.add_argument("--model", type=Path, default=MODEL_PATH, help="Output model path")
    parser.add_argument("--metrics", type=Path, default=METRICS_PATH, help="Output metrics path")
    args = parser.parse_args()

    report = train_and_save(args.data, args.model, args.metrics)
    best = report["best_model"]
    score = report["models"][best]["r2"]
    print(f"Training complete. Best model: {best} (R²={score:.4f})")
    print(f"Model saved to: {args.model}")


if __name__ == "__main__":
    main()
