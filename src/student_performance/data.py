from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import DATA_PATH, MODEL_FEATURES, TARGET


def load_dataset(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load the dataset and validate the fields needed by the model."""
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    frame = pd.read_csv(path)
    required = set(MODEL_FEATURES + [TARGET])
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {', '.join(missing)}")

    if frame[list(required)].isnull().any().any():
        raise ValueError("Dataset contains missing values in required columns")

    return frame
