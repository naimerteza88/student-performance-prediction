from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import pandas as pd

from .config import MODEL_PATH

EDUCATION_LEVELS = (
    "some high school",
    "high school",
    "some college",
    "associate's degree",
    "bachelor's degree",
    "master's degree",
)
LUNCH_OPTIONS = ("standard", "free/reduced")
PREPARATION_OPTIONS = ("none", "completed")


@dataclass(frozen=True)
class StudentInput:
    parental_level_of_education: str
    lunch: str
    test_preparation_course: str
    reading_score: float
    writing_score: float

    def validate(self) -> None:
        if self.parental_level_of_education not in EDUCATION_LEVELS:
            raise ValueError("Choose a valid parental education level")
        if self.lunch not in LUNCH_OPTIONS:
            raise ValueError("Choose a valid lunch option")
        if self.test_preparation_course not in PREPARATION_OPTIONS:
            raise ValueError("Choose a valid test preparation option")
        for field_name, score in (
            ("reading score", self.reading_score),
            ("writing score", self.writing_score),
        ):
            if not 0 <= score <= 100:
                raise ValueError(f"{field_name.capitalize()} must be between 0 and 100")

    def to_frame(self) -> pd.DataFrame:
        self.validate()
        return pd.DataFrame([asdict(self)])


class PerformancePredictor:
    def __init__(self, model_path: Path = MODEL_PATH) -> None:
        if not model_path.exists():
            raise FileNotFoundError(
                f"Trained model not found at {model_path}. Run: python -m student_performance.train"
            )
        self.pipeline = joblib.load(model_path)

    def predict(self, student: StudentInput) -> float:
        prediction = float(self.pipeline.predict(student.to_frame())[0])
        return round(max(0.0, min(100.0, prediction)), 1)
