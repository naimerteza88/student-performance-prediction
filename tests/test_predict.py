from pathlib import Path

import pytest

from student_performance.predict import PerformancePredictor, StudentInput
from student_performance.train import train_and_save


def valid_student() -> StudentInput:
    return StudentInput(
        parental_level_of_education="bachelor's degree",
        lunch="standard",
        test_preparation_course="completed",
        reading_score=78,
        writing_score=81,
    )


def test_student_validation_rejects_out_of_range_score():
    student = StudentInput(
        parental_level_of_education="high school",
        lunch="standard",
        test_preparation_course="none",
        reading_score=101,
        writing_score=60,
    )

    with pytest.raises(ValueError, match="between 0 and 100"):
        student.validate()


def test_train_and_predict(tmp_path: Path):
    model_path = tmp_path / "model.joblib"
    metrics_path = tmp_path / "metrics.json"
    report = train_and_save(model_path=model_path, metrics_path=metrics_path)

    assert model_path.exists()
    assert metrics_path.exists()
    assert report["best_model"] in report["models"]

    score = PerformancePredictor(model_path).predict(valid_student())
    assert 0 <= score <= 100
