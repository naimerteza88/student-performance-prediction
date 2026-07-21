from __future__ import annotations

import os

from flask import Flask, render_template, request

from student_performance.predict import (
    EDUCATION_LEVELS,
    LUNCH_OPTIONS,
    PREPARATION_OPTIONS,
    PerformancePredictor,
    StudentInput,
)

app = Flask(__name__)


@app.get("/")
def home():
    return render_template(
        "index.html",
        education_levels=EDUCATION_LEVELS,
        lunch_options=LUNCH_OPTIONS,
        preparation_options=PREPARATION_OPTIONS,
        form={},
    )


@app.post("/predict")
def predict():
    form = request.form.to_dict()
    prediction = None
    error = None
    try:
        student = StudentInput(
            parental_level_of_education=form.get("parental_level_of_education", ""),
            lunch=form.get("lunch", ""),
            test_preparation_course=form.get("test_preparation_course", ""),
            reading_score=float(form.get("reading_score", "")),
            writing_score=float(form.get("writing_score", "")),
        )
        prediction = PerformancePredictor().predict(student)
    except (TypeError, ValueError) as exc:
        error = str(exc) if str(exc) else "Enter valid values in every field."
    except FileNotFoundError:
        error = "Model is not trained yet. Run the training command, then refresh this page."

    return render_template(
        "index.html",
        education_levels=EDUCATION_LEVELS,
        lunch_options=LUNCH_OPTIONS,
        preparation_options=PREPARATION_OPTIONS,
        form=form,
        prediction=prediction,
        error=error,
    )


@app.get("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "5000")), debug=False)
