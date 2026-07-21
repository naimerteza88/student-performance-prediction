# Project Explanation in Easy English

## 1. What problem does this project solve?

This project estimates a student's math score by looking at the student's reading score, writing score, and three learning-context fields. It demonstrates how raw data can become a usable machine-learning web application.

The project is for learning. A real school should never use this small dataset or prediction alone to grade, admit, rank, or judge students.

## 2. What data is used?

The dataset contains 1,000 student records. Each row has background categories and three exam scores: math, reading, and writing. `math_score` is the value the model learns to predict.

Although the dataset contains gender and race/ethnicity, I did not give those fields to the model. I wanted the prediction form to avoid sensitive demographic inputs.

## 3. What happens during training?

1. Pandas loads the CSV file.
2. The program checks that every required column is present and has no missing value.
3. Scikit-learn separates the inputs (`X`) from the math score (`y`).
4. The data is split: 80% for training and 20% for testing.
5. Categorical values become machine-readable one-hot columns.
6. Numeric scores are standardized.
7. Three regression algorithms train on exactly the same training data.
8. Each model predicts the untouched test data.
9. The program calculates MAE, RMSE, and R².
10. The model with the best R² is saved together with preprocessing.

## 4. Why compare multiple models?

No algorithm is automatically best for every dataset. Linear regression is a strong simple baseline. Gradient boosting and random forest can learn more complex relationships. Comparing them with the same data and metrics lets the evidence choose the final model.

## 5. How does the web app work?

The Flask app shows an HTML form. When a user submits the form, Python validates every value, turns the values into a one-row pandas DataFrame, loads the saved pipeline, and returns a predicted score. The result is limited to the meaningful 0–100 range.

## 6. What do the main files do?

- `data.py`: loads and validates data.
- `train.py`: builds preprocessing, trains models, evaluates them, and saves the winner.
- `predict.py`: defines safe input validation and prediction logic.
- `app.py`: connects the HTML page to the Python predictor.
- `index.html`: defines the form and result area.
- `style.css`: makes the page clean and responsive.
- `tests/`: checks validation, training, prediction, and web routes.
- `ci.yml`: automatically tests the project on GitHub.

## 7. What is the biggest technical idea?

The preprocessing steps and the trained estimator are one saved `Pipeline`. The same transformation learned during training is therefore used during every future prediction. This avoids a common production bug where training and application inputs are processed differently.

## 8. What did I improve beyond the tutorial?

- Wrote the implementation independently with a compact package structure.
- Removed sensitive demographic fields from model inputs.
- Added validation and clearer errors.
- Added automated tests and GitHub Actions.
- Added Docker support.
- Added a detailed README and easy-English project explanation.
