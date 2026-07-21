# 2–5 Minute Demo Video Script

Use this as speaking guidance. Do not read it like a robot—practice it and explain the project in your own natural words.

## 0:00–0:25 — Introduction

Hello everyone. I am Naim Erteza. In this project, I built an end-to-end machine-learning application that predicts a student's math score. My goal was not only to train a model, but to complete the whole workflow: data validation, preprocessing, model comparison, prediction, testing, and a simple web interface.

## 0:25–0:55 — Problem and data

The dataset has 1,000 student records, including reading, writing, and math scores plus several context fields. Math score is my target. The model uses reading and writing scores, parent or guardian education, lunch type, and test-preparation status. I deliberately excluded gender and race or ethnicity from prediction because sensitive demographic information should not determine an individual result.

## 0:55–1:45 — Implementation

[Show the repository structure and open `train.py`.]

First, pandas loads the CSV and checks the required columns. I split the data into 80 percent training and 20 percent testing. My scikit-learn preprocessing pipeline one-hot encodes categorical fields and standardizes numeric scores.

I trained and compared three regression algorithms: linear regression, gradient boosting, and random forest. All models use the same train-test split. I compare them using MAE, RMSE, and R-squared, then save the best model together with preprocessing as one pipeline. This is important because the web app must transform new input exactly as the training code did.

## 1:45–2:20 — Application demo

[Run `python app.py` and show the browser.]

This is the Flask interface. I select the context values and enter reading and writing scores. After I click Predict, the form sends the data to Flask. The application validates it, loads the saved pipeline, and returns an estimated math score from zero to one hundred.

[Submit one prediction, then briefly show the validation by trying 101 if desired.]

## 2:20–2:50 — Challenges

One challenge was keeping preprocessing consistent between training and prediction. I solved it by saving the complete pipeline, not only the model. Another challenge was deciding which inputs were technically available but not responsible to use. That led me to exclude sensitive demographic fields. I also added tests so that invalid scores, model training, prediction, and application routes can be checked automatically.

## 2:50–3:20 — Learning and close

This project taught me that a real machine-learning implementation is more than algorithm accuracy. It also needs reproducible data processing, fair model comparison, input validation, documentation, testing, and responsible-use limits. The complete code and setup steps are available in my GitHub repository. Thank you for watching.

## Recording checklist

- Keep the GitHub repository, terminal, and web app ready before recording.
- Increase browser and editor text size so viewers can read it.
- Show the README, key training code, generated metrics, and one live prediction.
- Keep your webcam at eye level and use a quiet room.
- Record at 1080p if possible and finish between 2 and 5 minutes.
- Upload the final video to LinkedIn directly or to a public/unlisted video host.
