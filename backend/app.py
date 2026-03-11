from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load model
with open("student_performance_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load feature names
with open("feature_names.pkl", "rb") as f:
    feature_names = pickle.load(f)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    age = float(request.form["age"])
    study_hours = float(request.form["study_hours"])
    class_attendance = float(request.form["class_attendance"])
    sleep_hours = float(request.form["sleep_hours"])

    # Feature engineering (same as training)
    study_sleep_ratio = study_hours / sleep_hours
    study_attendance = study_hours * class_attendance
    productivity_score = (study_hours + class_attendance) / sleep_hours

    input_data = pd.DataFrame([[
        age,
        study_hours,
        class_attendance,
        sleep_hours,
        study_sleep_ratio,
        study_attendance,
        productivity_score
    ]], columns=feature_names)

    prediction = model.predict(input_data)[0]

    return render_template(
        "index.html",
        prediction_text=f"Predicted Performance: {prediction}"
    )


if __name__ == "__main__":
    app.run(debug=True)