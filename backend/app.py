from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
import pickle
import os

app = Flask(__name__)

# Load model
MODEL_PATH = os.path.join("ml_model", "preview_model.pkl")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Validate and get numeric inputs
        age = request.form.get("age")
        gender = request.form.get("gender")
        study_hours = request.form.get("study_hours")
        attendance = request.form.get("attendance")
        internet = request.form.get("internet")
        sleep_hours = request.form.get("sleep_hours")
        sleep_quality = request.form.get("sleep_quality")
        facility = request.form.get("facility")
        exam_diff = request.form.get("exam_diff")
        course = request.form.get("course")
        method = request.form.get("method")

        # Check for missing fields
        if not all([age, gender, study_hours, attendance, internet, sleep_hours, sleep_quality, facility, exam_diff, course, method]):
            return jsonify({
                "success": False,
                "error": "All fields are required"
            }), 400

        # Convert to appropriate types
        age = float(age)
        gender = int(gender)
        study_hours = float(study_hours)
        attendance = float(attendance)
        internet = int(internet)
        sleep_hours = float(sleep_hours)
        sleep_quality = float(sleep_quality)
        facility = float(facility)
        exam_diff = float(exam_diff)
        course = course.lower().strip()
        method = method.lower().strip()

        # One Hot Encoding
        courses = ["b.com","b.sc","b.tech","ba","bba","bca","diploma"]
        study_methods = ["coaching","group study","mixed","online videos","self-study"]

        course_encoded = [1 if course == c else 0 for c in courses]
        method_encoded = [1 if method == m else 0 for m in study_methods]

        # Combine features
        features = [
            age, gender, study_hours, attendance, internet,
            sleep_hours, sleep_quality, facility, exam_diff
        ]

        # Debug: Print model's expected feature names
        print("Model's expected feature names:")
        if hasattr(model, 'feature_names_in_'):
            print(model.feature_names_in_)
        else:
            print("No feature names found in model")

        # Create feature names to match model EXACTLY
        feature_names = [
            'age', 'gender', 'study_hours', 'class_attendance', 'internet_access',
            'sleep_hours', 'sleep_quality', 'facility_rating', 'exam_difficulty',
            'course_b.com', 'course_b.sc', 'course_b.tech', 'course_ba', 'course_bba', 'course_bca', 'course_diploma',
            'study_method_coaching', 'study_method_group study', 'study_method_mixed', 'study_method_online videos', 'study_method_self-study'
        ]

        # Create numpy array
        final_features_array = np.array(features + course_encoded + method_encoded).reshape(1, -1)
        
        # Convert to DataFrame with feature names
        final_features = pd.DataFrame(final_features_array, columns=feature_names)

        # Debug: Print feature shape and values
        print(f"Feature shape: {final_features.shape}")
        print(f"Features:\n{final_features}")
        print(f"Model expected features: {getattr(model, 'n_features_in_', 'Unknown')}")

        # Prediction (0=FAIL, 1=PASS)
        print("Making prediction...")
        try:
            prediction_result = model.predict(final_features)
            print(f"Prediction result: {prediction_result}, type: {type(prediction_result)}")
            
            # Model returns string categories directly
            performance = prediction_result[0]
            print(f"Performance category: {performance}")
            
            # Map performance string to category number
            performance_to_category = {
                "Excellent": 0,
                "Good": 1,
                "Average": 2,
                "Poor": 3
            }
            
            category = performance_to_category.get(performance, 0)
            print(f"Category number: {category}")
            
        except Exception as pred_error:
            print(f"ERROR in prediction: {pred_error}")
            print(f"Error type: {type(pred_error)}")
            raise pred_error

        # Get probability/confidence
        print("Getting probabilities...")
        try:
            probabilities = model.predict_proba(final_features)[0]
            print(f"Probabilities: {probabilities}")
            confidence = float(max(probabilities)) * 100
            print(f"Confidence: {confidence}")
        except Exception as prob_error:
            print(f"Error getting probabilities: {prob_error}")
            confidence = 0.0

        # Determine prediction result message
        if performance == "Excellent":
            result = "Student Performance is Excellent"
        elif performance == "Good":
            result = "Student Performance is Good"
        elif performance == "Average":
            result = "Student Performance is Average"
        else:  # Poor
            result = "Student Performance is Poor"
        
        print(f"Result: {result}")
        print(f"Category: {category}")

        # Performance and emoji mapping
        performance_map = {
            0: "Excellent",
            1: "Good",
            2: "Average",
            3: "Poor"
        }

        emojis = {
            0: "⭐",
            1: "😊",
            2: "👍",
            3: "⚠️"
        }

        response = {
            "success": True,
            "prediction": result,
            "performance": performance_map[category],
            "confidence": round(confidence, 2),
            "emoji": emojis[category],
            "category": category
        }
        
        print(f"Response: {response}")
        return jsonify(response)

    except ValueError as e:
        return jsonify({
            "success": False,
            "error": f"Invalid input format: {str(e)}"
        }), 400
    except Exception as e:
        error_msg = str(e)
        # Check for feature mismatch
        if "n_features" in error_msg or "feature" in error_msg:
            return jsonify({
                "success": False,
                "error": f"Model feature mismatch: {error_msg}. Please check that the model expects the correct number of features."
            }), 400
        return jsonify({
            "success": False,
            "error": error_msg
        }), 400


if __name__ == "__main__":
    app.run(debug=True)