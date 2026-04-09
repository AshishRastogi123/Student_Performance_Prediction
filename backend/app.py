from flask import Flask, render_template, request, jsonify
import pandas as pd
import os
import joblib
import traceback
import sys

app = Flask(__name__)

# Force immediate prints
sys.stdout.flush()

print("🚀 Flask app starting...", flush=True)

# ========= LOAD MODEL =========
MODEL_PATH = os.path.join("ml_model", "students.pkl")
try:
    model = joblib.load(MODEL_PATH)
    print("✅ Model loaded:", type(model).__name__, flush=True)
    print("📊 Model features expected:", getattr(model, 'n_features_in_', 'NA'), flush=True)
    if hasattr(model, 'feature_names_in_'):
        print("📋 Feature names:", list(model.feature_names_in_), flush=True)
except Exception as e:
    print("❌ Model load failed:", str(e), flush=True)
    traceback.print_exc()
    model = None

# ========= GLOBAL BEFORE REQUEST =========
@app.before_request
def log_request():
    print(f"\n📨 GLOBAL REQUEST: {request.method} {request.path}", flush=True)
    if request.form:
        print(f"📄 request.form: {dict(request.form)}", flush=True)
    print(f"🌐 Content-Type: {request.content_type}", flush=True)
    if request.is_json:
        print(f"📄 request.json: {request.json}", flush=True)
    sys.stdout.flush()

# ========= GLOBAL ERROR HANDLER =========
@app.errorhandler(Exception)
def handle_error(error):
    print("🚨 GLOBAL ERROR:", type(error).__name__, flush=True)
    print("💥 Error message:", str(error), flush=True)
    traceback.print_exc()
    sys.stdout.flush()
    
    return jsonify({
        "success": False,
        "error": str(error),
        "traceback": traceback.format_exc()
    }), 500

# ========= HOME =========
@app.route("/")
def home():
    print("🏠 Home route hit", flush=True)
    return render_template("index.html")

# ========= FAVICON =========
@app.route('/favicon.ico')
def favicon():
    return '', 404

# ========= TEST ROUTE =========
@app.route("/test", methods=["GET"])
def test():
    print("🧪 TEST ROUTE - SERVER WORKING ✅", flush=True)
    sys.stdout.flush()
    return jsonify({"status": "SERVER WORKING", "message": "All good!"})

# ========= PREDICT =========
@app.route("/predict", methods=["POST"])
def predict():
    print("\n🎯 API HIT - /predict called!", flush=True)
    
    try:
        # ===== STEP 1: LOG ALL INCOMING DATA =====
        print("📥 All form data:", dict(request.form), flush=True)
        data = {k: request.form.get(k, '').strip() for k in [
            "age", "gender", "study_hours", "attendance", "internet", 
            "sleep_hours", "sleep_quality", "facility", "exam_diff", 
            "course", "method"
        ]}
        print("🔍 Parsed data:", data, flush=True)

        # Validate
        missing = [k for k, v in data.items() if not v]
        if missing:
            error = f"Missing fields: {missing}"
            print(f"❌ Validation error: {error}", flush=True)
            return jsonify({"success": False, "error": error}), 400

        # ===== STEP 2: NUMERIC TRANSFORM =====
        numeric_map = {
            "age": float(data["age"]),
            "study_hours": float(data["study_hours"]),
            "attendance": float(data["attendance"]),
            "internet": int(data["internet"]),
            "sleep_hours": float(data["sleep_hours"]),
            "sleep_quality": int(data["sleep_quality"]),
            "facility": int(data["facility"]),
            "exam_diff": int(data["exam_diff"])
        }
        print("🔢 Numeric features:", numeric_map, flush=True)

        # ===== STEP 3: GENDER ONE-HOT =====
        g = data["gender"].lower()
        gender_features = {
            "gender_female": 1 if g in ["1", "female"] else 0,
            "gender_male": 1 if g in ["0", "male"] else 0,
            "gender_other": 1 if g in ["2", "other"] else 0
        }
        print("⚧️ Gender one-hot:", gender_features, flush=True)

        # ===== STEP 4: COURSE ONE-HOT =====
        courses = ["b.com", "b.sc", "b.tech", "ba", "bba", "bca", "diploma"]
        course_features = {f"course_{c}": 1 if data["course"].lower() == c else 0 for c in courses}
        print("📚 Course one-hot:", course_features, flush=True)

        # ===== STEP 5: METHOD ONE-HOT =====
        methods = ["coaching", "group study", "mixed", "online videos", "self-study"]
        method_features = {f"study_method_{m.replace(' ', '_')}": 1 if data["method"].lower() == m else 0 for m in methods}
        print("📖 Method one-hot:", method_features, flush=True)

        # ===== STEP 6: FINAL FEATURE DICT =====
        input_dict = {
            "age": numeric_map["age"],
            "study_hours": numeric_map["study_hours"],
            "class_attendance": numeric_map["attendance"],
            "internet_access": numeric_map["internet"],
            "sleep_hours": numeric_map["sleep_hours"],
            "sleep_quality": numeric_map["sleep_quality"],
            "facility_rating": numeric_map["facility"],
            "exam_difficulty": numeric_map["exam_diff"],
            **gender_features, **course_features, **method_features
        }
        print("🏗️ Final input dict keys:", list(input_dict.keys()), flush=True)
        print("📐 Input dict shape (len):", len(input_dict), flush=True)

        # ===== STEP 7: CREATE DATAFRAME =====
        df = pd.DataFrame([input_dict])
        print("📊 Initial DF shape:", df.shape, flush=True)
        print("📋 Initial DF columns:", list(df.columns), flush=True)

        # ===== STEP 8: ALIGN WITH MODEL FEATURES =====
        if model and hasattr(model, 'feature_names_in_'):
            expected_cols = model.feature_names_in_
            print(f"🎯 Model expects {len(expected_cols)} features: {list(expected_cols)}", flush=True)
            df = df.reindex(columns=expected_cols, fill_value=0)
            print("✅ Aligned DF shape:", df.shape, flush=True)
            print("📋 Final columns:", list(df.columns), flush=True)
        else:
            print("⚠️ No feature_names_in_, using raw DF", flush=True)

        sys.stdout.flush()

        # ===== STEP 9: MODEL PREDICTION =====
        if not model:
            raise ValueError("Model not loaded")
            
        pred = model.predict(df)
        print("🔮 Raw prediction:", pred, flush=True)
        
        label = int(pred[0])
        mapping = {0: "Poor", 1: "Average", 2: "Excellent"}
        performance = mapping.get(label, "Unknown")

        # ===== STEP 10: PROBABILITIES =====
        try:
            probs = model.predict_proba(df)[0]
            print("📈 Probabilities:", probs.tolist(), flush=True)
            confidence = max(probs) * 100
        except:
            print("⚠️ No predict_proba available", flush=True)
            confidence = 0

        print(f"🎉 FINAL RESULT: {performance} (confidence: {confidence:.2f}%)", flush=True)
        sys.stdout.flush()

        return jsonify({
            "success": True,
            "prediction": performance,
            "confidence": round(confidence, 2),
            "label": label,
            "probs": probs.tolist() if 'probs' in locals() else None
        })

    except Exception as e:
        print("🚨 PREDICT ERROR:", type(e).__name__, flush=True)
        print("💥 Error details:", str(e), flush=True)
        traceback.print_exc()
        sys.stdout.flush()
        return jsonify({"success": False, "error": str(e)}), 400

# ========= RUN SERVER =========
if __name__ == "__main__":
    print("🌐 Starting Flask server in DEBUG mode...", flush=True)
    app.run(debug=True, host='0.0.0.0', port=5000)
    sys.stdout.flush()
