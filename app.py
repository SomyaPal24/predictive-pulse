from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

model = joblib.load("model/logreg_model.pkl")
scaler = joblib.load("model/scaler.pkl")

stage_dict = {
0:"Normal",
1:"Stage-1 Hypertension",
2:"Stage-2 Hypertension",
3:"Hypertensive Crisis",
4:"Severe Hypertension",
}

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    features = [float(x) for x in request.form.values()]

    features = np.array([features])

    features_scaled = scaler.transform(features)

    prediction = model.predict(features_scaled)[0]

    confidence = np.max(model.predict_proba(features_scaled))*100

    result = stage_dict.get(int(prediction),"Unknown Stage")

    return render_template(
        "index.html",
        prediction=result,
        confidence=round(confidence,2)
    )


if __name__ == "__main__":
    app.run(debug=True)