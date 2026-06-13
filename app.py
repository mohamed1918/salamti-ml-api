import numpy as np
import joblib
from flask import Flask, request, jsonify

app = Flask(__name__)

# load trained model
model = joblib.load("model.pkl")


def gyro_magnitude(x, y):
    return np.sqrt(x**2 + y**2)


@app.route("/predict-severity", methods=["POST"])
def predict():

    data = request.get_json()

    peakG = float(data["peakG"])
    gyroX = float(data["gyroX"])
    gyroY = float(data["gyroY"])

    gyro_mag = gyro_magnitude(gyroX, gyroY)

    features = np.array([[peakG, gyro_mag]])

    prediction = model.predict(features)[0]

    return jsonify({
        "severity": prediction,
        "gyro_magnitude": float(gyro_mag)
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)