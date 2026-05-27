from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
from ultralytics import YOLO
import base64

app = Flask(__name__)

# Load model once
model = YOLO("best.pt")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    data = request.json["image"]

    image_data = base64.b64decode(data.split(",")[1])

    np_arr = np.frombuffer(image_data, np.uint8)

    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    results = model(frame, conf=0.25)

    annotated = results[0].plot()

    _, buffer = cv2.imencode(".jpg", annotated)

    encoded = base64.b64encode(buffer).decode("utf-8")

    return jsonify({"image": encoded})


if __name__ == "__main__":
    app.run(debug=True)