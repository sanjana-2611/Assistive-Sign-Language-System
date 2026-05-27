from flask import Flask, render_template, request, jsonify
from ultralytics import YOLO
import cv2
import numpy as np
import base64
import os
import requests
from collections import Counter
import time
from difflib import get_close_matches
from wordfreq import top_n_list

app = Flask(__name__)

# ----------------------------
# LOAD WORD LIST
# ----------------------------
english_words = set(top_n_list("en", 50000))

# ----------------------------
# MODEL PATHS
# ----------------------------
LETTERS_MODEL_PATH = "lettersbest.pt"
WORDS1_MODEL_PATH = "wordbest.pt"
WORDS2_MODEL_PATH = "word2best.pt"


def load_model(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model '{path}' not found")
    return YOLO(path)


letters_model = load_model(LETTERS_MODEL_PATH)
words1_model = load_model(WORDS1_MODEL_PATH)
words2_model = load_model(WORDS2_MODEL_PATH)

# ----------------------------
# GLOBALS
# ----------------------------
prediction_history = []
last_output = None
STABLE_TIME = 6

FRAME_SKIP = 2
frame_count = 0

# ----------------------------
# MISSING LETTERS PATCH
# Manually map common misdetections / similar-looking signs
# Add more pairs here if your model confuses specific letters
# ----------------------------
LETTER_ALIAS = {
    "kr": "k",   # your model outputs "kr" instead of "k"
    "j":  "j",
    "b":  "b",
    "e":  "e",
    "c":  "c",
    "l":  "l",
    "m":  "m",
    "d":  "d",
    "f":  "f",
    "g":  "g",
    "h":  "h",
    "i":  "i",
    "u":  "u",
}

def normalize_letter(label):
    """Fix model-specific quirks in letter labels."""
    label = str(label).strip().lower()
    return LETTER_ALIAS.get(label, label)


# ----------------------------
# LETTER → WORD FIX
# ----------------------------
def letters_to_word(tokens):
    word = "".join(tokens)
    matches = get_close_matches(word, english_words, n=1, cutoff=0.5)  # lowered cutoff
    if matches:
        return matches[0]
    return word


# ----------------------------
# CLEAN TOKENS
# ----------------------------
def clean_tokens(tokens):
    cleaned = []
    last = None
    for t in tokens:
        word = str(t).strip().lower()
        if not word:
            continue
        if word == last:
            continue
        cleaned.append(word)
        last = word
    return cleaned


# ----------------------------
# DETECTION — LETTERS + WORDS
# PATCH: Lower confidence thresholds significantly
# PATCH: Both word models run and best confidence wins
# ----------------------------
def run_all_models(frame):

    global prediction_history, last_output, frame_count

    frame_count += 1

    if frame_count % FRAME_SKIP != 0:
        return frame, []

    annotated = frame.copy()
    word_detections = []
    letter_detections = []

    # ----------------------------
    # WORD MODELS
    # PATCH: Lowered conf from 0.40 → 0.25 (both models)
    # PATCH: Accept if conf > 0.30 instead of 0.50
    # ----------------------------
    for model in [words1_model, words2_model]:
        results = model(frame, conf=0.25)   # <-- PATCHED (was 0.40)

        if results and results[0].boxes is not None:
            for box in results[0].boxes:
                cls_id = int(box.cls[0])
                label = model.names.get(cls_id)
                conf = float(box.conf[0])

                if conf > 0.30:             # <-- PATCHED (was 0.50)
                    word_detections.append((label, conf))

            if results[0].boxes is not None and len(results[0].boxes) > 0:
                annotated = results[0].plot()

    # ----------------------------
    # LETTER MODEL
    # PATCH: Lowered conf from 0.35 → 0.20
    # PATCH: Accept if conf > 0.25 instead of 0.40
    # PATCH: Normalize label to fix missing letters (b,e,c,l,m etc.)
    # ----------------------------
    results = letters_model(frame, conf=0.20)   # <-- PATCHED (was 0.35)

    if results and results[0].boxes is not None:
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            raw_label = letters_model.names.get(cls_id)
            label = normalize_letter(raw_label)  # <-- PATCHED (handles kr, aliases)
            conf = float(box.conf[0])

            if conf > 0.25:             # <-- PATCHED (was 0.40)
                letter_detections.append((label, conf))

        if results[0].boxes is not None and len(results[0].boxes) > 0:
            annotated = results[0].plot()

    # ----------------------------
    # PRIORITY: WORDS FIRST
    # If no confident word found, fall back to letters
    # ----------------------------
    if word_detections:
        detections = word_detections
    elif letter_detections:
        detections = letter_detections
    else:
        return annotated, []

    # ----------------------------
    # BEST CONFIDENCE
    # ----------------------------
    best_label, best_conf = max(detections, key=lambda x: x[1])

    current_time = time.time()
    prediction_history.append((best_label, best_conf, current_time))

    # keep only last STABLE_TIME seconds
    prediction_history = [
        (l, c, t) for l, c, t in prediction_history
        if current_time - t <= STABLE_TIME
    ]

    # ----------------------------
    # STABILITY CHECK
    # PATCH: Lowered minimum history from 5 → 3
    # PATCH: Lowered majority from 0.7 → 0.6
    # ----------------------------
    if len(prediction_history) >= 3:      # <-- PATCHED (was > 5)

        labels = [l for l, _, _ in prediction_history]
        most_common, count = Counter(labels).most_common(1)[0]

        if count >= int(0.6 * len(labels)):   # <-- PATCHED (was 0.7)

            if most_common != last_output:
                last_output = most_common
                prediction_history.clear()
                return annotated, [most_common]

    return annotated, []


# ----------------------------
# LLM SENTENCE
# ----------------------------
def call_llm_from_tokens(tokens):

    tokens = clean_tokens(tokens)
    if not tokens:
        return ""

    if all(len(t) == 1 for t in tokens):
        word = letters_to_word(tokens)
        prompt = f"Create one meaningful sentence using the word '{word}'. Reply with only the sentence, nothing else."
    else:
        token_str = " ".join(tokens)
        prompt = (
            f"Create one meaningful sentence using ALL these words: {token_str}. "
            f"Do not skip any word. Reply with only the sentence, nothing else."
        )

    try:
        response = requests.post(
            "http://localhost:11434/api/chat",
            json={
                "model": "gpt-oss:120b-cloud",
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a sentence generator. Always reply with exactly one sentence and nothing else."
                    },
                    {"role": "user", "content": prompt},
                ],
                "stream": False,
            },
            timeout=30,
        )

        data = response.json()
        if "message" in data:
            return data["message"]["content"].strip()
        return prompt

    except requests.exceptions.Timeout:
        return "LLM timeout — try again."
    except Exception as e:
        return f"Error: {str(e)}"


# ----------------------------
# ROUTES
# ----------------------------
@app.route("/")
def home():
    return render_template("home.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/predict-page")
def predict_page():
    return render_template("predict.html")


@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.get_json()
    img_data = data.get("image")

    image_bytes = base64.b64decode(img_data.split(",")[1])
    np_arr = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    annotated, tokens = run_all_models(frame)

    _, buffer = cv2.imencode(".jpg", annotated)
    encoded_image = base64.b64encode(buffer).decode("utf-8")

    return jsonify({
        "image": encoded_image,
        "tokens": tokens
    })


@app.route("/api/sentence", methods=["POST"])
def api_sentence():
    tokens = request.get_json().get("tokens", [])
    sentence = call_llm_from_tokens(tokens)
    return jsonify({"sentence": sentence})


# ----------------------------
# RUN
# ----------------------------
if __name__ == "__main__":
    app.run(debug=True)