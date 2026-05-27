from flask import Flask, render_template, request, jsonify
from ultralytics import YOLO
import cv2
import numpy as np
import base64
import os
import requests
from collections import Counter, deque
import time

app = Flask(__name__)

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
# VOCAB (KNOWN WORDS)
# ----------------------------

VOCAB = [
    "hungry", "thirsty", "milk", "water", "sleep",
    "chips", "chocolate", "biscuit", "icecream",
    "tv", "fan", "door", "phone", "toys",
    "mummy", "daddy", "akka", "washroom",
    "beautiful", "delicious",
    "thankyou", "iloveyou", "no", "yes"
]

# ----------------------------
# WORD MODE BUFFER
# ----------------------------

prediction_buffer = deque(maxlen=8)
last_output = None

# ----------------------------
# LETTER MODE VARIABLES
# ----------------------------

letter_buffer = []
letter_conf_buffer = []

letter_start_time = time.time()
last_detection_time = time.time()

# ----------------------------
# WORD MODE FUNCTION
# ----------------------------

def run_words_mode(frame):

    global last_output

    annotated = frame.copy()
    predictions = []

    for model in [words1_model, words2_model]:

        results = model(frame, conf=0.35)

        if results:
            r = results[0]

            if r.boxes is not None:
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    label = model.names.get(cls_id, str(cls_id))
                    predictions.append(label)

            annotated = r.plot()

    if not predictions:
        return annotated, []

    top_prediction = Counter(predictions).most_common(1)[0][0]
    prediction_buffer.append(top_prediction)

    if len(prediction_buffer) == prediction_buffer.maxlen:
        stable_prediction = Counter(prediction_buffer).most_common(1)[0][0]

        if stable_prediction != last_output:
            last_output = stable_prediction
            prediction_buffer.clear()
            return annotated, [stable_prediction]

    return annotated, []

# ----------------------------
# LETTER MODE FUNCTION
# ----------------------------

def run_letters_mode(frame):

    global letter_buffer, letter_conf_buffer
    global letter_start_time

    annotated = frame.copy()

    results = letters_model(frame, conf=0.30)

    if results:
        r = results[0]

        if r.boxes is not None:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                label = letters_model.names.get(cls_id, str(cls_id))
                letter_buffer.append(label)

        annotated = r.plot()

    current_time = time.time()

    # Reduce time (better UX)
    if current_time - letter_start_time >= 5 and letter_buffer:

        freq = Counter(letter_buffer)
        best_letter = freq.most_common(1)[0][0]

        letter_buffer.clear()
        letter_start_time = current_time

        return annotated, [best_letter]

    return annotated, []

# ----------------------------
# CLEAN TOKENS
# ----------------------------

def clean_tokens(tokens):
    return [str(t).strip().lower() for t in tokens if str(t).strip()]

# ----------------------------
# MATCH WORDS (ORDER INDEPENDENT)
# ----------------------------

def match_words_from_letters(tokens):

    letters = clean_tokens(tokens)
    if not letters:
        return []

    input_count = Counter(letters)
    matched_words = []

    for word in VOCAB:
        word_clean = word.replace(" ", "")
        word_count = Counter(word_clean)

        score = sum(min(input_count[c], word_count[c]) for c in word_count)

        # STRICT matching (avoid wrong guesses)
        if score >= len(word_clean) - 1:
            matched_words.append((word, score))

    matched_words.sort(key=lambda x: x[1], reverse=True)

    return [w[0] for w in matched_words[:2]]

# ----------------------------
# LLM SENTENCE GENERATION
# ----------------------------

def call_llm_from_tokens(tokens):

    tokens = clean_tokens(tokens)

    if not tokens:
        return ""

    words = match_words_from_letters(tokens)

    if not words:
        return "Unclear input"

    word_text = " ".join(words)

    print("Matched words:", word_text)

    try:
        response = requests.post(
            "http://localhost:11434/api/chat",
            json={
                "model": "llama3",
                "messages": [
                    {
                        "role": "system",
                        "content": """
You are a sign language interpreter.

Convert given words into a short meaningful sentence.

Rules:
- Keep it simple
- Combine words logically
- Ignore wrong words
- Return only sentence

Examples:
hungry → I am hungry.
milk → I want milk.
hungry milk → I am hungry and I want milk.
thankyou → Thank you.
iloveyou → I love you.
"""
                    },
                    {
                        "role": "user",
                        "content": word_text
                    }
                ],
                "stream": False,
            },
            timeout=20,
        )

        data = response.json()

        if "message" in data:
            return data["message"]["content"].strip()

    except Exception as e:
        print("LLM error:", e)

    return word_text.capitalize() + "."

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

# ----------------------------
# API
# ----------------------------

@app.route("/api/predict", methods=["POST"])
def api_predict():

    data = request.get_json()
    img_data = data.get("image")
    mode = data.get("mode", "letters")

    if not img_data:
        return jsonify({"error": "No image"}), 400

    image_bytes = base64.b64decode(img_data.split(",")[1])
    frame = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)

    if mode == "letters":
        annotated, tokens = run_letters_mode(frame)
    else:
        annotated, tokens = run_words_mode(frame)

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