from flask import Flask, render_template, request, jsonify
from ultralytics import YOLO
import cv2
import numpy as np
import base64
import os
import requests
from collections import Counter


app = Flask(__name__)


# Load YOLO models once at startup
#LETTERS_MODEL_PATH = "lettersbest.pt"
WORDS1_MODEL_PATH = "wordbest.pt"
WORDS2_MODEL_PATH = "word2best.pt"


def load_model(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Model file '{path}' not found. Make sure it is in the project folder."
        )
    return YOLO(path)


# letters_model = load_model(LETTERS_MODEL_PATH)
words1_model = load_model(WORDS1_MODEL_PATH)
words2_model = load_model(WORDS2_MODEL_PATH)


def run_all_models(frame: np.ndarray):
    """
    Run all three YOLO models on a single BGR frame.
    Returns an annotated frame (BGR) and at most one dominant detected label.
    """
    models = [words1_model, words2_model]
    all_labels = []
    annotated = frame

    for model in models:
        # Higher confidence to reduce noisy detections
        results = model(annotated, conf=0.5)
        if not results:
            continue

        r = results[0]
        if r.boxes is not None:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                label = model.names.get(cls_id, str(cls_id))
                all_labels.append(label)

        annotated = r.plot()

    if not all_labels:
        return annotated, []

    # Choose the most frequent label across all models for this frame
    counts = Counter(all_labels)
    top_label, _ = counts.most_common(1)[0]
    return annotated, [top_label]


def _clean_tokens(tokens):
    """
    Light cleanup before sending to the LLM:
    - Cast to strings
    - Strip spaces
    - Collapse consecutive duplicates (HELLO, HELLO -> HELLO)
    - Normalize to lowercase/gloss style words
    """
    cleaned = []
    last = None
    for t in tokens:
        word = str(t).strip()
        if not word:
            continue
        # Normalize HELLO / Hello / hello -> hello
        word_norm = word.lower()
        if word_norm == last:
            continue
        cleaned.append(word_norm)
        last = word_norm
    return cleaned


def call_llm_from_tokens(tokens):
    """
    Use a local Ollama model to convert a cleaned list of recognized
    words into a natural sentence. If Ollama is not available or
    fails, falls back to a simple joined string.
    """
    cleaned_tokens = _clean_tokens(tokens)
    if not cleaned_tokens:
        return ""

    token_str = " ".join(cleaned_tokens)

    # Fallback sentence used if Ollama is not reachable
    def _fallback():
        sentence = " ".join(cleaned_tokens).strip()
        if sentence:
            sentence = sentence[0].upper() + sentence[1:]
        if not sentence.endswith("."):
            sentence += "."
        return sentence

    ollama_model = os.getenv("OLLAMA_MODEL", "gpt-oss:120b-cloud")
    ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434/api/chat")

    prompt = (
        "You are assisting with sign-language translation.\n"
        "You receive a rough sequence of sign glosses (simple English words, "
        "often all lowercase, representing the order of signed concepts).\n"
        "Your job is to infer the user's most likely intended meaning and rewrite "
        "these glosses as ONE natural, fluent, grammatically correct English sentence.\n"
        "- Ignore obvious duplicates or noise.\n"
        "- Keep the meaning simple, clear, and respectful.\n"
        "- Do not explain what you are doing.\n"
        "- Only output the final sentence.\n\n"
        f"Sign gloss sequence: {token_str}"
    )

    try:
        response = requests.post(
            ollama_url,
            json={
                "model": ollama_model,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You convert sign-language gloss sequences into smooth, "
                            "natural English sentences for hearing users."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                "stream": False,
            },
            timeout=25,
        )
        response.raise_for_status()
        data = response.json()

        # Ollama chat API: { message: { content: "..." }, ... }
        if isinstance(data, dict) and "message" in data:
            content = data["message"].get("content", "")
            return str(content).strip()

        return _fallback()
    except Exception:
        # Safe fallback if Ollama is unavailable
        return _fallback()


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
    """
    Receives a base64-encoded image from the browser, runs all YOLO models,
    and returns an annotated image plus the list of detected tokens.
    """
    data = request.get_json(silent=True) or {}
    img_data = data.get("image")
    if not img_data or "," not in img_data:
        return jsonify({"error": "Invalid image data"}), 400

    try:
        image_bytes = base64.b64decode(img_data.split(",")[1])
        np_arr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    except Exception:
        return jsonify({"error": "Could not decode image"}), 400

    annotated, tokens = run_all_models(frame)

    # Encode annotated frame back to JPEG base64
    success, buffer = cv2.imencode(".jpg", annotated)
    if not success:
        return jsonify({"error": "Failed to encode image"}), 500

    encoded_image = base64.b64encode(buffer).decode("utf-8")
    return jsonify(
        {
            "image": encoded_image,
            "tokens": tokens,
        }
    )


@app.route("/api/sentence", methods=["POST"])
def api_sentence():
    """
    Receives a list of tokens and returns a refined sentence from the LLM.
    """
    data = request.get_json(silent=True) or {}
    tokens = data.get("tokens") or []
    if not isinstance(tokens, list):
        return jsonify({"error": "tokens must be a list"}), 400

    tokens = [str(t).strip() for t in tokens if str(t).strip()]
    sentence = call_llm_from_tokens(tokens)
    return jsonify({"sentence": sentence})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

