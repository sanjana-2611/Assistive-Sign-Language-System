from flask import Flask, render_template, request, jsonify
from ultralytics import YOLO
import cv2
import numpy as np
import base64
import os
import requests
from collections import Counter, deque
import time
from difflib import get_close_matches
from wordfreq import top_n_list

# ----------------------------
# HUGGINGFACE PRETRAINED MODELS
# ----------------------------
import torch
import mediapipe as mp
from transformers import AutoImageProcessor, SiglipForImageClassification
from PIL import Image

app = Flask(__name__)

# ----------------------------
# LOAD WORD LIST
# ----------------------------
english_words = set(top_n_list("en", 50000))

# ----------------------------
# YOLO MODEL PATHS
# ----------------------------
LETTERS_MODEL_PATH = "lettersbest.pt"
WORDS1_MODEL_PATH  = "wordbest.pt"
WORDS2_MODEL_PATH  = "word2best.pt"


def load_yolo(path):
    if not os.path.exists(path):
        print(f"[WARN] YOLO model '{path}' not found — skipping.")
        return None
    return YOLO(path)


letters_model = load_yolo(LETTERS_MODEL_PATH)
words1_model  = load_yolo(WORDS1_MODEL_PATH)
words2_model  = load_yolo(WORDS2_MODEL_PATH)

# ----------------------------
# DEVICE
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------
# HUGGINGFACE — ALPHABET (ASL letters)  [from preapp1.py]
# ----------------------------
HF_ALPHA_NAME = "prithivMLmods/Alphabet-Sign-Language-Detection"
try:
    hf_alpha_processor = AutoImageProcessor.from_pretrained(HF_ALPHA_NAME)
    hf_alpha_model     = SiglipForImageClassification.from_pretrained(HF_ALPHA_NAME)
    hf_alpha_model.to(device).eval()
    hf_alpha_id2label  = {int(k): v for k, v in hf_alpha_model.config.id2label.items()}
    HF_ALPHA_OK = True
    print(f"[OK] HF Alphabet model loaded | {len(hf_alpha_id2label)} classes | {device}")
except Exception as e:
    HF_ALPHA_OK = False
    hf_alpha_processor = hf_alpha_model = hf_alpha_id2label = None
    print(f"[WARN] HF Alphabet model failed to load: {e}")

# ----------------------------
# HUGGINGFACE — HAND GESTURE (19 gestures)  [from prewordsapp.py]
# ----------------------------
HF_GESTURE_NAME = "prithivMLmods/Hand-Gesture-19"
try:
    hf_gest_processor = AutoImageProcessor.from_pretrained(HF_GESTURE_NAME)
    hf_gest_model     = SiglipForImageClassification.from_pretrained(HF_GESTURE_NAME)
    hf_gest_model.to(device).eval()
    hf_gest_id2label  = {
        0: "call",   1: "dislike", 2: "fist",          3: "four",
        4: "like",   5: "mute",    6: "no_gesture",     7: "ok",
        8: "one",    9: "palm",    10: "peace",         11: "peace_inverted",
        12: "rock",  13: "stop",   14: "stop_inverted", 15: "three",
        16: "three2",17: "two_up", 18: "two_up_inverted"
    }
    GESTURE_MEANING = {
        "call":            "CALL",
        "dislike":         "NO",
        "fist":            "POWER",
        "four":            "FOUR",
        "like":            "YES",
        "mute":            "QUIET",
        "no_gesture":      "",
        "ok":              "OK",
        "one":             "ONE",
        "palm":            "STOP",
        "peace":           "PEACE",
        "peace_inverted":  "PEACE",
        "rock":            "ROCK",
        "stop":            "STOP",
        "stop_inverted":   "STOP",
        "three":           "THREE",
        "three2":          "THREE",
        "two_up":          "TWO",
        "two_up_inverted": "TWO",
    }
    HF_GEST_OK = True
    print(f"[OK] HF Gesture model loaded | {len(hf_gest_id2label)} gestures | {device}")
except Exception as e:
    HF_GEST_OK = False
    hf_gest_processor = hf_gest_model = hf_gest_id2label = None
    GESTURE_MEANING = {}
    print(f"[WARN] HF Gesture model failed to load: {e}")

# ----------------------------
# MEDIAPIPE HANDS (shared across HF models)
# ----------------------------
mp_hands       = mp.solutions.hands
mp_draw_utils  = mp.solutions.drawing_utils
hands_detector = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.65,
    min_tracking_confidence=0.5
)
MP_PADDING = 30


def get_hand_bbox(landmarks, fw, fh, pad=MP_PADDING):
    xs = [lm.x * fw for lm in landmarks.landmark]
    ys = [lm.y * fh for lm in landmarks.landmark]
    return (max(0, int(min(xs)) - pad),
            max(0, int(min(ys)) - pad),
            min(fw, int(max(xs)) + pad),
            min(fh, int(max(ys)) + pad))


# ----------------------------
# GLOBALS
# ----------------------------
prediction_history = []
last_output        = None
STABLE_TIME        = 5        # seconds — same feel as original
FRAME_SKIP         = 1        # process EVERY frame (no skip) — fixes simultaneous capture
HF_SKIP            = 3        # run heavy HF models only every 3rd frame to stay fast
frame_count        = 0
HF_CONF_THRESHOLD  = 0.50

# ----------------------------
# LETTER ALIAS PATCH  (same as original)
# ----------------------------
LETTER_ALIAS = {
    "kr": "k", "j": "j", "b": "b", "e": "e", "c": "c",
    "l": "l",  "m": "m", "d": "d", "f": "f", "g": "g",
    "h": "h",  "i": "i", "u": "u",
}


def normalize_letter(label):
    return LETTER_ALIAS.get(str(label).strip().lower(), str(label).strip().lower())


# ----------------------------
# LETTER → WORD FIX
# ----------------------------
def letters_to_word(tokens):
    word    = "".join(tokens)
    matches = get_close_matches(word, english_words, n=1, cutoff=0.5)
    return matches[0] if matches else word


# ----------------------------
# CLEAN TOKENS
# ----------------------------
def clean_tokens(tokens):
    cleaned, last = [], None
    for t in tokens:
        word = str(t).strip().lower()
        if not word or word == last:
            continue
        cleaned.append(word)
        last = word
    return cleaned


# ----------------------------
# HF INFERENCE HELPERS
# ----------------------------
def run_hf_alpha(pil_img):
    """Alphabet-Sign-Language-Detection inference. Returns (label, conf, 'hf_alpha') or None."""
    if not HF_ALPHA_OK:
        return None
    try:
        inputs = {k: v.to(device) for k, v in
                  hf_alpha_processor(images=pil_img, return_tensors="pt").items()}
        with torch.no_grad():
            logits = hf_alpha_model(**inputs).logits
        probs      = torch.softmax(logits, dim=1)[0]
        pred_id    = probs.argmax().item()
        confidence = probs[pred_id].item()
        label      = hf_alpha_id2label.get(pred_id, "?")
        return (label, confidence, "hf_alpha")
    except Exception as e:
        print(f"[hf_alpha error] {e}")
        return None


def run_hf_gesture(pil_img):
    """Hand-Gesture-19 inference. Returns (mapped_word, conf, 'hf_gesture') or None."""
    if not HF_GEST_OK:
        return None
    try:
        inputs = {k: v.to(device) for k, v in
                  hf_gest_processor(images=pil_img, return_tensors="pt").items()}
        with torch.no_grad():
            logits = hf_gest_model(**inputs).logits
        probs      = torch.softmax(logits, dim=1)[0]
        pred_id    = probs.argmax().item()
        confidence = probs[pred_id].item()
        label      = hf_gest_id2label.get(pred_id, "no_gesture")
        if label == "no_gesture":
            return None
        mapped = GESTURE_MEANING.get(label, label.upper())
        if not mapped:
            return None
        return (mapped, confidence, "hf_gesture")
    except Exception as e:
        print(f"[hf_gesture error] {e}")
        return None


# ----------------------------
# YOLO INFERENCE
# ----------------------------
def run_yolo_models(frame):
    """Run all YOLO models. Returns (annotated_frame, (label, conf, 'yolo')) or (frame, None)."""
    word_detections   = []
    letter_detections = []
    annotated         = frame.copy()

    for model in [m for m in [words1_model, words2_model] if m]:
        results = model(frame, conf=0.25)
        if results and results[0].boxes is not None:
            for box in results[0].boxes:
                cls_id = int(box.cls[0])
                label  = model.names.get(cls_id)
                conf   = float(box.conf[0])
                if conf > 0.30:
                    word_detections.append((label, conf))
            if len(results[0].boxes) > 0:
                annotated = results[0].plot()

    if letters_model:
        results = letters_model(frame, conf=0.20)
        if results and results[0].boxes is not None:
            for box in results[0].boxes:
                cls_id    = int(box.cls[0])
                raw_label = letters_model.names.get(cls_id)
                label     = normalize_letter(raw_label)
                conf      = float(box.conf[0])
                if conf > 0.25:
                    letter_detections.append((label, conf))
            if len(results[0].boxes) > 0:
                annotated = results[0].plot()

    detections = word_detections if word_detections else letter_detections
    if not detections:
        return annotated, None

    best_label, best_conf = max(detections, key=lambda x: x[1])
    return annotated, (best_label, best_conf, "yolo")


# ----------------------------
# MAIN DETECTION — YOLO + HF FUSION
#
# Flow per frame:
#   1. YOLO runs on full frame (words first, then letters)
#   2. MediaPipe finds hand bbox → HF models classify crop
#   3. All candidates compete: highest weighted-confidence wins
#   4. Stability buffer (6 s) gates final output (60 % majority, min 3 samples)
#
# Source priority weights:
#   yolo=1.0  hf_gesture=0.95  hf_alpha=0.90
# (YOLO wins ties; HF gesture preferred over HF alpha for whole-word signs)
# ----------------------------
# Cache last HF result so skipped frames still contribute to buffer
_last_hf_result = None


def run_all_models(frame):
    global prediction_history, last_output, frame_count, _last_hf_result

    frame_count  += 1
    current_time  = time.time()

    # ---- YOLO — runs every frame (fast, keeps buffer filling) ----
    annotated, yolo_result = run_yolo_models(frame)

    # ---- MEDIAPIPE — runs every frame (lightweight) ----
    hf_result  = _last_hf_result          # reuse last HF result by default
    rgb        = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_results = hands_detector.process(rgb)

    if mp_results.multi_hand_landmarks:
        hand_lm = mp_results.multi_hand_landmarks[0]
        h, w    = frame.shape[:2]
        x1, y1, x2, y2 = get_hand_bbox(hand_lm, w, h)
        hand_crop = frame[y1:y2, x1:x2]

        # ---- HF — runs only every HF_SKIP frames (heavy models) ----
        if hand_crop.size > 0 and frame_count % HF_SKIP == 0:
            pil_img     = Image.fromarray(cv2.cvtColor(hand_crop, cv2.COLOR_BGR2RGB))
            alpha_res   = run_hf_alpha(pil_img)
            gesture_res = run_hf_gesture(pil_img)
            candidates  = [r for r in [alpha_res, gesture_res]
                           if r and r[1] >= HF_CONF_THRESHOLD]
            _last_hf_result = max(candidates, key=lambda x: x[1]) if candidates else None
            hf_result       = _last_hf_result

        # Draw MP landmarks on annotated frame
        mp_draw_utils.draw_landmarks(annotated, hand_lm, mp_hands.HAND_CONNECTIONS)
    else:
        # No hand — clear cached HF result so stale signs don't bleed over
        _last_hf_result = None
        hf_result       = None

    # ---- FUSION — best weighted-confidence wins ----
    def source_weight(src):
        return {"yolo": 1.0, "hf_gesture": 0.95, "hf_alpha": 0.90}.get(src, 0.8)

    all_candidates = [r for r in [yolo_result, hf_result] if r]
    if not all_candidates:
        return annotated, []

    best_label, best_conf, best_src = max(
        all_candidates, key=lambda x: x[1] * source_weight(x[2])
    )

    # ---- STABILITY BUFFER (same thresholds as original working app) ----
    prediction_history.append((best_label, best_conf, current_time))
    prediction_history = [
        (l, c, t) for l, c, t in prediction_history
        if current_time - t <= STABLE_TIME
    ]

    # Original thresholds: min 3 samples, 60% majority
    if len(prediction_history) >= 3:
        labels = [l for l, _, _ in prediction_history]
        most_common, count = Counter(labels).most_common(1)[0]

        if count >= int(0.6 * len(labels)):
            if most_common != last_output:
                last_output = most_common
                prediction_history.clear()

                cv2.putText(
                    annotated,
                    f"[{best_src}] {most_common} ({best_conf:.2f})",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 255, 100), 2, cv2.LINE_AA
                )
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
        word   = letters_to_word(tokens)
        prompt = (f"Create one meaningful sentence using the word '{word}'. "
                  f"Reply with only the sentence, nothing else.")
    else:
        token_str = " ".join(tokens)
        prompt    = (f"Create one meaningful sentence using ALL these words: {token_str}. "
                     f"Do not skip any word. Reply with only the sentence, nothing else.")

    try:
        response = requests.post(
            "http://localhost:11434/api/chat",
            json={
                "model": "gpt-oss:120b-cloud",
                "messages": [
                    {"role": "system",
                     "content": "You are a sentence generator. Always reply with exactly one sentence and nothing else."},
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
    data        = request.get_json()
    img_data    = data.get("image")
    image_bytes = base64.b64decode(img_data.split(",")[1])
    np_arr      = np.frombuffer(image_bytes, np.uint8)
    frame       = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    annotated, tokens = run_all_models(frame)

    _, buffer     = cv2.imencode(".jpg", annotated)
    encoded_image = base64.b64encode(buffer).decode("utf-8")

    return jsonify({"image": encoded_image, "tokens": tokens})


@app.route("/api/sentence", methods=["POST"])
def api_sentence():
    tokens   = request.get_json().get("tokens", [])
    sentence = call_llm_from_tokens(tokens)
    return jsonify({"sentence": sentence})


# ----------------------------
# STATUS ENDPOINT
# Lets the frontend know which models actually loaded
# ----------------------------
@app.route("/api/status")
def api_status():
    return jsonify({
        "yolo_letters": letters_model is not None,
        "yolo_words1":  words1_model  is not None,
        "yolo_words2":  words2_model  is not None,
        "hf_alphabet":  HF_ALPHA_OK,
        "hf_gesture":   HF_GEST_OK,
        "device":       str(device),
    })


# ----------------------------
# RUN
# ----------------------------
if __name__ == "__main__":
    app.run(debug=True)