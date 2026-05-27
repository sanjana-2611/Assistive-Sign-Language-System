from flask import Flask, render_template, request, jsonify
from ultralytics import YOLO
import cv2
import numpy as np
import base64
import os
import requests
from collections import Counter
import time
from wordfreq import top_n_list, word_frequency
import mediapipe as mp

app = Flask(__name__)

# ----------------------------
# LOAD WORD LIST
# ----------------------------
english_words = set(top_n_list("en", 50000))

# ----------------------------
# MODEL PATHS
# ----------------------------
LETTERS_MODEL_PATH = "lettersbest.pt"


def load_model(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model '{path}' not found")
    return YOLO(path)


letters_model = load_model(LETTERS_MODEL_PATH)

# ----------------------------
# MEDIAPIPE SETUP
# Letters that YOLO misses most — we use MediaPipe geometry for these
# ----------------------------
WEAK_LETTERS = set("b e c l m d f g h i j k r u".split())

mp_hands = mp.solutions.hands
hands_detector = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.5,
)

# ----------------------------
# LANDMARK GEOMETRY HELPERS
# ----------------------------

def _dist(a, b):
    return np.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)


def _angle(a, b, c):
    """Angle at vertex b formed by points a-b-c (degrees)."""
    ba = np.array([a.x - b.x, a.y - b.y])
    bc = np.array([c.x - b.x, c.y - b.y])
    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))


def _finger_extended(lm, tip, pip):
    """True if finger tip is above its PIP joint (extended)."""
    return lm[tip].y < lm[pip].y


def _thumb_extended(lm):
    return lm[4].x < lm[3].x  # works for right hand; flipped for left


def mediapipe_classify(lm):
    """
    Rule-based ASL letter classifier using hand landmarks.
    Covers the 14 letters YOLO misses: b e c l m d f g h i j k r u
    Returns a letter string or None.
    """

    # Finger extension flags
    index_ext  = _finger_extended(lm, 8,  6)
    middle_ext = _finger_extended(lm, 12, 10)
    ring_ext   = _finger_extended(lm, 16, 14)
    pinky_ext  = _finger_extended(lm, 20, 18)
    thumb_ext  = _thumb_extended(lm)

    # Tip distances
    thumb_index_dist  = _dist(lm[4], lm[8])
    thumb_middle_dist = _dist(lm[4], lm[12])
    index_middle_dist = _dist(lm[8], lm[12])

    # ---------- B ----------
    # All four fingers extended and together, thumb tucked across palm
    if (index_ext and middle_ext and ring_ext and pinky_ext and not thumb_ext):
        spread = (
            _dist(lm[8], lm[12]) +
            _dist(lm[12], lm[16]) +
            _dist(lm[16], lm[20])
        )
        if spread < 0.18:
            return "b"

    # ---------- C ----------
    # Hand curved like holding a ball — all fingers partially curled, gap between thumb & index
    if (not index_ext and not middle_ext and not ring_ext and not pinky_ext):
        if 0.08 < thumb_index_dist < 0.25:
            thumb_curve = _dist(lm[4], lm[2])
            if thumb_curve > 0.06:
                return "c"

    # ---------- D ----------
    # Index extended, rest curled, thumb touches middle fingertip
    if (index_ext and not middle_ext and not ring_ext and not pinky_ext):
        if thumb_middle_dist < 0.06:
            return "d"

    # ---------- E ----------
    # All fingers curled down (like a claw), thumb tucked under
    all_curled = not index_ext and not middle_ext and not ring_ext and not pinky_ext
    if all_curled and thumb_index_dist < 0.08:
        return "e"

    # ---------- F ----------
    # Index and thumb touching (O shape), other three extended
    if (middle_ext and ring_ext and pinky_ext and not index_ext):
        if thumb_index_dist < 0.06:
            return "f"

    # ---------- G ----------
    # Index pointing sideways (horizontal), thumb parallel, others curled
    if (index_ext and not middle_ext and not ring_ext and not pinky_ext and not thumb_ext):
        horizontal = abs(lm[8].x - lm[5].x) > abs(lm[8].y - lm[5].y)
        if horizontal:
            return "g"

    # ---------- H ----------
    # Index and middle pointing sideways together, others curled
    if (index_ext and middle_ext and not ring_ext and not pinky_ext):
        horizontal_i = abs(lm[8].x - lm[5].x) > abs(lm[8].y - lm[5].y)
        horizontal_m = abs(lm[12].x - lm[9].x) > abs(lm[12].y - lm[9].y)
        if horizontal_i and horizontal_m:
            return "h"

    # ---------- I ----------
    # Only pinky extended, others curled
    if (not index_ext and not middle_ext and not ring_ext and pinky_ext):
        tilt = lm[20].x - lm[17].x
        if abs(tilt) <= 0.05:   # not tilted → I (not J)
            return "i"

    # ---------- J ----------
    # Like I but pinky traces a J arc — approximate by pinky extended + lateral tilt
    if (not index_ext and not middle_ext and not ring_ext and pinky_ext):
        tilt = lm[20].x - lm[17].x
        if abs(tilt) > 0.05:
            return "j"

    # ---------- K ----------
    # Index and middle extended in V, thumb touching middle, spread wide
    if (index_ext and middle_ext and not ring_ext and not pinky_ext):
        v_spread = _dist(lm[8], lm[12])
        if v_spread > 0.10 and thumb_middle_dist < 0.09:
            return "k"

    # ---------- L ----------
    # Index pointing up, thumb pointing sideways (L shape), others curled
    if (index_ext and not middle_ext and not ring_ext and not pinky_ext and thumb_ext):
        angle = _angle(lm[8], lm[5], lm[4])
        if angle > 70:
            return "l"

    # ---------- M ----------
    # Three fingers (index, middle, ring) folded over tucked thumb
    if (not index_ext and not middle_ext and not ring_ext and not pinky_ext):
        thumb_under = (
            lm[4].y > lm[8].y and
            lm[4].y > lm[12].y and
            lm[4].y > lm[16].y
        )
        if thumb_under and thumb_index_dist < 0.12:
            return "m"

    # ---------- R ----------
    # Index and middle crossed (overlap in X), both extended
    if (index_ext and middle_ext and not ring_ext and not pinky_ext):
        cross = abs(lm[8].x - lm[12].x) < 0.04 and index_middle_dist < 0.06
        if cross:
            return "r"

    # ---------- U ----------
    # Index and middle extended together (not spread, not crossed), others curled
    if (index_ext and middle_ext and not ring_ext and not pinky_ext):
        u_close     = _dist(lm[8], lm[12]) < 0.07
        not_crossed = abs(lm[8].x - lm[12].x) >= 0.04
        if u_close and not_crossed:
            return "u"

    return None


def run_mediapipe(frame):
    """Run MediaPipe on frame, return predicted letter or None."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands_detector.process(rgb)
    if not result.multi_hand_landmarks:
        return None
    lm = result.multi_hand_landmarks[0].landmark
    return mediapipe_classify(lm)


# ----------------------------
# GLOBALS
# ----------------------------
prediction_history = []
last_output = None
STABLE_TIME = 6

FRAME_SKIP = 2
frame_count = 0


# ----------------------------
# SMART LETTER → WORD
# ----------------------------
def letters_to_word(tokens):
    from itertools import permutations

    letters = [t.lower() for t in tokens if len(t) == 1]

    if not letters:
        return " ".join(tokens)

    candidates = set()

    exact = "".join(letters)
    if exact in english_words:
        candidates.add(exact)

    if len(letters) <= 7:
        for perm in permutations(letters):
            word = "".join(perm)
            if word in english_words:
                candidates.add(word)

    if len(letters) <= 8:
        for i in range(len(letters)):
            subset = letters[:i] + letters[i + 1:]
            for perm in permutations(subset):
                word = "".join(perm)
                if word in english_words:
                    candidates.add(word)

    if candidates:
        best = max(candidates, key=lambda w: word_frequency(w, "en"))
        return best

    return exact


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
# DETECTION — YOLO + MEDIAPIPE FUSION
# ----------------------------
def run_all_models(frame):
    global prediction_history, last_output, frame_count

    frame_count += 1

    if frame_count % FRAME_SKIP != 0:
        return frame, []

    annotated = frame.copy()
    letter_detections = []

    # ── Step 1: YOLO ──────────────────────────────────────────
    results = letters_model(frame, conf=0.35)

    yolo_best_label = None
    yolo_best_conf  = 0.0

    if results and results[0].boxes is not None:
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            label  = letters_model.names.get(cls_id)
            conf   = float(box.conf[0])
            if conf > 0.4:
                letter_detections.append((label, conf))

        annotated = results[0].plot()

        if letter_detections:
            yolo_best_label, yolo_best_conf = max(letter_detections, key=lambda x: x[1])

    # ── Step 2: MediaPipe fallback for weak letters ────────────
    # Trigger MediaPipe when:
    #   (a) YOLO detected nothing, OR
    #   (b) YOLO's best guess is a known-weak letter with low confidence
    use_mediapipe = (
        yolo_best_label is None
        or (yolo_best_label.lower() in WEAK_LETTERS and yolo_best_conf < 0.70)
    )

    final_label = yolo_best_label
    final_conf  = yolo_best_conf

    if use_mediapipe:
        mp_label = run_mediapipe(frame)
        if mp_label is not None:
            if yolo_best_label and yolo_best_label.lower() == mp_label:
                # YOLO + MediaPipe agree → boost confidence
                final_label = yolo_best_label
                final_conf  = min(1.0, yolo_best_conf + 0.20)
            else:
                # Only MediaPipe fires → use its result
                final_label = mp_label.upper()
                final_conf  = 0.72

    if final_label is None:
        return annotated, []

    # ── Step 3: Temporal smoothing ────────────────────────────
    current_time = time.time()
    prediction_history.append((final_label, final_conf, current_time))
    prediction_history = [
        (l, c, t) for l, c, t in prediction_history
        if current_time - t <= STABLE_TIME
    ]

    if len(prediction_history) > 5:
        labels = [l for l, _, _ in prediction_history]
        most_common, count = Counter(labels).most_common(1)[0]

        if count >= int(0.7 * len(labels)):
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
        prompt = (
            f"You are helping a sign language learner practice. "
            f"They signed the word '{word}'. "
            f"Write one short, simple, encouraging sentence (max 15 words) "
            f"that naturally uses the word '{word}'. "
            f"Only return the sentence, nothing else."
        )
    else:
        token_str = " ".join(tokens)
        prompt = (
            f"You are helping a sign language learner practice. "
            f"They signed these words: {token_str}. "
            f"Write one short, simple, encouraging sentence (max 15 words) "
            f"using these words. Only return the sentence, nothing else."
        )

    try:
        response = requests.post(
            "http://localhost:11434/api/chat",
            json={
                "model": "gpt-oss:120b-cloud",
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are a helpful assistant for a sign language coaching app. "
                            "Always produce short, clear, meaningful sentences. "
                            "Never produce random or nonsensical output."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                "stream": False,
            },
            timeout=20,
        )

        data = response.json()

        if "message" in data:
            return data["message"]["content"].strip()

        return prompt

    except Exception:
        return prompt


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
    data     = request.get_json()
    img_data = data.get("image")

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
# RUN
# ----------------------------
if __name__ == "__main__":
    app.run(debug=True)