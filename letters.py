from flask import Flask, render_template, request, jsonify
from ultralytics import YOLO
import cv2
import numpy as np
import base64
import os
import requests
from collections import Counter, deque
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
    ba = np.array([a.x - b.x, a.y - b.y])
    bc = np.array([c.x - b.x, c.y - b.y])
    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))


def _finger_extended(lm, tip, pip):
    return lm[tip].y < lm[pip].y


def _thumb_extended(lm):
    return lm[4].x < lm[3].x


def mediapipe_classify(lm):
    index_ext  = _finger_extended(lm, 8,  6)
    middle_ext = _finger_extended(lm, 12, 10)
    ring_ext   = _finger_extended(lm, 16, 14)
    pinky_ext  = _finger_extended(lm, 20, 18)
    thumb_ext  = _thumb_extended(lm)

    thumb_index_dist  = _dist(lm[4], lm[8])
    thumb_middle_dist = _dist(lm[4], lm[12])
    index_middle_dist = _dist(lm[8], lm[12])

    # B
    if (index_ext and middle_ext and ring_ext and pinky_ext and not thumb_ext):
        spread = _dist(lm[8], lm[12]) + _dist(lm[12], lm[16]) + _dist(lm[16], lm[20])
        if spread < 0.18:
            return "b"

    # C
    if (not index_ext and not middle_ext and not ring_ext and not pinky_ext):
        if 0.08 < thumb_index_dist < 0.25:
            if _dist(lm[4], lm[2]) > 0.06:
                return "c"

    # D
    if (index_ext and not middle_ext and not ring_ext and not pinky_ext):
        if thumb_middle_dist < 0.06:
            return "d"

    # E
    all_curled = not index_ext and not middle_ext and not ring_ext and not pinky_ext
    if all_curled and thumb_index_dist < 0.08:
        return "e"

    # F
    if (middle_ext and ring_ext and pinky_ext and not index_ext):
        if thumb_index_dist < 0.06:
            return "f"

    # G
    if (index_ext and not middle_ext and not ring_ext and not pinky_ext and not thumb_ext):
        if abs(lm[8].x - lm[5].x) > abs(lm[8].y - lm[5].y):
            return "g"

    # H
    if (index_ext and middle_ext and not ring_ext and not pinky_ext):
        if (abs(lm[8].x - lm[5].x) > abs(lm[8].y - lm[5].y) and
                abs(lm[12].x - lm[9].x) > abs(lm[12].y - lm[9].y)):
            return "h"

    # I
    if (not index_ext and not middle_ext and not ring_ext and pinky_ext):
        if abs(lm[20].x - lm[17].x) <= 0.05:
            return "i"

    # J
    if (not index_ext and not middle_ext and not ring_ext and pinky_ext):
        if abs(lm[20].x - lm[17].x) > 0.05:
            return "j"

    # K
    if (index_ext and middle_ext and not ring_ext and not pinky_ext):
        if _dist(lm[8], lm[12]) > 0.10 and thumb_middle_dist < 0.09:
            return "k"

    # L
    if (index_ext and not middle_ext and not ring_ext and not pinky_ext and thumb_ext):
        if _angle(lm[8], lm[5], lm[4]) > 70:
            return "l"

    # M
    if (not index_ext and not middle_ext and not ring_ext and not pinky_ext):
        thumb_under = (lm[4].y > lm[8].y and lm[4].y > lm[12].y and lm[4].y > lm[16].y)
        if thumb_under and thumb_index_dist < 0.12:
            return "m"

    # R
    if (index_ext and middle_ext and not ring_ext and not pinky_ext):
        if abs(lm[8].x - lm[12].x) < 0.04 and index_middle_dist < 0.06:
            return "r"

    # U
    if (index_ext and middle_ext and not ring_ext and not pinky_ext):
        if _dist(lm[8], lm[12]) < 0.07 and abs(lm[8].x - lm[12].x) >= 0.04:
            return "u"

    return None


def run_mediapipe(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands_detector.process(rgb)
    if not result.multi_hand_landmarks:
        return None
    lm = result.multi_hand_landmarks[0].landmark
    return mediapipe_classify(lm)


# ----------------------------
# HOLD-DURATION STATE
# One global state (single-user app). Extend to dict-per-session if needed.
# ----------------------------
HOLD_DURATION    = 1.5     # seconds to hold a sign before letter is committed
HISTORY_LEN      = 12
CONF_THRESHOLD   = 0.35
FRAME_SKIP       = 2

_state = {
    "history":     deque(maxlen=HISTORY_LEN),  # (label, conf) recent predictions
    "hold_label":  None,    # letter currently being held
    "hold_start":  None,    # timestamp when current hold began
    "just_added":  False,   # debounce: True after committing, until sign changes
    "frame_count": 0,
}


def detect_letter(frame):
    """
    Run YOLO + MediaPipe fusion on a single frame.
    Returns (annotated_frame, label_or_None, conf_or_0).
    """
    results = letters_model(frame, conf=0.35)

    yolo_best_label = None
    yolo_best_conf  = 0.0
    letter_detections = []

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
    else:
        annotated = frame.copy()

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
                final_label = yolo_best_label
                final_conf  = min(1.0, yolo_best_conf + 0.20)
            else:
                final_label = mp_label.upper()
                final_conf  = 0.72

    return annotated, final_label, final_conf


def process_frame(frame):
    """
    Full pipeline: detect → update hold state → return result.

    Returns dict:
      annotated_b64   – JPEG image for display
      stable_label    – letter currently being held (or None)
      stable_conf     – confidence of stable label
      hold_progress   – 0.0 – 1.0 (how close to committing)
      committed       – letter string if just committed this frame, else ""
    """
    s = _state
    s["frame_count"] += 1

    annotated = frame.copy()
    committed = ""

    # Frame-skip: still return last annotated quickly
    if s["frame_count"] % FRAME_SKIP != 0:
        _, buf = cv2.imencode(".jpg", annotated)
        return {
            "annotated_b64": base64.b64encode(buf).decode("utf-8"),
            "stable_label":  s["hold_label"],
            "stable_conf":   0.0,
            "hold_progress": _current_hold_progress(s),
            "committed":     "",
        }

    annotated, raw_label, raw_conf = detect_letter(frame)

    # Update rolling history
    if raw_label and raw_conf >= CONF_THRESHOLD:
        s["history"].append((raw_label, raw_conf))
    elif not raw_label:
        # No detection — decay history slowly (don't clear instantly)
        if s["history"]:
            s["history"].popleft()

    # Derive stable label from history majority
    stable_label = None
    stable_conf  = 0.0
    if s["history"]:
        hist_labels  = [e[0] for e in s["history"]]
        stable_label = Counter(hist_labels).most_common(1)[0][0]
        stable_conf  = sum(c for l, c in s["history"] if l == stable_label) / hist_labels.count(stable_label)

    now = time.time()

    if stable_label:
        if stable_label == s["hold_label"]:
            # Same sign — check hold duration
            hold_elapsed = now - s["hold_start"]

            if hold_elapsed >= HOLD_DURATION and not s["just_added"]:
                # ✅ Commit the letter
                committed       = stable_label.upper()
                s["just_added"] = True
        else:
            # New sign — reset hold timer
            s["hold_label"] = stable_label
            s["hold_start"] = now
            s["just_added"] = False
    else:
        # No sign detected — reset
        s["hold_label"] = None
        s["hold_start"] = None
        s["just_added"] = False

    hold_progress = _current_hold_progress(s)

    _, buf = cv2.imencode(".jpg", annotated)
    return {
        "annotated_b64": base64.b64encode(buf).decode("utf-8"),
        "stable_label":  s["hold_label"],
        "stable_conf":   round(stable_conf, 3),
        "hold_progress": hold_progress,
        "committed":     committed,
    }


def _current_hold_progress(s):
    if s["hold_label"] and s["hold_start"] and not s["just_added"]:
        elapsed = time.time() - s["hold_start"]
        return round(min(elapsed / HOLD_DURATION, 1.0), 3)
    return 0.0


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
        return max(candidates, key=lambda w: word_frequency(w, "en"))

    return exact


# ----------------------------
# CLEAN TOKENS
# ----------------------------
def clean_tokens(tokens):
    cleaned = []
    last = None
    for t in tokens:
        word = str(t).strip().lower()
        if not word or word == last:
            continue
        cleaned.append(word)
        last = word
    return cleaned


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
    return render_template("redict.html")


@app.route("/api/redict", methods=["POST"])
def api_predict():
    """
    Accepts: { image: <base64 jpeg> }

    Returns:
    {
      image:         <base64 annotated jpeg>,
      stable_label:  "A" | null,
      stable_conf:   0.87,
      hold_progress: 0.45,   # 0.0 – 1.0
      committed:     "A"     # non-empty only when letter just committed
    }
    """
    data      = request.get_json()
    img_data  = data.get("image")

    image_bytes = base64.b64decode(img_data.split(",")[1])
    np_arr      = np.frombuffer(image_bytes, np.uint8)
    frame       = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    result = process_frame(frame)

    return jsonify({
        "image":         result["annotated_b64"],
        "stable_label":  result["stable_label"],
        "stable_conf":   result["stable_conf"],
        "hold_progress": result["hold_progress"],
        "committed":     result["committed"],
    })


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