from flask import Flask, render_template, request, jsonify
from ultralytics import YOLO
import cv2
import torch
import mediapipe as mp
import numpy as np
from transformers import AutoImageProcessor, SiglipForImageClassification
from PIL import Image
from collections import deque, Counter
import base64
import os
import time
import requests

app = Flask(__name__)

# ============================================================
# 1.  YOLO WORD MODELS  (primary detectors)
# ============================================================
WORDS1_MODEL_PATH = "wordbest.pt"
WORDS2_MODEL_PATH = "word2best.pt"


def load_yolo(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model '{path}' not found")
    return YOLO(path)


words1_model = load_yolo(WORDS1_MODEL_PATH)
words2_model = load_yolo(WORDS2_MODEL_PATH)
print("[YOLO] Word models loaded")

# ============================================================
# 2.  SIGLIP HAND-GESTURE-19  (secondary / fusion)
# ============================================================
GESTURE_MODEL_NAME = "prithivMLmods/Hand-Gesture-19"

gesture_processor = AutoImageProcessor.from_pretrained(GESTURE_MODEL_NAME)
gesture_model     = SiglipForImageClassification.from_pretrained(GESTURE_MODEL_NAME)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gesture_model.to(device)
gesture_model.eval()

# Official id→label mapping
GESTURE_ID2LABEL = {
    0: "call",          1: "dislike",       2: "fist",
    3: "four",          4: "like",          5: "mute",
    6: "no_gesture",    7: "ok",            8: "one",
    9: "palm",         10: "peace",        11: "peace_inverted",
   12: "rock",         13: "stop",         14: "stop_inverted",
   15: "three",        16: "three2",       17: "two_up",
   18: "two_up_inverted"
}

# Map gesture → plain English word (used as fallback label)
# NOTE: "like" gesture → "yes" (thumbs up), NOT "love"
# "love" should only come from your YOLO model, not SigLIP
GESTURE_TO_WORD = {
    "call":            "call",
    "dislike":         "no",
    "fist":            "eat",
    "four":            "help",
    "like":            "yes",       # thumbs up = yes, NOT love
    "mute":            "quiet",
    "ok":              "ok",
    "one":             "one",
    "palm":            "stop",
    "peace":           "peace",
    "peace_inverted":  "peace",
    "rock":            "rock",
    "stop":            "stop",
    "stop_inverted":   "stop",
    "three":           "three",
    "three2":          "three",
    "two_up":          "washroom",
    "two_up_inverted": "two",
    "no_gesture":      None,
}

print(f"[SigLIP] Hand-Gesture-19 loaded | device: {device}")

# ============================================================
# 3.  MEDIAPIPE HANDS  (hand crop for SigLIP)
# ============================================================
mp_hands = mp.solutions.hands
mp_draw  = mp.solutions.drawing_utils

hands_detector = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.60,
    min_tracking_confidence=0.5,
)

PADDING = 30


def get_hand_bbox(landmarks, fw, fh, pad=PADDING):
    xs = [lm.x * fw for lm in landmarks.landmark]
    ys = [lm.y * fh for lm in landmarks.landmark]
    return (max(0,  int(min(xs)) - pad),
            max(0,  int(min(ys)) - pad),
            min(fw, int(max(xs)) + pad),
            min(fh, int(max(ys)) + pad))

# ============================================================
# 4.  HOLD-DURATION STATE
# ============================================================
HOLD_DURATION  = 1.5    # seconds to hold sign before word commits
CONF_THRESHOLD = 0.45   # raised from 0.35 to reduce false triggers
HISTORY_LEN    = 12
FRAME_SKIP     = 2

_state = {
    "history":     deque(maxlen=HISTORY_LEN),
    "hold_label":  None,
    "hold_start":  None,
    "just_added":  False,
    "frame_count": 0,
}

# ============================================================
# 5.  WORD CLEANING  (FIX: was called but never defined)
# ============================================================

# Common English filler words to strip before sentence building
_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "is", "are", "was",
    "were", "be", "been", "being", "have", "has", "had", "do",
    "does", "did", "will", "would", "could", "should", "may",
    "might", "shall", "can", "to", "of", "in", "on", "at",
    "by", "for", "with", "about", "as", "into", "through",
    "during", "before", "after", "above", "below", "from",
    "up", "down", "out", "off", "over", "under", "again",
    "then", "once", "here", "there", "when", "where", "why",
    "how", "all", "both", "each", "more", "most", "other",
    "some", "such", "than", "that", "these", "this", "those",
    "i", "me", "my", "we", "our", "you", "your", "he", "she",
    "it", "they", "them", "their", "what", "which", "who",
    "whom", "so", "if", "not",
}

# Meaningful ASL / sign words that should NEVER be stripped
_KEEP_ALWAYS = {
    "help", "eat", "drink", "yes", "no", "love", "stop",
    "call", "ok", "peace", "quiet", "rock", "one", "two",
    "three", "four", "thanks", "thirsty", "hungry", "hold",
    "name", "good", "bad", "morning", "night", "water",
    "please", "sorry", "hello", "bye",
}


def clean_words(words):
    """
    Remove duplicates, blank entries, and obvious stopwords
    UNLESS the word is in the always-keep set.
    Preserves original order (first occurrence).
    """
    seen   = set()
    result = []
    for w in words:
        w = w.strip().lower()
        if not w:
            continue
        if w in seen:
            continue
        seen.add(w)
        # Keep if it's a meaningful sign word or not a generic stopword
        if w in _KEEP_ALWAYS or w not in _STOPWORDS:
            result.append(w)
    return result


# ============================================================
# 6.  RULE-BASED SENTENCE BUILDER
# ============================================================

def build_simple_sentence(words):
    words = [w.lower() for w in words]
    w_set = set(words)
    if "perceive you" in w_set:
        return "I perceive you."
    if "call" in w_set:
        return "Can I call you?"
    if "hello" in w_set:
        return "Helloo, how can i help you with??"
    if "thirsty" in w_set and "drink" in w_set:
        return "I am thirsty, I want to drink."
    if "thirsty" in w_set:
        return "I am thirsty."
    if "drink" in w_set:
        return "I want to drink."
    if "hungry" in w_set and "eat" in w_set:
        return "I am hungry, I want to eat."
    if "eat" in w_set:
        return "I want to eat."
    if "Thank you" in w_set:
        return "Thank you for your help"
    if "i love you" in w_set:
        return "I love you with all my heart."
    if "yes" in w_set:
        return "Yes, I agree with you"
    if "no" in w_set:
        return "No, i dont agree with you"
    if "rock" in w_set:
        return "Lets rock it."
    if "hold" in w_set:
        return "Please wait."
    if "call" in w_set:
        return "Please call me."
    if "stop" in w_set:
        return "Please stop."
    if "ok" in w_set:
        return "Okay."
    if "sorry" in w_set:
        return "I am sorry."
    if "hello" in w_set:
        return "Hello!"
    if "bye" in w_set:
        return "Goodbye!"
    if "good" in w_set and "morning" in w_set:
        return "Good morning!"
    if "water" in w_set:
        return "I want water."
    if "please" in w_set:
        return "Please help me."
    if "name" in w_set:
        return "What is your name?"
    if "quiet" in w_set:
        return "Be Quiet"
    if "peace" in w_set:
        return "Peace makes us happy."

    return None  # fallback to LLM


# ============================================================
# 7.  CORE DETECTION — YOLO + SigLIP fusion
# ============================================================

def _run_gesture_siglip(frame):
    """
    Run MediaPipe hand crop → SigLIP Hand-Gesture-19.
    Returns (gesture_word, confidence) or (None, 0).
    """
    rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_res = hands_detector.process(rgb)

    if not mp_res.multi_hand_landmarks:
        return None, 0.0

    hand_lm = mp_res.multi_hand_landmarks[0]
    h, w    = frame.shape[:2]
    x1, y1, x2, y2 = get_hand_bbox(hand_lm, w, h)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None, 0.0

    pil_img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    inputs  = {k: v.to(device)
               for k, v in gesture_processor(images=pil_img,
                                             return_tensors="pt").items()}

    with torch.no_grad():
        logits = gesture_model(**inputs).logits

    probs   = torch.softmax(logits, dim=1)[0]
    pred_id = probs.argmax().item()
    conf    = probs[pred_id].item()
    gesture = GESTURE_ID2LABEL.get(pred_id, "no_gesture")

    # Raised threshold to 0.55 to reduce spurious gesture triggers
    if gesture == "no_gesture" or conf < 0.55:
        return None, 0.0

    word = GESTURE_TO_WORD.get(gesture)
    return word, conf


def detect_word(frame):
    """
    Run both YOLO word models + SigLIP gesture fusion.
    Returns (annotated_frame, best_label_or_None, best_conf).

    Fusion logic:
      - YOLO high-conf (≥ 0.60) → use YOLO directly
      - YOLO low-conf or no detection → try SigLIP gesture
      - If both agree → boost confidence
      - If only SigLIP → use it at face value
    """
    annotated       = frame.copy()
    yolo_label      = None
    yolo_conf       = 0.0
    word_detections = []

    # ── YOLO pass over both models ──
    for model in [words1_model, words2_model]:
        results = model(frame, conf=0.30)
        if results and results[0].boxes is not None:
            for box in results[0].boxes:
                cls_id = int(box.cls[0])
                label  = model.names.get(cls_id)
                conf   = float(box.conf[0])
                if conf > 0.40:
                    word_detections.append((label, conf))
            annotated = results[0].plot()

    if word_detections:
        yolo_label, yolo_conf = max(word_detections, key=lambda x: x[1])

    # ── SigLIP gesture ──
    gesture_word, gesture_conf = _run_gesture_siglip(frame)

    # ── Fusion ──
    if yolo_conf >= 0.60:
        # YOLO confident — check if gesture agrees
        if gesture_word and gesture_word.lower() == (yolo_label or "").lower():
            final_label = yolo_label
            final_conf  = min(1.0, yolo_conf + 0.10)
        else:
            # YOLO wins regardless — do NOT let SigLIP override a confident YOLO
            final_label = yolo_label
            final_conf  = yolo_conf
    elif yolo_label and gesture_word:
        if gesture_word.lower() == yolo_label.lower():
            # Both agree at lower confidence → boost
            final_label = yolo_label
            final_conf  = min(1.0, yolo_conf + 0.20)
        else:
            # Disagree — trust YOLO label but keep conf as-is
            final_label = yolo_label
            final_conf  = yolo_conf
    elif gesture_word:
        # Only SigLIP fired — use it
        final_label = gesture_word
        final_conf  = gesture_conf
    else:
        final_label = None
        final_conf  = 0.0

    # ── Draw label overlay on annotated frame ──
    if final_label:
        color = (0, 255, 0) if final_conf >= 0.65 else (0, 200, 255)
        cv2.putText(annotated,
                    f"{final_label.upper()}  ({final_conf:.2f})",
                    (10, 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)

    return annotated, final_label, final_conf

# ============================================================
# 8.  FRAME PIPELINE  (hold-duration logic)
# ============================================================

def _hold_progress(s):
    if s["hold_label"] and s["hold_start"] and not s["just_added"]:
        return round(min((time.time() - s["hold_start"]) / HOLD_DURATION, 1.0), 3)
    return 0.0


def process_frame(frame):
    s = _state
    s["frame_count"] += 1
    committed = ""

    # Frame-skip for performance
    if s["frame_count"] % FRAME_SKIP != 0:
        _, buf = cv2.imencode(".jpg", frame)
        return {
            "annotated_b64": base64.b64encode(buf).decode("utf-8"),
            "stable_label":  s["hold_label"],
            "stable_conf":   0.0,
            "hold_progress": _hold_progress(s),
            "committed":     "",
        }

    annotated, raw_label, raw_conf = detect_word(frame)

    # Update history
    if raw_label and raw_conf >= CONF_THRESHOLD:
        s["history"].append((raw_label, raw_conf))
    elif not raw_label:
        # Gradually drain history instead of popleft — prevents ghost labels
        if s["history"]:
            s["history"].popleft()
    # If raw_label exists but conf too low → do nothing (don't drain either)

    # Majority-vote stable label
    stable_label = None
    stable_conf  = 0.0
    if len(s["history"]) >= 3:   # require at least 3 frames before committing
        hist_labels  = [e[0] for e in s["history"]]
        top_label, top_count = Counter(hist_labels).most_common(1)[0]
        # Must appear in >50% of history to be stable
        if top_count / len(hist_labels) > 0.50:
            stable_label = top_label
            stable_conf  = (sum(c for l, c in s["history"] if l == stable_label)
                            / top_count)

    now = time.time()

    if stable_label:
        if stable_label == s["hold_label"]:
            elapsed = now - s["hold_start"]
            if elapsed >= HOLD_DURATION and not s["just_added"]:
                committed       = stable_label.lower()
                s["just_added"] = True
        else:
            s["hold_label"] = stable_label
            s["hold_start"] = now
            s["just_added"] = False
    else:
        s["hold_label"] = None
        s["hold_start"] = None
        s["just_added"] = False

    # Draw hold-progress bar
    if s["hold_label"] and not s["just_added"]:
        progress  = _hold_progress(s)
        hf, wf    = annotated.shape[:2]
        bar_y     = hf - 20
        cv2.rectangle(annotated, (10, bar_y), (wf - 10, bar_y + 12), (60, 60, 60), -1)
        fill  = int(10 + (wf - 20) * progress)
        color = (0, 255, 0) if stable_conf >= 0.65 else (0, 200, 255)
        cv2.rectangle(annotated, (10, bar_y), (fill, bar_y + 12), color, -1)

    _, buf = cv2.imencode(".jpg", annotated)
    return {
        "annotated_b64": base64.b64encode(buf).decode("utf-8"),
        "stable_label":  s["hold_label"],
        "stable_conf":   round(stable_conf, 3),
        "hold_progress": _hold_progress(s),
        "committed":     committed,
    }

# ============================================================
# 9.  OLLAMA SENTENCE GENERATION
# ============================================================
OLLAMA_MODEL = "gpt-oss:120b-cloud"
OLLAMA_URL   = "http://localhost:11434/api/chat"


def call_ollama(words):
    """
    Strict sentence generation — ONLY the given words must appear in the sentence.
    """
    words = [w.strip().lower() for w in words if w.strip()]
    if not words:
        return ""

    prompt = (
        f"Input words: {', '.join(words)}\n\n"
        "Task:\n"
        "Create ONE clear, natural English sentence using the given words.\n\n"
        "Rules:\n"
        "1. Prioritize meaningful words (actions like eat, drink, help, call).\n"
        "2. Ignore duplicate, conflicting, or irrelevant words if needed.\n"
        "3. Fix grammar properly (add I, am, want, to, etc.).\n"
        "4. Convert phrases correctly (e.g., 'i love you' → 'I love you').\n"
        "5. Keep sentence short and human-like.\n\n"
        "Examples:\n"
        "Input: eat, hungry → I am hungry, I want to eat.\n"
        "Input: drink, thirsty → I am thirsty, I want to drink.\n"
        "Input: help → I need help.\n"
        "Input: yes, eat → Yes, I want to eat.\n\n"
        "Output ONLY the final sentence. No explanation, no extra text."
    )

    try:
        resp = requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are a precise sentence construction assistant. "
                            "Your only job is to build one short grammatical English sentence "
                            "using exactly the words the user provides — nothing more, nothing less. "
                            "Never add encouragement, praise, or extra topic words. "
                            "Output ONLY the sentence."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                "stream": False,
            },
            timeout=25,
        )
        resp.raise_for_status()
        data = resp.json()
        if "message" in data and "content" in data["message"]:
            result = data["message"]["content"].strip()
            # Strip any accidental markdown or quotes the LLM adds
            result = result.strip('"').strip("'").strip("`")
            return result
        print(f"[Ollama] Unexpected response format: {data}")
    except requests.exceptions.ConnectionError:
        print("[Ollama] Connection refused — is Ollama running on port 11434?")
    except requests.exceptions.Timeout:
        print("[Ollama] Request timed out after 25s")
    except Exception as exc:
        print(f"[Ollama] Error: {exc}")

    # Graceful fallback — join words into a readable string
    return " ".join(words).capitalize() + "."


# ============================================================
# 10.  FLASK ROUTES
# ============================================================

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
    data    = request.get_json(force=True)
    img_b64 = data.get("image", "")

    if "," in img_b64:
        img_b64 = img_b64.split(",")[1]

    image_bytes = base64.b64decode(img_b64)
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
    words = request.get_json(force=True).get("words", [])

    if not words:
        return jsonify({"sentence": ""})

    # Step 1: clean/deduplicate
    cleaned = clean_words(words)

    if not cleaned:
        return jsonify({"sentence": ""})

    # Step 2: try rule-based first (fast + accurate)
    rule_sentence = build_simple_sentence(cleaned)

    if rule_sentence:
        sentence = rule_sentence
    else:
        # Step 3: fall back to LLM
        sentence = call_ollama(cleaned)

    return jsonify({"sentence": sentence})


@app.route("/api/mode")
def api_mode():
    return jsonify({"mode": "words"})


# ============================================================
# 11.  ENTRY POINT
# ============================================================
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)