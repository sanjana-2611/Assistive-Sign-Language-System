from flask import Flask, render_template, request, jsonify
import cv2
import torch
import mediapipe as mp
import numpy as np
from transformers import AutoImageProcessor, SiglipForImageClassification
from PIL import Image
from collections import deque, Counter
import base64
import time
import requests
from wordfreq import top_n_list, word_frequency

app = Flask(__name__)

# ============================================================
# 1.  SIGLIP MODEL  (primary detector)
# ============================================================
MODEL_NAME = "prithivMLmods/Alphabet-Sign-Language-Detection"

processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
siglip_model = SiglipForImageClassification.from_pretrained(MODEL_NAME)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
siglip_model.to(device)
siglip_model.eval()

id2label = {int(k): v for k, v in siglip_model.config.id2label.items()}
print(f"[SigLIP] Loaded | {len(id2label)} classes | device: {device}")

# ============================================================
# 2.  MEDIAPIPE HANDS  (used for hand crop + optional fusion)
# ============================================================
mp_hands = mp.solutions.hands
mp_draw  = mp.solutions.drawing_utils

hands_detector = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.65,
    min_tracking_confidence=0.5,
)

PADDING = 30

# Letters where MediaPipe geometry cross-check helps reduce confusion
WEAK_LETTERS = set("b e c l m d f g h i j k r u".split())

# ============================================================
# 3.  MEDIAPIPE LANDMARK GEOMETRY HELPERS
# ============================================================

def _dist(a, b):
    return np.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)


def _angle(a, b, c):
    ba = np.array([a.x - b.x, a.y - b.y])
    bc = np.array([c.x - b.x, c.y - b.y])
    cos_a = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0)))


def _finger_extended(lm, tip, pip):
    return lm[tip].y < lm[pip].y


def _thumb_extended(lm):
    return lm[4].x < lm[3].x


def mediapipe_classify(lm):
    """Rule-based geometry classifier for weak/confused letters."""
    index_ext  = _finger_extended(lm, 8,  6)
    middle_ext = _finger_extended(lm, 12, 10)
    ring_ext   = _finger_extended(lm, 16, 14)
    pinky_ext  = _finger_extended(lm, 20, 18)
    thumb_ext  = _thumb_extended(lm)

    ti_dist  = _dist(lm[4], lm[8])   # thumb-index
    tm_dist  = _dist(lm[4], lm[12])  # thumb-middle
    im_dist  = _dist(lm[8], lm[12])  # index-middle

    # B — four fingers up, together, thumb tucked
    if index_ext and middle_ext and ring_ext and pinky_ext and not thumb_ext:
        spread = (_dist(lm[8], lm[12]) + _dist(lm[12], lm[16]) + _dist(lm[16], lm[20]))
        if spread < 0.18:
            return "b"

    # C — rounded open hand
    if not index_ext and not middle_ext and not ring_ext and not pinky_ext:
        if 0.08 < ti_dist < 0.25 and _dist(lm[4], lm[2]) > 0.06:
            return "c"

    # D — index up, thumb touches middle
    if index_ext and not middle_ext and not ring_ext and not pinky_ext:
        if tm_dist < 0.06:
            return "d"

    # E — all curled, thumb tucked under
    if not index_ext and not middle_ext and not ring_ext and not pinky_ext and ti_dist < 0.08:
        return "e"

    # F — middle/ring/pinky up, thumb-index pinch
    if middle_ext and ring_ext and pinky_ext and not index_ext and ti_dist < 0.06:
        return "f"

    # G — index pointing sideways
    if index_ext and not middle_ext and not ring_ext and not pinky_ext and not thumb_ext:
        if abs(lm[8].x - lm[5].x) > abs(lm[8].y - lm[5].y):
            return "g"

    # H — index + middle pointing sideways
    if index_ext and middle_ext and not ring_ext and not pinky_ext:
        if (abs(lm[8].x - lm[5].x) > abs(lm[8].y - lm[5].y) and
                abs(lm[12].x - lm[9].x) > abs(lm[12].y - lm[9].y)):
            return "h"

    # I — only pinky up, vertical
    if not index_ext and not middle_ext and not ring_ext and pinky_ext:
        if abs(lm[20].x - lm[17].x) <= 0.05:
            return "i"

    # J — pinky up, lateral movement (approximate)
    if not index_ext and not middle_ext and not ring_ext and pinky_ext:
        if abs(lm[20].x - lm[17].x) > 0.05:
            return "j"

    # K — index + middle up spread, thumb near middle
    if index_ext and middle_ext and not ring_ext and not pinky_ext:
        if im_dist > 0.10 and tm_dist < 0.09:
            return "k"

    # L — index up, thumb extended, L-shape
    if index_ext and not middle_ext and not ring_ext and not pinky_ext and thumb_ext:
        if _angle(lm[8], lm[5], lm[4]) > 70:
            return "l"

    # M — three fingers over thumb
    if not index_ext and not middle_ext and not ring_ext and not pinky_ext:
        thumb_under = (lm[4].y > lm[8].y and lm[4].y > lm[12].y and lm[4].y > lm[16].y)
        if thumb_under and ti_dist < 0.12:
            return "m"

    # R — index + middle crossed/close
    if index_ext and middle_ext and not ring_ext and not pinky_ext:
        if abs(lm[8].x - lm[12].x) < 0.04 and im_dist < 0.06:
            return "r"

    # U — index + middle up, parallel, close
    if index_ext and middle_ext and not ring_ext and not pinky_ext:
        if im_dist < 0.07 and abs(lm[8].x - lm[12].x) >= 0.04:
            return "u"

    return None

# ============================================================
# 4.  HAND BBOX HELPER
# ============================================================

def get_hand_bbox(landmarks, frame_w, frame_h, padding=PADDING):
    xs = [lm.x * frame_w for lm in landmarks.landmark]
    ys = [lm.y * frame_h for lm in landmarks.landmark]
    x1 = max(0,        int(min(xs)) - padding)
    y1 = max(0,        int(min(ys)) - padding)
    x2 = min(frame_w,  int(max(xs)) + padding)
    y2 = min(frame_h,  int(max(ys)) + padding)
    return x1, y1, x2, y2

# ============================================================
# 5.  HOLD-DURATION STATE  (single-user; extend to per-session dict if needed)
# ============================================================
HOLD_DURATION  = 1.5   # seconds to hold sign before letter commits
CONF_THRESHOLD = 0.50
HISTORY_LEN    = 12
FRAME_SKIP     = 2     # process every Nth frame for performance

_state = {
    "history":     deque(maxlen=HISTORY_LEN),
    "hold_label":  None,
    "hold_start":  None,
    "just_added":  False,
    "frame_count": 0,
}

# ============================================================
# 6.  CORE DETECTION — SigLIP + MediaPipe fusion
# ============================================================

def detect_letter(frame):
    """
    Run SigLIP on the hand crop.
    For weak letters, cross-check with MediaPipe geometry.
    Returns (annotated_frame, label_or_None, conf_or_0).
    """
    rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_res  = hands_detector.process(rgb)
    annotated = frame.copy()

    if not mp_res.multi_hand_landmarks:
        return annotated, None, 0.0

    hand_lm = mp_res.multi_hand_landmarks[0]
    h, w    = frame.shape[:2]

    # Draw skeleton
    mp_draw.draw_landmarks(annotated, hand_lm, mp_hands.HAND_CONNECTIONS)

    x1, y1, x2, y2 = get_hand_bbox(hand_lm, w, h)
    hand_crop = frame[y1:y2, x1:x2]
    if hand_crop.size == 0:
        return annotated, None, 0.0

    # --- SigLIP inference ---
    pil_img = Image.fromarray(cv2.cvtColor(hand_crop, cv2.COLOR_BGR2RGB))
    inputs  = {k: v.to(device)
               for k, v in processor(images=pil_img, return_tensors="pt").items()}

    with torch.no_grad():
        logits = siglip_model(**inputs).logits

    probs      = torch.softmax(logits, dim=1)[0]
    pred_id    = probs.argmax().item()
    siglip_conf  = probs[pred_id].item()
    siglip_label = id2label.get(pred_id, "?")

    final_label = siglip_label
    final_conf  = siglip_conf

    # --- MediaPipe geometry cross-check for weak letters ---
    if siglip_label.lower() in WEAK_LETTERS and siglip_conf < 0.70:
        mp_label = mediapipe_classify(hand_lm.landmark)
        if mp_label is not None:
            if mp_label == siglip_label.lower():
                # Both agree → boost confidence
                final_conf = min(1.0, siglip_conf + 0.20)
            else:
                # Geometry override (more reliable for these letters)
                final_label = mp_label.upper()
                final_conf  = 0.72

    # Draw bbox + label on annotated frame
    color = (0, 255, 0) if final_conf >= 0.65 else (0, 200, 255)
    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
    cv2.putText(annotated,
                f"{final_label} ({final_conf:.2f})",
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)

    return annotated, final_label, final_conf


# ============================================================
# 7.  FRAME PIPELINE  (hold-duration logic)
# ============================================================

def _hold_progress(s):
    if s["hold_label"] and s["hold_start"] and not s["just_added"]:
        return round(min((time.time() - s["hold_start"]) / HOLD_DURATION, 1.0), 3)
    return 0.0


def process_frame(frame):
    """
    Full pipeline: detect → update hold state → return dict.

    Returns:
      annotated_b64  – JPEG bytes (base64) for browser display
      stable_label   – letter currently being held (or None)
      stable_conf    – confidence of stable label
      hold_progress  – 0.0–1.0 fill bar
      committed      – non-empty string when a letter is committed this frame
    """
    s = _state
    s["frame_count"] += 1
    committed = ""

    # Frame-skip: return lightweight response on skipped frames
    if s["frame_count"] % FRAME_SKIP != 0:
        _, buf = cv2.imencode(".jpg", frame)
        return {
            "annotated_b64": base64.b64encode(buf).decode("utf-8"),
            "stable_label":  s["hold_label"],
            "stable_conf":   0.0,
            "hold_progress": _hold_progress(s),
            "committed":     "",
        }

    annotated, raw_label, raw_conf = detect_letter(frame)

    # Update history
    if raw_label and raw_conf >= CONF_THRESHOLD:
        s["history"].append((raw_label, raw_conf))
    elif not raw_label and s["history"]:
        s["history"].popleft()   # gentle decay

    # Majority-vote stable label
    stable_label = None
    stable_conf  = 0.0
    if s["history"]:
        hist_labels  = [e[0] for e in s["history"]]
        stable_label = Counter(hist_labels).most_common(1)[0][0]
        stable_conf  = (sum(c for l, c in s["history"] if l == stable_label)
                        / hist_labels.count(stable_label))

    now = time.time()

    if stable_label:
        if stable_label == s["hold_label"]:
            elapsed = now - s["hold_start"]
            if elapsed >= HOLD_DURATION and not s["just_added"]:
                committed       = stable_label.upper()
                s["just_added"] = True
        else:
            s["hold_label"] = stable_label
            s["hold_start"] = now
            s["just_added"] = False
    else:
        s["hold_label"] = None
        s["hold_start"] = None
        s["just_added"] = False

    # Draw hold-progress bar if a sign is being held
    if s["hold_label"] and not s["just_added"]:
        progress = _hold_progress(s)
        h_frame, w_frame = annotated.shape[:2]
        bar_y = h_frame - 20
        cv2.rectangle(annotated, (10, bar_y), (w_frame - 10, bar_y + 12), (60, 60, 60), -1)
        fill = int(10 + (w_frame - 20) * progress)
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
# 8.  WORD LIST  (for smart word reconstruction)
# ============================================================
english_words = set(top_n_list("en", 50000))


def _anagram_resolve(letters):
    """
    Stage-1 fast path: try exact → anagram → drop-one anagram.
    Returns best candidate string, or None if nothing found.
    """
    from itertools import permutations

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

    return None


def _llm_resolve(letters):
    """
    Stage-2 LLM path: ask Ollama to find the closest real English word
    given the signed letters (which may be out-of-order or have 1 wrong letter).

    Returns the corrected word string, or the raw joined letters on failure.
    """
    raw = "".join(letters).upper()
    prompt = (
        f"A sign language learner signed these letters one by one: {', '.join(letters).upper()}.\n"
        f"The letters may be slightly out of order or have one mistake because "
        f"sign detection isn't perfect.\n"
        f"What is the single most likely real English word they intended?\n"
        f"Rules:\n"
        f"  - Return ONLY the word, lowercase, nothing else.\n"
        f"  - No punctuation, no explanation, no sentence.\n"
        f"  - Must be a real common English word.\n"
        f"  - Prefer words phonetically or visually close to '{raw}'.\n"
        f"Word:"
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
                            "You are a spelling correction assistant for a sign language app. "
                            "The user gives you a sequence of letters that may be scrambled or "
                            "have one error. You must respond with exactly ONE real English word "
                            "and nothing else — no punctuation, no explanation."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                "stream": False,
            },
            timeout=15,
        )
        data = resp.json()
        if "message" in data:
            word = data["message"]["content"].strip().lower()
            # Safety: strip any accidental punctuation / spaces
            word = "".join(c for c in word if c.isalpha())
            if word:
                print(f"[LLM word] {raw} → '{word}'")
                return word
    except Exception as exc:
        print(f"[LLM word] Error: {exc}")

    # Hard fallback: return raw letters joined
    return raw.lower()


def letters_to_word(letters):
    """
    Two-stage word resolver:
      1. Anagram / wordfreq fast-path  (instant, no network)
      2. LLM correction via Ollama     (handles scrambled / noisy input)

    Example:  Z A E L  →  stage-1 finds 'zeal'  (anagram match)
              X A E L  →  stage-1 finds nothing  →  stage-2 LLM returns 'zeal'
    """
    letters = [l.lower() for l in letters if l.strip()]
    if not letters:
        return ""

    # Stage 1 — fast anagram match
    fast = _anagram_resolve(letters)
    if fast:
        print(f"[Anagram] {''.join(letters).upper()} → '{fast}'")
        return fast

    # Stage 2 — LLM correction (always runs when stage-1 fails)
    return _llm_resolve(letters)

# ============================================================
# 9.  OLLAMA SENTENCE GENERATION
# ============================================================
OLLAMA_MODEL = "gpt-oss:120b-cloud"
OLLAMA_URL   = "http://localhost:11434/api/chat"


def call_ollama(words):
    """
    Given a list of words (already resolved from letter groups),
    ask Ollama to produce one coherent, encouraging sentence.
    """
    words = [w.strip() for w in words if w.strip()]
    if not words:
        return ""

    word_str = " ".join(words)
    prompt = (
        f"A sign language learner signed these words: {word_str}. "
        f"Write ONE short, simple, encouraging sentence (max 15 words) "
        f"that naturally uses all of these words. "
        f"Return only the sentence, nothing else."
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
                            "You are a helpful assistant for a sign language coaching app. "
                            "Always produce short, clear, meaningful English sentences. "
                            "Never produce random or nonsensical output."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                "stream": False,
            },
            timeout=25,
        )
        data = resp.json()
        if "message" in data:
            return data["message"]["content"].strip()
    except Exception as exc:
        print(f"[Ollama] Error: {exc}")

    return word_str   # graceful fallback

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


# ------------------------------------------------------------------
# POST /api/predict
#   Body : { "image": "<base64 data-URL jpeg>" }
#   Returns:
#     image         – annotated frame (base64 JPEG)
#     stable_label  – letter currently being held  ("A" | null)
#     stable_conf   – float 0-1
#     hold_progress – float 0-1  (fill bar)
#     committed     – letter string when just committed, else ""
# ------------------------------------------------------------------
@app.route("/api/predict", methods=["POST"])
def api_predict():
    data     = request.get_json(force=True)
    img_b64  = data.get("image", "")

    # Strip the data-URL prefix if present
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


# ------------------------------------------------------------------
# POST /api/word
#   Body : { "letters": ["H","E","L","L","O"] }
#   Returns: { "word": "hello" }
#
#   Call this when the user completes a word (space gesture).
# ------------------------------------------------------------------
@app.route("/api/word", methods=["POST"])
def api_word():
    data    = request.get_json(force=True)
    letters = data.get("letters", [])
    if not letters:
        return jsonify({"word": ""})
    word = letters_to_word(letters)
    return jsonify({"word": word})


# ------------------------------------------------------------------
# POST /api/sentence
#   Body : { "words": ["hello", "world"] }
#   Returns: { "sentence": "Hello, world!" }
#
#   Call this when the user triggers sentence generation (space gesture
#   on an already-empty current word, or a dedicated UI button).
# ------------------------------------------------------------------
@app.route("/api/sentence", methods=["POST"])
def api_sentence():
    words    = request.get_json(force=True).get("words", [])
    sentence = call_ollama(words)
    return jsonify({"sentence": sentence})


# ============================================================
# 11.  ENTRY POINT
# ============================================================
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)