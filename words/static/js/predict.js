/**
 * predict.js  — WORDS mode
 * ──────────────────────────────────────────────────────────────
 * Pipeline:
 *   1. Webcam frame  →  POST /api/predict
 *      ← { annotated_b64, stable_label, stable_conf,
 *           hold_progress, committed }
 *
 *   2. When "committed" arrives → it's already a full word
 *      Push directly into completedWords[]
 *
 *   3. "Add Word" button → manually commits whatever is currently
 *      being held (stable_label) into completedWords[]
 *
 *   4. "Build Sentence" button → POST /api/sentence { words: [...] }
 *      ← { sentence }  → render with typewriter effect
 *
 *   5. "Speak" → Web Speech API TTS
 *   6. "✕" on word chip → remove that word
 *   7. "Clear" → wipe completedWords[]
 * ──────────────────────────────────────────────────────────────
 */

document.addEventListener("DOMContentLoaded", () => {

  // ─── DOM refs ───────────────────────────────────────────────
  const video           = document.getElementById("camera");
  const canvas          = document.getElementById("capture-canvas");
  const ctx             = canvas.getContext("2d");
  const annotatedImg    = document.getElementById("annotated-image");
  const statusText      = document.getElementById("status-text");
  const livePill        = document.getElementById("live-pill");

  const holdBar         = document.getElementById("hold-bar");
  const holdLabelText   = document.getElementById("hold-label-text");
  const detLetter       = document.getElementById("det-letter");
  const detConf         = document.getElementById("det-conf");
  const detBadge        = document.getElementById("detection-badge");

  const currentWordEl   = document.getElementById("current-word-display");
  const letterChipsEl   = document.getElementById("letter-chips");   // reused — shows current held word
  const tokensList      = document.getElementById("tokens-list");
  const sentenceBox     = document.getElementById("sentence-box");
  const sentenceSpinner = document.getElementById("sentence-spinner");

  const btnStart        = document.getElementById("btn-start");
  const btnStop         = document.getElementById("btn-stop");
  const btnBackspace    = document.getElementById("btn-backspace");   // "Remove last word"
  const btnWordBreak    = document.getElementById("btn-word-break");  // "Add Word"
  const btnClearWords   = document.getElementById("btn-clear-words");
  const btnSentence     = document.getElementById("btn-sentence");
  const btnSpeak        = document.getElementById("btn-speak");

  // Update button labels for words mode
  if (btnWordBreak)  btnWordBreak.textContent  = "Add Word ›";
  if (btnBackspace)  btnBackspace.textContent  = "⌫ Remove";

  // ─── State ──────────────────────────────────────────────────
  let stream         = null;
  let loopId         = null;
  let running        = false;
  let busy           = false;

  let currentHeld    = null;   // word currently being held (stable_label from server)
  let completedWords = [];     // committed words list
  let lastSentence   = "";

  const FRAME_INTERVAL = 120; // ms (~8 fps)
  let   lastSendTime   = 0;

  // ─── Camera ─────────────────────────────────────────────────
  btnStart.addEventListener("click", startCamera);
  btnStop .addEventListener("click", stopCamera);

  async function startCamera() {
    try {
      stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480, facingMode: "user" },
        audio: false,
      });
      video.srcObject = stream;
      await video.play();

      running = true;
      btnStart.disabled = true;
      btnStop .disabled = false;
      setStatus("🟢 Detecting words…");
      livePill.classList.add("active");
      loop();
    } catch (err) {
      setStatus("❌ Camera error: " + err.message);
      console.error(err);
    }
  }

  function stopCamera() {
    running = false;
    if (loopId) cancelAnimationFrame(loopId);
    if (stream) stream.getTracks().forEach(t => t.stop());
    video.srcObject = null;
    btnStart.disabled = false;
    btnStop .disabled = true;
    setStatus("Camera stopped");
    livePill.classList.remove("active");
    resetHoldBar();
  }

  // ─── Main loop ──────────────────────────────────────────────
  function loop() {
    if (!running) return;
    loopId = requestAnimationFrame(loop);
    const now = performance.now();
    if (now - lastSendTime < FRAME_INTERVAL) return;
    if (busy) return;
    lastSendTime = now;
    sendFrame();
  }

  async function sendFrame() {
    busy = true;
    try {
      canvas.width  = video.videoWidth  || 640;
      canvas.height = video.videoHeight || 480;
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      const b64 = canvas.toDataURL("image/jpeg", 0.8);

      const res  = await fetch("/api/predict", {
        method:  "POST",
        headers: { "Content-Type": "application/json" },
        body:    JSON.stringify({ image: b64 }),
      });

      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();

      // ── Annotated feed ──
      if (data.image) {
        annotatedImg.src = "data:image/jpeg;base64," + data.image;
      }

      // ── Detection badge ──
      currentHeld = data.stable_label || null;

      if (currentHeld) {
        detLetter.textContent = currentHeld.toUpperCase();
        detConf  .textContent = Math.round((data.stable_conf || 0) * 100) + "%";
        detBadge .classList.add("visible");
        renderCurrentHeld(currentHeld);
      } else {
        detBadge.classList.remove("visible");
        detLetter.textContent = "?";
        detConf  .textContent = "0%";
        renderCurrentHeld(null);
      }

      // ── Hold bar ──
      const progress = data.hold_progress || 0;
      holdBar.style.width      = (progress * 100) + "%";
      holdBar.style.background = progress > 0.85
        ? "var(--color-accent-green, #00e676)"
        : "var(--color-accent, #7c6aff)";

      holdLabelText.textContent = currentHeld
        ? `Holding: ${currentHeld.toUpperCase()} (${Math.round(progress * 100)}%)`
        : "—";

      // ── Auto-committed word from server ──
      if (data.committed && data.committed.trim()) {
        const word = data.committed.trim().toLowerCase();
        completedWords.push(word);
        renderWords();
        flashBadge();
        setStatus(`✔ Word committed: "${word}"`);
      }

    } catch (err) {
      console.error("Frame error:", err);
      setStatus("⚠ " + err.message);
    } finally {
      busy = false;
    }
  }

  // ─── "Add Word" button — manually commit currently held word ─
  btnWordBreak.addEventListener("click", () => {
    if (!currentHeld) {
      setStatus("⚠ No word being held yet.");
      return;
    }
    const word = currentHeld.trim().toLowerCase();
    completedWords.push(word);
    renderWords();
    renderCurrentHeld(null);
    setStatus(`✔ Word added: "${word}"`);
  });

  // ─── Remove last word ────────────────────────────────────────
  btnBackspace.addEventListener("click", () => {
    if (completedWords.length > 0) {
      const removed = completedWords.pop();
      renderWords();
      setStatus(`↩ Removed: "${removed}"`);
    }
  });

  // ─── Clear all words ─────────────────────────────────────────
  btnClearWords.addEventListener("click", () => {
    completedWords = [];
    renderWords();
    sentenceBox.innerHTML = '<span class="sentence-placeholder">Waiting for words…</span>';
    btnSpeak.disabled = true;
    lastSentence = "";
    setStatus("Cleared.");
  });

  // ─── Build Sentence ──────────────────────────────────────────
  btnSentence.addEventListener("click", buildSentence);

  async function buildSentence() {
    const allWords = [...completedWords];
    if (currentHeld) allWords.push(currentHeld.toLowerCase());

    if (allWords.length === 0) {
      setStatus("⚠ No words to build from.");
      return;
    }

    btnSentence.disabled = true;
    sentenceSpinner.style.display = "block";
    sentenceBox.innerHTML = "";

    try {
      const res = await fetch("/api/sentence", {
        method:  "POST",
        headers: { "Content-Type": "application/json" },
        body:    JSON.stringify({ words: allWords }),
      });
      const data = await res.json();
      lastSentence = data.sentence || allWords.join(" ");

      sentenceBox.innerHTML = "";
      typewriterEffect(sentenceBox, lastSentence);
      btnSpeak.disabled = false;
      setStatus("✦ Sentence ready!");
    } catch (err) {
      sentenceBox.textContent = "⚠ Failed to get sentence.";
      console.error(err);
    } finally {
      btnSentence.disabled = false;
      sentenceSpinner.style.display = "none";
    }
  }

  // ─── Speak ───────────────────────────────────────────────────
  btnSpeak.addEventListener("click", () => {
    if (!lastSentence) return;
    const utt = new SpeechSynthesisUtterance(lastSentence);
    utt.lang = "en-US";
    utt.rate = 0.95;
    speechSynthesis.speak(utt);
  });

  // ─── Renderers ───────────────────────────────────────────────

  function renderCurrentHeld(word) {
    // "Current Word" panel shows what is being held right now
    if (!word) {
      currentWordEl.innerHTML = '<span class="cw-placeholder">Hold a sign…</span>';
      if (letterChipsEl) letterChipsEl.innerHTML = "";
      return;
    }
    currentWordEl.innerHTML =
      `<span class="cw-text">${word.toUpperCase()}</span><span class="cw-cursor">|</span>`;
  }

  function renderWords() {
    if (completedWords.length === 0) {
      tokensList.innerHTML = '<span class="tokens-placeholder">No words yet.</span>';
      return;
    }

    tokensList.innerHTML = completedWords
      .map((w, i) =>
        `<span class="token-chip" data-idx="${i}">
           ${w}
           <button class="token-del" data-idx="${i}" title="Remove">×</button>
         </span>`
      )
      .join("");

    tokensList.querySelectorAll(".token-del").forEach(btn => {
      btn.addEventListener("click", e => {
        const idx = parseInt(e.target.dataset.idx);
        completedWords.splice(idx, 1);
        renderWords();
      });
    });
  }

  // ─── Typewriter ──────────────────────────────────────────────
  function typewriterEffect(el, text, speed = 28) {
    let i = 0;
    el.innerHTML = '<span class="tw-text"></span><span class="cw-cursor">|</span>';
    const span   = el.querySelector(".tw-text");
    const cursor = el.querySelector(".cw-cursor");

    const tick = setInterval(() => {
      if (i < text.length) {
        span.textContent += text[i++];
      } else {
        clearInterval(tick);
        cursor.style.display = "none";
      }
    }, speed);
  }

  // ─── Flash badge ─────────────────────────────────────────────
  function flashBadge() {
    detBadge.classList.add("flash");
    setTimeout(() => detBadge.classList.remove("flash"), 400);
  }

  // ─── Helpers ─────────────────────────────────────────────────
  function setStatus(msg) { statusText.textContent = msg; }
  function resetHoldBar() {
    holdBar.style.width = "0%";
    holdLabelText.textContent = "—";
  }

});