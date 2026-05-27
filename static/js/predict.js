/**
 * predict.js
 * ──────────────────────────────────────────────────────────────
 * Pipeline:
 *   1. Webcam frame  →  POST /api/predict
 *      ← { annotated_b64, stable_label, stable_conf,
 *           hold_progress, committed }
 *
 *   2. When "committed" arrives → append letter to currentLetters[]
 *      Visual: letter chips + large word display update live
 *
 *   3. "Word Break" button →  POST /api/word  { letters: [...] }
 *      ← { word }  → push word chip into completedWords[]
 *      Reset currentLetters[]
 *
 *   4. "Build Sentence" button → POST /api/sentence { words: [...] }
 *      ← { sentence }  → render in sentence-box
 *
 *   5. "Speak" button  → Web Speech API TTS
 *   6. "⌫ Back"        → remove last letter from currentLetters[]
 *   7. "Clear"         → wipe completedWords[]
 * ──────────────────────────────────────────────────────────────
 */

document.addEventListener("DOMContentLoaded", () => {

  // ─── DOM refs ───────────────────────────────────────────────
  const video          = document.getElementById("camera");
  const canvas         = document.getElementById("capture-canvas");
  const ctx            = canvas.getContext("2d");
  const annotatedImg   = document.getElementById("annotated-image");
  const statusText     = document.getElementById("status-text");
  const livePill       = document.getElementById("live-pill");

  const holdBar        = document.getElementById("hold-bar");
  const holdLabelText  = document.getElementById("hold-label-text");
  const detLetter      = document.getElementById("det-letter");
  const detConf        = document.getElementById("det-conf");
  const detBadge       = document.getElementById("detection-badge");

  const currentWordEl  = document.getElementById("current-word-display");
  const letterChipsEl  = document.getElementById("letter-chips");
  const tokensList     = document.getElementById("tokens-list");
  const sentenceBox    = document.getElementById("sentence-box");
  const sentenceSpinner = document.getElementById("sentence-spinner");

  const btnStart       = document.getElementById("btn-start");
  const btnStop        = document.getElementById("btn-stop");
  const btnBackspace   = document.getElementById("btn-backspace");
  const btnWordBreak   = document.getElementById("btn-word-break");
  const btnClearWords  = document.getElementById("btn-clear-words");
  const btnSentence    = document.getElementById("btn-sentence");
  const btnSpeak       = document.getElementById("btn-speak");

  // ─── State ──────────────────────────────────────────────────
  let stream          = null;
  let loopId          = null;       // requestAnimationFrame handle
  let running         = false;
  let busy            = false;      // throttle concurrent requests

  let currentLetters  = [];         // letters in the word being built
  let completedWords  = [];         // finished words
  let lastSentence    = "";

  const FRAME_INTERVAL = 120;       // ms between frames sent to server (~8 fps)
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
      setStatus("🟢 Detecting…");
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
      // Capture frame
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

      // ── Update annotated feed ──
      if (data.image) {
        annotatedImg.src = "data:image/jpeg;base64," + data.image;
      }

      // ── Update detection badge ──
      if (data.stable_label) {
        detLetter.textContent = data.stable_label;
        detConf  .textContent = Math.round((data.stable_conf || 0) * 100) + "%";
        detBadge .classList.add("visible");
      } else {
        detBadge.classList.remove("visible");
        detLetter.textContent = "?";
        detConf  .textContent = "0%";
      }

      // ── Update hold bar ──
      const progress = data.hold_progress || 0;
      holdBar.style.width = (progress * 100) + "%";
      holdBar.style.background = progress > 0.85
        ? "var(--color-accent-green)"
        : "var(--color-accent)";

      if (data.stable_label) {
        holdLabelText.textContent = `Holding: ${data.stable_label} (${Math.round(progress * 100)}%)`;
      } else {
        holdLabelText.textContent = "—";
      }

      // ── Committed letter? ──
      if (data.committed && data.committed.trim()) {
        const letter = data.committed.trim().toUpperCase();

        // Special built-in "del" from model
        if (letter === "DEL") {
          currentLetters.pop();
        } else if (letter === "SPACE") {
          // model-side space (not used in button mode, but handle gracefully)
          triggerWordBreak();
        } else {
          currentLetters.push(letter);
          flashBadge();
        }

        renderCurrentWord();
      }

    } catch (err) {
      console.error("Frame error:", err);
      setStatus("⚠ " + err.message);
    } finally {
      busy = false;
    }
  }

  // ─── Word Break button ───────────────────────────────────────
  btnWordBreak.addEventListener("click", triggerWordBreak);

  async function triggerWordBreak() {
    if (currentLetters.length === 0) {
      // Double-space with no letters → treat as sentence trigger
      if (completedWords.length > 0) buildSentence();
      return;
    }

    btnWordBreak.disabled = true;
    btnWordBreak.textContent = "…";

    try {
      const res = await fetch("/api/word", {
        method:  "POST",
        headers: { "Content-Type": "application/json" },
        body:    JSON.stringify({ letters: currentLetters }),
      });
      const data = await res.json();
      const word = (data.word || currentLetters.join("")).toLowerCase();

      completedWords.push(word);
      renderWords();

      currentLetters = [];
      renderCurrentWord();
      setStatus(`✔ Word added: "${word}"`);
    } catch (err) {
      console.error("Word break error:", err);
      setStatus("⚠ Word resolve failed");
    } finally {
      btnWordBreak.disabled = false;
      btnWordBreak.textContent = "Space ›";
    }
  }

  // ─── Backspace ───────────────────────────────────────────────
  btnBackspace.addEventListener("click", () => {
    if (currentLetters.length > 0) {
      currentLetters.pop();
      renderCurrentWord();
    }
  });

  // ─── Clear completed words ───────────────────────────────────
  btnClearWords.addEventListener("click", () => {
    completedWords = [];
    renderWords();
    sentenceBox.innerHTML = '<span class="sentence-placeholder">Waiting for words…</span>';
    btnSpeak.disabled = true;
    lastSentence = "";
  });

  // ─── Build Sentence ──────────────────────────────────────────
  btnSentence.addEventListener("click", buildSentence);

  async function buildSentence() {
    // Include any unsaved letters as-is
    const allWords = [...completedWords];
    if (currentLetters.length > 0) {
      allWords.push(currentLetters.join("").toLowerCase());
    }

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
  function renderCurrentWord() {
    const word = currentLetters.join("");

    if (word.length === 0) {
      currentWordEl.innerHTML = '<span class="cw-placeholder">Sign a letter…</span>';
      letterChipsEl.innerHTML = "";
      return;
    }

    // Big display word with blinking cursor
    currentWordEl.innerHTML =
      `<span class="cw-text">${word}</span><span class="cw-cursor">|</span>`;

    // Individual letter chips with remove button
    letterChipsEl.innerHTML = currentLetters
      .map((l, i) =>
        `<span class="lchip" data-idx="${i}">${l}<button class="lchip-del" data-idx="${i}" title="Remove">×</button></span>`
      )
      .join("");

    // Attach remove handlers
    letterChipsEl.querySelectorAll(".lchip-del").forEach(btn => {
      btn.addEventListener("click", e => {
        const idx = parseInt(e.target.dataset.idx);
        currentLetters.splice(idx, 1);
        renderCurrentWord();
      });
    });
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
           <button class="token-del" data-idx="${i}" title="Remove word">×</button>
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

  // ─── Typewriter effect for sentence ─────────────────────────
  function typewriterEffect(el, text, speed = 28) {
    let i = 0;
    el.innerHTML = '<span class="tw-text"></span><span class="cw-cursor">|</span>';
    const span = el.querySelector(".tw-text");
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

  // ─── Flash badge on commit ───────────────────────────────────
  function flashBadge() {
    detBadge.classList.add("flash");
    setTimeout(() => detBadge.classList.remove("flash"), 400);
  }

  // ─── Helpers ─────────────────────────────────────────────────
  function setStatus(msg) {
    statusText.textContent = msg;
  }

  function resetHoldBar() {
    holdBar.style.width = "0%";
    holdLabelText.textContent = "—";
  }

});