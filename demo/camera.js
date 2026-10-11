/**
 * Camera Handler & Mini Game-Reaction Demo Logic
 * Implements camera access, centered guide framing, cropped capture,
 * and integration reaction testing.
 */

const ARABIC_LETTERS = [
  { char: "ا", name: "ألف (Alef)" },
  { char: "ب", name: "باء (Baa)" },
  { char: "ت", name: "تاء (Taa)" },
  { char: "ث", name: "ثاء (Thaa)" },
  { char: "ج", name: "جيم (Jeem)" },
  { char: "ح", name: "حاء (Haa)" },
  { char: "خ", name: "خاء (Khaa)" },
  { char: "د", name: "دال (Daal)" },
  { char: "ذ", name: "ذال (Dhaal)" },
  { char: "ر", name: "راء (Raa)" },
  { char: "ز", name: "زاي (Zaay)" },
  { char: "س", name: "سين (Seen)" },
  { char: "ش", name: "شين (Sheen)" },
  { char: "ص", name: "صاد (Saad)" },
  { char: "ض", name: "ضاد (Daad)" },
  { char: "ط", name: "طاء (Taa)" },
  { char: "ظ", name: "ظاء (Dhaa)" },
  { char: "ع", name: "عين (Ain)" },
  { char: "غ", name: "غين (Ghain)" },
  { char: "ف", name: "فاء (Faa)" },
  { char: "ق", name: "قاف (Qaaf)" },
  { char: "ك", name: "كاف (Kaaf)" },
  { char: "ل", name: "لام (Laam)" },
  { char: "م", name: "ميم (Meem)" },
  { char: "ن", name: "نون (Noon)" },
  { char: "ه", name: "هاء (Haa)" },
  { char: "و", name: "واو (Waaw)" },
  { char: "ي", name: "ياء (Yaa)" },
];

let currentTargetIndex = 1; // Default: 'ب' (Baa)
let videoStream = null;
let capturedBlob = null;
const api = new ArabicRecognitionAPI();

// DOM elements
const videoElem = document.getElementById("video-feed");
const canvasElem = document.getElementById("capture-canvas");
const capturedImgElem = document.getElementById("captured-image");
const guideFrame = document.getElementById("guide-frame");

const btnStartCamera = document.getElementById("btn-start-camera");
const btnCapture = document.getElementById("btn-capture");
const btnRecognize = document.getElementById("btn-recognize");
const btnRetake = document.getElementById("btn-retake");
const btnNextChallenge = document.getElementById("btn-next-letter");
const fileInput = document.getElementById("file-input");

const targetLetterElem = document.getElementById("target-letter");
const targetNameElem = document.getElementById("target-name");

const reactionBanner = document.getElementById("reaction-banner");
const reactionIcon = document.getElementById("reaction-icon");
const reactionTitle = document.getElementById("reaction-title");
const reactionDesc = document.getElementById("reaction-desc");

const statDetectedChar = document.getElementById("stat-detected-char");
const statConfidence = document.getElementById("stat-confidence");
const statLatency = document.getElementById("stat-latency");
const statAccepted = document.getElementById("stat-accepted");
const candidateList = document.getElementById("candidate-list");

function updateTargetPrompt() {
  const current = ARABIC_LETTERS[currentTargetIndex];
  targetLetterElem.textContent = current.char;
  targetNameElem.textContent = `اكتب الحرف: ${current.name}`;
}

async function startCamera() {
  try {
    const constraints = {
      video: {
        facingMode: "environment", // Prefer back camera on phones/tablets
        width: { ideal: 1280 },
        height: { ideal: 720 },
      },
    };
    videoStream = await navigator.mediaDevices.getUserMedia(constraints);
    videoElem.srcObject = videoStream;
    videoElem.style.display = "block";
    capturedImgElem.style.display = "none";
    guideFrame.style.display = "block";

    btnStartCamera.disabled = true;
    btnCapture.disabled = false;
    btnRetake.disabled = true;
    btnRecognize.disabled = true;
  } catch (err) {
    alert("Could not access camera: " + err.message + "\nYou can also use the 'Upload Image' button to test.");
    console.error("Camera error:", err);
  }
}

function captureFrame() {
  if (!videoStream) return;

  const vWidth = videoElem.videoWidth;
  const vHeight = videoElem.videoHeight;
  if (!vWidth || !vHeight) return;

  // Compute square guide coordinates in video coordinate space
  const guideSizeVideo = Math.min(vWidth, vHeight) * 0.55;
  const startX = (vWidth - guideSizeVideo) / 2;
  const startY = (vHeight - guideSizeVideo) / 2;

  canvasElem.width = guideSizeVideo;
  canvasElem.height = guideSizeVideo;

  const ctx = canvasElem.getContext("2d");
  // Crop only the central guide frame
  ctx.drawImage(
    videoElem,
    startX, startY, guideSizeVideo, guideSizeVideo,
    0, 0, guideSizeVideo, guideSizeVideo
  );

  canvasElem.toBlob((blob) => {
    capturedBlob = blob;
    capturedImgElem.src = URL.createObjectURL(blob);
    capturedImgElem.style.display = "block";
    videoElem.style.display = "none";
    guideFrame.style.display = "none";

    btnCapture.disabled = true;
    btnRecognize.disabled = false;
    btnRetake.disabled = false;
  }, "image/jpeg", 0.95);
}

function retakeFrame() {
  capturedBlob = null;
  capturedImgElem.style.display = "none";
  if (videoStream) {
    videoElem.style.display = "block";
    guideFrame.style.display = "block";
    btnCapture.disabled = false;
  } else {
    btnStartCamera.disabled = false;
  }
  btnRecognize.disabled = true;
  btnRetake.disabled = true;
  resetReaction();
}

async function performRecognition() {
  if (!capturedBlob) return;

  btnRecognize.disabled = true;
  btnRecognize.textContent = "جاري التحليل...";

  try {
    const res = await api.recognizeLetter(capturedBlob);
    displayResult(res);
  } catch (err) {
    alert("Recognition failed: " + err.message);
    console.error(err);
  } finally {
    btnRecognize.disabled = false;
    btnRecognize.textContent = "✨ تعرّف على الحرف (Recognize)";
  }
}

function displayResult(result) {
  const pred = result.prediction;
  const expected = ARABIC_LETTERS[currentTargetIndex].char;

  statDetectedChar.textContent = pred.letter || "—";
  statConfidence.textContent = `${(pred.confidence * 100).toFixed(1)}%`;
  statLatency.textContent = `${result.processing_time_ms} ms`;
  statAccepted.textContent = pred.accepted ? "مقبول (Accepted)" : "مرفوض (Retry)";

  // Update candidate list
  candidateList.innerHTML = "";
  if (result.top_candidates) {
    result.top_candidates.forEach((c) => {
      const li = document.createElement("li");
      li.className = "candidate-item";
      li.innerHTML = `
        <span class="char">${c.letter} (${c.name_en})</span>
        <span>${(c.confidence * 100).toFixed(1)}%</span>
      `;
      candidateList.appendChild(li);
    });
  }

  // Mini-Game Reaction logic (Section 14 of Brief)
  reactionBanner.className = "reaction-banner";
  if (!pred.accepted) {
    reactionBanner.classList.add("warning");
    reactionIcon.textContent = "🔍";
    reactionTitle.textContent = "لم أستطع قراءة الحرف بوضوح";
    reactionDesc.textContent = result.message || "يرجى التقاط صورة أخرى بإضاءة أفضل وثبات.";
  } else if (pred.letter === expected) {
    // Game Victory Reaction
    reactionBanner.classList.add("success");
    reactionIcon.textContent = "⭐ 🌟 🎉";
    reactionTitle.textContent = `ممتاز رائع! كتبتَ الحرف (${pred.letter}) بشكل صحيح!`;
    reactionDesc.textContent = "تفاعل اللعبة: حصلتَ على نجمة ذهبية!";
  } else {
    // Incorrect letter written
    reactionBanner.classList.add("error");
    reactionIcon.textContent = "✏️";
    reactionTitle.textContent = "حاول مرة أخرى!";
    reactionDesc.textContent = `المطلوب: (${expected}) — المكتشف: (${pred.letter})`;
  }
}

function resetReaction() {
  reactionBanner.className = "reaction-banner neutral";
  reactionIcon.textContent = "🎮";
  reactionTitle.textContent = "بانتظار التقاط الحرف...";
  reactionDesc.textContent = "اكتب الحرف المطلوب على ورقة بيضاء والتقطه بالكاميرا.";
  statDetectedChar.textContent = "-";
  statConfidence.textContent = "-";
  statLatency.textContent = "-";
  statAccepted.textContent = "-";
  candidateList.innerHTML = "";
}

// Event Listeners
btnStartCamera.addEventListener("click", startCamera);
btnCapture.addEventListener("click", captureFrame);
btnRetake.addEventListener("click", retakeFrame);
btnRecognize.addEventListener("click", performRecognition);

btnNextChallenge.addEventListener("click", () => {
  currentTargetIndex = (currentTargetIndex + 1) % ARABIC_LETTERS.length;
  updateTargetPrompt();
  resetReaction();
});

fileInput.addEventListener("change", (e) => {
  const file = e.target.files[0];
  if (file) {
    capturedBlob = file;
    capturedImgElem.src = URL.createObjectURL(file);
    capturedImgElem.style.display = "block";
    videoElem.style.display = "none";
    guideFrame.style.display = "none";

    btnCapture.disabled = true;
    btnRecognize.disabled = false;
    btnRetake.disabled = false;
    resetReaction();
  }
});

// Initialize on page load
updateTargetPrompt();
resetReaction();
