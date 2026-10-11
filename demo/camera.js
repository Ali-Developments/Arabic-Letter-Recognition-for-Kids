/**
 * Camera Handler & Mini Game-Reaction Demo Logic
 * Implements camera access, centered guide framing, cropped capture,
 * environment diagnostics, error analysis, one-button Capture & Recognize,
 * and game reaction integration.
 */

// Priority testing sequence: dotted/similar groups first, then remaining letters
const ARABIC_LETTERS = [
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
  { char: "ن", name: "نون (Noon)" },
  { char: "ي", name: "ياء (Yaa)" },
  { char: "ا", name: "ألف (Alef)" },
  { char: "ك", name: "كاف (Kaaf)" },
  { char: "ل", name: "لام (Laam)" },
  { char: "م", name: "ميم (Meem)" },
  { char: "ه", name: "هاء (Haa)" },
  { char: "و", name: "واو (Waaw)" },
];

let currentTargetIndex = 0; // Default: 'ب' (Baa)
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
const btnCaptureRecognize = document.getElementById("btn-capture-recognize");
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

const statExpectedChar = document.getElementById("stat-expected-char");
const statDetectedChar = document.getElementById("stat-detected-char");
const statConfidence = document.getElementById("stat-confidence");
const statLatency = document.getElementById("stat-latency");
const statAccepted = document.getElementById("stat-accepted");
const statRequestId = document.getElementById("stat-request-id");
const candidateList = document.getElementById("candidate-list");

// Diagnostic elements
const diagSecure = document.getElementById("diag-secure");
const diagHost = document.getElementById("diag-host");
const diagGum = document.getElementById("diag-gum");
const diagUa = document.getElementById("diag-ua");
const diagStreamActive = document.getElementById("diag-stream-active");
const diagVideoDim = document.getElementById("diag-video-dim");
const diagFacing = document.getElementById("diag-facing");
const diagPipeState = document.getElementById("diag-pipe-state");
const diagBlobInfo = document.getElementById("diag-blob-info");
const diagApiUrl = document.getElementById("diag-api-url");
const diagErrName = document.getElementById("diag-err-name");
const diagErrMsg = document.getElementById("diag-err-msg");
const debugErrorDetail = document.getElementById("debug-error-detail");
const debugSecurityAlert = document.getElementById("debug-security-alert");
const rawCaptureBox = document.getElementById("raw-capture-box");
const rawCapturedThumb = document.getElementById("raw-captured-thumb");

function setPipelineState(state, color = "#38bdf8") {
  console.log(`[PIPELINE STATE] -> ${state}`);
  if (diagPipeState) {
    diagPipeState.textContent = state;
    diagPipeState.style.color = color;
  }
}

function renderEnvironmentDiagnostics() {
  const isSecure = window.isSecureContext === true;
  const hasMediaDevices = !!navigator.mediaDevices;
  const hasGum = !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia);
  const proto = location.protocol;
  const host = location.hostname + (location.port ? `:${location.port}` : "");

  console.log("Environment Diagnostics:", {
    isSecureContext: window.isSecureContext,
    mediaDevices: navigator.mediaDevices,
    getUserMedia: hasGum,
    userAgent: navigator.userAgent,
    protocol: location.protocol,
    hostname: location.hostname,
    apiBaseUrl: api.baseUrl,
  });

  if (diagSecure) {
    diagSecure.textContent = isSecure ? "true (Secure Context)" : "false (INSECURE CONTEXT)";
    diagSecure.style.color = isSecure ? "#4ade80" : "#f87171";
  }
  if (diagHost) {
    diagHost.textContent = `${proto}//${host}`;
  }
  if (diagGum) {
    diagGum.textContent = hasGum ? "Available (Yes)" : "Unavailable (No)";
    diagGum.style.color = hasGum ? "#4ade80" : "#f87171";
  }
  if (diagUa) {
    diagUa.textContent = navigator.userAgent;
  }
  if (diagApiUrl) {
    diagApiUrl.textContent = `${api.baseUrl}/api/v1/recognize-letter`;
  }

  // Security Alert Banner
  if (debugSecurityAlert) {
    if (!isSecure && location.hostname !== "localhost" && location.hostname !== "127.0.0.1") {
      debugSecurityAlert.className = "debug-alert";
      debugSecurityAlert.innerHTML = `
        <strong>⚠️ تحذير بيئة غير آمنة (Insecure Context):</strong><br>
        Live camera is unavailable because this page is not running in a trusted secure context (window.isSecureContext = false).<br>
        Mobile browsers strictly disable <code>navigator.mediaDevices.getUserMedia</code> on plain HTTP or untrusted HTTPS outside localhost.
      `;
    } else if (!hasGum) {
      debugSecurityAlert.className = "debug-alert warning";
      debugSecurityAlert.innerHTML = `
        <strong>⚠️ getUserMedia غير متاح:</strong><br>
        <code>navigator.mediaDevices.getUserMedia</code> is not supported or was blocked by browser security policy.
      `;
    } else {
      debugSecurityAlert.className = "debug-alert success";
      debugSecurityAlert.innerHTML = `
        <strong>✅ البيئة جاهزة للتشغيل (Ready):</strong><br>
        Secure context and getUserMedia are supported. Click "فتح الكاميرا (Open Camera)" to start.
      `;
    }
  }
}

function updateTargetPrompt() {
  const current = ARABIC_LETTERS[currentTargetIndex];
  targetLetterElem.textContent = current.char;
  targetNameElem.textContent = `اكتب الحرف: ${current.name}`;
  if (statExpectedChar) {
    statExpectedChar.textContent = current.char;
  }
}

function stopCurrentStream() {
  if (videoStream) {
    videoStream.getTracks().forEach((track) => {
      try {
        track.stop();
      } catch (e) {
        console.warn("Error stopping track:", e);
      }
    });
    videoStream = null;
  }
  if (diagStreamActive) {
    diagStreamActive.textContent = "No (Stopped)";
    diagStreamActive.style.color = "#94a3b8";
  }
}

function mapCameraError(err) {
  const errName = err.name || "Error";
  let explanation = "";

  switch (errName) {
    case "NotAllowedError":
    case "PermissionDeniedError":
      explanation = "Camera permission was denied. Enable camera access in browser/site settings.";
      break;
    case "NotFoundError":
    case "DevicesNotFoundError":
      explanation = "No camera was detected on this device.";
      break;
    case "NotReadableError":
    case "TrackStartError":
      explanation = "The camera may already be in use by another application.";
      break;
    case "OverconstrainedError":
    case "ConstraintNotSatisfiedError":
      explanation = "Requested camera constraints are not supported by this sensor.";
      break;
    case "SecurityError":
      explanation = "The browser does not trust the current HTTPS context.";
      break;
    case "AbortError":
      explanation = "Camera hardware operation was aborted.";
      break;
    case "TypeError":
      explanation = "Invalid constraints or camera invocation type error.";
      break;
    default:
      explanation = err.message || "Unknown camera access error.";
  }

  return { errName, message: err.message || "", explanation, stack: err.stack || "" };
}

async function startCamera() {
  stopCurrentStream();

  btnStartCamera.disabled = true;
  btnStartCamera.textContent = "جاري فتح الكاميرا...";

  if (diagErrName) diagErrName.textContent = "None";
  if (diagErrMsg) diagErrMsg.textContent = "None";
  if (debugErrorDetail) debugErrorDetail.style.display = "none";

  if (!window.isSecureContext && location.hostname !== "localhost" && location.hostname !== "127.0.0.1") {
    const msg = "Live camera unavailable because this page is not running in a trusted secure context.";
    alert(msg);
    if (diagErrName) diagErrName.textContent = "InsecureContextError";
    if (diagErrMsg) diagErrMsg.textContent = msg;
    btnStartCamera.disabled = false;
    btnStartCamera.textContent = "📹 فتح الكاميرا (Open Camera)";
    return;
  }

  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    const msg = "navigator.mediaDevices.getUserMedia is not supported or blocked in this browser.";
    alert(msg);
    if (diagErrName) diagErrName.textContent = "NotSupportedError";
    if (diagErrMsg) diagErrMsg.textContent = msg;
    btnStartCamera.disabled = false;
    btnStartCamera.textContent = "📹 فتح الكاميرا (Open Camera)";
    return;
  }

  const constraintOptions = [
    {
      audio: false,
      video: {
        facingMode: { ideal: "environment" },
        width: { ideal: 1280 },
        height: { ideal: 720 },
      },
    },
    {
      audio: false,
      video: {
        facingMode: "environment",
      },
    },
    {
      audio: false,
      video: true,
    },
  ];

  let acquiredStream = null;
  let lastError = null;

  for (const constraints of constraintOptions) {
    try {
      console.log("Requesting getUserMedia with:", JSON.stringify(constraints));
      acquiredStream = await navigator.mediaDevices.getUserMedia(constraints);
      if (acquiredStream) break;
    } catch (err) {
      lastError = err;
      console.warn("Constraint attempt failed:", constraints, err);
      if (err.name === "NotAllowedError" || err.name === "PermissionDeniedError" || err.name === "SecurityError") {
        break;
      }
    }
  }

  if (!acquiredStream) {
    const mapped = mapCameraError(lastError || new Error("Unknown camera initialization error"));
    console.error("Camera getUserMedia failed:", lastError);
    if (diagErrName) diagErrName.textContent = mapped.errName;
    if (diagErrMsg) diagErrMsg.textContent = `${mapped.explanation} (${mapped.message})`;
    if (debugErrorDetail) {
      debugErrorDetail.textContent = `Name: ${mapped.errName}\nMessage: ${mapped.message}\nExplanation: ${mapped.explanation}\nStack: ${mapped.stack}`;
      debugErrorDetail.style.display = "block";
    }

    alert(`فشل فتح الكاميرا (Camera Error):\n${mapped.errName}: ${mapped.explanation}\n\nيمكنك استخدام خيار رفع صورة للتجربة.`);

    btnStartCamera.disabled = false;
    btnStartCamera.textContent = "📹 فتح الكاميرا (Open Camera)";
    return;
  }

  videoStream = acquiredStream;

  videoElem.autoplay = true;
  videoElem.playsInline = true;
  videoElem.setAttribute("playsinline", "true");
  videoElem.setAttribute("webkit-playsinline", "true");
  videoElem.muted = true;
  videoElem.srcObject = videoStream;

  await new Promise((resolve) => {
    if (videoElem.readyState >= 1) {
      resolve();
    } else {
      videoElem.onloadedmetadata = () => resolve();
    }
  });

  try {
    await videoElem.play();
  } catch (playErr) {
    console.warn("video.play() rejected:", playErr);
  }

  const vWidth = videoElem.videoWidth;
  const vHeight = videoElem.videoHeight;
  console.log(`Video ready: ${vWidth}x${vHeight}`);

  const videoTrack = videoStream.getVideoTracks()[0];
  const settings = videoTrack && videoTrack.getSettings ? videoTrack.getSettings() : {};

  if (diagStreamActive) {
    diagStreamActive.textContent = "Yes (Active)";
    diagStreamActive.style.color = "#4ade80";
  }
  if (diagVideoDim) {
    diagVideoDim.textContent = `${vWidth} x ${vHeight}`;
  }
  if (diagFacing) {
    diagFacing.textContent = settings.facingMode || "default";
  }

  videoElem.style.display = "block";
  capturedImgElem.style.display = "none";
  guideFrame.style.display = "block";

  btnStartCamera.textContent = "📹 الكاميرا تعمل (Active)";
  btnStartCamera.disabled = true;
  btnCapture.disabled = false;
  if (btnCaptureRecognize) btnCaptureRecognize.disabled = false;
  btnRetake.disabled = true;
  btnRecognize.disabled = true;
  setPipelineState("STREAM_READY", "#4ade80");
}

/**
 * Capture frame from live video feed into a Blob
 * @returns {Promise<Blob>} The cropped JPEG blob
 */
async function captureFrameToBlob() {
  if (!videoStream) {
    throw new Error("No active camera stream available.");
  }

  const vWidth = videoElem.videoWidth;
  const vHeight = videoElem.videoHeight;
  if (!vWidth || !vHeight) {
    throw new Error("Video dimensions not ready yet.");
  }

  setPipelineState("CAPTURING_FRAME", "#f59e0b");
  console.log(`[CAPTURE_CLICKED] Video buffer: ${vWidth}x${vHeight}`);

  const videoRect = videoElem.getBoundingClientRect();
  const guideRect = guideFrame.getBoundingClientRect();

  const containerAspect = videoRect.width / videoRect.height;
  const videoAspect = vWidth / vHeight;

  let renderWidth, renderHeight, offsetX, offsetY;
  if (videoAspect > containerAspect) {
    renderHeight = videoRect.height;
    renderWidth = renderHeight * videoAspect;
    offsetX = (videoRect.width - renderWidth) / 2;
    offsetY = 0;
  } else {
    renderWidth = videoRect.width;
    renderHeight = renderWidth / videoAspect;
    offsetX = 0;
    offsetY = (videoRect.height - renderHeight) / 2;
  }

  const guideDisplayLeft = guideRect.left - videoRect.left - offsetX;
  const guideDisplayTop = guideRect.top - videoRect.top - offsetY;
  const guideDisplayWidth = guideRect.width;
  const guideDisplayHeight = guideRect.height;

  const scale = vWidth / renderWidth;
  let sourceX = guideDisplayLeft * scale;
  let sourceY = guideDisplayTop * scale;
  let sourceW = guideDisplayWidth * scale;
  let sourceH = guideDisplayHeight * scale;

  // Add 15% safety padding around the box so strokes near edges are never clipped
  const padX = sourceW * 0.15;
  const padY = sourceH * 0.15;
  sourceX = Math.max(0, sourceX - padX);
  sourceY = Math.max(0, sourceY - padY);
  sourceW = Math.min(vWidth - sourceX, sourceW + 2 * padX);
  sourceH = Math.min(vHeight - sourceY, sourceH + 2 * padY);

  const cropDim = Math.min(sourceW, sourceH);
  canvasElem.width = Math.round(cropDim);
  canvasElem.height = Math.round(cropDim);

  console.log(`[CROP_PARAMS] sourceX: ${Math.round(sourceX)}, sourceY: ${Math.round(sourceY)}, cropDim: ${Math.round(cropDim)}, canvas: ${canvasElem.width}x${canvasElem.height}`);

  const ctx = canvasElem.getContext("2d");
  ctx.drawImage(
    videoElem,
    sourceX, sourceY, cropDim, cropDim,
    0, 0, Math.round(cropDim), Math.round(cropDim)
  );

  return new Promise((resolve, reject) => {
    canvasElem.toBlob((blob) => {
      if (!blob || blob.size === 0) {
        setPipelineState("CAPTURE_BLOB_EMPTY", "#ef4444");
        reject(new Error("Captured frame resulted in empty blob."));
        return;
      }

      console.log(`[BLOB_CREATED] size: ${blob.size} bytes, type: ${blob.type}`);
      if (diagBlobInfo) {
        diagBlobInfo.textContent = `${blob.size} bytes (${blob.type})`;
      }

      capturedBlob = blob;
      const blobUrl = URL.createObjectURL(blob);
      capturedImgElem.src = blobUrl;
      capturedImgElem.style.display = "block";
      videoElem.style.display = "none";
      guideFrame.style.display = "none";

      if (rawCaptureBox && rawCapturedThumb) {
        rawCapturedThumb.src = blobUrl;
        rawCaptureBox.style.display = "block";
      }

      btnCapture.disabled = true;
      if (btnCaptureRecognize) btnCaptureRecognize.disabled = true;
      btnRecognize.disabled = false;
      btnRetake.disabled = false;
      setPipelineState("CAPTURED", "#4ade80");

      resolve(blob);
    }, "image/jpeg", 0.95);
  });
}

async function handleManualCapture() {
  try {
    await captureFrameToBlob();
  } catch (err) {
    alert("Capture failed: " + err.message);
    console.error(err);
  }
}

function retakeFrame() {
  capturedBlob = null;
  capturedImgElem.style.display = "none";
  if (rawCaptureBox) {
    rawCaptureBox.style.display = "none";
  }

  if (videoStream && videoStream.active) {
    videoElem.style.display = "block";
    guideFrame.style.display = "block";
    btnCapture.disabled = false;
    if (btnCaptureRecognize) btnCaptureRecognize.disabled = false;
    setPipelineState("STREAM_ACTIVE", "#4ade80");
  } else {
    btnStartCamera.disabled = false;
    btnStartCamera.textContent = "📹 فتح الكاميرا (Open Camera)";
    btnCapture.disabled = true;
    if (btnCaptureRecognize) btnCaptureRecognize.disabled = true;
    setPipelineState("READY", "#38bdf8");
  }
  btnRecognize.disabled = true;
  btnRetake.disabled = true;
  resetReaction();
}

async function performRecognition() {
  if (!capturedBlob) {
    alert("No captured image found. Please capture an image first.");
    return;
  }

  setPipelineState("SENDING_TO_API", "#f59e0b");
  btnRecognize.disabled = true;
  btnRecognize.textContent = "جاري التحليل...";
  if (btnCaptureRecognize) btnCaptureRecognize.disabled = true;

  try {
    const res = await api.recognizeLetter(capturedBlob);
    setPipelineState("RESPONSE_RECEIVED", "#4ade80");
    displayResult(res);

    // Auto-scroll on mobile screens to results card
    if (window.innerWidth <= 860 && reactionBanner) {
      reactionBanner.scrollIntoView({ behavior: "smooth", block: "nearest" });
    }
  } catch (err) {
    setPipelineState("API_ERROR", "#ef4444");
    if (diagErrName) diagErrName.textContent = err.name || "APIError";
    if (diagErrMsg) diagErrMsg.textContent = err.message;
    if (debugErrorDetail) {
      debugErrorDetail.textContent = `API Error:\n${err.message}\nStack: ${err.stack || "N/A"}`;
      debugErrorDetail.style.display = "block";
    }
    alert("Recognition failed: " + err.message);
    console.error("[RECOGNITION ERROR]", err);
  } finally {
    btnRecognize.disabled = false;
    btnRecognize.textContent = "✨ تعرّف (Recognize)";
    if (btnCaptureRecognize && videoStream && videoStream.active) {
      btnCaptureRecognize.disabled = false;
    }
  }
}

/**
 * Primary Unified Interaction: One-Button Capture & Recognize
 */
async function handleCaptureAndRecognize() {
  try {
    const blob = await captureFrameToBlob();
    if (blob) {
      await performRecognition();
    }
  } catch (err) {
    alert("Capture & Recognize failed: " + err.message);
    console.error(err);
  }
}

function displayResult(result) {
  const pred = result.prediction;
  const expected = ARABIC_LETTERS[currentTargetIndex].char;

  const detectedLetter = pred.letter || (result.top_candidates && result.top_candidates[0] ? result.top_candidates[0].letter : null);
  const confidenceVal = pred.confidence !== undefined ? pred.confidence : 0;

  statDetectedChar.textContent = detectedLetter || "—";
  statConfidence.textContent = `${(confidenceVal * 100).toFixed(1)}%`;
  statLatency.textContent = `${result.processing_time_ms} ms`;
  statAccepted.textContent = pred.accepted ? "مقبول (Accepted)" : "مرفوض (Retry)";
  if (statRequestId) {
    statRequestId.textContent = result.request_id || "—";
  }

  // Update candidate list
  candidateList.innerHTML = "";
  if (result.top_candidates && result.top_candidates.length > 0) {
    result.top_candidates.forEach((c) => {
      const li = document.createElement("li");
      li.className = "candidate-item";
      li.innerHTML = `
        <span class="char">${c.letter} (${c.name_en || ""})</span>
        <span>${(c.confidence * 100).toFixed(1)}%</span>
      `;
      candidateList.appendChild(li);
    });
  } else {
    candidateList.innerHTML = '<li class="candidate-item" style="color: var(--text-muted); justify-content: center;">لا توجد تنبؤات مطابقة</li>';
  }

  // Mini-Game Reaction logic
  reactionBanner.className = "reaction-banner";
  if (!pred.accepted) {
    reactionBanner.classList.add("warning");
    reactionIcon.textContent = "⚠️";
    reactionTitle.textContent = "يرجى المحاولة مرة أخرى (Please capture again)";
    reactionDesc.textContent = result.message || "الكتابة غير واضحة أو الإضاءة خافتة. اضغط إعادة وثبّت الورقة جيداً.";
  } else if (detectedLetter === expected) {
    reactionBanner.classList.add("success");
    reactionIcon.textContent = "✅ ⭐ 🎉";
    reactionTitle.textContent = `صحيح وممتاز! كتبتَ الحرف (${detectedLetter}) بدقة!`;
    reactionDesc.textContent = "تفاعل اللعبة: أحسنت! حصلتَ على نقطة كاملة!";
  } else {
    reactionBanner.classList.add("error");
    reactionIcon.textContent = "❌";
    reactionTitle.textContent = `المطلوب (${expected}) — والمكتشف (${detectedLetter})`;
    reactionDesc.textContent = "حاول كتابة الحرف المطلوب مجدداً داخل المربع المخصص.";
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
  if (statRequestId) {
    statRequestId.textContent = "-";
  }
  candidateList.innerHTML = '<li class="candidate-item" style="color: var(--text-muted); justify-content: center;">لا توجد تنبؤات بعد</li>';
}

document.addEventListener("visibilitychange", () => {
  if (document.hidden) {
    console.log("App moved to background.");
  } else {
    console.log("App returned to foreground.");
    if (videoStream && !videoStream.active) {
      stopCurrentStream();
      btnStartCamera.disabled = false;
      btnStartCamera.textContent = "📹 إعادة فتح الكاميرا (Restart Camera)";
      btnCapture.disabled = true;
      if (btnCaptureRecognize) btnCaptureRecognize.disabled = true;
    }
  }
});

// Event Listeners
btnStartCamera.addEventListener("click", startCamera);
btnCapture.addEventListener("click", handleManualCapture);
if (btnCaptureRecognize) btnCaptureRecognize.addEventListener("click", handleCaptureAndRecognize);
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
    const blobUrl = URL.createObjectURL(file);
    capturedImgElem.src = blobUrl;
    capturedImgElem.style.display = "block";
    videoElem.style.display = "none";
    guideFrame.style.display = "none";

    if (rawCaptureBox && rawCapturedThumb) {
      rawCapturedThumb.src = blobUrl;
      rawCaptureBox.style.display = "block";
    }

    if (diagBlobInfo) {
      diagBlobInfo.textContent = `${file.size} bytes (${file.type})`;
    }

    btnCapture.disabled = true;
    if (btnCaptureRecognize) btnCaptureRecognize.disabled = true;
    btnRecognize.disabled = false;
    btnRetake.disabled = false;
    setPipelineState("FILE_UPLOADED", "#4ade80");
    resetReaction();
  }
});

// Initialize on page load
renderEnvironmentDiagnostics();
updateTargetPrompt();
resetReaction();
