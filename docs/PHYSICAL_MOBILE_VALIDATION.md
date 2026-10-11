# Physical Mobile Camera Validation Report

## 1. Executive Summary & Mandatory Evaluation Notice

> [!WARNING]
> **Mandatory Evaluation Scope Notice:**
> In accordance with Section 17 of the Project Master Execution Plan:
> **Physical mobile validation could not be completed because no real device captures were provided.**
>
> The software pipeline (FastAPI inference engine, live browser camera interface, dynamic guide-frame coordinate projection, and ruled-notebook line suppression) is **fully configured and ready to receive real mobile samples**, but **production readiness cannot yet be claimed** until physical testing with real mobile hardware is conducted.

---

## 2. Tested & Verified Environment

### Software & Hardware Platform
- **Test Server:** Windows 10/11 x86_64, Python 3.13.5
- **Inference Runtime:** PyTorch 2.x CPU runtime
- **Inference Checkpoint:** `ml/models/final_candidate_model.pt` (ArabicCNN, 2,346,076 parameters, 8.97 MB)
- **API Server:** FastAPI 0.115+ on Uvicorn (`--host 0.0.0.0 --port 8000`)
- **Automated Tests:** 21 automated unit/integration tests passing (100% pass rate)

### Physical Device Status
- **Devices Physically Connected/Captured:** None provided in local repository.
- **Physical Camera Photographs Provided:** 0 real camera photographs.
- **Controlled Synthetic Stress Benchmark:** 70 samples evaluated in Step 4.

---

## 3. Directory Structure Prepared for Physical Ingestion

The repository directory structure has been prepared to receive physical device captures:

```text
tests/real_mobile_samples/
├── clean_white_paper/   # Condition A: White paper + dark pen + normal light
├── lined_paper/         # Condition B: Ruled notebook paper with blue/red lines
├── pencil/              # Condition C: Low-contrast graphite pencil strokes
├── pen/                 # Dark ink ballpoint/gel pen
├── shadow/              # Condition D: Harsh ambient or device cast shadows
├── rotated/             # Condition E: Skewed and rotated paper (up to 20°)
├── low_light/           # Dim ambient room illumination
├── small_letter/        # Letter drawn small within the guide frame
├── large_letter/        # Letter drawn large touching frame boundaries
├── faint_dots/          # Faint/weak Arabic diacritical dots
└── no_letter/           # Invalid inputs: blank sheet, random marks, fingers
```

---

## 4. Camera Permission & HTTPS Requirements

### Mobile Browser Behavior & Security Requirements
When testing the live demo (`/demo/index.html`) from a smartphone or tablet:

1. **Localhost vs. Network IP:**
   - Modern browsers (Chrome, Safari, Edge) enforce `navigator.mediaDevices.getUserMedia` as a **Secure Context API**.
   - `getUserMedia` works over plain HTTP **only** on `http://localhost` and `http://127.0.0.1`.
   - When connecting from a remote mobile device via local network Wi-Fi (e.g., `http://192.168.1.X:8000`), **mobile browsers may block camera access** unless:
     - HTTPS is enabled (e.g., self-signed SSL certificate, `mkcert`, or reverse proxy), or
     - Chrome Flag `chrome://flags/#unsafely-treat-insecure-origin-as-secure` is explicitly enabled on the Android device for `http://192.168.1.X:8000`.

2. **Mobile Safari (iOS / iPadOS):**
   - Strictly requires HTTPS for camera access over non-localhost LAN IPs.
   - Requires explicit user interaction (touch event) to prompt for camera permissions.
   - Video element requires `playsinline` and `muted` attributes to display inline video without forcing fullscreen (both are configured in `demo/index.html`).

3. **Fallback Upload Mechanism:**
   - For environments where HTTPS is not yet provisioned on the local network, the demo UI includes a **"📁 رفع صورة للتجربة (Upload Image)"** button. On mobile devices, tapping this button opens the native phone camera directly via file input (`accept="image/*"`).

---

## 5. Privacy & Debug Configuration

In compliance with children's privacy requirements (Section 15):
- The `DEBUG_CAMERA_IMAGES` environment variable governs whether captured camera frames and intermediate preprocessing stages are saved to disk.
- **Development Mode (`DEBUG_CAMERA_IMAGES=true`):** Saves diagnostic stages to `debug/real_camera/<request_id>/` for debugging.
- **Production Mode (`DEBUG_CAMERA_IMAGES=false`):** Purely in-memory processing. Zero disk persistence of raw images or child handwriting.
- Configured in `.env.example` and read by `backend/app/api/recognition.py`.

---

## 6. Real-Device Evaluation Checklist (For Step 5 Readiness)

To transition from developmental completion to full production readiness, the following physical test protocol must be executed once physical mobile hardware is available:

- [ ] Capture 15 priority dotted letter groups across physical Android phone (Chrome)
- [ ] Capture 15 priority dotted letter groups across physical iPhone / iPad (Safari)
- [ ] Record exact device model, OS version, browser version, and camera resolution
- [ ] Test Condition A (white paper + pen), Condition B (lined notebook paper), Condition C (pencil), Condition D (cast shadow), Condition E (slight rotation)
- [ ] Test no-letter samples (blank lined paper, blank white paper, finger intrusion, stray scribble)
- [ ] Evaluate real false acceptance rate and tune `ACCEPTANCE_THRESHOLD` (currently 0.75) and `MIN_TOP2_MARGIN` (currently 0.20) if needed
- [ ] Log granular results to `ml/evaluation/real_mobile_validation.csv` and `ml/evaluation/real_mobile_validation.json`
