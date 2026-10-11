# Mobile Phone / Tablet Physical Camera Testing Setup Guide

This guide describes how to connect a real smartphone or tablet to your development computer on the same Wi-Fi network, access the live camera recognition demo over a **Secure Context (HTTPS)**, and collect real physical-camera validation samples.

---

## 1. Network Architecture & Startup Options

The FastAPI backend serves both the API endpoints (`/api/v1/recognize-letter`) and the static web demo (`/demo/index.html`).

```text
Laptop / PC (Running Backend & Demo)
  │ IP: 192.168.1.34
  │
  ├── Option A (Recommended): HTTPS with local self-signed SSL on Port 8443
  └── Option B: HTTP on Port 8000 (with native mobile camera photo upload fallback)
        ▲
        │ Wi-Fi Local Area Network (Same Router / Subnet)
        │
Smartphone / Tablet (Android Chrome or iOS Safari)
  │ Navigates to: https://192.168.1.34:8443/demo/index.html
  │ Grants live camera access
```

---

## 2. Option A: Secure Context (HTTPS) for Live Mobile Camera (Recommended)

Modern mobile browsers (Chrome on Android, Safari on iOS) **require a Secure Context (HTTPS)** to allow access to `navigator.mediaDevices.getUserMedia()` when accessing any IP other than `localhost` / `127.0.0.1`.

### Step 1: Ensure SSL Certificates Exist
Local self-signed development certificates have been generated with Subject Alternative Names (SAN) for `127.0.0.1`, `localhost`, and your LAN IP (`192.168.1.34`).
If your IP address ever changes, regenerate them anytime by running:
```bash
python scripts/generate_ssl_cert.py
```
*(Creates `certs/cert.pem` and `certs/key.pem`)*

### Step 2: Start the HTTPS Backend & Demo
To enable full debug tracing for physical validation, run with `DEBUG_CAMERA_IMAGES=true`:

**On Windows (PowerShell):**
```powershell
$env:DEBUG_CAMERA_IMAGES="true"
.venv\Scripts\uvicorn.exe --app-dir backend app.main:app --host 0.0.0.0 --port 8443 --ssl-keyfile certs/key.pem --ssl-certfile certs/cert.pem --reload
```

**On Linux / macOS (Bash):**
```bash
DEBUG_CAMERA_IMAGES=true .venv/bin/uvicorn --app-dir backend app.main:app --host 0.0.0.0 --port 8443 --ssl-keyfile certs/key.pem --ssl-certfile certs/cert.pem --reload
```

### Step 3: Open on Mobile Browser
1. Connect your phone or tablet to the **same Wi-Fi network** as your PC.
2. In your phone's browser (Chrome or Safari), navigate to:
   ```text
   https://192.168.1.34:8443/demo/index.html
   ```
3. **Accepting Self-Signed Certificate Warning:**
   - **Chrome on Android:** You will see a *"Your connection is not private"* warning. Tap **Advanced** (متقدم) → **Proceed to 192.168.1.34 (unsafe)** (متابعة).
   - **Safari on iOS:** Tap **Show Details** → **visit this website** → Confirm with Passcode/TouchID.
4. Tap **"📹 فتح الكاميرا (Open Camera)"** and tap **Allow** on the camera permission prompt.

> [!WARNING]
> **Mobile OS Secure Context Constraint on Self-Signed Certificates:**
> On iOS Safari (and certain Android Chrome builds), even if the user clicks "Proceed anyway" to bypass the self-signed SSL warning, the browser engine marks the origin as `Not Secure` and **strictly disables `navigator.mediaDevices.getUserMedia()`** (`window.isSecureContext = false` or `NotAllowedError`).
> If your device blocks the live camera on the self-signed IP address, use **Section 2.1 (Trusted HTTPS Tunnel)** below.

---

## 2.1. Option A.2: Trusted HTTPS Development Tunnel (Localtunnel / Cloudflared)

If mobile Safari or Chrome blocks `getUserMedia()` on the self-signed IP, you can use a temporary trusted HTTPS tunnel. This provides a valid Public CA certificate (Let's Encrypt / Cloudflare) recognized natively by iOS and Android without any certificate warnings.

### Method 1: Using `localtunnel` (No installation or sign-up needed via npx)
1. Start the HTTP backend on your computer:
   ```powershell
   $env:DEBUG_CAMERA_IMAGES="true"
   .venv\Scripts\uvicorn.exe --app-dir backend app.main:app --host 0.0.0.0 --port 8000 --reload
   ```
2. In another terminal, run:
   ```bash
   npx localtunnel --port 8000
   ```
   *Output will display:* `your url is: https://<random-subdomain>.localtunnel.me`
3. On your phone, open:
   ```text
   https://<random-subdomain>.localtunnel.me/demo/index.html
   ```
   *(If prompted by Localtunnel for a password, enter your external IP displayed on the screen).*

### Method 2: Using `cloudflared` (Cloudflare Quick Tunnel)
If you have `cloudflared` installed:
```bash
cloudflared tunnel --url http://localhost:8000
```
Open the generated `https://<random-subdomain>.trycloudflare.com/demo/index.html` on your mobile device. Live camera access will immediately work under a 100% trusted secure context.

---

## 3. Option B: Standard HTTP on Port 8000 (File Upload Fallback)

If you prefer not to accept self-signed certificate warnings, you can run over standard HTTP:

```powershell
$env:DEBUG_CAMERA_IMAGES="true"
.venv\Scripts\uvicorn.exe --app-dir backend app.main:app --host 0.0.0.0 --port 8000 --reload
```

On your phone, open:
```text
http://192.168.1.34:8000/demo/index.html
```

> [!NOTE]
> In HTTP mode on a non-localhost IP, modern mobile browsers will block the WebRTC live camera stream. However, you can tap **"📁 رفع صورة للتجربة (Upload Image)"** which triggers the phone's native camera shutter to take and upload a real photograph directly.

---

## 4. Windows Firewall & Network Troubleshooting

If your phone cannot load `http://192.168.1.34:8000` or `https://192.168.1.34:8443`:

1. **Verify Local IP Address:**
   Run `ipconfig` in PowerShell to confirm your current IPv4 address (e.g. `192.168.1.34`).
2. **Allow Ports in Windows Defender Firewall:**
   If Windows blocks inbound connections, run PowerShell as Administrator:
   ```powershell
   New-NetFirewallRule -DisplayName "Arabic Handwriting AI Demo" -Direction Inbound -LocalPort 8000,8443 -Protocol TCP -Action Allow
   ```
3. **Wi-Fi AP Isolation:**
   Ensure your Wi-Fi router does not have "Client Isolation" or "Guest Mode" enabled, which prevents local devices from communicating with each other.

---

## 5. Physical Validation Step-by-Step Procedure

To collect the real-device test dataset and validate the pipeline:

### Step 1: Open Target Challenge
1. Open the demo on your mobile device:
   ```text
   https://192.168.1.34:8443/demo/index.html
   ```
2. The demo prompts the target letter in the **Priority Testing Sequence**:
   `ب`, `ت`, `ث`, `ج`, `ح`, `خ`, `د`, `ذ`, `ر`, `ز`, `س`, `ش`, `ص`, `ض`, `ط`, `ظ`, `ع`, `غ`, `ف`, `ق`, `ن`, `ي`, `ا`, `ك`, `ل`, `م`, `ه`, `و`
3. Tap **"🎲 حرف آخر (Next Letter)"** to advance through the letters.

### Step 2: Write on Paper & Capture
Capture the handwritten letter under the required evaluation conditions:
- **Condition A (`white_paper_pen`):** White unlined paper, dark ink pen, balanced lighting.
- **Condition B (`lined_paper_pen`):** Standard ruled notebook paper (tests line suppression).
- **Condition C (`white_paper_pencil`):** Graphite pencil (tests low-contrast stroke detection).
- **Condition D (`mild_shadow`):** Cast shadow across the paper or phone body shadow.
- **Condition E (`slight_rotation`):** Paper slightly rotated ($\le 20^\circ$).
- **Condition F (`faint_dots`):** Letters with small or light dots (ب, ت, ث, ج, خ, ش, etc.).

### Step 3: Inspect Prediction & Record Result
After tapping **"✨ تعرّف على الحرف (Recognize)"**, observe the technical prediction details:
- **Expected Letter** (الحرف المطلوب)
- **Detected Letter** (الحرف المكتشف)
- **Confidence Score** (نسبة الثقة)
- **Accepted / Rejected** (الحالة)
- **Trace ID** (`stat-request-id`, e.g. `req_1791728900_e4b1a8f2`)

Record the test trial into [ml/evaluation/real_mobile_validation.csv](file:///d:/Freelancing-Projects/Writing-Detection/ml/evaluation/real_mobile_validation.csv):
```csv
request_id,device,browser,condition,true_letter,predicted_letter,confidence,top2_margin,accepted,correct,failure_stage,notes
req_1791728900_e4b1a8f2,Samsung Galaxy S22,Chrome,white_paper_pen,ب,ب,0.992,0.910,True,True,none,Centered crisp capture
```

### Step 4: Invalid Input Rejection Verification
Capture and verify that the system correctly rejects invalid inputs:
1. **Blank white paper:** Must return rejection (`accepted: false`).
2. **Blank lined notebook paper:** Must return rejection (`accepted: false`).
3. **Random scribble / stray mark:** Must return rejection.
4. **Two letters side-by-side:** Must trigger multi-character rejection (`multiple_letters`).
5. **Finger partially in frame:** Must be rejected.

---

## 6. Request ID Debugging Workflow

If an attempt fails or produces an unexpected result, use its **Trace ID** to inspect the internal CV pipeline:

1. Copy the `request_id` displayed on the mobile screen (e.g. `req_1791728900_e4b1a8f2`).
2. On your computer, open:
   ```text
   debug/real_camera/<request_id>/
   ```
3. Inspect the 6 diagnostic artifacts:
   - `original_received.jpg` — Did the camera crop capture the full letter without clipping dots?
   - `grayscale.png` — Was the lighting too dark or washed out?
   - `threshold.png` — Did adaptive thresholding isolate the strokes cleanly?
   - `cleaned.png` — Did the line suppression remove notebook lines while preserving the letter basin?
   - `components.png` — Are green bounding boxes surrounding the main stroke and all associated dots?
   - `final_32x32.png` — Is the letter proportionally centered with margin=8?
4. Identify the failure stage from the taxonomy:
   - `frontend_crop`
   - `lighting`
   - `threshold`
   - `line_removal`
   - `connected_components`
   - `dot_loss`
   - `centering`
   - `model_confusion`
   - `confidence_logic`
