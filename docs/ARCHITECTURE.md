# Architecture & System Design — Arabic Handwritten Letter AI

> **Reference**: [ARABIC_HANDWRITING_AI_PROJECT_BRIEF.md](file:///d:/Freelancing-Projects/Writing-Detection/ARABIC_HANDWRITING_AI_PROJECT_BRIEF.md)

---

## 1. High-Level System Architecture

The AI module is decoupled from the client's educational platform and game engine. It operates as an independent microservice accessed via HTTP REST endpoints.

```text
┌────────────────────────────────────────────────────────┐
│               Child Client (Browser)                   │
│  - WebRTC Camera Capture with Centered Framing Guide   │
│  - Guided UI (Target Letter, Audio Cue, Frame Box)     │
│  - Binary Blob Extracted from Guide Region             │
└───────────────────────────┬────────────────────────────┘
                            │ POST multipart/form-data
                            │ /api/v1/recognize-letter
                            ▼
┌────────────────────────────────────────────────────────┐
│                 FastAPI Backend Service                │
│                                                        │
│  ┌──────────────────────────────────────────────────┐  │
│  │ 1. Memory Ingestion & Validation                 │  │
│  │    - File format, size, dimension checks         │  │
│  │    - ZERO disk storage (Child Privacy Compliance)│  │
│  └──────────────────────────┬───────────────────────┘  │
│                             │                          │
│  ┌──────────────────────────▼───────────────────────┐  │
│  │ 2. Computer Vision Preprocessing Pipeline        │  │
│  │    - Grayscale conversion                        │  │
│  │    - Illumination normalization & CLAHE          │  │
│  │    - Adaptive binarization (Otsu / Sauvola)      │  │
│  │    - Connected Component Analysis (CCA)          │  │
│  │    - CRITICAL: Arabic Dot Preservation Guard     │  │
│  │    - Aspect-ratio preserving square padding      │  │
│  │    - Resizing to model input (32x32) & norm      │  │
│  └──────────────────────────┬───────────────────────┘  │
│                             │                          │
│  ┌──────────────────────────▼───────────────────────┐  │
│  │ 3. Deep Learning Classifier (PyTorch)            │  │
│  │    - Lightweight CNN (Baseline / MobileNetV3)    │  │
│  │    - 28-class Arabic character softmax output    │  │
│  │    - Top-K ranking                               │  │
│  └──────────────────────────┬───────────────────────┘  │
│                             │                          │
│  ┌──────────────────────────▼───────────────────────┐  │
│  │ 4. Confidence Engine & Acceptance Decision       │  │
│  │    - Confidence threshold evaluation             │  │
│  │    - Calibration & ambiguous sample rejection    │  │
│  │    - Informative feedback message formulation    │  │
│  └──────────────────────────────────────────────────┘  │
└───────────────────────────┬────────────────────────────┘
                            │ JSON Response
                            ▼
┌────────────────────────────────────────────────────────┐
│            Client's Existing Game Engine               │
│  - Checks `prediction.accepted`                        │
│  - Compares `prediction.letter` against expected       │
│  - Triggers game celebration or retry prompt           │
└────────────────────────────────────────────────────────┘
```

---

## 2. Preprocessing & Dot Preservation Rationale

In the Arabic writing system, 15 out of 28 letters differ solely based on the number and positioning of dots:
- Baa (`ب`), Taa (`ت`), Thaa (`ث`) share the same base stroke (حوض/صحن).
- Jeem (`ج`), Haa (`ح`), Khaa (`خ`) share the same base body.
- Daal (`د`), Dhaal (`ذ`) differ by one dot.
- Raa (`ر`), Zaay (`ز`) differ by one dot.
- Seen (`س`), Sheen (`ش`) differ by three dots.
- Saad (`ص`), Daad (`ض`) differ by one dot.
- Taa emphatic (`ط`), Dhaa emphatic (`ظ`) differ by one dot.
- Ain (`ع`), Ghain (`غ`) differ by one dot.
- Faa (`ف`), Qaaf (`ق`) differ by body curvature and dots (1 vs 2).

Standard noise reduction techniques (e.g., area-threshold component filtering) routinely delete isolated dots as salt-and-pepper noise.
Our preprocessing algorithm computes spatial proximities between components: any small component situated within the vertical/horizontal bounding envelope of the primary stroke is protected from noise filtering.

---

## 3. Privacy-by-Design Architecture

To adhere to COPPA and child safety regulations:
- Inbound camera frames are handled exclusively in RAM (`io.BytesIO`).
- No images are persisted to the filesystem, object stores, or logging streams.
- Prediction latency is optimized to < 50ms on modern standard CPUs.
