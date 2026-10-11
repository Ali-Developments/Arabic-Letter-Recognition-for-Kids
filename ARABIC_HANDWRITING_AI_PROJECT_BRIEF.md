# Arabic Handwritten Letter Recognition — AI Module Project Brief

## 1. Project Context

This project is **NOT** a full educational platform and **NOT** a full game.

The client already has an existing educational platform for children and an existing game.

Our responsibility is to build only the **AI handwriting-recognition module** that will later be integrated into the client's existing platform.

The required feature is:

1. A child writes one Arabic letter by hand on a white sheet of paper.
2. The child opens the camera from a mobile phone or iPad browser.
3. The browser captures the handwritten letter.
4. The AI system recognizes the handwritten Arabic character.
5. The AI returns the detected letter and confidence score.
6. The client's existing game receives the result and reacts to it.

Example:

Child writes:

`ب`

AI response:

```json
{
  "success": true,
  "prediction": {
    "letter": "ب",
    "class_id": 1,
    "confidence": 0.95,
    "accepted": true
  }
}
```

The educational game itself is **outside our scope**.

---

# 2. Scope

## What We Build

We build:

- Arabic handwritten letter recognition model.
- Image preprocessing pipeline.
- Camera capture demo.
- Backend inference API.
- Browser-based demonstration page.
- Integration contract/API for the client's existing platform.
- Model evaluation and confidence logic.

## What We Do NOT Build

Do NOT build:

- A new educational platform.
- A complete children's game.
- User accounts.
- Student dashboards.
- Teacher dashboards.
- Course management.
- Payment systems.
- LMS functionality.
- Game logic beyond a very small demo reaction.
- Full Arabic OCR for sentences or words.

This project recognizes **one isolated handwritten Arabic letter at a time**.

---

# 3. Main AI Task

Treat the problem primarily as:

> 28-class Arabic handwritten character image classification.

Target classes:

```text
ا ب ت ث ج ح خ د ذ ر ز س ش ص ض ط ظ ع غ ف ق ك ل م ن ه و ي
```

Do NOT start with a generic OCR engine such as Tesseract as the main recognition method.

The input is a single handwritten character captured by a camera.

A lightweight CNN-based classifier is preferred.

---

# 4. Recommended Dataset Strategy

## Primary Dataset — Hijja

Download and use **Hijja** as the primary dataset.

Repository:

https://github.com/israksu/Hijja2

Hijja is important because it was collected from Arabic-speaking school children aged approximately 7–12 years.

It contains tens of thousands of handwritten Arabic character samples from hundreds of children.

This makes it much more representative of the target users than adult handwriting datasets.

### Important Dataset Rule

The client asks the child to write an **isolated letter on paper**.

Therefore, when preparing Hijja:

- Focus primarily on isolated/disconnected character forms.
- Do not blindly mix initial, medial, final and isolated forms without documenting the mapping.
- Verify class labels carefully.
- Exclude Hamza from the main 28-class model unless a future requirement explicitly includes it.

Expected primary classes:

```text
28 Arabic letters
```

---

## Secondary Dataset — AHCD

Also obtain the Arabic Handwritten Characters Dataset (AHCD).

Use it as a secondary/support dataset.

AHCD contains adult handwriting and can help:

- increase variation,
- improve robustness,
- provide additional samples,
- compare child-only vs mixed training.

However:

**Hijja remains the priority because the target users are children.**

Run experiments with:

### Experiment A

```text
Train: Hijja
Validation: Hijja
Test: Hijja
```

### Experiment B

```text
Train: Hijja + AHCD
Validation: mixed
Test: Hijja child samples
```

The final decision must be based on accuracy on **child handwriting**, not only global accuracy.

---

# 5. Research Background

Relevant research has shown that child handwriting should be treated separately from adult handwriting because children's writing contains larger variations.

Useful references:

## Hijja Dataset

GitHub:

https://github.com/israksu/Hijja2

## Child Arabic Handwriting Recognition Research

Towards Accurate Children's Arabic Handwriting Recognition via Deep Learning:

https://www.mdpi.com/2076-3417/13/3/1692

Deep Learning-Based Child Handwritten Arabic Character Recognition and Handwriting Discrimination:

https://pmc.ncbi.nlm.nih.gov/articles/PMC10422599/

Handwritten Arabic Character Recognition for Children Writing Using CNN:

https://arxiv.org/abs/2211.02119

Research findings suggest that combining child and adult datasets can improve robustness, but evaluation must remain focused on child handwriting.

---

# 6. Recommended Model Development Strategy

Do NOT begin with a very large model.

Evaluate these architectures:

1. Small custom CNN.
2. MobileNetV3-Small.
3. ResNet18 or another lightweight residual CNN.

## Recommended Initial Baseline

Start with:

```text
Custom CNN
```

Purpose:

- verify the dataset pipeline,
- verify the labels,
- establish baseline accuracy,
- detect preprocessing problems.

Then compare against:

```text
MobileNetV3-Small
```

or

```text
ResNet18
```

The final model should prioritize:

- high accuracy,
- fast inference,
- low memory usage,
- CPU compatibility,
- easy API deployment.

---

# 7. Camera Input Is Different from Dataset Images

This is critical.

Training datasets contain relatively clean character images.

The real client scenario will contain:

- paper background,
- shadows,
- perspective distortion,
- uneven lighting,
- weak pencil strokes,
- pen strokes,
- rotation,
- different distances from the camera,
- large or small handwritten letters,
- noise around the paper,
- possible fingers or table background.

Therefore training accuracy alone is NOT enough.

A separate real-camera preprocessing pipeline must be developed.

---

# 8. Image Preprocessing Pipeline

Recommended pipeline:

```text
Camera Image
      ↓
Crop Guide Region
      ↓
Orientation Correction
      ↓
Perspective Correction if needed
      ↓
Grayscale
      ↓
Lighting Normalization
      ↓
Denoising
      ↓
Adaptive Threshold / Otsu
      ↓
Foreground Extraction
      ↓
Connected Component Analysis
      ↓
Letter Bounding Box
      ↓
Preserve Arabic Dots
      ↓
Aspect-Ratio Resize
      ↓
Padding
      ↓
Center Character
      ↓
Resize to Model Input
      ↓
Normalize
      ↓
Classifier
```

Possible model input sizes:

```text
32x32
48x48
64x64
```

Do not select the final resolution until experiments are performed.

---

# 9. Critical Arabic-Language Requirement — Preserve Dots

Arabic dots are part of the character identity.

Examples:

```text
ب / ت / ث

ج / ح / خ

د / ذ

ر / ز

س / ش

ص / ض

ط / ظ

ع / غ

ف / ق
```

Do NOT remove small connected components simply because they look like noise.

The preprocessing algorithm must distinguish between:

- noise,
- meaningful Arabic dots.

Connected components near the main character body should be preserved.

This requirement must have dedicated tests.

---

# 10. Data Augmentation

Use realistic augmentation during training.

Recommended augmentation:

- small rotation,
- translation,
- scale changes,
- brightness variation,
- contrast variation,
- mild blur,
- light Gaussian noise,
- slight perspective transformation,
- variable stroke intensity.

Avoid unrealistic augmentation that changes the character identity.

Do NOT use horizontal flipping.

Arabic letters must not be mirrored.

---

# 11. Confidence Handling

The classifier must NOT always force a prediction.

Use confidence-based acceptance.

Example:

```python
if confidence >= ACCEPTANCE_THRESHOLD:
    accepted = True
else:
    accepted = False
```

Example response:

```json
{
  "success": true,
  "prediction": {
    "letter": "ث",
    "confidence": 0.58,
    "accepted": false
  },
  "message": "Please capture the letter again."
}
```

The confidence threshold must be configurable.

Later, calibrate the model confidence using validation data.

---

# 12. Required Backend

Use FastAPI.

Required endpoints:

## Health

```http
GET /health
```

Example:

```json
{
  "status": "ok"
}
```

---

## Model Information

```http
GET /api/v1/model-info
```

Example:

```json
{
  "model": "arabic-character-classifier",
  "version": "0.1.0",
  "classes": 28
}
```

---

## Recognize Letter

```http
POST /api/v1/recognize-letter
```

Input:

```text
multipart/form-data
image=<captured image>
```

Output:

```json
{
  "success": true,
  "prediction": {
    "letter": "ب",
    "class_id": 1,
    "confidence": 0.947,
    "accepted": true
  },
  "processing_time_ms": 42
}
```

---

# 13. Required Client Demo

A small browser demo must be created for presentation to the client.

This demo is NOT the final educational platform.

It only proves that the AI feature works.

## Demo Flow

The demo page should contain:

### Step 1

Button:

```text
Open Camera
```

Use:

```javascript
navigator.mediaDevices.getUserMedia()
```

The feature should work from a modern phone/tablet browser where browser permissions allow camera access.

---

### Step 2

Show live camera preview.

Display a clear square/rectangle guide in the center.

Instruction:

```text
Place the handwritten letter inside the frame.
```

---

### Step 3

Button:

```text
Capture Letter
```

Capture only the guide region when practical.

---

### Step 4

Show captured image.

Buttons:

```text
Recognize
Retake
```

---

### Step 5

Send image to:

```text
POST /api/v1/recognize-letter
```

---

### Step 6

Show prediction.

Example UI:

```text
Detected Letter

        ب

Confidence: 95%
```

---

# 14. Mini Game-Reaction Demo

To demonstrate integration to the client, include a very small fake game interaction.

Do NOT build a real game.

Example:

Before recognition:

```text
Write the letter ب
```

The child writes `ب` on paper.

After AI recognition:

If correct:

```text
✅ Great!
You wrote ب
```

Show a simple animation, star or success state.

If incorrect:

```text
Try again.
Expected: ب
Detected: ت
```

If confidence is low:

```text
I could not read the letter clearly.
Please take another photo.
```

This small interaction proves that the client's game can receive and use the AI result.

---

# 15. Integration Contract with Client Platform

Keep the AI system independent from the client's current codebase.

The client's developers should only need to do something similar to:

```javascript
const formData = new FormData();
formData.append("image", imageBlob);

const response = await fetch(
    "https://AI-SERVICE/api/v1/recognize-letter",
    {
        method: "POST",
        body: formData
    }
);

const result = await response.json();

if (result.prediction.accepted) {
    game.handleArabicLetter(
        result.prediction.letter
    );
}
```

The AI service must not depend on the client's game engine.

---

# 16. Recommended Project Structure

```text
arabic-handwriting-ai/
│
├── README.md
├── PROJECT_BRIEF.md
│
├── backend/
│   ├── app/
│   │   ├── main.py
│   │   ├── api/
│   │   │   └── recognition.py
│   │   ├── ml/
│   │   │   ├── predictor.py
│   │   │   ├── preprocessing.py
│   │   │   ├── confidence.py
│   │   │   └── labels.py
│   │   └── schemas/
│   ├── tests/
│   └── requirements.txt
│
├── ml/
│   ├── data/
│   │   ├── raw/
│   │   ├── processed/
│   │   └── external/
│   │
│   ├── training/
│   ├── evaluation/
│   ├── experiments/
│   └── models/
│
├── demo/
│   ├── index.html
│   ├── camera.js
│   ├── api.js
│   └── styles.css
│
├── scripts/
│   ├── prepare_hijja.py
│   ├── prepare_ahcd.py
│   └── evaluate_camera_samples.py
│
├── docs/
│   ├── ARCHITECTURE.md
│   ├── DATASETS.md
│   ├── API_SPEC.md
│   └── EXPERIMENTS.md
│
└── tests/
```

---

# 17. Development Stages

Follow these stages.

Do NOT jump directly to the final system.

## Stage 1 — Research and Architecture

Deliver:

- architecture,
- requirements,
- dataset plan,
- model strategy,
- API specification.

---

## Stage 2 — Dataset Preparation

Download and inspect:

```text
Hijja
AHCD
```

Tasks:

- verify class mappings,
- extract isolated Hijja characters,
- remove unwanted classes,
- map labels to 28 Arabic characters,
- create train/validation/test splits,
- visualize random samples,
- detect corrupted samples,
- document class balance.

---

## Stage 3 — Baseline Model

Train a small CNN.

Deliver:

- training curves,
- validation accuracy,
- confusion matrix,
- per-class precision/recall/F1,
- misclassified samples.

---

## Stage 4 — Improved Model

Compare baseline against:

```text
MobileNetV3-Small
or
ResNet18
```

Evaluate specifically on Hijja child samples.

---

## Stage 5 — Camera Preprocessing

Create real camera test samples.

Test:

- white paper,
- shadows,
- pencil,
- pen,
- rotated page,
- different distances,
- different lighting.

The camera pipeline must preserve dots.

---

## Stage 6 — FastAPI Service

Implement production-style inference endpoint.

---

## Stage 7 — Browser Camera Demo

Implement camera capture and recognition.

---

## Stage 8 — Client Demo Interaction

Add only a tiny example reaction proving the AI can control game behavior.

---

## Stage 9 — Testing

Test:

- Chrome Android,
- iPhone Safari where available,
- iPad Safari where available,
- desktop Chrome.

Also test incorrect inputs such as:

- blank paper,
- multiple letters,
- extremely small character,
- extremely dark image,
- no paper,
- unsupported image.

---

## Stage 10 — Deployment Preparation

Prepare:

- Dockerfile,
- environment variables,
- model versioning,
- CORS configuration,
- health endpoint,
- deployment instructions.

---

# 18. Success Criteria

The prototype is considered successful when:

1. The browser can access a phone/tablet camera.
2. The child can capture one handwritten Arabic letter.
3. The backend preprocesses the camera image.
4. The AI returns one of the 28 Arabic letters.
5. A confidence score is returned.
6. Low-confidence predictions can be rejected.
7. The demo visually reacts to the prediction.
8. The API is independent and can be integrated into the client's existing platform.
9. Captured images are not permanently stored by default.
10. Recognition latency is reasonable for an interactive educational experience.

---

# 19. Privacy

Because the target users are children:

- Do not store camera images by default.
- Process images in memory.
- Do not perform face recognition.
- Do not collect biometric information.
- Avoid unnecessary personal data.
- Do not log raw images.
- Encourage the camera frame to capture the paper only.

---

# 20. Important Instruction to Antigravity

The client's platform already exists.

You are building:

```text
CAMERA
   ↓
IMAGE PROCESSING
   ↓
AI CHARACTER RECOGNITION
   ↓
API RESULT
```

You are NOT building:

```text
THE CLIENT'S PLATFORM
```

and you are NOT building:

```text
THE CLIENT'S COMPLETE GAME
```

The browser demo exists only to prove the AI module works and to demonstrate how the client's existing platform can integrate it.

Do not expand the scope unless explicitly instructed.

---

# 21. First Required Action

Before implementing the final system:

1. Read this document completely.
2. Inspect the Hijja dataset repository.
3. Download Hijja.
4. Identify and extract the isolated/disconnected Arabic letter samples needed for this use case.
5. Obtain AHCD as the secondary dataset.
6. Build a dataset inspection script.
7. Produce a class-distribution report.
8. Display samples of all 28 target classes.
9. Confirm the exact label mapping.
10. Create `DATASET_REPORT.md`.

Then stop and provide a completion report before moving to model training.

The report must include:

```text
- datasets downloaded
- exact number of usable images
- excluded classes/forms
- final 28-class mapping
- train/validation/test counts
- detected dataset issues
- screenshots/sample grids generated
- recommended next step
```

Do not train the production model until the dataset mapping is verified.

---

# Final Product Vision

The final technical component should behave like this:

```text
Child
  ↓
Writes Arabic Letter on Real Paper
  ↓
Phone / iPad Browser Camera
  ↓
Capture
  ↓
Preprocessing
  ↓
Arabic Handwriting AI Model
  ↓
Confidence Check
  ↓
FastAPI
  ↓
JSON Result
  ↓
Client's Existing Game
  ↓
Game Reacts
```

That is the complete scope of this AI project.
