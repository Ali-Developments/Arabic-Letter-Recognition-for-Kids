# Arabic Handwritten Letter Recognition — Master Execution Plan

## 0. Project Rule

This project is an **AI module only**.

The client already owns and operates the educational platform and game.

Our responsibility is limited to:

```text
Camera Input
→ Image Preprocessing
→ Arabic Handwritten Letter Recognition
→ Confidence Decision
→ API Response
→ Existing Client Game
```

Do NOT build a new platform.
Do NOT build a full game.
Do NOT build LMS features, accounts, dashboards, payments, or unrelated frontend features.

The target task is:

> Recognize ONE isolated handwritten Arabic letter written by a child on paper.

Target classes:

```text
ا ب ت ث ج ح خ د ذ ر ز س ش ص ض ط ظ ع غ ف ق ك ل م ن ه و ي
```

Primary users:
Arabic-speaking children.

Primary dataset:
Hijja.

Secondary dataset:
AHCD.

---

# GLOBAL EXECUTION RULE

Antigravity must execute this project **one step at a time**.

For every step:

1. Read this file before starting.
2. Read the latest previous completion report.
3. Execute ONLY the requested step.
4. Run all required checks/tests for that step.
5. Create/update the required documentation.
6. Print the required completion report.
7. STOP.
8. Do NOT automatically continue to the next step.

The next step starts only when explicitly instructed.

If a requirement cannot be completed:
- state the exact reason,
- do not fabricate results,
- do not silently skip the requirement.

---

# CURRENT PROJECT STATUS

Already completed:

- Project directory initialized.
- Hijja downloaded and inspected.
- AHCD downloaded and inspected.
- 28-class Arabic mapping created.
- Hijja isolated child samples extracted.
- Train / validation / test split created.
- Preprocessing pipeline implemented.
- Arabic-dot preservation tests implemented.
- FastAPI skeleton implemented.
- Camera demo skeleton implemented.
- Baseline ArabicCNN architecture exists.
- Unit/integration tests passed.

Current next step:

> STEP 2 — Baseline Model Training & Evaluation

---

# STEP 1 — Dataset & Architecture Readiness

## Goal

Prepare the project, datasets, mappings, preprocessing rules, and architecture.

## Inputs

- `ARABIC_HANDWRITING_AI_PROJECT_BRIEF.md`
- Hijja dataset
- AHCD dataset

## Tasks

- Initialize project structure.
- Download Hijja.
- Download AHCD.
- Inspect dataset structures.
- Extract Hijja isolated forms.
- Exclude Hamza.
- Map both datasets to the canonical 28 Arabic classes.
- Create train / validation / test split.
- Prefer writer-aware split if metadata allows.
- Create sample grids.
- Validate preprocessing.
- Preserve Arabic dots.
- Prepare API and demo architecture.

## Required Deliverables

```text
README.md
DATASET_REPORT.md
docs/DATASETS.md
docs/ARCHITECTURE.md
docs/API_SPEC.md
docs/EXPERIMENTS.md

scripts/prepare_hijja.py
scripts/prepare_ahcd.py
scripts/inspect_datasets.py

backend/app/ml/labels.py
backend/app/ml/preprocessing.py
backend/app/ml/confidence.py
```

## Required Validation

- 28 classes exist.
- No incorrect label mapping.
- Hamza excluded.
- Sample grid contains all 28 classes.
- Dot-preservation tests pass.
- Dataset counts reported honestly.

## Completion Report

```text
STEP 1 COMPLETION REPORT

DATASETS:
...

FINAL CLASS MAPPING:
...

SPLITS:
...

FILES CREATED:
...

TEST RESULTS:
...

KNOWN ISSUES:
...

READY FOR STEP 2:
Yes / No
```

## Stop Condition

STOP after dataset and architecture readiness.

Do not train a model.

---

# STEP 2 — Baseline Model Training & Evaluation

## Goal

Train the first real Arabic handwritten-character classifier and produce trustworthy evaluation metrics.

This step creates the **first usable trained model**.

## Primary Training Dataset

Use:

```text
Hijja isolated child handwriting
```

Train split:
Use the prepared Hijja training split.

Validation:
Use the prepared Hijja validation split.

Final baseline test:
Use Hijja child test split only.

Do NOT use test data during training or hyperparameter tuning.

---

## Baseline Architecture

Use the existing lightweight custom `ArabicCNN`.

If necessary, clean or improve its implementation before training, but do NOT replace it with MobileNet or ResNet in this step.

Recommended characteristics:

- Input: 1-channel grayscale.
- Input size consistent with current preprocessing.
- 28 output classes.
- Convolution blocks.
- Batch normalization.
- ReLU.
- Max pooling where appropriate.
- Dropout.
- Final linear classifier.

---

## Training Requirements

Use PyTorch.

Set deterministic seeds where practical.

Track:

- training loss,
- validation loss,
- training accuracy,
- validation accuracy,
- learning rate.

Use:

- CrossEntropyLoss.
- Adam or AdamW.
- learning-rate scheduler.
- early stopping.
- best-model checkpointing.

Do not overwrite the best model with a worse final epoch.

Save:

```text
ml/models/baseline_arabic_cnn_best.pt
```

Also save training metadata.

Example:

```text
ml/experiments/baseline_cnn/
```

with:

```text
config.json
metrics.json
training_history.csv
```

---

## Data Augmentation

Use only realistic augmentations.

Allowed examples:

- small rotations,
- small translation,
- slight scale changes,
- mild brightness/contrast changes,
- slight blur/noise.

Do NOT use:

- horizontal flipping,
- vertical flipping,
- extreme rotations,
- transforms that change Arabic letter identity.

---

## Evaluation Requirements

After training, evaluate on the untouched Hijja test set.

Required metrics:

- overall accuracy,
- macro precision,
- macro recall,
- macro F1,
- per-class precision,
- per-class recall,
- per-class F1,
- confusion matrix.

Pay special attention to:

```text
ب ت ث
ج ح خ
د ذ
ر ز
س ش
ص ض
ط ظ
ع غ
ف ق
```

Generate:

```text
ml/evaluation/baseline_confusion_matrix.png
ml/evaluation/baseline_per_class_metrics.csv
ml/evaluation/baseline_misclassified_samples.png
ml/evaluation/baseline_report.json
```

---

## Misclassification Analysis

Create a short analysis identifying:

- most-confused class pairs,
- dotted-letter errors,
- weak-stroke errors,
- likely preprocessing errors,
- ambiguous handwriting examples.

Update:

```text
docs/EXPERIMENTS.md
```

with baseline results.

---

## Sanity Inference Test

After saving the best model:

1. Load the model from disk.
2. Run inference on several test samples.
3. Verify that returned class IDs map correctly to Arabic letters.
4. Verify probability distribution sums correctly.
5. Verify low-confidence logic still works.
6. Verify model can be loaded by `backend/app/ml/predictor.py`.

---

## Step 2 Success Criteria

Step 2 is complete only if:

- training completed successfully,
- best checkpoint saved,
- real test-set metrics generated,
- confusion matrix generated,
- per-class metrics generated,
- sample predictions generated,
- predictor loads the trained weights,
- no label-index mismatch exists.

Do NOT claim the model is client-ready based only on training accuracy.

---

## Step 2 Completion Report

Print:

```text
==================================================
STEP 2 COMPLETION REPORT
==================================================

MODEL:
Architecture:
Checkpoint:
Input size:
Classes:

TRAINING:
Epochs completed:
Best epoch:
Best validation accuracy:
Best validation loss:

TEST RESULTS:
Accuracy:
Macro Precision:
Macro Recall:
Macro F1:

MOST CONFUSED CLASSES:
1.
2.
3.
4.
5.

DOTTED LETTER PERFORMANCE:
...

FILES CREATED:
...

MODEL LOAD TEST:
Passed / Failed

API PREDICTOR USING TRAINED WEIGHTS:
Yes / No

READY FOR REAL CAMERA TESTING:
Yes / No

KNOWN LIMITATIONS:
...

RECOMMENDED NEXT STEP:
Step 3 — Improved Model Comparison
==================================================
```

## Stop Condition

STOP.

Do NOT train MobileNetV3.
Do NOT train ResNet18.
Do NOT start real camera benchmarking.

---

# STEP 3 — Improved Model Comparison

## Goal

Determine whether a lightweight pretrained architecture improves child handwriting recognition.

## Models

Compare baseline ArabicCNN against ONE or BOTH:

```text
MobileNetV3-Small
ResNet18
```

Adapt first layer for grayscale if needed, or replicate grayscale to 3 channels consistently.

Output:
28 classes.

---

## Experiments

Run at least:

### Experiment A
Hijja only.

### Experiment B
Hijja + AHCD training support.

Important:
Final evaluation must still include a child-only Hijja test score.

Do not allow AHCD adult handwriting to dominate the objective.

---

## Required Comparison

Compare:

- test accuracy,
- macro F1,
- dotted-letter performance,
- confusion matrix,
- model size,
- inference latency,
- CPU memory,
- loading time.

Create:

```text
ml/evaluation/model_comparison.csv
docs/MODEL_SELECTION.md
```

Select ONE final model.

Explain why.

The winner should balance:

- child handwriting accuracy,
- robustness,
- latency,
- model size,
- CPU deployment suitability.

---

## Step 3 Completion Report

```text
STEP 3 COMPLETION REPORT

MODELS COMPARED:
...

BASELINE RESULT:
...

MOBILENET/RESNET RESULT:
...

CHILD-ONLY TEST PERFORMANCE:
...

SELECTED FINAL MODEL:
...

WHY SELECTED:
...

MODEL SIZE:
...

CPU LATENCY:
...

READY FOR STEP 4:
Yes / No
```

## Stop Condition

STOP before camera-domain optimization.

---

# STEP 4 — Real Camera Preprocessing & Robustness

## Goal

Make dataset-trained recognition work on actual phone/tablet camera images.

This is one of the most important project stages.

## Test Conditions

Collect or create real samples representing:

- white paper,
- black pen,
- blue pen,
- pencil,
- weak pencil strokes,
- strong shadows,
- uneven lighting,
- tilted paper,
- rotated letter,
- near camera,
- far camera,
- small character,
- large character,
- clutter outside guide box.

---

## Preprocessing Pipeline

Evaluate:

- crop guide region,
- grayscale,
- CLAHE / illumination correction,
- thresholding,
- adaptive threshold,
- Otsu,
- connected-component analysis,
- dot preservation,
- bounding box extraction,
- aspect-ratio preservation,
- centering,
- normalization.

Do NOT delete Arabic dots.

---

## Camera Benchmark

Create:

```text
tests/camera_samples/
```

and benchmark script:

```text
scripts/evaluate_camera_samples.py
```

Report:

- camera sample accuracy,
- rejection rate,
- accepted-correct rate,
- accepted-wrong rate,
- preprocessing failures,
- latency.

---

## Confidence Calibration

Calibrate:

```text
ACCEPTANCE_THRESHOLD
```

Do not choose threshold arbitrarily.

Use validation/camera results.

Goal:
Reduce confident wrong predictions.

---

## Step 4 Completion Report

```text
STEP 4 COMPLETION REPORT

CAMERA SAMPLES:
...

CAMERA ACCURACY:
...

ACCEPTED CORRECT:
...

ACCEPTED WRONG:
...

REJECTION RATE:
...

SELECTED CONFIDENCE THRESHOLD:
...

PREPROCESSING CHANGES:
...

DOT PRESERVATION:
Passed / Failed

READY FOR API INTEGRATION:
Yes / No
```

## Stop Condition

STOP before production API integration.

---

# STEP 5 — Production Inference API

## Goal

Connect the trained model to FastAPI.

## Required Endpoints

```text
GET /health
GET /api/v1/model-info
POST /api/v1/recognize-letter
```

## Recognition Response

Example:

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

## Requirements

- Load model once at startup.
- CPU-compatible.
- Process image in memory.
- Validate content type.
- Limit image size.
- Handle blank images.
- Handle invalid images.
- No raw-image logging.
- Configurable confidence threshold.
- CORS configurable.

---

## API Tests

Test:

- valid image,
- invalid image,
- unsupported file,
- blank page,
- low-confidence image,
- model unavailable,
- health endpoint,
- model info endpoint.

---

## Step 5 Completion Report

```text
STEP 5 COMPLETION REPORT

MODEL LOADED:
...

API ENDPOINTS:
...

TEST RESULTS:
...

AVERAGE INFERENCE TIME:
...

ERROR HANDLING:
...

READY FOR BROWSER DEMO:
Yes / No
```

## Stop Condition

STOP before browser UX completion.

---

# STEP 6 — Browser Camera Demo

## Goal

Create a small client-facing demo proving the AI module works.

This is NOT the client's real game.

## Flow

```text
Open Camera
→ Position paper inside guide
→ Capture
→ Preview
→ Recognize
→ Display result
→ Demo game reaction
```

## Requirements

Use:

```javascript
navigator.mediaDevices.getUserMedia()
```

Provide:

- camera preview,
- framing guide,
- capture button,
- retake button,
- recognize button,
- loading state,
- detected letter,
- confidence,
- accepted/rejected state.

---

## Mini Game Reaction

Example prompt:

```text
Write the letter ب
```

If AI returns `ب`:

```text
✅ Great!
You wrote ب
```

If AI returns another character:

```text
Try again
Expected: ب
Detected: ت
```

If rejected:

```text
I couldn't read it clearly.
Please capture again.
```

Do not build a complete game.

---

## Browser Testing

Test where possible:

- desktop Chrome,
- Android Chrome,
- iPhone Safari,
- iPad Safari.

Document HTTPS/camera-permission requirements.

---

## Step 6 Completion Report

```text
STEP 6 COMPLETION REPORT

CAMERA ACCESS:
...

CAPTURE:
...

API CONNECTION:
...

RECOGNITION:
...

GAME REACTION DEMO:
...

BROWSERS TESTED:
...

KNOWN ISSUES:
...

READY FOR CLIENT DEMO:
Yes / No
```

## Stop Condition

STOP before deployment packaging.

---

# STEP 7 — End-to-End Validation

## Goal

Test the complete system:

```text
Paper
→ Camera
→ Preprocessing
→ Model
→ API
→ Browser
→ Game Reaction
```

## Required Test Scenarios

At minimum:

- correct clean letter,
- weak pencil,
- shadow,
- tilted paper,
- wrong letter,
- blank page,
- multiple letters,
- letter outside frame,
- low confidence,
- repeated attempts.

Create:

```text
docs/END_TO_END_TEST_REPORT.md
```

Track:

- successful recognition,
- incorrect recognition,
- rejection,
- latency,
- UX issue.

---

## Step 7 Completion Report

```text
STEP 7 COMPLETION REPORT

END-TO-END TESTS:
...

PASS RATE:
...

FAILURES:
...

AVERAGE LATENCY:
...

CRITICAL ISSUES:
...

READY FOR DEPLOYMENT:
Yes / No
```

## Stop Condition

STOP before deployment.

---

# STEP 8 — Deployment Packaging

## Goal

Prepare the AI module for integration/deployment.

## Deliverables

- Dockerfile.
- `.env.example`.
- production requirements.
- CORS configuration.
- model path configuration.
- confidence configuration.
- startup instructions.
- API documentation.
- health check.
- deployment guide.

Create:

```text
docs/DEPLOYMENT.md
docs/CLIENT_INTEGRATION.md
```

---

## Client Integration Example

```javascript
const formData = new FormData();
formData.append("image", imageBlob);

const response = await fetch(
  `${AI_API_URL}/api/v1/recognize-letter`,
  {
    method: "POST",
    body: formData
  }
);

const result = await response.json();

if (result.prediction.accepted) {
  game.handleArabicLetter(result.prediction.letter);
}
```

---

## Step 8 Completion Report

```text
STEP 8 COMPLETION REPORT

DOCKER:
...

CONFIGURATION:
...

API DOCUMENTATION:
...

CLIENT INTEGRATION DOC:
...

DEPLOYMENT TEST:
...

FINAL TECHNICAL STATUS:
...
```

## Stop Condition

STOP.

---

# STEP 9 — Final Client Demo Package

## Goal

Prepare a clean demo package for presenting the feature to the client.

## Required Demo Story

```text
1. Open browser demo.
2. Game asks for a specific Arabic letter.
3. Child writes it on paper.
4. Camera captures it.
5. AI recognizes it.
6. Confidence is displayed.
7. Demo reacts to correct/incorrect answer.
```

## Client-Facing Explanation

Explain simply:

```text
The educational platform already exists.
This AI service adds real handwritten-letter interaction.

The child's real handwriting becomes an input to the digital game.
```

Do not expose technical complexity unnecessarily in the client UI.

---

# FINAL PROJECT ACCEPTANCE CRITERIA

The project is technically complete only when:

- a trained model exists,
- child handwriting test results are documented,
- real camera input is tested,
- Arabic dots remain preserved,
- low-confidence predictions are rejected,
- API serves the trained model,
- browser camera works,
- demo game reacts,
- images are not stored by default,
- integration documentation exists.

---

# MASTER RULE

Never use phrases such as:

```text
"done"
"production ready"
"client ready"
```

unless the required tests for that milestone actually passed.

Never fabricate:

- accuracy,
- latency,
- dataset counts,
- browser compatibility,
- evaluation metrics.

Every step must end with a report and STOP.
