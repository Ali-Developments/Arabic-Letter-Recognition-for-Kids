# Experiments Log — Arabic Handwriting Recognition

## Experiment Matrix

| Exp ID | Architecture | Training Dataset | Validation | Test Set | Input Size | Target Children Accuracy | Status | Test Accuracy | Macro F1 | Dotted Acc |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **EXP-01** | Custom ArabicCNN (Baseline) | Hijja Isolated Train (8,632) | Hijja Val (1,840) | Hijja Test (1,879) | 32x32 | > 88.0% | **Completed (Selected)** | **90.15%** | **90.10%** | **87.69%** |
| **EXP-02** | MobileNetV3-Small | Hijja Isolated Train (8,632) | Hijja Val (1,840) | Hijja Test (1,879) | 32x32 | > 92.0% | **Completed** | **76.32%** | **76.25%** | **70.11%** |
| **EXP-03** | MobileNetV3-Small | Hijja (70%) + AHCD (30%) Mixed | Hijja Val (1,840) | Hijja Test (1,879) | 32x32 | Comparison vs EXP-01 | **Completed** | **74.83%** | **74.73%** | **69.91%** |
| **EXP-04** | ResNet18-Light | Hijja Isolated Train | Hijja Val | Hijja Test | 32x32 | > 92.0% | Not Needed (32x32 Downsample Flaw) | - | - | - |

---

## EXP-01: Baseline ArabicCNN on Hijja Child-Handwriting

### 1. Configuration & Setup
- **Architecture**: 4-Block Convolutional Network (`ArabicCNN`)
  - Conv Block 1: `Conv2d(1, 32, 3)` + BatchNorm + ReLU + Conv + BN + ReLU + MaxPool(2) + Dropout(0.2)
  - Conv Block 2: `Conv2d(32, 64, 3)` + BatchNorm + ReLU + Conv + BN + ReLU + MaxPool(2) + Dropout(0.2)
  - Conv Block 3: `Conv2d(64, 128, 3)` + BatchNorm + ReLU + MaxPool(2) + Dropout(0.3)
  - Conv Block 4: `Conv2d(128, 256, 3)` + BatchNorm + ReLU + AdaptiveAvgPool2d(2, 2) + Dropout(0.3)
  - Classifier: `Linear(1024, 256)` + ReLU + Dropout(0.4) + `Linear(256, 28)`
  - Parameter Count: **2,346,076**
- **Data**: Untouched Hijja isolated splits:
  - Training: 8,632 samples
  - Validation: 1,840 samples
  - Test: 1,879 samples
- **Hyperparameters**:
  - Optimizer: `AdamW(lr=1e-3, weight_decay=1e-4)`
  - Scheduler: `ReduceLROnPlateau(mode='max', factor=0.5, patience=2)`
  - Loss: `CrossEntropyLoss`
  - Batch Size: 64
  - Max Epochs: 25
  - Best Checkpoint Criterion: Validation Accuracy
- **Augmentation**:
  - Rotation: `[-10 deg, +10 deg]`
  - Translation: `[-3 px, +3 px]`
  - Scaling: `[0.9, 1.1]`
  - Gaussian Noise: `std=0.03`
  - Horizontal/Vertical Flips: **Strictly Disabled** (to preserve Arabic letter identity)

### 2. Training Dynamics
- **Epochs Completed**: 25
- **Best Epoch**: 24
- **Best Validation Accuracy**: **90.49%** (Loss: 0.3370)
- **Training Progression**:
  - Epoch 1: Train Acc 47.98%, Val Acc 72.88%
  - Epoch 5: Train Acc 80.20%, Val Acc 85.05%
  - Epoch 10: Train Acc 86.04%, Val Acc 87.55%
  - Epoch 20: Train Acc 91.07%, Val Acc 90.00%
  - Epoch 24: Train Acc 92.42%, Val Acc 90.49% (Best Checkpoint saved)

### 3. Test Evaluation (Untouched Hijja Test Set — 1,879 samples)
- **Accuracy**: **90.15%** (1,694 / 1,879 correct)
- **Macro Precision**: **90.41%**
- **Macro Recall**: **90.13%**
- **Macro F1-Score**: **90.10%**

### 4. Top Confusions Analysis
The most frequent misclassifications reflect specific handwriting challenges common among young children (ages 7–12):
1. **ت (taa) vs ن (noon)** — 10 instances:
   - Young children frequently write the body of `ت` as a rounded bowl identical to `ن`, and merge the two dots into a single dash or dot.
2. **ذ (dhaal) vs ز (zaay)** — 7 instances:
   - Children struggle with slant differentiation; a short slanted `ذ` with an upper dot is visually indistinguishable from a steep `ز`.
3. **ص (saad) vs س (seen)** — 7 instances:
   - Children frequently omit the small defining "tooth" on `ص`, causing it to match the multi-tooth loop of `س`.
4. **ظ (dhah) vs ط (tah)** — 7 instances:
   - Weak or omitted upper dot on `ظ` converts the character body directly into `ط`.
5. **ذ (dhaal) vs د (daal)** — 6 instances:
   - Omitted or faint pencil dot on `ذ`.
6. **ت (taa) vs ك (kaaf)** — 6 instances:
   - Children drawing the inner hamza of isolated `ك` as two dots or drawing flat kaaf bases.
7. **ب / ت / ث vs ن / ي** — Dot count ambiguities:
   - Multi-dot merging (2 or 3 dots rendered as a blob or horizontal bar) causes systematic confusion in the `ب - ت - ث` family.

### 5. Dotted Letter Observations
- Letters with unique body structures (e.g., `ا`, `ل`, `م`, `هـ`) achieve >95% precision and recall.
- Characters whose sole differentiator is dot presence or count (`د`/`ذ`, `ط`/`ظ`, `ص`/`ض`, `ع`/`غ`, `ر`/`ز`) represent over 65% of all misclassifications.
- Dot preservation in preprocessing is critical: Otsu binarization must protect high-frequency peripheral dots from kernel erosion.

### 6. Limitations & Path Forward
- **Clean Dataset Bias**: All training and test images in EXP-01 are centered, pre-segmented 32x32 crops. Full camera frames introduce paper tilt, shadow gradients, background noise, and varied stroke thickness.
- **Model Checkpoint**: Saved to [ml/models/baseline_arabic_cnn_best.pt](file:///d:/Freelancing-Projects/Writing-Detection/ml/models/baseline_arabic_cnn_best.pt) and selected as final candidate.

---

## EXP-02: MobileNetV3-Small (Hijja Child Only)

### 1. Configuration & Setup
- **Architecture**: `MobileNetV3-Small` adapted for 1-channel grayscale 32x32 input.
  - ImageNet pretrained weights transfer with first-layer RGB filter channel averaging.
  - Replaced classifier head with 28-class linear output (1,081,628 parameters).
- **Training Data**: Hijja Isolated Train (8,632 samples).
- **Validation Data**: Hijja Isolated Val (1,840 samples).
- **Test Set**: Untouched Hijja Isolated Child Test (1,879 samples).
- **Hyperparameters**: AdamW (lr=1e-3, weight_decay=1e-4), ReduceLROnPlateau, 20 epochs.

### 2. Results
- **Best Validation Accuracy**: 77.34% (Epoch 18, Loss: 0.6874)
- **Hijja Test Accuracy**: **76.32%** (Macro F1: 76.25%)
- **Dotted Letter Accuracy**: **70.11%**
- **CPU Latency (mean)**: **4.55 ms** (Median: 4.16 ms, P95: 6.51 ms)
- **Accepted Wrong Rate**: **7.93%**
- **Top Confusions**: `ث` -> `ت` (15), `ص` -> `س` (15), `ن` -> `ت` (15), `ظ` -> `ط` (13), `خ` -> `غ` (11).

---

## EXP-03: MobileNetV3-Small (Mixed Balanced: 70% Hijja + 30% AHCD)

### 1. Configuration & Setup
- **Architecture**: `MobileNetV3-Small` (same as EXP-02).
- **Training Strategy**: Controlled Weighted Random Sampling:
  - 70% Hijja Child handwriting + 30% AHCD Adult handwriting support per batch.
  - Prevents AHCD (13,440 adult samples) from dominating Hijja (8,632 child samples).
- **Validation**: Hijja Isolated Val (1,840 child samples).
- **Test Set**: Untouched Hijja Isolated Child Test (1,879 child samples).

### 2. Results
- **Best Validation Accuracy**: 75.82% (Epoch 20, Loss: 0.7226)
- **Hijja Test Accuracy**: **74.83%** (Macro F1: 74.73%)
- **Dotted Letter Accuracy**: **69.91%**
- **CPU Latency (mean)**: **4.75 ms** (Median: 4.46 ms, P95: 6.84 ms)
- **Accepted Wrong Rate**: **8.46%**
- **Top Confusions**: `ث` -> `ت` (22), `ح` -> `ج` (16), `ط` -> `ظ` (12), `ن` -> `ت` (12), `خ` -> `غ` (11).

---

## Comparative Analysis & Architectural Conclusions

1. **Spatial Resolution Collapse:** Standard MobileNetV3 utilizes 5 spatial downsampling stages (stride 2), reducing 32x32 images to 1x1 before the classification head. This destroys the high-frequency diacritical dot structures vital for Arabic letters (`ث` vs `ت`, `ح` vs `ج`).
2. **Receptive Field Advantage:** Custom `ArabicCNN` maintains higher spatial feature resolution across its 4 blocks, achieving **90.15%** child accuracy vs **76.32%** for MobileNetV3 (+13.83% improvement) and **87.69%** dotted accuracy vs **70.11%** (+17.58% improvement).
4. **Final Model Choice:** `ArabicCNN` was selected as the final candidate model for Step 4 real camera testing and deployed at [`ml/models/final_candidate_model.pt`](file:///d:/Freelancing-Projects/Writing-Detection/ml/models/final_candidate_model.pt).

---

## EXP-04: Real Camera Preprocessing & Robustness (Step 4)

### 1. Preprocessing Variants Evaluated

Evaluated on structured camera stress test dataset (70 samples across 14 conditions: clean, shadow, low light, bright light, rotated, perspective, pencil, pen, small letter, large letter, faint dots, clutter, blank, invalid):

- **Variant A (Current Baseline):** Grayscale -> CLAHE -> Otsu Global Binarization -> Connected Components
  - Valid Accuracy: **39.68%**
  - Accepted Accuracy: **57.14%**
  - Shadow Condition Accuracy: **0 / 6 (0.0%)** (completely failed under non-uniform shadow)
  - Faint Dots Preserved: 3 / 8 (37.5%)
  - Mean Preprocessing Latency: 6.08 ms
  - Invalid Rejection: 85.7%

- **Variant B (Selected Winner — Adaptive Gaussian):** Grayscale -> Bilateral Filter -> Adaptive Gaussian (21x21, C=10) -> Frame Filtering -> CC Dot Preservation -> Margin=8 Scaling
  - Valid Accuracy: **71.43%**
  - Accepted Accuracy: **78.85%** (85.11% after calibration)
  - Shadow Condition Accuracy: **5 / 6 (83.3%)**
  - Faint Dots Preserved: **8 / 8 (100.0%)**
  - Mean Preprocessing Latency: **3.07 ms**
  - Invalid Rejection: **100.0%** (7 / 7)

- **Variant C (Morphological Background Field Division):** Grayscale -> 31x31 Ellipse Morphological Close Background Estimation -> Image Division -> CLAHE -> Adaptive Threshold
  - Valid Accuracy: **58.73%**
  - Accepted Accuracy: **71.11%**
  - Shadow Condition Accuracy: 4 / 6 (66.7%)
  - Faint Dots Preserved: 7 / 8 (87.5%)
  - Mean Preprocessing Latency: 22.07 ms (heavy morphological operations)
  - Invalid Rejection: 85.7%

### 2. Confidence Calibration

Evaluated across threshold grid to minimize false acceptances for children's learning:
- Prioritized: **Correct Accepted > Rejection/Retry > Confident Wrong**
- At `ACCEPTANCE_THRESHOLD = 0.75` and `MIN_TOP2_MARGIN = 0.20`:
  - Accepted Accuracy: **85.11%**
  - False Acceptance Rate: **11.11%**
  - Invalid Input Rejection: **100.0%**

### 3. Latency Metrics
- Preprocessing mean: **4.56 ms**
- Model inference mean: **3.70 ms**
- End-to-end latency: **8.26 ms**

---

## Step 4.5: Physical Mobile Camera Validation Status

- **Status:** Pipeline prepared & hardened; physical hardware evaluation pending real device captures.
- **Model Checkpoint:** `ml/models/final_candidate_model.pt` (ArabicCNN, 8.97 MB, strictly preserved).
- **Camera Preprocessing Enhancements:**
  - Browser guide-frame coordinate mapping with 15% safety padding implemented.
  - Multi-stage morphological and component-level ruled notebook line suppression active.
  - Edge-touching border margin removal active.
  - Arabic dot preservation confirmed 100% across canonical dotted letters.
  - Debug request tracing with unique `request_id` operational.
  - Configurable privacy debug logging via `DEBUG_CAMERA_IMAGES` environment variable.
- **Physical Samples Evaluated:** 0 real mobile captures provided in workspace.
- **Production Status:** Ready for client demo & local evaluation; physical device collection required prior to full production deployment claim.
- **Reference Document:** See [docs/PHYSICAL_MOBILE_VALIDATION.md](file:///d:/Freelancing-Projects/Writing-Detection/docs/PHYSICAL_MOBILE_VALIDATION.md) for full protocol and requirements.


