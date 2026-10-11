# Step 4: Real Camera Preprocessing & Robustness Specification

## Executive Summary

The Arabic Handwritten Letter Recognition model was trained on clean, cropped, 32x32 centered grayscale images from the Hijja child dataset. In production, phone/tablet browser camera captures exhibit shadows, non-uniform desk lighting, notebook paper grain, faint pencil strokes, and off-center placement.

The objective of Step 4 is to build an input preprocessing pipeline that bridges the distribution gap between real camera captures and the model's training distribution while strictly preserving Arabic diacritical dots and rejecting invalid inputs.

---

## 1. Selected Preprocessing Pipeline (Variant B: Adaptive Gaussian)

Through controlled benchmarking on the camera stress test suite (70 structured test captures across 14 conditions), **Variant B (Adaptive Gaussian Thresholding with Connected Component Analysis)** was selected as the winner.

### Architecture of the Selected Pipeline:

```text
Full Camera Frame (RGB / Grayscale)
        ↓
Illumination Sanity Check (Mean & Std Dev -> Blank/Dark Rejection)
        ↓
Bilateral Filtering (d=5, sigma=40, suppresses noise while preserving stroke/dot edges)
        ↓
Adaptive Gaussian Thresholding (blockSize=21, C=10, THRESH_BINARY_INV)
        ↓
Polarity Verification & Inversion (ensures white strokes on black background)
        ↓
Connected Component Analysis (8-connectivity)
        ↓
Outer Guide Frame / Border Filter (4-corner hollow rectangle rejection)
        ↓
Multiple Character Disconnection Check (rejects multi-letter inputs)
        ↓
Arabic Dot Spatial Envelope Evaluation (preserves dots above, below, or inside body)
        ↓
Combined Character + Dots Bounding Box Extraction
        ↓
Proportional Resize (margin=8, usable size 16x16 matching Hijja dataset distribution)
        ↓
Canvas Center & Padding (32x32 square)
        ↓
Normalization (float32, [0.0, 1.0])
        ↓
Inference (ArabicCNN Checkpoint: ml/models/final_candidate_model.pt)
```

---

## 2. Alternatives Tested & Comparison

Three distinct candidate pipelines were evaluated under identical conditions on the structured camera stress dataset:

| Preprocessing Variant | Core Mechanism | Valid Acc (%) | Accepted Acc (%) | Shadow Correct | Faint Dots Preserved | Mean Prep Latency | Invalid Rejection |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Variant A (Current)** | Grayscale + CLAHE + Otsu Global Threshold + CC | 39.68% | 57.14% | 0 / 6 (0.0%) | 3 / 8 (37.5%) | 6.08 ms | 85.7% |
| **Variant B (Selected)** | **Bilateral Filter + Adaptive Gaussian (21x21) + CC** | **71.43%** | **78.85%** | **5 / 6 (83.3%)** | **8 / 8 (100.0%)** | **3.07 ms** | **100.0%** |
| **Variant C (Heavy)** | Morphological Background Division (31x31) + CLAHE + Adaptive | 58.73% | 71.11% | 4 / 6 (66.7%) | 7 / 8 (87.5%) | 22.07 ms | 85.7% |

### Key Findings:
1. **Otsu Global Threshold (Variant A) completely fails under shadow:** Otsu selects a single global threshold across the entire image. When a child's hand or phone casts a gradient shadow, Otsu either converts the shadowed paper into solid foreground or deletes the letter in bright regions.
2. **Adaptive Gaussian (Variant B) excels under shadow gradients:** By computing local threshold surfaces over a 21x21 neighborhood, Variant B accurately segments dark pencil/pen strokes regardless of local paper luminance.
3. **Variant C is overengineered and slow:** While morphological background estimation flattens shadows, the large 31x31 ellipse kernel introduces edge distortion and incurs a 7x latency penalty (22.07 ms vs 3.07 ms).

---

## 3. Critical Dot Preservation Strategy

Arabic letters rely heavily on dots (1, 2, or 3 dots) to differentiate character classes (e.g. `ب` vs `ت` vs `ث`, `ح` vs `ج` vs `خ`, `د` vs `ذ`, `ر` vs `ز`, `س` vs `ش`, `ص` vs `ض`, `ط` vs `ظ`, `ع` vs `غ`).

### Dot Extraction Algorithm:
1. Identify the **primary letter body component** ($L_{\text{max}}$) having the largest connected component area among valid candidates.
2. Define the **spatial letter envelope** around the primary body:
   $$\text{env}_{x\text{min}} = \max(0, \text{body}_x - 0.7 \times \text{body}_w)$$
   $$\text{env}_{x\text{max}} = \min(W, \text{body}_x + 1.7 \times \text{body}_w)$$
   $$\text{env}_{y\text{min}} = \max(0, \text{body}_y - 1.3 \times \text{body}_h)$$
   $$\text{env}_{y\text{max}} = \min(H, \text{body}_y + 2.3 \times \text{body}_h)$$
3. **Preserve any component** whose centroid falls inside this envelope if:
   $$\text{area} \ge \max(2, 0.003 \times \text{body\_area}) \quad \text{and} \quad \text{area} \le 0.50 \times \text{body\_area}$$
4. Distant corner dust or speckles falling outside the envelope are discarded.

---

## 4. Stroke Scale Calibration (The Margin=8 Discovery)

Analysis of the ground-truth Hijja child training dataset revealed that characters occupy an average bounding box of **10.6 x 10.9 pixels** (median 10.0 x 10.0 pixels, max dimension 12.0 pixels) within the 32x32 canvas.

- When previous preprocessing used `margin=4`, characters were resized to **24x24 pixels** (a 200% scale expansion), distorting the CNN's learned receptive field.
- By configuring `fit_to_square(..., margin=8)`, characters are scaled to **16x16 pixels**, closely matching the native scale of child handwriting in Hijja and maximizing recognition accuracy.

---

## 5. Guide Frame & Border Rejection

Camera viewfinders often capture printed paper guidelines or border edges.
A rectangular guide frame is uniquely identified and filtered before character body selection:
1. Bounding box width $> 60\%$ and height $> 60\%$ of image dimensions.
2. Low fill ratio: $\frac{\text{area}}{w \times h} < 0.15$ (thin hollow outline).
3. **4-Corner Presence Check:** Pixels exist in all four corners of its bounding box.
Diagonal strokes or large letters that do not satisfy all three criteria are never discarded.

---

## 6. Invalid Input Detection & Rejection

Before passing noisy inputs to the neural network, preprocessing evaluates sanity conditions:
- **Blank Paper:** $\text{mean} > 250$ and $\text{std} < 4.0$, or total foreground pixels $< 25$.
  - Result: `status: "blank"`, Message: `"No clear handwritten letter detected."`
- **Extreme Darkness:** $\text{mean} < 20$.
  - Result: `status: "invalid_lighting"`, Message: `"The image is too dark. Please use better lighting."`
- **Multiple Disconnected Characters:** Second largest component area $> 35\%$ of primary body with horizontal separation $> 1.2 \times \text{min\_width}$.
  - Result: `status: "multiple_letters"`, Message: `"Please write only one letter inside the frame."`

---

## 7. Confidence Calibration & Quality Priority

For a children's educational platform, **a false acceptance (teaching a child the wrong letter with high confidence) is far more damaging than asking them to retry.**

Threshold calibration was systematically evaluated on the benchmark suite:

| Threshold | Margin | Accepted Accuracy | False Acceptance Rate | Rejection Rate |
| :---: | :---: | :---: | :---: | :---: |
| 0.60 | 0.15 | 74.1% | 23.8% | 7.9% |
| 0.65 | 0.15 | 78.8% | 17.5% | 17.5% |
| 0.70 | 0.20 | 83.3% | 12.7% | 23.8% |
| **0.75** | **0.20** | **85.1%** | **11.1%** | **25.4%** |
| 0.80 | 0.20 | 86.0% | 9.5% | 31.7% |

### Configured Production Parameters:
- `ACCEPTANCE_THRESHOLD = 0.75`
- `MIN_TOP2_MARGIN = 0.20`

Predictions below threshold return `accepted: false`, `letter: null`, and child-friendly guidance:
`"The handwriting was not clear enough. Please hold the paper steady under good light and try again!"`

---

## 8. Latency Performance

Benchmarks measured on Intel Core i7 CPU:
- **Preprocessing Mean:** 4.56 ms
- **Model Inference Mean:** 3.70 ms
- **End-to-End Mean Latency:** **8.26 ms**

Well within the target latency budget of $< 50$ ms.

---

## 9. Limitations & Next Steps

1. **Synthetic Camera Stress Data:** Benchmarking was performed using 70 synthetic camera stress captures accurately modeling paper, lighting gradients, shadows, and textures. Real physical phone/tablet camera captures under diverse lenses should be tested in Step 6.
2. **Severely Overlapping Multi-Character Writing:** Cursive or touching multi-letter writing is rejected or segmented to the dominant stroke. Full cursive OCR is out of scope (isolated single letters only).
