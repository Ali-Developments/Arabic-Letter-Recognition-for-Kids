# Model Selection Report — Step 3 Evaluation & Tradeoffs

## Executive Summary

To determine whether a modern lightweight pretrained vision model improves isolated Arabic character recognition on children's handwriting (ages 7–12), we empirically evaluated three architectures on the untouched Hijja child test split (1,879 samples):

1. **ArabicCNN (Baseline)** — Custom lightweight 4-block CNN with batch normalization and adaptive pooling.
2. **MobileNetV3-Small (Hijja)** — ImageNet-pretrained MobileNetV3-Small fine-tuned exclusively on child handwriting (8,632 samples).
3. **MobileNetV3-Small (Mixed)** — MobileNetV3-Small fine-tuned with 70/30 balanced weighted sampling of Hijja child handwriting and AHCD adult handwriting.

### Selected Final Model: **ArabicCNN (Baseline)**
**Checkpoint:** [`ml/models/final_candidate_model.pt`](file:///d:/Freelancing-Projects/Writing-Detection/ml/models/final_candidate_model.pt)

---

## Benchmark Matrix

All models evaluated under identical hardware conditions (Intel Core CPU, single thread, batch size = 1, 1,879 test evaluations).

| Model Architecture | Training Strategy | Hijja Child Test Acc | Macro F1 | Dotted Letter Acc | CPU Latency (Mean) | P95 Latency | Checkpoint Size | False Accept Rate | Decision |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **ArabicCNN (Baseline)** | Hijja Child Train (8,632) | **90.15%** | **90.10%** | **87.69%** | **1.10 ms** | **1.48 ms** | 8.97 MB | **4.68%** | **SELECTED WINNER** |
| **MobileNetV3-Small** | Hijja Child Train (8,632) | 76.32% | 76.25% | 70.11% | 4.55 ms | 6.51 ms | 4.25 MB | 7.93% | Rejected |
| **MobileNetV3-Small** | Mixed (70% Hijja + 30% AHCD) | 74.83% | 74.73% | 69.91% | 4.75 ms | 6.84 ms | 4.25 MB | 8.46% | Rejected |

---

## Architectural & Tradeoff Analysis

### 1. Child Handwriting Accuracy (+13.83% Advantage for ArabicCNN)
- **ArabicCNN:** Achieves **90.15%** test accuracy on the untouched Hijja child test set.
- **MobileNetV3 (Hijja):** Reaches only **76.32%** test accuracy.
- **Root Cause:** MobileNetV3 employs 5 consecutive spatial downsamplings (stride 2) designed for 224x224 ImageNet inputs. At a 32x32 character resolution, five downsamplings reduce the feature map to 1x1 before the classification head, erasing fine geometric features. In contrast, ArabicCNN uses standard 3x3 convolutions with controlled pooling and retains rich spatial feature maps throughout its 4 blocks.

### 2. Dotted-Letter Discrimination (+17.58% Advantage for ArabicCNN)
- **ArabicCNN:** **87.69%** accuracy across the 15 dotted Arabic letters (`ب`, `ت`, `ث`, `ج`, `خ`, `ذ`, `ز`, `ش`, `ض`, `ظ`, `غ`, `ف`, `ق`, `ن`, `ي`).
- **MobileNetV3:** Only **70.11%** (Hijja) and **69.91%** (Mixed).
- **Dot Blurring:** In MobileNetV3, multiple inverted residual bottleneck blocks with depthwise convolutions smooth out isolated high-frequency components (diacritical dots). As a result:
  - `ث` (3 dots) collapsed into `ت` (2 dots) 15 times in MobileNetV3-Hijja and 22 times in MobileNetV3-Mixed (vs only 3 times in ArabicCNN).
  - `ح` (0 dots) collapsed into `ج` (1 dot) 16 times in MobileNetV3-Mixed.

### 3. CPU Inference Latency (4.1x Faster for ArabicCNN)
- **ArabicCNN:** **1.10 ms** average CPU latency (1.03 ms median, 1.48 ms p95).
- **MobileNetV3:** **4.55 ms** average CPU latency (4.16 ms median, 6.51 ms p95).
- **Computational Overhead:** While MobileNetV3 has fewer parameters (1.08M vs 2.35M), its depthwise-separable convolutions and squeeze-and-excitation branches require substantially more memory passes and non-contiguous kernel operations on standard CPU hardware, resulting in 4x higher CPU latency per character.

### 4. Adult Handwriting Mixing Effect (AHCD Domain Shift)
- Training with 30% adult handwriting (AHCD) decreased child test accuracy from 76.32% to 74.83%.
- Adult handwriting features distinct cursive slant, connected ligature habits, and fluid stroke dynamics that contrast with the hesitant, isolated, angular strokes characteristic of children aged 7–12. Child-domain specificity is paramount.

### 5. False Acceptance Safety (Accepted Wrong Rate)
- In children's educational platforms, wrongly accepting an incorrect letter degrades the learning experience.
- ArabicCNN exhibits a low **4.68%** accepted-wrong rate at threshold 0.65, whereas MobileNetV3 misclassifies and falsely accepts 7.93%–8.46% of test images.

---

## Conclusion & Selection Decision

**`ArabicCNN` is conclusively selected as the final candidate model.**
It dominates across all critical production criteria:
1. **Accuracy:** Superior child-domain accuracy (90.15% vs 76.32%).
2. **Dotted Letters:** Superior preservation and distinction of Arabic dots (87.69% vs 70.11%).
3. **Speed:** 4.1x faster CPU inference (1.10 ms vs 4.55 ms).
4. **Reliability:** Half the false acceptance rate on incorrect predictions.

The winning weights are saved as [`ml/models/final_candidate_model.pt`](file:///d:/Freelancing-Projects/Writing-Detection/ml/models/final_candidate_model.pt) and integrated directly into the FastAPI backend predictor.
