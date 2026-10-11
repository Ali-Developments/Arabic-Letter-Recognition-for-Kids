# Arabic Handwritten Letter Recognition — Dataset Inspection & Mapping Report

> **Stage**: Stage 2 — Dataset Preparation & Verification
> **Date**: 2026-10-11
> **Specification Source**: [ARABIC_HANDWRITING_AI_PROJECT_BRIEF.md](file:///d:/Freelancing-Projects/Writing-Detection/ARABIC_HANDWRITING_AI_PROJECT_BRIEF.md)

---

## 1. Executive Summary

This report documents the inspection, validation, and class-mapping of the two Arabic handwriting datasets for the isolated Arabic letter recognition AI module:

1. **Hijja2 (Primary Dataset)**: Collected specifically from Arabic-speaking school children aged 7–12 years in Riyadh, Saudi Arabia (591 participants). Contains `45726` total samples across 29 folders, of which **`12351` isolated character samples** across the 28 target classes have been isolated and verified.
2. **AHCD (Secondary/Support Dataset)**: Collected from 60 adult participants aged 19–40 years. Contains **`16800` total images** (13,440 train + 3,360 test), with a balanced distribution of 480 train and 120 test samples per class across all 28 classes.

---

## 2. Dataset Overview & Inventory

| Metric | Hijja2 (Primary - Children) | AHCD (Secondary - Adult) | Total Combined |
| :--- | :--- | :--- | :--- |
| **Target Demographic** | School children (ages 7–12) | Adults (ages 19–40) | Multi-age |
| **Total Raw Samples** | 47434 | 16800 | 64234 |
| **Usable Isolated Samples (28 Classes)** | **12351** | **16800** | **29151** |
| **Excluded Classes** | Hamza (`29 hamza`: 1708 samples) | None | 1708 |
| **Excluded Forms in Hijja (B, M, E)** | 33375 connected samples | N/A (all isolated) | 33375 |
| **Native Image Resolution** | [[32, 32]] | [32, 32] | Varied |
| **Native Color Mode** | ['L'] | Grayscale (L) | Grayscale |

---

## 3. Authoritative 28-Class Label Mapping

Per section 3 of the Project Brief, the target classes are the 28 isolated Arabic letters in canonical alphabetical order. The table below defines the exact mapping used across all modules:

| Class ID | Letter | Name (EN) | Name (AR) | Unicode | Hijja Folder | Hijja Isolated Subdir | Hijja Samples | AHCD Train | AHCD Test | Total Samples | Dots & Position |
| :---: | :---: | :--- | :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| 0 | **ا** | alef | ألف | `U+0627` | `1 alif` | `1.1` | 456 | 480 | 120 | 1056 | 0 (none) |
| 1 | **ب** | baa | باء | `U+0628` | `2 ba` | `2.1` | 452 | 480 | 120 | 1052 | 1 (below) |
| 2 | **ت** | taa | تاء | `U+062A` | `3 ta` | `3.1` | 451 | 480 | 120 | 1051 | 2 (above) |
| 3 | **ث** | thaa | ثاء | `U+062B` | `4 tha` | `4.1` | 454 | 480 | 120 | 1054 | 3 (above) |
| 4 | **ج** | jeem | جيم | `U+062C` | `5 gim` | `5.1` | 472 | 480 | 120 | 1072 | 1 (inside) |
| 5 | **ح** | haa | حاء | `U+062D` | `6 ha` | `6.1` | 462 | 480 | 120 | 1062 | 0 (none) |
| 6 | **خ** | khaa | خاء | `U+062E` | `7 kha` | `7.1` | 471 | 480 | 120 | 1071 | 1 (above) |
| 7 | **د** | daal | دال | `U+062F` | `8 dal` | `8.1` | 449 | 480 | 120 | 1049 | 0 (none) |
| 8 | **ذ** | dhaal | ذال | `U+0630` | `9 thal` | `9.1` | 429 | 480 | 120 | 1029 | 1 (above) |
| 9 | **ر** | raa | راء | `U+0631` | `10 ra` | `10.1` | 424 | 480 | 120 | 1024 | 0 (none) |
| 10 | **ز** | zaay | زاي | `U+0632` | `11 zay` | `11.1` | 426 | 480 | 120 | 1026 | 1 (above) |
| 11 | **س** | seen | سين | `U+0633` | `12 sin` | `12.1` | 440 | 480 | 120 | 1040 | 0 (none) |
| 12 | **ش** | sheen | شين | `U+0634` | `13 shin` | `13.1` | 439 | 480 | 120 | 1039 | 3 (above) |
| 13 | **ص** | saad | صاد | `U+0635` | `14 sad` | `14.1` | 438 | 480 | 120 | 1038 | 0 (none) |
| 14 | **ض** | daad | ضاد | `U+0636` | `15 dad` | `15.1` | 432 | 480 | 120 | 1032 | 1 (above) |
| 15 | **ط** | tah | طاء | `U+0637` | `16 da` | `16.1` | 452 | 480 | 120 | 1052 | 0 (none) |
| 16 | **ظ** | dhah | ظاء | `U+0638` | `17 za` | `17.1` | 422 | 480 | 120 | 1022 | 1 (above) |
| 17 | **ع** | ain | عين | `U+0639` | `18 ayn` | `18.1` | 434 | 480 | 120 | 1034 | 0 (none) |
| 18 | **غ** | ghain | غين | `U+063A` | `19 gayn` | `19.1` | 427 | 480 | 120 | 1027 | 1 (above) |
| 19 | **ف** | faa | فاء | `U+0641` | `20 fa` | `20.1` | 434 | 480 | 120 | 1034 | 1 (above) |
| 20 | **ق** | qaaf | قاف | `U+0642` | `21 qaf` | `21.1` | 435 | 480 | 120 | 1035 | 2 (above) |
| 21 | **ك** | kaaf | كاف | `U+0643` | `22 kaf` | `22.1` | 438 | 480 | 120 | 1038 | 0 (none) |
| 22 | **ل** | laam | لام | `U+0644` | `23 lam` | `23.1` | 437 | 480 | 120 | 1037 | 0 (none) |
| 23 | **م** | meem | ميم | `U+0645` | `24 mim` | `24.1` | 435 | 480 | 120 | 1035 | 0 (none) |
| 24 | **ن** | noon | نون | `U+0646` | `25 non` | `25.1` | 440 | 480 | 120 | 1040 | 1 (above/inside) |
| 25 | **ه** | haa_end | هاء | `U+0647` | `26 ha` | `26.1` | 428 | 480 | 120 | 1028 | 0 (none) |
| 26 | **و** | waaw | واو | `U+0648` | `27 waw` | `27.1` | 439 | 480 | 120 | 1039 | 0 (none) |
| 27 | **ي** | yaa | ياء | `U+064A` | `28 ya` | `28.1` | 435 | 480 | 120 | 1035 | 2 (below) |

---

## 4. Class Balance Analysis

- **Hijja Isolated Mean Samples per Class**: 441.1
- **Standard Deviation**: 13.12
- **Minimum Class Count**: 422 (Class 16: ظ)
- **Maximum Class Count**: 472 (Class 4: ج)
- **Balance Verdict**: Hijja isolated subset is remarkably well-balanced, ranging between 422 and 472 samples per class across all 28 classes.
- **AHCD Balance**: Perfectly balanced with exactly 480 training and 120 testing samples per class.

---

## 5. Excluded Classes and Forms

### 5.1 Hamza (`29 hamza`)
- Total excluded samples: `1708`
- **Rationale**: Per Project Brief Section 4 & Section 21, the client target is strictly the 28 standard Arabic letters taught to children. Hamza is not one of the 28 base alphabet letters in this curriculum.

### 5.2 Connected Letter Forms (Beginning, Middle, End)
- Total connected samples preserved in raw dataset but excluded from the 28-class isolated model: `33375` samples.
- **Rationale**: In the target game/learning scenario, the child writes an isolated letter on a piece of paper. Mixing medial and final forms (which include horizontal connection ligatures) would corrupt isolated letter classification.

---

## 6. Train / Validation / Test Split Strategy

We establish a stratified 70% / 15% / 15% split on the isolated child samples:

### Split 1: Child-Only Benchmark (Hijja)
- **Train (70%)**: ~8,645 images (~308 per class)
- **Validation (15%)**: ~1,852 images (~66 per class)
- **Test (15%)**: ~1,852 images (~66 per class)

### Split 2: Augmented / Mixed Training (Hijja Train + AHCD Train)
- **Combined Train**: 22,085 images (Hijja Train + AHCD 13,440)
- **Child Validation**: ~1,852 images (Hijja Val)
- **Child Test (Ground Truth)**: ~1,852 images (Hijja Test)

> [!IMPORTANT]
> Per Project Brief Section 4, the **Test set must always consist of child handwriting samples (Hijja)** to reflect real-world performance with school children.

---

## 7. Critical Domain Characteristics & Detected Issues

1. **Arabic Dot Preservation**:
   - 15 out of 28 letters rely on dots (1 dot: Baa, Jeem, Khaa, Dhaal, Zaay, Daad, Dhaa, Ghain, Faa, Noon; 2 dots: Taa, Qaaf, Yaa; 3 dots: Thaa, Sheen).
   - Traditional CV noise-removal filters (e.g. morphological opening with large kernel, or removing small connected components) would erase dots, fatally confusing `ب` with `ت` or `ث`, or `د` with `ذ`.
2. **Child Stroke Variations**:
   - Young children frequently write with varying pen pressure, irregular stroke slant, and separated or overlapping dots.
3. **Inversion & Background Contrast**:
   - Hijja raw images are black ink on white paper (or inverted in CSVs). Real camera capture produces dark ink on bright/shadowed paper.
   - The preprocessing pipeline must standardize all inputs to white stroke on black background (or normalized black on white) matching model training.

---

## 8. Sample Grids Generated

- Hijja 28-Class Sample Grid: `docs/samples/hijja_28_classes_grid.png`
- AHCD 28-Class Sample Grid: `docs/samples/ahcd_28_classes_grid.png`

---

## 9. Recommended Next Step

Proceed to **Stage 3 & Stage 4**:
1. Implement dataset splitting and caching script (`scripts/prepare_hijja.py` & `scripts/prepare_ahcd.py`).
2. Build custom CNN baseline classifier and train on Hijja child handwriting.
3. Evaluate child test accuracy and confusion matrix.
