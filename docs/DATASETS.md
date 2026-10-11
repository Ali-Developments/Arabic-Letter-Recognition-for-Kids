# Datasets Specification — Hijja & AHCD

> **Detailed Audit Report**: [DATASET_REPORT.md](file:///d:/Freelancing-Projects/Writing-Detection/DATASET_REPORT.md)

---

## 1. Primary Dataset — Hijja2

- **Origin**: King Saud University, Riyadh, Saudi Arabia (Altwaijry et al., 2019–2020)
- **Repository**: [https://github.com/israksu/Hijja2](https://github.com/israksu/Hijja2)
- **Target Demographic**: 591 school children (approx. ages 7 to 12).
- **Total Characters**: 47,434 characters across 29 classes.
- **Isolated Subset**: **12,351 samples** across the 28 canonical letters (subfolder `.1`).
- **Resolution**: 32x32 pixels, single-channel grayscale (`L`).
- **Relevance**: Represents authentic children's motor skills, stroke instability, and handwriting habits.

### Excluded Classes in Hijja:
1. `29 hamza` (1,708 samples): Excluded because curriculum focuses on the 28 isolated letters.
2. Connected forms `x.2` (beginning), `x.3` (middle), `x.4` (end): Excluded to prevent ligature bias when children write isolated characters on paper.

---

## 2. Secondary Dataset — AHCD

- **Origin**: El-Sawy, Loey, El-Bakry (WSEAS, 2017)
- **Repository**: [https://github.com/mloey/Arabic-Handwritten-Characters-Dataset](https://github.com/mloey/Arabic-Handwritten-Characters-Dataset)
- **Demographic**: 60 adult writers (ages 19–40).
- **Total Characters**: 16,800 characters across 28 classes.
  - Training: 13,440 (480 per class)
  - Testing: 3,360 (120 per class)
- **Resolution**: 32x32 pixels, single-channel grayscale.
- **Role**: Support dataset to enrich stroke variability and prevent overfitting on small child sample counts during mixed training experiments.
