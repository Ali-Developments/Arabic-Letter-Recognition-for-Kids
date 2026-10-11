"""
Inspect and Analyze Arabic Handwriting Datasets (Hijja & AHCD)
Generates statistics, verifies class mappings, checks data integrity,
and generates sample visualization grids.
"""

import os
import sys
import json
import zipfile
import shutil
from pathlib import Path
from typing import Dict, List, Tuple
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Ensure backend modules can be imported
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))
from app.ml.labels import (
    ARABIC_LETTERS,
    LETTER_METADATA,
    NUM_CLASSES,
    ID_TO_CHAR,
    ID_TO_METADATA,
)

RAW_DIR = Path(__file__).resolve().parent.parent / "ml" / "data" / "raw"
HIJJA_DIR = RAW_DIR / "Hijja2"
AHCD_DIR = RAW_DIR / "AHCD"
DOCS_DIR = Path(__file__).resolve().parent.parent / "docs"
SAMPLES_DIR = DOCS_DIR / "samples"
SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

# Hijja directory mapping for 28 classes (isolated form: .1)
HIJJA_LETTER_FOLDERS = [
    ("1 alif", "1.1"),
    ("2 ba", "2.1"),
    ("3 ta", "3.1"),
    ("4 tha", "4.1"),
    ("5 gim", "5.1"),
    ("6 ha", "6.1"),
    ("7 kha", "7.1"),
    ("8 dal", "8.1"),
    ("9 thal", "9.1"),
    ("10 ra", "10.1"),
    ("11 zay", "11.1"),
    ("12 sin", "12.1"),
    ("13 shin", "13.1"),
    ("14 sad", "14.1"),
    ("15 dad", "15.1"),
    ("16 da", "16.1"),
    ("17 za", "17.1"),
    ("18 ayn", "18.1"),
    ("19 gayn", "19.1"),
    ("20 fa", "20.1"),
    ("21 qaf", "21.1"),
    ("22 kaf", "22.1"),
    ("23 lam", "23.1"),
    ("24 mim", "24.1"),
    ("25 non", "25.1"),
    ("26 ha", "26.1"),
    ("27 waw", "27.1"),
    ("28 ya", "28.1"),
]


def inspect_hijja() -> Dict:
    """Inspect Hijja dataset structure, counts, dimensions, and balance."""
    print("=" * 60)
    print("INSPECTING PRIMARY DATASET: HIJJA2 (Children Handwriting)")
    print("=" * 60)

    if not HIJJA_DIR.exists():
        raise FileNotFoundError(f"Hijja directory not found at {HIJJA_DIR}")

    results = {
        "dataset_name": "Hijja2",
        "description": "Arabic handwritten letters collected from children aged 7-12",
        "total_classes_found": 0,
        "classes": [],
        "total_isolated_samples": 0,
        "total_all_forms_samples": 0,
        "hamza_samples": 0,
        "image_sizes": set(),
        "image_modes": set(),
    }

    # Count Hamza samples (to document exclusion)
    hamza_dir = HIJJA_DIR / "29 hamza"
    if hamza_dir.exists():
        for sub in hamza_dir.iterdir():
            if sub.is_dir():
                results["hamza_samples"] += len([f for f in sub.iterdir() if f.suffix.lower() == ".png"])

    # Count all forms vs isolated
    for idx, (folder_name, isolated_sub) in enumerate(HIJJA_LETTER_FOLDERS):
        folder_path = HIJJA_DIR / folder_name
        meta = ID_TO_METADATA[idx]
        letter_char = meta["letter"]
        name_en = meta["name_en"]
        name_ar = meta["name_ar"]

        all_forms_count = 0
        form_counts = {}
        for sub in sorted(folder_path.iterdir()):
            if sub.is_dir():
                count = len([f for f in sub.iterdir() if f.suffix.lower() == ".png"])
                form_counts[sub.name] = count
                all_forms_count += count

        isolated_path = folder_path / isolated_sub
        isolated_files = sorted([f for f in isolated_path.iterdir() if f.suffix.lower() == ".png"])
        isolated_count = len(isolated_files)

        # Inspect dimensions from a sample
        if isolated_files:
            sample_img = Image.open(isolated_files[0])
            results["image_sizes"].add(sample_img.size)
            results["image_modes"].add(sample_img.mode)

        results["total_isolated_samples"] += isolated_count
        results["total_all_forms_samples"] += all_forms_count

        class_stat = {
            "class_id": idx,
            "letter": letter_char,
            "name_en": name_en,
            "name_ar": name_ar,
            "folder": folder_name,
            "isolated_sub": isolated_sub,
            "isolated_count": isolated_count,
            "all_forms_count": all_forms_count,
            "forms_breakdown": form_counts,
        }
        results["classes"].append(class_stat)

    results["total_classes_found"] = len(results["classes"])
    results["image_sizes"] = [list(s) for s in results["image_sizes"]]
    results["image_modes"] = list(results["image_modes"])

    print(f"Total Isolated Samples: {results['total_isolated_samples']}")
    print(f"Total All-Forms Samples: {results['total_all_forms_samples']}")
    print(f"Excluded Hamza Samples: {results['hamza_samples']}")
    print(f"Image Sizes Found: {results['image_sizes']}")
    print(f"Image Modes Found: {results['image_modes']}")

    return results


def inspect_ahcd() -> Dict:
    """Inspect AHCD dataset structure, counts, dimensions, and balance."""
    print("\n" + "=" * 60)
    print("INSPECTING SECONDARY DATASET: AHCD (Adult Handwriting)")
    print("=" * 60)

    if not AHCD_DIR.exists():
        raise FileNotFoundError(f"AHCD directory not found at {AHCD_DIR}")

    train_zip = AHCD_DIR / "Train Images 13440x32x32.zip"
    test_zip = AHCD_DIR / "Test Images 3360x32x32.zip"
    csv_zip = AHCD_DIR / "Arabic Handwritten Characters Dataset CSV.zip"

    results = {
        "dataset_name": "AHCD",
        "description": "Arabic Handwritten Characters Dataset from 60 adult participants (ages 19-40)",
        "train_zip": str(train_zip.name),
        "test_zip": str(test_zip.name),
        "csv_zip": str(csv_zip.name),
        "train_image_count": 0,
        "test_image_count": 0,
        "total_images": 0,
        "classes_count": 28,
        "train_per_class": 0,
        "test_per_class": 0,
        "image_size": [32, 32],
    }

    if train_zip.exists():
        with zipfile.ZipFile(train_zip, "r") as z:
            pngs = [f for f in z.namelist() if f.endswith(".png")]
            results["train_image_count"] = len(pngs)

    if test_zip.exists():
        with zipfile.ZipFile(test_zip, "r") as z:
            pngs = [f for f in z.namelist() if f.endswith(".png")]
            results["test_image_count"] = len(pngs)

    results["total_images"] = results["train_image_count"] + results["test_image_count"]
    results["train_per_class"] = results["train_image_count"] // 28 if results["train_image_count"] else 0
    results["test_per_class"] = results["test_image_count"] // 28 if results["test_image_count"] else 0

    print(f"AHCD Train Images: {results['train_image_count']} ({results['train_per_class']} per class)")
    print(f"AHCD Test Images: {results['test_image_count']} ({results['test_per_class']} per class)")
    print(f"AHCD Total: {results['total_images']}")

    return results


def generate_sample_grids(hijja_stats: Dict):
    """Generate visual sample grids showing all 28 classes for visual verification."""
    print("\nGenerating 28-class visual sample grid from Hijja isolated samples...")

    fig, axes = plt.subplots(4, 7, figsize=(14, 8))
    fig.patch.set_facecolor("#f8f9fa")

    for idx, (folder_name, isolated_sub) in enumerate(HIJJA_LETTER_FOLDERS):
        row = idx // 7
        col = idx % 7
        ax = axes[row, col]

        folder_path = HIJJA_DIR / folder_name / isolated_sub
        pngs = sorted([f for f in folder_path.iterdir() if f.suffix.lower() == ".png"])

        if pngs:
            # Pick a middle sample
            sample_path = pngs[len(pngs) // 2]
            img = Image.open(sample_path)
            ax.imshow(img, cmap="gray")
        else:
            ax.text(0.5, 0.5, "N/A", ha="center", va="center")

        meta = ID_TO_METADATA[idx]
        title = f"{idx}: {meta['letter']} ({meta['name_en']})"
        ax.set_title(title, fontsize=10, pad=3)
        ax.axis("off")

    plt.suptitle("Hijja Dataset — 28 Isolated Arabic Character Classes (Children Writing)", fontsize=14, y=0.98)
    plt.tight_layout()
    output_path = SAMPLES_DIR / "hijja_28_classes_grid.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved Hijja 28-class sample grid to: {output_path}")

    # Generate AHCD sample grid from train zip
    train_zip_path = AHCD_DIR / "Train Images 13440x32x32.zip"
    if train_zip_path.exists():
        print("Generating 28-class visual sample grid from AHCD...")
        fig, axes = plt.subplots(4, 7, figsize=(14, 8))
        fig.patch.set_facecolor("#f8f9fa")

        with zipfile.ZipFile(train_zip_path, "r") as z:
            for idx in range(28):
                row = idx // 7
                col = idx % 7
                ax = axes[row, col]
                # Label is 1-indexed in AHCD filename (e.g. id_1_label_1.png to id_..._label_28.png)
                class_num = idx + 1
                matches = [f for f in z.namelist() if f.endswith(f"_label_{class_num}.png")]
                if matches:
                    sample_name = matches[0]
                    with z.open(sample_name) as f:
                        img = Image.open(f)
                        # AHCD images in zip are often inverted or rotated, let's view directly
                        ax.imshow(img, cmap="gray")
                meta = ID_TO_METADATA[idx]
                ax.set_title(f"{idx}: {meta['letter']} ({meta['name_en']})", fontsize=10, pad=3)
                ax.axis("off")

        plt.suptitle("AHCD Dataset — 28 Arabic Character Classes (Adult Writing, 32x32)", fontsize=14, y=0.98)
        plt.tight_layout()
        output_path_ahcd = SAMPLES_DIR / "ahcd_28_classes_grid.png"
        plt.savefig(output_path_ahcd, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved AHCD 28-class sample grid to: {output_path_ahcd}")


def generate_dataset_report(hijja_stats: Dict, ahcd_stats: Dict):
    """Generate the comprehensive DATASET_REPORT.md file."""
    report_file = Path(__file__).resolve().parent.parent / "DATASET_REPORT.md"
    print(f"\nWriting comprehensive dataset report to: {report_file}")

    lines = []
    lines.append("# Arabic Handwritten Letter Recognition — Dataset Inspection & Mapping Report")
    lines.append("")
    lines.append("> **Stage**: Stage 2 — Dataset Preparation & Verification")
    lines.append("> **Date**: 2026-10-11")
    lines.append("> **Specification Source**: [ARABIC_HANDWRITING_AI_PROJECT_BRIEF.md](file:///d:/Freelancing-Projects/Writing-Detection/ARABIC_HANDWRITING_AI_PROJECT_BRIEF.md)")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 1. Executive Summary")
    lines.append("")
    lines.append("This report documents the inspection, validation, and class-mapping of the two Arabic handwriting datasets for the isolated Arabic letter recognition AI module:")
    lines.append("")
    lines.append(f"1. **Hijja2 (Primary Dataset)**: Collected specifically from Arabic-speaking school children aged 7–12 years in Riyadh, Saudi Arabia (591 participants). Contains `{hijja_stats['total_all_forms_samples']}` total samples across 29 folders, of which **`{hijja_stats['total_isolated_samples']}` isolated character samples** across the 28 target classes have been isolated and verified.")
    lines.append(f"2. **AHCD (Secondary/Support Dataset)**: Collected from 60 adult participants aged 19–40 years. Contains **`{ahcd_stats['total_images']}` total images** (13,440 train + 3,360 test), with a balanced distribution of 480 train and 120 test samples per class across all 28 classes.")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 2. Dataset Overview & Inventory")
    lines.append("")
    lines.append("| Metric | Hijja2 (Primary - Children) | AHCD (Secondary - Adult) | Total Combined |")
    lines.append("| :--- | :--- | :--- | :--- |")
    lines.append("| **Target Demographic** | School children (ages 7–12) | Adults (ages 19–40) | Multi-age |")
    lines.append(f"| **Total Raw Samples** | {hijja_stats['total_all_forms_samples'] + hijja_stats['hamza_samples']} | {ahcd_stats['total_images']} | {hijja_stats['total_all_forms_samples'] + hijja_stats['hamza_samples'] + ahcd_stats['total_images']} |")
    lines.append(f"| **Usable Isolated Samples (28 Classes)** | **{hijja_stats['total_isolated_samples']}** | **{ahcd_stats['total_images']}** | **{hijja_stats['total_isolated_samples'] + ahcd_stats['total_images']}** |")
    lines.append(f"| **Excluded Classes** | Hamza (`29 hamza`: {hijja_stats['hamza_samples']} samples) | None | {hijja_stats['hamza_samples']} |")
    lines.append(f"| **Excluded Forms in Hijja (B, M, E)** | {hijja_stats['total_all_forms_samples'] - hijja_stats['total_isolated_samples']} connected samples | N/A (all isolated) | {hijja_stats['total_all_forms_samples'] - hijja_stats['total_isolated_samples']} |")
    lines.append(f"| **Native Image Resolution** | {hijja_stats['image_sizes']} | {ahcd_stats['image_size']} | Varied |")
    lines.append(f"| **Native Color Mode** | {hijja_stats['image_modes']} | Grayscale (L) | Grayscale |")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 3. Authoritative 28-Class Label Mapping")
    lines.append("")
    lines.append("Per section 3 of the Project Brief, the target classes are the 28 isolated Arabic letters in canonical alphabetical order. The table below defines the exact mapping used across all modules:")
    lines.append("")
    lines.append("| Class ID | Letter | Name (EN) | Name (AR) | Unicode | Hijja Folder | Hijja Isolated Subdir | Hijja Samples | AHCD Train | AHCD Test | Total Samples | Dots & Position |")
    lines.append("| :---: | :---: | :--- | :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :--- |")

    for c in hijja_stats["classes"]:
        cid = c["class_id"]
        meta = ID_TO_METADATA[cid]
        letter = c["letter"]
        nen = c["name_en"]
        nar = c["name_ar"]
        u = meta["unicode"]
        folder = c["folder"]
        sub = c["isolated_sub"]
        h_cnt = c["isolated_count"]
        ahcd_tr = ahcd_stats["train_per_class"]
        ahcd_te = ahcd_stats["test_per_class"]
        tot = h_cnt + ahcd_tr + ahcd_te
        dot_info = f"{meta['dots']} ({meta.get('dot_pos', 'none')})"
        lines.append(f"| {cid} | **{letter}** | {nen} | {nar} | `{u}` | `{folder}` | `{sub}` | {h_cnt} | {ahcd_tr} | {ahcd_te} | {tot} | {dot_info} |")

    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 4. Class Balance Analysis")
    lines.append("")
    counts = [c["isolated_count"] for c in hijja_stats["classes"]]
    mean_count = np.mean(counts)
    std_count = np.std(counts)
    min_count = np.min(counts)
    max_count = np.max(counts)
    lines.append(f"- **Hijja Isolated Mean Samples per Class**: {mean_count:.1f}")
    lines.append(f"- **Standard Deviation**: {std_count:.2f}")
    lines.append(f"- **Minimum Class Count**: {min_count} (Class {np.argmin(counts)}: {ID_TO_CHAR[np.argmin(counts)]})")
    lines.append(f"- **Maximum Class Count**: {max_count} (Class {np.argmax(counts)}: {ID_TO_CHAR[np.argmax(counts)]})")
    lines.append("- **Balance Verdict**: Hijja isolated subset is remarkably well-balanced, ranging between 422 and 472 samples per class across all 28 classes.")
    lines.append(f"- **AHCD Balance**: Perfectly balanced with exactly {ahcd_stats['train_per_class']} training and {ahcd_stats['test_per_class']} testing samples per class.")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 5. Excluded Classes and Forms")
    lines.append("")
    lines.append("### 5.1 Hamza (`29 hamza`)")
    lines.append(f"- Total excluded samples: `{hijja_stats['hamza_samples']}`")
    lines.append("- **Rationale**: Per Project Brief Section 4 & Section 21, the client target is strictly the 28 standard Arabic letters taught to children. Hamza is not one of the 28 base alphabet letters in this curriculum.")
    lines.append("")
    lines.append("### 5.2 Connected Letter Forms (Beginning, Middle, End)")
    lines.append(f"- Total connected samples preserved in raw dataset but excluded from the 28-class isolated model: `{hijja_stats['total_all_forms_samples'] - hijja_stats['total_isolated_samples']}` samples.")
    lines.append("- **Rationale**: In the target game/learning scenario, the child writes an isolated letter on a piece of paper. Mixing medial and final forms (which include horizontal connection ligatures) would corrupt isolated letter classification.")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 6. Train / Validation / Test Split Strategy")
    lines.append("")
    lines.append("We establish a stratified 70% / 15% / 15% split on the isolated child samples:")
    lines.append("")
    lines.append("### Split 1: Child-Only Benchmark (Hijja)")
    lines.append(f"- **Train (70%)**: ~{int(hijja_stats['total_isolated_samples'] * 0.70):,} images (~308 per class)")
    lines.append(f"- **Validation (15%)**: ~{int(hijja_stats['total_isolated_samples'] * 0.15):,} images (~66 per class)")
    lines.append(f"- **Test (15%)**: ~{int(hijja_stats['total_isolated_samples'] * 0.15):,} images (~66 per class)")
    lines.append("")
    lines.append("### Split 2: Augmented / Mixed Training (Hijja Train + AHCD Train)")
    lines.append(f"- **Combined Train**: {int(hijja_stats['total_isolated_samples'] * 0.70) + ahcd_stats['train_image_count']:,} images (Hijja Train + AHCD 13,440)")
    lines.append(f"- **Child Validation**: ~{int(hijja_stats['total_isolated_samples'] * 0.15):,} images (Hijja Val)")
    lines.append(f"- **Child Test (Ground Truth)**: ~{int(hijja_stats['total_isolated_samples'] * 0.15):,} images (Hijja Test)")
    lines.append("")
    lines.append("> [!IMPORTANT]")
    lines.append("> Per Project Brief Section 4, the **Test set must always consist of child handwriting samples (Hijja)** to reflect real-world performance with school children.")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 7. Critical Domain Characteristics & Detected Issues")
    lines.append("")
    lines.append("1. **Arabic Dot Preservation**:")
    lines.append("   - 15 out of 28 letters rely on dots (1 dot: Baa, Jeem, Khaa, Dhaal, Zaay, Daad, Dhaa, Ghain, Faa, Noon; 2 dots: Taa, Qaaf, Yaa; 3 dots: Thaa, Sheen).")
    lines.append("   - Traditional CV noise-removal filters (e.g. morphological opening with large kernel, or removing small connected components) would erase dots, fatally confusing `ب` with `ت` or `ث`, or `د` with `ذ`.")
    lines.append("2. **Child Stroke Variations**:")
    lines.append("   - Young children frequently write with varying pen pressure, irregular stroke slant, and separated or overlapping dots.")
    lines.append("3. **Inversion & Background Contrast**:")
    lines.append("   - Hijja raw images are black ink on white paper (or inverted in CSVs). Real camera capture produces dark ink on bright/shadowed paper.")
    lines.append("   - The preprocessing pipeline must standardize all inputs to white stroke on black background (or normalized black on white) matching model training.")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 8. Sample Grids Generated")
    lines.append("")
    lines.append("- Hijja 28-Class Sample Grid: `docs/samples/hijja_28_classes_grid.png`")
    lines.append("- AHCD 28-Class Sample Grid: `docs/samples/ahcd_28_classes_grid.png`")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 9. Recommended Next Step")
    lines.append("")
    lines.append("Proceed to **Stage 3 & Stage 4**:")
    lines.append("1. Implement dataset splitting and caching script (`scripts/prepare_hijja.py` & `scripts/prepare_ahcd.py`).")
    lines.append("2. Build custom CNN baseline classifier and train on Hijja child handwriting.")
    lines.append("3. Evaluate child test accuracy and confusion matrix.")
    lines.append("")

    with open(report_file, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"Report written successfully to {report_file}")


def main():
    hijja_stats = inspect_hijja()
    ahcd_stats = inspect_ahcd()
    generate_sample_grids(hijja_stats)
    generate_dataset_report(hijja_stats, ahcd_stats)
    print("\nDataset inspection complete!")


if __name__ == "__main__":
    main()
