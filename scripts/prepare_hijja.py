"""
Prepare Hijja Dataset for Model Training
Extracts isolated letter samples (.1 subfolders), verifies labels (0 to 27),
creates stratified Train (70%), Val (15%), Test (15%) splits, and saves metadata manifests.
"""

import os
import sys
import json
import random
import shutil
from pathlib import Path
from typing import Dict, List, Tuple
from PIL import Image
import numpy as np

# Ensure backend imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))
from app.ml.labels import (
    ARABIC_LETTERS,
    LETTER_METADATA,
    NUM_CLASSES,
    ID_TO_METADATA,
    ID_TO_CHAR,
)

RAW_DIR = Path(__file__).resolve().parent.parent / "ml" / "data" / "raw" / "Hijja2"
PROCESSED_DIR = Path(__file__).resolve().parent.parent / "ml" / "data" / "processed" / "hijja"

HIJJA_ISOLATED_DIRS = [
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


def prepare_hijja_splits(seed: int = 42, train_ratio: float = 0.70, val_ratio: float = 0.15):
    random.seed(seed)
    np.random.seed(seed)

    print("=" * 60)
    print("PREPARING HIJJA ISOLATED DATASET SPLITS (Children)")
    print(f"Ratios: Train {train_ratio*100:.0f}%, Val {val_ratio*100:.0f}%, Test {(1-train_ratio-val_ratio)*100:.0f}%")
    print("=" * 60)

    train_dir = PROCESSED_DIR / "train"
    val_dir = PROCESSED_DIR / "val"
    test_dir = PROCESSED_DIR / "test"

    for d in [train_dir, val_dir, test_dir]:
        d.mkdir(parents=True, exist_ok=True)

    manifest = {"train": [], "val": [], "test": []}
    class_stats = []

    for class_id, (letter_folder, isolated_sub) in enumerate(HIJJA_ISOLATED_DIRS):
        source_dir = RAW_DIR / letter_folder / isolated_sub
        files = sorted([f for f in source_dir.iterdir() if f.suffix.lower() == ".png"])
        total_class_files = len(files)

        random.shuffle(files)
        n_train = int(total_class_files * train_ratio)
        n_val = int(total_class_files * val_ratio)
        n_test = total_class_files - n_train - n_val

        train_files = files[:n_train]
        val_files = files[n_train:n_train + n_val]
        test_files = files[n_train + n_val:]

        meta = ID_TO_METADATA[class_id]

        for split_name, split_files in [("train", train_files), ("val", val_files), ("test", test_files)]:
            split_class_dir = PROCESSED_DIR / split_name / f"{class_id:02d}"
            split_class_dir.mkdir(parents=True, exist_ok=True)

            for src in split_files:
                dest = split_class_dir / src.name
                shutil.copy2(src, dest)
                manifest[split_name].append({
                    "image_path": str(dest.relative_to(PROCESSED_DIR.parent.parent)),
                    "class_id": class_id,
                    "letter": meta["letter"],
                    "name_en": meta["name_en"],
                })

        class_stats.append({
            "class_id": class_id,
            "letter": meta["letter"],
            "name_en": meta["name_en"],
            "total": total_class_files,
            "train": len(train_files),
            "val": len(val_files),
            "test": len(test_files),
        })

    # Save manifest JSON
    manifest_path = PROCESSED_DIR / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump({
            "dataset": "Hijja Isolated Letters (28 classes)",
            "splits": {
                "train_count": len(manifest["train"]),
                "val_count": len(manifest["val"]),
                "test_count": len(manifest["test"]),
                "total_count": len(manifest["train"]) + len(manifest["val"]) + len(manifest["test"]),
            },
            "class_stats": class_stats,
            "samples": manifest,
        }, f, ensure_ascii=False, indent=2)

    print(f"Preparation complete!")
    print(f"Train samples: {len(manifest['train'])}")
    print(f"Validation samples: {len(manifest['val'])}")
    print(f"Test samples: {len(manifest['test'])}")
    print(f"Manifest written to: {manifest_path}")


if __name__ == "__main__":
    prepare_hijja_splits()
