"""
Prepare AHCD (Arabic Handwritten Characters Dataset)
Extracts 32x32 images from Train/Test zip archives, maps labels (1..28 -> 0..27),
and organizes into train and test directories with a manifest.
"""

import os
import sys
import json
import zipfile
from pathlib import Path
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))
from app.ml.labels import (
    ARABIC_LETTERS,
    LETTER_METADATA,
    NUM_CLASSES,
    ID_TO_METADATA,
    ID_TO_CHAR,
)

RAW_DIR = Path(__file__).resolve().parent.parent / "ml" / "data" / "raw" / "AHCD"
PROCESSED_DIR = Path(__file__).resolve().parent.parent / "ml" / "data" / "processed" / "ahcd"


def prepare_ahcd():
    print("=" * 60)
    print("PREPARING AHCD DATASET (Adult Handwriting)")
    print("=" * 60)

    train_zip_path = RAW_DIR / "Train Images 13440x32x32.zip"
    test_zip_path = RAW_DIR / "Test Images 3360x32x32.zip"

    if not train_zip_path.exists() or not test_zip_path.exists():
        raise FileNotFoundError("AHCD zip files not found in raw directory.")

    train_dir = PROCESSED_DIR / "train"
    test_dir = PROCESSED_DIR / "test"

    for d in [train_dir, test_dir]:
        d.mkdir(parents=True, exist_ok=True)

    manifest = {"train": [], "test": []}

    # Extract Train
    print("Extracting Train images (13,440)...")
    with zipfile.ZipFile(train_zip_path, "r") as z:
        for fname in z.namelist():
            if fname.endswith(".png"):
                # File naming format: id_<writer>_label_<label_1_to_28>.png
                parts = fname[:-4].split("_")
                label_1_based = int(parts[-1])
                class_id = label_1_based - 1  # 0-indexed class

                dest_class_dir = train_dir / f"{class_id:02d}"
                dest_class_dir.mkdir(parents=True, exist_ok=True)
                dest_file = dest_class_dir / Path(fname).name

                with z.open(fname) as src, open(dest_file, "wb") as dst:
                    dst.write(src.read())

                meta = ID_TO_METADATA[class_id]
                manifest["train"].append({
                    "image_path": str(dest_file.relative_to(PROCESSED_DIR.parent.parent)),
                    "class_id": class_id,
                    "letter": meta["letter"],
                    "name_en": meta["name_en"],
                })

    # Extract Test
    print("Extracting Test images (3,360)...")
    with zipfile.ZipFile(test_zip_path, "r") as z:
        for fname in z.namelist():
            if fname.endswith(".png"):
                parts = fname[:-4].split("_")
                label_1_based = int(parts[-1])
                class_id = label_1_based - 1

                dest_class_dir = test_dir / f"{class_id:02d}"
                dest_class_dir.mkdir(parents=True, exist_ok=True)
                dest_file = dest_class_dir / Path(fname).name

                with z.open(fname) as src, open(dest_file, "wb") as dst:
                    dst.write(src.read())

                meta = ID_TO_METADATA[class_id]
                manifest["test"].append({
                    "image_path": str(dest_file.relative_to(PROCESSED_DIR.parent.parent)),
                    "class_id": class_id,
                    "letter": meta["letter"],
                    "name_en": meta["name_en"],
                })

    manifest_path = PROCESSED_DIR / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump({
            "dataset": "AHCD (28 classes)",
            "splits": {
                "train_count": len(manifest["train"]),
                "test_count": len(manifest["test"]),
                "total_count": len(manifest["train"]) + len(manifest["test"]),
            },
            "samples": manifest,
        }, f, ensure_ascii=False, indent=2)

    print(f"AHCD preparation complete!")
    print(f"Train samples: {len(manifest['train'])}")
    print(f"Test samples: {len(manifest['test'])}")
    print(f"Manifest written to: {manifest_path}")


if __name__ == "__main__":
    prepare_ahcd()
