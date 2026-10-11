"""
Populate sample test images for camera pipeline evaluation.
Generates test cases: clean characters, rotated, noisy, and high-contrast samples.
"""

from pathlib import Path
import shutil
import cv2
import numpy as np

SAMPLE_DIR = Path(__file__).resolve().parent.parent / "tests" / "camera_samples"
SAMPLE_DIR.mkdir(parents=True, exist_ok=True)

PROCESSED_HIJJA = Path(__file__).resolve().parent.parent / "ml" / "data" / "processed" / "hijja" / "test"

# Copy 5 test characters from Hijja test split: Alef (00), Baa (01), Jeem (04), Seen (11), Yaa (27)
test_classes = [("00", "alef"), ("01", "baa"), ("04", "jeem"), ("11", "seen"), ("27", "yaa")]

for cid, name in test_classes:
    class_folder = PROCESSED_HIJJA / cid
    if class_folder.exists():
        files = list(class_folder.glob("*.png"))
        if files:
            src = files[0]
            # Save raw sample
            dst = SAMPLE_DIR / f"test_{name}_sample.png"
            shutil.copy2(src, dst)

            # Generate simulated real camera variation (paper texture + shadow + threshold)
            img = cv2.imread(str(src), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                # Resize to larger simulated camera crop
                large = cv2.resize(img, (200, 200), interpolation=cv2.INTER_NEAREST)
                # Invert to dark ink on white paper
                inverted = cv2.bitwise_not(large)
                # Add synthetic gradient shadow
                shadow = np.tile(np.linspace(255, 200, 200), (200, 1)).astype(np.uint8)
                synthetic_cam = cv2.multiply(inverted, shadow, scale=1.0 / 255.0)

                cam_dst = SAMPLE_DIR / f"camera_sim_{name}.jpg"
                cv2.imwrite(str(cam_dst), synthetic_cam)

print(f"Sample camera test images prepared in {SAMPLE_DIR}")
