import cv2
import torch
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path("backend").resolve()))
sys.path.insert(0, ".")
from app.ml.architectures import ArabicCNN
from app.ml.labels import ID_TO_CHAR, ID_TO_METADATA
from scripts.compare_preprocessing_variants import preprocess_variant_c, fit_to_square

ckpt = torch.load("ml/models/final_candidate_model.pt", map_location="cpu")
m = ArabicCNN()
m.load_state_dict(ckpt["model_state_dict"])
m.eval()

test_cids = [0, 1, 2, 4, 11, 13, 24, 27]
print(f"Testing {len(test_cids)} character classes with margin=8...")

for test_cid in test_cids:
    name = ID_TO_METADATA[test_cid]["name_en"]
    img_path = Path(f"tests/camera_samples/clean/clean_{test_cid:02d}_{name}.jpg")
    if not img_path.exists():
        continue
    img = cv2.imread(str(img_path))
    p32, meta = preprocess_variant_c(img)
    if meta["status"] != "valid":
        print(f"True {test_cid:02d} ({name:8s}) -> REJECTED: {meta['status']}")
        continue
    coords = cv2.findNonZero(meta["stages"]["cleaned"])
    if coords is None:
        continue
    x, y, bw, bh = cv2.boundingRect(coords)
    cropped = meta["stages"]["cleaned"][y:y+bh, x:x+bw]
    for mar in [8, 9, 10]:
        fitted = fit_to_square(cropped, 32, margin=mar)
        tensor = torch.from_numpy(fitted.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            probs = torch.softmax(m(tensor), dim=1).numpy()[0]
        top1 = int(probs.argmax())
        t_name = ID_TO_METADATA[test_cid]["name_en"]
        p_name = ID_TO_METADATA[top1]["name_en"]
        is_match = (test_cid == top1)
        print(f"True {test_cid:02d} ({t_name:8s}) mar={mar:2d} -> Pred {top1:02d} ({p_name:8s}) | Conf: {probs[top1]*100:5.1f}% | Match: {is_match}")
