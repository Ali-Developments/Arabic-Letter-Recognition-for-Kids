import sys
from pathlib import Path

backend_dir = Path(__file__).resolve().parent.parent / "backend"
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

import cv2
import numpy as np
from app.ml.predictor import ArabicPredictor

predictor = ArabicPredictor()
letters = {
    'Baa': ('\u0628', lambda img: (cv2.ellipse(img, (200, 220), (50, 20), 0, 0, 180, 30, 6), cv2.circle(img, (200, 275), 6, 30, -1))),
    'Taa': ('\u062a', lambda img: (cv2.ellipse(img, (200, 220), (50, 20), 0, 0, 180, 30, 6), cv2.circle(img, (190, 180), 6, 30, -1), cv2.circle(img, (210, 180), 6, 30, -1))),
    'Thaa': ('\u062b', lambda img: (cv2.ellipse(img, (200, 220), (50, 20), 0, 0, 180, 30, 6), cv2.circle(img, (190, 180), 6, 30, -1), cv2.circle(img, (210, 180), 6, 30, -1), cv2.circle(img, (200, 165), 6, 30, -1))),
    'Jeem': ('\u062c', lambda img: (cv2.line(img, (160, 160), (240, 160), 30, 6), cv2.ellipse(img, (200, 210), (40, 35), 0, -45, 135, 30, 6), cv2.circle(img, (200, 210), 6, 30, -1))),
    'Khaa': ('\u062e', lambda img: (cv2.line(img, (160, 160), (240, 160), 30, 6), cv2.ellipse(img, (200, 210), (40, 35), 0, -45, 135, 30, 6), cv2.circle(img, (200, 135), 6, 30, -1))),
    'Dhaal': ('\u0630', lambda img: (cv2.line(img, (180, 180), (220, 220), 30, 6), cv2.line(img, (220, 220), (240, 220), 30, 6), cv2.circle(img, (185, 160), 6, 30, -1))),
    'Zaay': ('\u0632', lambda img: (cv2.ellipse(img, (200, 190), (30, 45), 30, 0, 90, 30, 6), cv2.circle(img, (200, 165), 6, 30, -1))),
    'Sheen': ('\u0634', lambda img: (cv2.ellipse(img, (200, 220), (45, 20), 0, 0, 180, 30, 6), cv2.circle(img, (190, 180), 5, 30, -1), cv2.circle(img, (210, 180), 5, 30, -1), cv2.circle(img, (200, 165), 5, 30, -1))),
    'Daad': ('\u0636', lambda img: (cv2.ellipse(img, (180, 210), (30, 20), 0, 0, 360, 30, 6), cv2.ellipse(img, (225, 220), (25, 25), 0, 0, 180, 30, 6), cv2.circle(img, (180, 175), 6, 30, -1))),
    'Dhaa': ('\u0638', lambda img: (cv2.ellipse(img, (180, 215), (35, 20), 0, 0, 360, 30, 6), cv2.line(img, (205, 140), (205, 225), 30, 6), cv2.circle(img, (215, 145), 6, 30, -1))),
    'Ghain': ('\u063a', lambda img: (cv2.ellipse(img, (190, 175), (20, 20), 45, 0, 180, 30, 6), cv2.ellipse(img, (200, 220), (35, 35), 0, -45, 135, 30, 6), cv2.circle(img, (190, 140), 6, 30, -1))),
    'Faa': ('\u0641', lambda img: (cv2.circle(img, (175, 190), 15, 30, 6), cv2.ellipse(img, (210, 210), (40, 15), 0, 0, 180, 30, 6), cv2.circle(img, (175, 160), 6, 30, -1))),
    'Qaaf': ('\u0642', lambda img: (cv2.circle(img, (175, 190), 15, 30, 6), cv2.ellipse(img, (210, 215), (35, 25), 0, 0, 180, 30, 6), cv2.circle(img, (170, 160), 5, 30, -1), cv2.circle(img, (185, 160), 5, 30, -1))),
    'Noon': ('\u0646', lambda img: (cv2.ellipse(img, (200, 200), (35, 30), 0, 0, 180, 30, 6), cv2.circle(img, (200, 190), 6, 30, -1))),
    'Yaa': ('\u064a', lambda img: (cv2.ellipse(img, (190, 175), (20, 15), 0, 0, 180, 30, 6), cv2.ellipse(img, (205, 210), (35, 25), 0, 0, 180, 30, 6), cv2.circle(img, (195, 245), 5, 30, -1), cv2.circle(img, (215, 245), 5, 30, -1))),
}

print("=== EVALUATING CANONICAL DOTTED LETTERS (CLEAN vs RULED NOTEBOOK) ===")
for name, (char, draw_fn) in letters.items():
    img_clean = np.full((400, 400), 245, dtype=np.uint8)
    draw_fn(img_clean)
    _, enc_c = cv2.imencode('.jpg', img_clean)
    res_c = predictor.predict_from_bytes(enc_c.tobytes(), request_id=f"test_clean_{name}")

    img_ruled = img_clean.copy()
    cv2.line(img_ruled, (0, 260), (399, 260), 120, 2)
    _, enc_r = cv2.imencode('.jpg', img_ruled)
    res_r = predictor.predict_from_bytes(enc_r.tobytes(), request_id=f"test_ruled_{name}")

    c_match = (res_c["letter"] == char)
    r_match = (res_r["letter"] == char)
    c_p = res_c["letter"].encode("unicode_escape").decode() if res_c["letter"] else "None"
    r_p = res_r["letter"].encode("unicode_escape").decode() if res_r["letter"] else "None"
    c_conf = res_c["confidence"]
    r_conf = res_r["confidence"]
    print(f"{name:6s} | Clean: match={c_match} pred={c_p} conf={c_conf:.2f} | Ruled: match={r_match} pred={r_p} conf={r_conf:.2f}")
