"""
Arabic Alphabet Label Mapping (28 Classes)
Primary specification based on ARABIC_HANDWRITING_AI_PROJECT_BRIEF.md.
"""

from typing import Dict, List, Optional, Tuple

# 28 Standard Arabic Letters in canonical alphabetical order
ARABIC_LETTERS: List[str] = [
    "ا",  # 0: Alef
    "ب",  # 1: Baa
    "ت",  # 2: Taa
    "ث",  # 3: Thaa
    "ج",  # 4: Jeem
    "ح",  # 5: Haa
    "خ",  # 6: Khaa
    "د",  # 7: Daal
    "ذ",  # 8: Dhaal
    "ر",  # 9: Raa
    "ز",  # 10: Zaay
    "س",  # 11: Seen
    "ش",  # 12: Sheen
    "ص",  # 13: Saad
    "ض",  # 14: Daad
    "ط",  # 15: Taa (Emphatic)
    "ظ",  # 16: Dhaa (Emphatic)
    "ع",  # 17: Ain
    "غ",  # 18: Ghain
    "ف",  # 19: Faa
    "ق",  # 20: Qaaf
    "ك",  # 21: Kaaf
    "ل",  # 22: Laam
    "م",  # 23: Meem
    "ن",  # 24: Noon
    "ه",  # 25: Haa
    "و",  # 26: Waaw
    "ي",  # 27: Yaa
]

LETTER_METADATA: List[Dict[str, any]] = [
    {"class_id": 0, "letter": "ا", "name_en": "alef", "name_ar": "ألف", "unicode": "U+0627", "dots": 0},
    {"class_id": 1, "letter": "ب", "name_en": "baa", "name_ar": "باء", "unicode": "U+0628", "dots": 1, "dot_pos": "below"},
    {"class_id": 2, "letter": "ت", "name_en": "taa", "name_ar": "تاء", "unicode": "U+062A", "dots": 2, "dot_pos": "above"},
    {"class_id": 3, "letter": "ث", "name_en": "thaa", "name_ar": "ثاء", "unicode": "U+062B", "dots": 3, "dot_pos": "above"},
    {"class_id": 4, "letter": "ج", "name_en": "jeem", "name_ar": "جيم", "unicode": "U+062C", "dots": 1, "dot_pos": "inside"},
    {"class_id": 5, "letter": "ح", "name_en": "haa", "name_ar": "حاء", "unicode": "U+062D", "dots": 0},
    {"class_id": 6, "letter": "خ", "name_en": "khaa", "name_ar": "خاء", "unicode": "U+062E", "dots": 1, "dot_pos": "above"},
    {"class_id": 7, "letter": "د", "name_en": "daal", "name_ar": "دال", "unicode": "U+062F", "dots": 0},
    {"class_id": 8, "letter": "ذ", "name_en": "dhaal", "name_ar": "ذال", "unicode": "U+0630", "dots": 1, "dot_pos": "above"},
    {"class_id": 9, "letter": "ر", "name_en": "raa", "name_ar": "راء", "unicode": "U+0631", "dots": 0},
    {"class_id": 10, "letter": "ز", "name_en": "zaay", "name_ar": "زاي", "unicode": "U+0632", "dots": 1, "dot_pos": "above"},
    {"class_id": 11, "letter": "س", "name_en": "seen", "name_ar": "سين", "unicode": "U+0633", "dots": 0},
    {"class_id": 12, "letter": "ش", "name_en": "sheen", "name_ar": "شين", "unicode": "U+0634", "dots": 3, "dot_pos": "above"},
    {"class_id": 13, "letter": "ص", "name_en": "saad", "name_ar": "صاد", "unicode": "U+0635", "dots": 0},
    {"class_id": 14, "letter": "ض", "name_en": "daad", "name_ar": "ضاد", "unicode": "U+0636", "dots": 1, "dot_pos": "above"},
    {"class_id": 15, "letter": "ط", "name_en": "tah", "name_ar": "طاء", "unicode": "U+0637", "dots": 0},
    {"class_id": 16, "letter": "ظ", "name_en": "dhah", "name_ar": "ظاء", "unicode": "U+0638", "dots": 1, "dot_pos": "above"},
    {"class_id": 17, "letter": "ع", "name_en": "ain", "name_ar": "عين", "unicode": "U+0639", "dots": 0},
    {"class_id": 18, "letter": "غ", "name_en": "ghain", "name_ar": "غين", "unicode": "U+063A", "dots": 1, "dot_pos": "above"},
    {"class_id": 19, "letter": "ف", "name_en": "faa", "name_ar": "فاء", "unicode": "U+0641", "dots": 1, "dot_pos": "above"},
    {"class_id": 20, "letter": "ق", "name_en": "qaaf", "name_ar": "قاف", "unicode": "U+0642", "dots": 2, "dot_pos": "above"},
    {"class_id": 21, "letter": "ك", "name_en": "kaaf", "name_ar": "كاف", "unicode": "U+0643", "dots": 0},
    {"class_id": 22, "letter": "ل", "name_en": "laam", "name_ar": "لام", "unicode": "U+0644", "dots": 0},
    {"class_id": 23, "letter": "م", "name_en": "meem", "name_ar": "ميم", "unicode": "U+0645", "dots": 0},
    {"class_id": 24, "letter": "ن", "name_en": "noon", "name_ar": "نون", "unicode": "U+0646", "dots": 1, "dot_pos": "above/inside"},
    {"class_id": 25, "letter": "ه", "name_en": "haa_end", "name_ar": "هاء", "unicode": "U+0647", "dots": 0},
    {"class_id": 26, "letter": "و", "name_en": "waaw", "name_ar": "واو", "unicode": "U+0648", "dots": 0},
    {"class_id": 27, "letter": "ي", "name_en": "yaa", "name_ar": "ياء", "unicode": "U+064A", "dots": 2, "dot_pos": "below"},
]

NUM_CLASSES: int = len(ARABIC_LETTERS)
assert NUM_CLASSES == 28, f"Expected 28 classes, got {NUM_CLASSES}"

CHAR_TO_ID: Dict[str, int] = {item["letter"]: item["class_id"] for item in LETTER_METADATA}
ID_TO_CHAR: Dict[int, str] = {item["class_id"]: item["letter"] for item in LETTER_METADATA}
ID_TO_METADATA: Dict[int, Dict[str, any]] = {item["class_id"]: item for item in LETTER_METADATA}


def get_letter(class_id: int) -> str:
    if class_id not in ID_TO_CHAR:
        raise ValueError(f"Invalid class_id: {class_id}. Must be between 0 and 27.")
    return ID_TO_CHAR[class_id]


def get_class_id(letter: str) -> int:
    # Normalize common variants
    normalized = normalize_arabic_letter(letter)
    if normalized not in CHAR_TO_ID:
        raise ValueError(f"Unknown Arabic letter: '{letter}'. Expected one of 28 isolated characters.")
    return CHAR_TO_ID[normalized]


def normalize_arabic_letter(char: str) -> str:
    """Normalize input letter representation (e.g. Alef with hamza -> plain Alef if given)."""
    char = char.strip()
    # Normalize alef variants to bare alef for isolated letter classification
    if char in ["أ", "إ", "آ", "ٱ"]:
        return "ا"
    # Normalize taa marbuta to haa or taa
    if char == "ة":
        return "ه"
    # Normalize alif maqsura to yaa
    if char == "ى":
        return "ي"
    return char
