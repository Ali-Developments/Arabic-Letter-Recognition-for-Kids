"""
Unit tests for Arabic 28-class label definitions and mappings.
"""

import pytest
from app.ml.labels import (
    ARABIC_LETTERS,
    CHAR_TO_ID,
    ID_TO_CHAR,
    ID_TO_METADATA,
    LETTER_METADATA,
    NUM_CLASSES,
    get_class_id,
    get_letter,
    normalize_arabic_letter,
)


def test_num_classes_is_28():
    assert NUM_CLASSES == 28
    assert len(ARABIC_LETTERS) == 28
    assert len(LETTER_METADATA) == 28
    assert len(CHAR_TO_ID) == 28
    assert len(ID_TO_CHAR) == 28


def test_canonical_alphabet_ordering():
    expected_first_five = ["ا", "ب", "ت", "ث", "ج"]
    expected_last_five = ["م", "ن", "ه", "و", "ي"]
    assert ARABIC_LETTERS[:5] == expected_first_five
    assert ARABIC_LETTERS[-5:] == expected_last_five


def test_bidirectional_mapping():
    for idx, char in enumerate(ARABIC_LETTERS):
        assert ID_TO_CHAR[idx] == char
        assert CHAR_TO_ID[char] == idx
        assert get_letter(idx) == char
        assert get_class_id(char) == idx


def test_arabic_letter_normalization():
    # Alef variants should normalize to plain Alef
    assert normalize_arabic_letter("أ") == "ا"
    assert normalize_arabic_letter("إ") == "ا"
    assert normalize_arabic_letter("آ") == "ا"
    # Taa marbuta and alif maqsura
    assert normalize_arabic_letter("ة") == "ه"
    assert normalize_arabic_letter("ى") == "ي"


def test_invalid_class_id():
    with pytest.raises(ValueError):
        get_letter(28)
    with pytest.raises(ValueError):
        get_letter(-1)
