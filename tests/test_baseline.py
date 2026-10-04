"""
Fast, local unit tests that don't require Docker to be running -- these test
pure logic (chunking math, feature engineering math), not the live services.
For end-to-end validation against the live stack, use
scripts/validate_baseline.py instead (that one needs Docker up).

Run with:
    pytest tests/ -v
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from api.pipeline.chunking import chunk_text  # noqa: E402


def test_chunk_text_basic_split():
    text = " ".join(f"word{i}" for i in range(500))
    chunks = chunk_text(text, doc_id="doc1", chunk_size_words=100, chunk_overlap_words=20)

    assert len(chunks) > 1
    assert chunks[0]["chunk_id"] == "doc1::chunk0"
    assert chunks[0]["position"] == 0
    # first chunk should have exactly 100 words
    assert len(chunks[0]["text"].split()) == 100


def test_chunk_text_overlap_preserved():
    text = " ".join(f"word{i}" for i in range(300))
    chunks = chunk_text(text, doc_id="doc1", chunk_size_words=100, chunk_overlap_words=20)

    first_words = chunks[0]["text"].split()
    second_words = chunks[1]["text"].split()
    # last 20 words of chunk 0 should equal first 20 words of chunk 1
    assert first_words[-20:] == second_words[:20]


def test_chunk_text_empty_string():
    assert chunk_text("", doc_id="doc1") == []


def test_chunk_text_shorter_than_chunk_size():
    text = "just a few words here"
    chunks = chunk_text(text, doc_id="doc1", chunk_size_words=100, chunk_overlap_words=20)
    assert len(chunks) == 1
    assert chunks[0]["text"] == text


def test_chunk_text_rejects_overlap_ge_size():
    import pytest
    with pytest.raises(ValueError):
        chunk_text("some text", doc_id="doc1", chunk_size_words=50, chunk_overlap_words=50)


def test_chunk_id_format_is_stable():
    text = " ".join(f"word{i}" for i in range(50))
    chunks = chunk_text(text, doc_id="CVE-2021-44228")
    assert chunks[0]["chunk_id"] == "CVE-2021-44228::chunk0"


def test_text_stat_features_shape():
    from defense4_xgboost.feature_engineering import text_stat_features
    feats = text_stat_features("Hello world, this is a test sentence with numb3rs 123.")
    expected_keys = {
        "text_length_chars", "text_length_words", "avg_word_length",
        "char_entropy", "digit_ratio", "punctuation_ratio",
    }
    assert set(feats.keys()) == expected_keys
    assert feats["text_length_words"] == 10


def test_smote_falls_back_on_tiny_minority():
    import pandas as pd
    from defense4_xgboost.smote_utils import apply_smote

    X = pd.DataFrame({"f1": range(20), "f2": range(20)})
    y = pd.Series([0] * 19 + [1])  # only 1 minority sample

    X_res, y_res = apply_smote(X, y)
    # should skip SMOTE and return input unchanged
    assert len(X_res) == 20
    assert y_res.sum() == 1
