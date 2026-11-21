"""
Unit tests for PDF extraction module.
"""

from src.pdf_parser import extract_text_from_pdf


def test_empty_bytes_handling():
    """Empty byte input should return an error message, not crash."""
    text, err = extract_text_from_pdf(b"")
    assert text == ""
    assert "empty" in err.lower()


def test_corrupted_pdf_bytes():
    """Corrupted / non-PDF data should be handled gracefully."""
    garbage_bytes = b"This is not a real PDF file."
    text, err = extract_text_from_pdf(garbage_bytes)
    assert text == ""
    assert len(err) > 0