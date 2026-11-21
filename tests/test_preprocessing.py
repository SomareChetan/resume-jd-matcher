"""
Unit tests for text preprocessing functions.
"""

from src.text_preprocessor import clean_text_basic, preprocess_for_tfidf, split_into_sentences


def test_clean_text_basic():
    """Normalizes excessive whitespace and weird control characters."""
    raw = "Software    Engineer\n\n\n\r\twith   5 years\tof experience."
    cleaned = clean_text_basic(raw)
    assert "Software Engineer" in cleaned
    assert "  " not in cleaned


def test_preprocess_for_tfidf():
    """Checks lemmatization and removal of common stopwords."""
    text = "Developing machine learning models using Python and PyTorch."
    processed = preprocess_for_tfidf(text)
    # Stop words like 'and', 'using' should be removed
    assert "and" not in processed.split()
    assert "python" in processed
    assert "model" in processed  # lemmatized from 'models'


def test_split_into_sentences():
    """Checks that text is segmented into distinct meaningful sentences."""
    paragraph = (
        "Experienced software developer specializing in Python. "
        "Built distributed REST APIs deployed on AWS. "
        "Passionate about clean code and testing."
    )
    sents = split_into_sentences(paragraph)
    assert len(sents) >= 2
    assert any("REST APIs" in s for s in sents)