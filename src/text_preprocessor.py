"""
Text preprocessing module using spaCy.
Provides tokenization, lemmatization, stopword removal, and sentence splitting.
"""

import re
from typing import List
import spacy

# Load spaCy English pipeline
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    # Fallback to blank model if needed
    nlp = spacy.blank("en")
    if "sentencizer" not in nlp.pipe_names:
        nlp.add_pipe("sentencizer")


def clean_text_basic(text: str) -> str:
    """
    Normalizes whitespace and removes unwanted control characters
    while preserving technical tokens and punctuation needed for sentence splitting.
    """
    if not text:
        return ""
    # Replace non-breaking spaces and tabs with normal spaces
    cleaned = re.sub(r"[\r\t\f\v]+", " ", text)
    # Normalize multiple newlines to double newlines
    cleaned = re.sub(r"\n\s*\n+", "\n\n", cleaned)
    # Collapse multiple consecutive spaces
    cleaned = re.sub(r"[ ]{2,}", " ", cleaned)
    return cleaned.strip()


def preprocess_for_tfidf(text: str) -> str:
    """
    Prepares text for TF-IDF vectorization:
    - Lowercased
    - Lemmatized
    - Punctuation and stop words removed (except meaningful alphanumeric tokens)
    """
    if not text:
        return ""

    doc = nlp(text)
    tokens = [
        token.lemma_.lower()
        for token in doc
        if not token.is_stop
        and not token.is_punct
        and not token.is_space
        and len(token.text.strip()) > 1
    ]
    return " ".join(tokens)


def split_into_sentences(text: str) -> List[str]:
    """
    Segments text into distinct, readable sentences.
    Preserves original sentence text for user evidence.
    """
    cleaned = clean_text_basic(text)
    if not cleaned:
        return []

    doc = nlp(cleaned)
    sentences = [
        sent.text.strip()
        for sent in doc.sents
        if len(sent.text.strip()) > 15  # Filter out trivial fragments
    ]
    return sentences