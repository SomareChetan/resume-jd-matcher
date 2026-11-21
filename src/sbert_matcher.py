"""
Sentence-BERT semantic similarity module using Sentence-Transformers.
Uses pretrained all-MiniLM-L6-v2 for dense contextual embeddings.
"""

from typing import Tuple
from sentence_transformers import SentenceTransformer, util
import numpy as np

# Cache model instance to avoid reloading into memory on every request
_MODEL_CACHE = None


def get_sbert_model(model_name: str = "all-MiniLM-L6-v2") -> SentenceTransformer:
    """
    Loads or returns the cached SentenceTransformer model.
    Downloads on first use; cached locally afterwards.
    """
    global _MODEL_CACHE
    if _MODEL_CACHE is None:
        _MODEL_CACHE = SentenceTransformer(model_name)
    return _MODEL_CACHE


def compute_semantic_similarity(resume_text: str, jd_text: str) -> Tuple[float, str]:
    """
    Computes overall semantic cosine similarity between the full resume and JD.

    Args:
        resume_text (str): Cleaned raw text of resume.
        jd_text (str): Cleaned raw text of job description.

    Returns:
        Tuple[float, str]: (similarity_score_between_0_and_1, error_message)
    """
    if not resume_text.strip() or not jd_text.strip():
        return 0.0, "One or both input texts are empty."

    try:
        model = get_sbert_model()

        # Generate embeddings
        resume_embedding = model.encode(resume_text, convert_to_tensor=True, show_progress_bar=False)
        jd_embedding = model.encode(jd_text, convert_to_tensor=True, show_progress_bar=False)

        # Calculate cosine similarity
        cosine_score = util.cos_sim(resume_embedding, jd_embedding).item()
        
        # Ensure score stays in 0.0 to 1.0 range
        score = max(0.0, min(1.0, float(cosine_score)))
        return round(score, 4), ""

    except Exception as e:
        return 0.0, f"Error computing semantic embeddings: {str(e)}"