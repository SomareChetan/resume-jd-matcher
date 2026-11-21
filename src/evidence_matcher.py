"""
Evidence matcher module for explainable recommendations.
Finds specific sentences in the candidate's resume that are semantically
closest to the job requirements.
"""

from typing import Dict, List
from sentence_transformers import util
from src.sbert_matcher import get_sbert_model
from src.text_preprocessor import split_into_sentences


def extract_top_evidence_sentences(
    resume_text: str,
    jd_text: str,
    top_k: int = 5
) -> List[Dict[str, any]]:
    """
    Extracts the top-K resume sentences semantically most relevant to the JD.

    Returns:
        List of dicts: [
            {"sentence": str, "similarity": float, "rank": int},
            ...
        ]
    """
    sentences = split_into_sentences(resume_text)
    if not sentences or not jd_text.strip():
        return []

    model = get_sbert_model()

    # Encode JD text and all resume sentences
    jd_embedding = model.encode(jd_text, convert_to_tensor=True, show_progress_bar=False)
    sentence_embeddings = model.encode(sentences, convert_to_tensor=True, show_progress_bar=False)

    # Compute cosine similarity between JD and every sentence
    cosine_scores = util.cos_sim(sentence_embeddings, jd_embedding).squeeze(-1).tolist()

    if isinstance(cosine_scores, float):
        cosine_scores = [cosine_scores]

    # Pair each sentence with its score
    scored_sentences = [
        {"sentence": sent, "similarity": round(max(0.0, float(score)), 3)}
        for sent, score in zip(sentences, cosine_scores)
    ]

    # Sort descending by similarity score
    scored_sentences.sort(key=lambda x: x["similarity"], reverse=True)

    # Return top K unique evidence sentences
    results = []
    seen = set()
    for item in scored_sentences:
        if item["sentence"].lower() not in seen and item["similarity"] > 0.15:
            seen.add(item["sentence"].lower())
            results.append({
                "rank": len(results) + 1,
                "sentence": item["sentence"],
                "similarity": item["similarity"]
            })
        if len(results) >= top_k:
            break

    return results
    