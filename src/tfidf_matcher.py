"""
TF-IDF baseline similarity module using scikit-learn.
Computes token/keyword frequency overlap using cosine similarity.
"""

from typing import Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def compute_tfidf_similarity(resume_preprocessed: str, jd_preprocessed: str) -> Tuple[float, list]:
    """
    Calculates TF-IDF cosine similarity between preprocessed resume and JD text.

    Args:
        resume_preprocessed (str): Lemmatized, stopword-free resume text.
        jd_preprocessed (str): Lemmatized, stopword-free job description text.

    Returns:
        Tuple[float, list]: (similarity_score_between_0_and_1, top_shared_keywords)
    """
    if not resume_preprocessed.strip() or not jd_preprocessed.strip():
        return 0.0, []

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
    tfidf_matrix = vectorizer.fit_transform([resume_preprocessed, jd_preprocessed])

    # Cosine similarity between vector 0 (resume) and vector 1 (JD)
    sim_matrix = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
    score = float(sim_matrix[0][0])

    # Identify shared terms with non-zero TF-IDF in both docs
    feature_names = vectorizer.get_feature_names_out()
    r_vec = tfidf_matrix[0].toarray()[0]
    j_vec = tfidf_matrix[1].toarray()[0]

    shared_terms = []
    for idx, (r_val, j_val) in enumerate(zip(r_vec, j_vec)):
        if r_val > 0 and j_val > 0:
            shared_terms.append((feature_names[idx], float(r_val * j_val)))

    # Sort shared terms by combined importance
    shared_terms.sort(key=lambda x: x[1], reverse=True)
    top_terms = [term for term, _ in shared_terms[:15]]

    return round(score, 4), top_terms