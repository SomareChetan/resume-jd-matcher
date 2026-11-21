"""
Benchmark Evaluation Runner with Precision, Recall & F1
======================================================
Runs TF-IDF and Sentence-BERT against test pairs, computes
similarity scores, and calculates standard Information Retrieval
classification metrics: Accuracy, Precision, Recall, and F1-Score.
"""

import json
import sys
from pathlib import Path
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.text_preprocessor import clean_text_basic, preprocess_for_tfidf
from src.skill_extractor import SkillExtractor
from src.tfidf_matcher import compute_tfidf_similarity
from src.sbert_matcher import compute_semantic_similarity, get_sbert_model


def load_pairs():
    """Loads benchmark pairs from data directory."""
    hf_path = PROJECT_ROOT / "data" / "hf_benchmark_pairs.json"
    local_path = PROJECT_ROOT / "data" / "benchmark_pairs.json"

    if hf_path.exists():
        print(f"📄 Dataset: Hugging Face (nonameee12233/job-resume-matching)", flush=True)
        with open(hf_path, "r", encoding="utf-8") as f:
            return json.load(f)
    elif local_path.exists():
        print(f"📄 Dataset: Curated Domain Pairs", flush=True)
        with open(local_path, "r", encoding="utf-8") as f:
            return json.load(f)
    else:
        print("❌ No benchmark JSON found in data/ directory.", flush=True)
        sys.exit(1)


def calculate_ir_metrics(y_true, y_pred, model_name: str):
    """Calculates Accuracy, Precision, Recall, and F1-Score safely."""
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    print(f"\n📊 {model_name} Classification Metrics:")
    print(f"  • Accuracy        : {acc * 100:.1f}%")
    print(f"  • Precision       : {prec:.3f}  ({prec * 100:.1f}%)  [TP / (TP + FP)]")
    print(f"  • Recall          : {rec:.3f}  ({rec * 100:.1f}%)  [TP / (TP + FN)]")
    print(f"  • F1-Score        : {f1:.3f}  ({f1 * 100:.1f}%)  [Harmonic mean]")
    print(f"  • Breakdown       : TP={tp}, FP={fp}, TN={tn}, FN={fn}")


def main():
    print("\n⏳ Initializing NLP models and loading S-BERT weights into memory...", flush=True)
    get_sbert_model()
    extractor = SkillExtractor()
    pairs = load_pairs()

    print("\n" + "=" * 90, flush=True)
    print("  MODEL BENCHMARK EVALUATION RESULTS", flush=True)
    print("=" * 90, flush=True)

    header = f"{'ID':<8} {'Pair Name':<38} {'Expected':<10} {'TF-IDF':<10} {'S-BERT':<10} {'Skills':<8}"
    print(header, flush=True)
    print("-" * 90, flush=True)

    y_true = []        # Ground truth: 1 for Match, 0 for Non-match
    y_pred_sbert = []  # S-BERT prediction: 1 if score >= 0.48 else 0
    y_pred_tfidf = []  # TF-IDF prediction: 1 if score >= 0.05 else 0
    skill_recalls = []

    for idx, item in enumerate(pairs, 1):
        print(f"⏳ Evaluating [{idx}/{len(pairs)}] {item['id']}...", end="\r", flush=True)

        clean_resume = clean_text_basic(item["resume"])
        clean_jd = clean_text_basic(item["job_description"])

        tfidf_resume = preprocess_for_tfidf(clean_resume)
        tfidf_jd = preprocess_for_tfidf(clean_jd)

        tfidf_score, _ = compute_tfidf_similarity(tfidf_resume, tfidf_jd)
        sbert_score, _ = compute_semantic_similarity(clean_resume, clean_jd)

        # Skill Recall (Matched / Total JD Skills)
        r_skills = extractor.extract_skills(clean_resume)["all"]
        j_skills = extractor.extract_skills(clean_jd)["all"]
        comp = SkillExtractor.compare_skills(r_skills, j_skills)
        coverage = round((len(comp["matched_skills"]) / len(j_skills) * 100), 1) if j_skills else 0.0
        skill_recalls.append(coverage)

        # Display row
        lbl = item["label"][:36]
        row_str = f"{item['id']:<8} {lbl:<38} {item['expected']:<10} {tfidf_score:<10.3f} {sbert_score:<10.3f} {coverage:<8.1f}%"
        print(f"{row_str:<90}", flush=True)

        # Ground truth classification (HIGH or MEDIUM = 1, LOW or NONE = 0)
        is_match_ground_truth = 1 if item["expected"].upper() in ("HIGH", "MEDIUM") else 0
        y_true.append(is_match_ground_truth)

        # Binary predictions based on model thresholds
        y_pred_sbert.append(1 if sbert_score >= 0.48 else 0)
        y_pred_tfidf.append(1 if tfidf_score >= 0.05 else 0)

    print("-" * 90, flush=True)

    # 1. Print S-BERT classification metrics
    calculate_ir_metrics(y_true, y_pred_sbert, "Sentence-BERT (Threshold = 0.48)")

    # 2. Print TF-IDF classification metrics
    calculate_ir_metrics(y_true, y_pred_tfidf, "TF-IDF Baseline (Threshold = 0.05)")

    # 3. Print Skill Extraction Recall
    avg_skill_recall = sum(skill_recalls) / len(skill_recalls) if skill_recalls else 0.0
    print("\n🛠️ Skill Extraction Performance:")
    print(f"  • Average Skill Recall / Coverage : {avg_skill_recall:.1f}%")
    print("=" * 90 + "\n", flush=True)

if __name__ == "__main__":
    main()