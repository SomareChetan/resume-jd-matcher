"""
Hugging Face Dataset Downloader & Sampler
=========================================
Downloads cv.csv (185MB), job.csv (31MB), and matches.csv (18MB) from
https://huggingface.co/datasets/nonameee12233/job-resume-matching
directly into the data/ directory, then samples balanced test pairs.
"""

import json
import urllib.request
from pathlib import Path
import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

BASE_URL = "https://huggingface.co/datasets/nonameee12233/job-resume-matching/resolve/main"

FILES = [
    ("matches.csv", "18.8 MB"),
    ("job.csv", "31.4 MB"),
    ("cv.csv", "185.0 MB")
]


def download_file_with_progress(filename: str, label: str):
    """Downloads a file with an interactive terminal progress bar."""
    dest_path = DATA_DIR / filename
    if dest_path.exists():
        print(f"  ✓ {filename} already exists in data/ (skipping download).")
        return dest_path

    url = f"{BASE_URL}/{filename}"
    print(f"\n⬇️ Downloading {filename} ({label}) into data/ ...")

    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as response:
        total_size = int(response.headers.get("content-length", 0))
        block_size = 1024 * 1024  # 1 MB chunk

        with open(dest_path, "wb") as f, tqdm(
            total=total_size, unit="B", unit_scale=True, desc=filename
        ) as bar:
            while True:
                buffer = response.read(block_size)
                if not buffer:
                    break
                f.write(buffer)
                bar.update(len(buffer))

    print(f"  ✓ Saved to: {dest_path}")
    return dest_path


def main():
    print("=" * 70)
    print("  HUGGING FACE DATASET DOWNLOADER (nonameee12233/job-resume-matching)")
    print("=" * 70)

    # 1. Download the three dataset CSV files into data/
    for filename, label in FILES:
        download_file_with_progress(filename, label)

    print("\n🔄 Processing and merging dataset pairs...")

    matches_path = DATA_DIR / "matches.csv"
    job_path = DATA_DIR / "job.csv"
    cv_path = DATA_DIR / "cv.csv"

    # 2. Read matches table
    print("  -> Reading matches table...")
    matches_df = pd.read_csv(matches_path)

    # Sample top 4 highest matches, 4 medium matches, and 4 lowest matches
    high_matches = matches_df[matches_df["final_score"] >= 0.70].head(4)
    med_matches = matches_df[(matches_df["final_score"] >= 0.40) & (matches_df["final_score"] < 0.60)].head(4)
    low_matches = matches_df[matches_df["final_score"] < 0.30].head(4)

    sampled_matches = pd.concat([high_matches, med_matches, low_matches])
    target_cands = set(sampled_matches["candidate_id"].unique())
    target_jobs = set(sampled_matches["job_id"].unique())

    # 3. Read jobs table
    print("  -> Reading job postings...")
    job_df = pd.read_csv(job_path)
    job_df = job_df[job_df["job_id"].isin(target_jobs)]

    # 4. Read CV table (only the needed candidate IDs)
    print("  -> Reading candidate resumes...")
    cv_chunks = []
    # Read cv.csv in chunks to minimize memory usage
    for chunk in pd.read_csv(cv_path, chunksize=1000):
        matched_chunk = chunk[chunk["candidate_id"].isin(target_cands)]
        if not matched_chunk.empty:
            cv_chunks.append(matched_chunk)
            if len(set().union(*[set(c["candidate_id"]) for c in cv_chunks])) >= len(target_cands):
                break

    cv_df = pd.concat(cv_chunks, ignore_index=True) if cv_chunks else pd.DataFrame()

    # 5. Merge tables into test pairs
    merged = sampled_matches.merge(
        cv_df[["candidate_id", "target_position", "resume_text"]],
        on="candidate_id",
        how="inner"
    ).merge(
        job_df[["job_id", "job_title", "job_description"]],
        on="job_id",
        how="inner"
    )

    pairs = []
    for idx, row in merged.iterrows():
        gt_score = float(row["final_score"])
        expected = "HIGH" if gt_score >= 0.65 else ("MEDIUM" if gt_score >= 0.40 else "LOW")

        pairs.append({
            "id": f"HF-{len(pairs)+1:02d}",
            "label": f"{row['target_position']} vs {row['job_title']}",
            "category": f"Ground Truth Score: {gt_score:.2f}",
            "resume": str(row["resume_text"]),
            "job_description": str(row["job_description"]),
            "expected": expected,
            "ground_truth_score": round(gt_score, 4)
        })

    output_path = DATA_DIR / "hf_benchmark_pairs.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(pairs, f, indent=2)

    print("\n" + "=" * 70)
    print(f"🎉 Success! Generated {len(pairs)} balanced pairs.")
    print(f"📁 Raw files saved in: {DATA_DIR}")
    print(f"📄 Clean benchmark file: {output_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()