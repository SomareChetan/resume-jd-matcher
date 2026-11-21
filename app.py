"""
Resume–Job Description Matcher
Clean, functional user interface for text and semantic analysis.
"""

import streamlit as st
from src.pdf_parser import extract_text_from_pdf
from src.text_preprocessor import clean_text_basic, preprocess_for_tfidf
from src.skill_extractor import SkillExtractor
from src.tfidf_matcher import compute_tfidf_similarity
from src.sbert_matcher import compute_semantic_similarity
from src.evidence_matcher import extract_top_evidence_sentences
from src.utils import create_skill_coverage_chart, format_score_badge

# Page Configuration
st.set_page_config(
    page_title="Resume–Job Description Matcher",
    layout="wide",
    initial_sidebar_state="collapsed"
)


@st.cache_resource
def get_cached_skill_extractor():
    """Cache the skill extractor to load vocabulary once."""
    return SkillExtractor()


def main():
    # Header Section
    st.title("Resume – Job Description Matcher")
    st.markdown(
        "Quantitative evaluation of textual overlap, dense semantic alignment, "
        "and skill coverage between candidate resumes and job descriptions."
    )
    st.divider()

    # Input Section
    col_input_left, col_input_right = st.columns(2, gap="large")

    with col_input_left:
        st.subheader("1. Candidate Resume")
        uploaded_file = st.file_uploader(
            "Upload resume (PDF format)",
            type=["pdf"],
            help="Select a machine-readable text PDF resume."
        )

    with col_input_right:
        st.subheader("2. Job Description")
        jd_text_input = st.text_area(
            "Target job description text:",
            height=200,
            placeholder="Paste technical requirements, responsibilities, and qualifications..."
        )

    analyze_btn = st.button("Run Alignment Analysis", type="primary", use_container_width=True)

    # Execution Flow
    if analyze_btn:
        if uploaded_file is None:
            st.error("Error: Please upload a PDF resume before proceeding.")
            return

        if not jd_text_input.strip():
            st.error("Error: Please paste a job description before proceeding.")
            return

        with st.spinner("Extracting text and computing NLP representations..."):
            # 1. PDF Extraction
            file_bytes = uploaded_file.read()
            raw_resume_text, parse_error = extract_text_from_pdf(file_bytes)

            if parse_error:
                st.error(f"PDF Extraction Error: {parse_error}")
                return

            # 2. Text Preprocessing
            cleaned_resume = clean_text_basic(raw_resume_text)
            cleaned_jd = clean_text_basic(jd_text_input)

            tfidf_resume = preprocess_for_tfidf(cleaned_resume)
            tfidf_jd = preprocess_for_tfidf(cleaned_jd)

            # 3. Skill Extraction
            extractor = get_cached_skill_extractor()
            resume_skill_data = extractor.extract_skills(cleaned_resume)
            jd_skill_data = extractor.extract_skills(cleaned_jd)

            skill_comparison = SkillExtractor.compare_skills(
                resume_skill_data["all"],
                jd_skill_data["all"]
            )

            # 4. Model Similarities
            tfidf_score, _ = compute_tfidf_similarity(tfidf_resume, tfidf_jd)
            semantic_score, sbert_error = compute_semantic_similarity(cleaned_resume, cleaned_jd)

            if sbert_error:
                st.warning(f"Sentence-BERT Notice: {sbert_error}")

            # 5. Evidence Extraction
            evidence_sentences = extract_top_evidence_sentences(cleaned_resume, cleaned_jd, top_k=4)

        # Results Display
        st.divider()
        st.subheader("Analysis Metrics")

        # Metric Cards
        metric_col1, metric_col2, metric_col3 = st.columns(3)
        with metric_col1:
            st.metric(
                label="TF-IDF Keyword Similarity",
                value=format_score_badge(tfidf_score),
                help="Sparse n-gram vector cosine similarity."
            )
        with metric_col2:
            st.metric(
                label="Sentence-BERT Semantic Similarity",
                value=format_score_badge(semantic_score),
                help="Dense embedding cosine similarity via all-MiniLM-L6-v2."
            )
        with metric_col3:
            matched_len = len(skill_comparison["matched_skills"])
            jd_total_len = len(jd_skill_data["all"])
            coverage_pct = round((matched_len / jd_total_len) * 100, 1) if jd_total_len > 0 else 0
            st.metric(
                label="Job Skill Recall / Coverage",
                value=f"{coverage_pct}%",
                delta=f"{matched_len} of {jd_total_len} JD Skills Detected"
            )

        st.write("")

        # Skill Breakdown and Visual Coverage
        skill_col_left, skill_col_right = st.columns([1.4, 1.0], gap="medium")

        with skill_col_left:
            st.markdown("### Skill Extraction Summary")

            # Matched Skills
            st.markdown("**Matched Skills (Present in both Resume and JD):**")
            if skill_comparison["matched_skills"]:
                matched_badges = " ".join([f"`{s}`" for s in skill_comparison["matched_skills"]])
                st.markdown(matched_badges)
            else:
                st.info("No overlapping vocabulary skills detected.")

            st.write("")
            # Missing Skills
            st.markdown("**Skills Mentioned in JD Missing from Resume:**")
            if skill_comparison["missing_skills"]:
                missing_badges = " ".join([f"`{s}`" for s in skill_comparison["missing_skills"]])
                st.markdown(missing_badges)
            else:
                st.success("All identified JD skills were present in the resume.")

            st.write("")
            # Additional Skills
            st.markdown("**Additional Candidate Skills:**")
            if skill_comparison["additional_skills"]:
                other_badges = " ".join([f"`{s}`" for s in skill_comparison["additional_skills"][:12]])
                st.markdown(other_badges)
            else:
                st.caption("No additional vocabulary skills detected.")

        with skill_col_right:
            st.markdown("### Skill Coverage Ratio")
            if jd_total_len > 0:
                fig = create_skill_coverage_chart(
                    len(skill_comparison["matched_skills"]),
                    len(skill_comparison["missing_skills"])
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No technical skills were identified in the Job Description to plot.")

        # Evidence Section
        st.divider()
        st.markdown("### Evidence Sentences")
        st.caption("Candidate resume excerpts with highest semantic cosine relevance to the job requirements:")

        if evidence_sentences:
            for item in evidence_sentences:
                with st.expander(f"Rank {item['rank']} — Cosine Relevance: {item['similarity']:.3f}", expanded=True):
                    st.markdown(f"> *\"{item['sentence']}\"*")
        else:
            st.info("No strong sentence matches found above the relevance threshold.")

        # Raw Extracted Text Expander
        st.divider()
        with st.expander("Extracted Resume Raw Text"):
            st.text_area("Parsed Text Content", raw_resume_text, height=200, disabled=True)

        # Footer Disclaimer
        st.caption(
            "Note: This tool calculates textual and semantic similarity metrics for research and demonstration "
            "purposes. It does not measure candidate competency or automate hiring decisions."
        )


if __name__ == "__main__":
    main()