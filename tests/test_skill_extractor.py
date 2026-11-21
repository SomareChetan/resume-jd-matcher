"""
Unit tests for skill extraction and comparison.
"""

from src.skill_extractor import SkillExtractor


def test_skill_extraction_and_alias():
    """Extracts known skills and normalizes aliases."""
    extractor = SkillExtractor()
    sample_text = "Proficient in Python, sklearn, Docker, and PostgreSQL."
    result = extractor.extract_skills(sample_text)
    
    # Should detect skills
    assert "python" in result["all"]
    assert "docker" in result["all"]
    assert "postgresql" in result["all"]

    # Alias normalization: 'sklearn' should normalize to 'scikit-learn'
    assert "scikit-learn" in result["all"]


def test_compare_skills():
    """Calculates matched, missing, and additional skill sets."""
    resume_skills = {"python", "docker", "sql"}
    jd_skills = {"python", "kubernetes", "sql"}

    diff = SkillExtractor.compare_skills(resume_skills, jd_skills)

    assert set(diff["matched_skills"]) == {"python", "sql"}
    assert set(diff["missing_skills"]) == {"kubernetes"}
    assert set(diff["additional_skills"]) == {"docker"}
    