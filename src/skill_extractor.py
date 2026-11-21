"""
Skill extraction module using spaCy PhraseMatcher and vocabulary dictionary.
Extracts known technical skills, categorizes them, and identifies overlaps.
"""

import json
from pathlib import Path
from typing import Dict, List, Set
import spacy
from spacy.matcher import PhraseMatcher

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    nlp = spacy.blank("en")

# Canonical aliases dictionary
SKILL_ALIASES = {
    "sklearn": "scikit-learn",
    "node.js": "nodejs",
    "postgres": "postgresql",
    "gcp": "google cloud",
    "nlp": "natural language processing",
    "oop": "object-oriented programming",
    "ci/cd": "ci/cd"
}


def load_skills_vocabulary(json_path: Path = None) -> Dict[str, List[str]]:
    """Loads categorized skills from data/skills.json."""
    if json_path is None:
        json_path = Path(__file__).resolve().parent.parent / "data" / "skills.json"

    if not json_path.exists():
        return {}

    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


class SkillExtractor:
    def __init__(self, vocabulary_path: Path = None):
        self.categories_vocab = load_skills_vocabulary(vocabulary_path)
        self.matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
        self._build_matcher()

    def _build_matcher(self):
        for category, skills in self.categories_vocab.items():
            patterns = [nlp.make_doc(skill) for skill in skills]
            self.matcher.add(category, patterns)

    def extract_skills(self, text: str) -> Dict[str, Set[str]]:
        """
        Extracts skills from text.

        Returns:
            Dict with:
                'all': set of canonical skill names found
                'by_category': dict of category -> set of skills
        """
        if not text:
            return {"all": set(), "by_category": {}}

        doc = nlp(text)
        matches = self.matcher(doc)

        all_skills = set()
        by_category: Dict[str, Set[str]] = {}

        for match_id, start, end in matches:
            rule_id = nlp.vocab.strings[match_id]
            span_text = doc[start:end].text.strip().lower()

            canonical = SKILL_ALIASES.get(span_text, span_text)
            all_skills.add(canonical)

            if rule_id not in by_category:
                by_category[rule_id] = set()
            by_category[rule_id].add(canonical)

        return {"all": all_skills, "by_category": by_category}

    @staticmethod
    def compare_skills(resume_skills: Set[str], jd_skills: Set[str]) -> Dict[str, List[str]]:
        """
        Compares skills between Resume and Job Description.
        """
        matched = sorted(list(resume_skills.intersection(jd_skills)))
        missing = sorted(list(jd_skills.difference(resume_skills)))
        additional = sorted(list(resume_skills.difference(jd_skills)))

        return {
            "matched_skills": matched,
            "missing_skills": missing,
            "additional_skills": additional,
        }