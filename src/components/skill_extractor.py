import re
import sys
from typing import List, Set, Dict
from src.constants.skill_taxonomy import SKILL_TAXONOMY
from src.entity.artifact_entity import SkillGapAnalysis
from src.exception.custom_exception import CustomException
from src.logger.logging import logger


class SkillExtractor:
    """
    Component for extracting normalized skills from text and calculating skill gap analysis.
    """

    def __init__(self, custom_taxonomy: Dict[str, str] = None):
        logger.info("Initializing SkillExtractor component with ontology mappings")
        self.taxonomy = custom_taxonomy if custom_taxonomy is not None else SKILL_TAXONOMY
        self._compiled_patterns = self._build_compiled_patterns()

    def _build_compiled_patterns(self) -> Dict[re.Pattern, str]:
        compiled = {}
        for raw_key, normalized_label in self.taxonomy.items():
            escaped_key = re.escape(raw_key)
            if raw_key in ["c++", "cpp", "c#", ".net", "r", "c", "go"]:
                if raw_key == "c++":
                    pattern = re.compile(r'(?:\b|(?<=\s))c\+\+(?:\b|(?=[\s,;.]))', re.IGNORECASE)
                elif raw_key == "c#":
                    pattern = re.compile(r'(?:\b|(?<=\s))c\#(?:\b|(?=[\s,;.]))', re.IGNORECASE)
                elif raw_key == ".net":
                    pattern = re.compile(r'(?:\b|(?<=\s))\.(?:net)(?:\b|(?=[\s,;.]))', re.IGNORECASE)
                elif raw_key == "r":
                    pattern = re.compile(r'(?:\b|(?<=\s))R(?:\b(?=[,\s;.]programming|\blanguage))', re.IGNORECASE)
                elif raw_key == "go":
                    pattern = re.compile(r'\b(?:golang|go\s+programming|go\s+language)\b', re.IGNORECASE)
                else:
                    pattern = re.compile(r'\b' + escaped_key + r'\b', re.IGNORECASE)
            else:
                pattern = re.compile(r'(?<!\w)' + escaped_key + r'(?!\w)', re.IGNORECASE)
            compiled[pattern] = normalized_label
        return compiled

    def extract(self, text: str) -> List[str]:
        """
        Extracts all identified skills from input text with normalized labels.
        """
        try:
            if not text:
                return []

            matched_skills: Set[str] = set()
            for pattern, normalized_skill in self._compiled_patterns.items():
                if pattern.search(text):
                    matched_skills.add(normalized_skill)

            return sorted(list(matched_skills))
        except Exception as e:
            logger.error(f"Error during skill extraction: {str(e)}")
            raise CustomException(e, sys)

    def analyze_gap(self, required_skills: List[str], candidate_skills: List[str]) -> SkillGapAnalysis:
        """
        Calculates the coverage ratio, matched skills, and missing skills.
        """
        try:
            req_set = set(required_skills)
            cand_set = set(candidate_skills)

            if not req_set:
                return SkillGapAnalysis(
                    matched_skills=sorted(list(cand_set)),
                    missing_skills=[],
                    coverage_ratio=1.0,
                    total_required_skills=0,
                    matched_skills_count=len(cand_set)
                )

            matched = sorted(list(req_set.intersection(cand_set)))
            missing = sorted(list(req_set.difference(cand_set)))
            ratio = len(matched) / len(req_set) if len(req_set) > 0 else 0.0

            return SkillGapAnalysis(
                matched_skills=matched,
                missing_skills=missing,
                coverage_ratio=ratio,
                total_required_skills=len(req_set),
                matched_skills_count=len(matched)
            )
        except Exception as e:
            logger.error(f"Error during skill gap analysis: {str(e)}")
            raise CustomException(e, sys)
