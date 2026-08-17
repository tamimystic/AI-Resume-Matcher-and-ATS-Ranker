from dataclasses import dataclass, field
from typing import List, Dict, Any


@dataclass
class ParsedDocument:
    filename: str
    raw_text: str
    word_count: int
    char_count: int


@dataclass
class SkillGapAnalysis:
    matched_skills: List[str]
    missing_skills: List[str]
    coverage_ratio: float
    total_required_skills: int
    matched_skills_count: int


@dataclass
class AIVerdict:
    verdict_badge: str
    verdict_description: str
    badge_style: str
    strengths: List[str]
    gaps: List[str]


@dataclass
class CandidateEvaluationArtifact:
    filename: str
    raw_text: str
    match_percentage: float
    fit_tier: str
    fit_color: str
    semantic_score: float
    skill_score: float
    keyword_score: float
    skill_gap: SkillGapAnalysis
    verdict: AIVerdict
    executive_summary: str
    metadata: Dict[str, Any] = field(default_factory=dict)
