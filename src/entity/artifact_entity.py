from dataclasses import dataclass, field
from typing import List, Dict, Any


@dataclass
class ParsedDocument:
    filename: str
    raw_text: str
    word_count: int
    char_count: int
    passages: List[str] = field(default_factory=list)


@dataclass
class RequirementEvidence:
    requirement_text: str
    category: str  # "Core / Mandatory", "Preferred / Good-to-Have", "Soft Skills / General"
    matched_evidence_snippet: str
    raw_similarity: float
    calibrated_score: float
    is_satisfied: bool
    status_label: str


@dataclass
class DynamicRequirementAnalysis:
    total_requirements: int
    core_requirements_count: int
    preferred_requirements_count: int
    soft_skills_count: int
    core_score: float
    preferred_score: float
    soft_skills_score: float
    overall_coverage_score: float
    requirement_evidence_list: List[RequirementEvidence]
    satisfied_requirements: List[str]
    unmet_requirements: List[str]
    matched_domain_terms: List[str]
    missing_domain_terms: List[str]


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
    core_qualifications_score: float
    preferred_qualifications_score: float
    experiential_evidence_score: float
    requirement_analysis: DynamicRequirementAnalysis
    verdict: AIVerdict
    executive_summary: str
    metadata: Dict[str, Any] = field(default_factory=dict)
