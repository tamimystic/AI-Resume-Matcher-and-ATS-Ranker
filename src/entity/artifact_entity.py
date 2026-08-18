from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


@dataclass
class ParsedDocument:
    filename: str
    raw_text: str
    word_count: int
    char_count: int
    chunks: List[str] = field(default_factory=list)


@dataclass
class RequirementEvidence:
    requirement_text: str
    matched_evidence_snippet: str
    raw_similarity: float
    calibrated_score: float
    is_satisfied: bool
    status_label: str


@dataclass
class DynamicRequirementAnalysis:
    total_requirements: int
    satisfied_count: int
    partial_count: int
    unmet_count: int
    requirement_evidence_list: List[RequirementEvidence]
    satisfied_requirements: List[str]
    unmet_requirements: List[str]
    coverage_score: float
    extracted_keyphrases: List[str]
    matched_keyphrases: List[str]
    missing_keyphrases: List[str]


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
    requirement_coverage_score: float
    macro_semantic_score: float
    domain_terminology_score: float
    requirement_analysis: DynamicRequirementAnalysis
    verdict: AIVerdict
    executive_summary: str
    metadata: Dict[str, Any] = field(default_factory=dict)
