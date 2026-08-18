from dataclasses import dataclass
from typing import Any


class MatchingWeightsConfig:
    """
    Robust, backwards-compatible configuration for ATS evaluation weighting.
    """
    core_requirements_weight: float
    preferred_qualifications_weight: float
    experiential_evidence_weight: float

    def __init__(
        self,
        core_requirements_weight: float = 0.60,
        preferred_qualifications_weight: float = 0.25,
        experiential_evidence_weight: float = 0.15,
        **kwargs: Any
    ):
        self.core_requirements_weight = float(kwargs.get("requirement_weight", core_requirements_weight))
        self.preferred_qualifications_weight = float(kwargs.get("macro_semantic_weight", preferred_qualifications_weight))
        self.experiential_evidence_weight = float(kwargs.get("terminology_weight", experiential_evidence_weight))


@dataclass
class ModelConfig:
    model_name: str = "all-MiniLM-L6-v2"
    device: str = "cpu"
    batch_size: int = 32
