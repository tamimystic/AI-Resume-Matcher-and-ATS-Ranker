from dataclasses import dataclass


@dataclass(frozen=True)
class MatchingWeightsConfig:
    core_requirements_weight: float = 0.60
    preferred_qualifications_weight: float = 0.25
    experiential_evidence_weight: float = 0.15


@dataclass(frozen=True)
class ModelConfig:
    model_name: str = "all-MiniLM-L6-v2"
    device: str = "cpu"
    batch_size: int = 32
