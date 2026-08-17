from dataclasses import dataclass


@dataclass(frozen=True)
class MatchingWeightsConfig:
    semantic_weight: float = 0.45
    skills_weight: float = 0.40
    keywords_weight: float = 0.15


@dataclass(frozen=True)
class ModelConfig:
    model_name: str = "all-MiniLM-L6-v2"
    device: str = "cpu"
    batch_size: int = 32
