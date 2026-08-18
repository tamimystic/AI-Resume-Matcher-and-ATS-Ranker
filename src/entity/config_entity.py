from dataclasses import dataclass


@dataclass(frozen=True)
class MatchingWeightsConfig:
    requirement_weight: float = 0.55
    macro_semantic_weight: float = 0.25
    terminology_weight: float = 0.20


@dataclass(frozen=True)
class ModelConfig:
    model_name: str = "all-MiniLM-L6-v2"
    device: str = "cpu"
    batch_size: int = 32
