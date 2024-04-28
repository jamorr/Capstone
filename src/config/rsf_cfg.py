from dataclasses import dataclass
from .base import SurvParams

@dataclass
class RSFParams(SurvParams):
    max_depth: int = 3
    min_samples_leaf: int = 10
    n_estimators: int = 50
