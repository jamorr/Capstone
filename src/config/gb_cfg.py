from .base import SurvParams
from dataclasses import dataclass, field


@dataclass
class GradientBoostParams(SurvParams):
    loss: str = "coxph"
    learning_rate: float = 0.1
    n_estimators: int = 100
    subsample: float = 0.05
    criterion: str = "friedman_mse"
    min_samples_split: int = 2
    min_samples_leaf: int = 1
    min_weight_fraction_leaf: float = 0.0
    max_depth: int = 3
    min_impurity_decrease: float = 0.0
    random_state: int = 14
    max_features: int = None
    max_leaf_nodes: int = None
    validation_fraction: float = 0.1
    n_iter_no_change: int = None
    tol: float = 0.0001
    dropout_rate: float = 0.0
    ccp_alpha: float = 0.0
