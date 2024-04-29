from dataclasses import dataclass, field
from .base import SurvParams

@dataclass
class ElasticParams(SurvParams):
    n_alphas:int=100
    alphas:list=None
    alpha_min_ratio:int=field(default='auto')
    l1_ratio:float=0.5
    penalty_factor:float=None
    normalize:bool=False
    copy_X:bool=True
    tol:float=1e-07
    max_iter:int=5000
    verbose:bool=False
    fit_baseline_model:bool=True