import datetime
import heapq
import json
import multiprocessing as mp
import pathlib
from dataclasses import dataclass, field
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
import seaborn as sns
from lifelines import CoxPHFitter, utils
from matplotlib import pyplot as plt
from preprocessing.pre_survival import prep_data_for_surv_analysis
from scipy.integrate import trapezoid
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GridSearchCV, train_test_split
from sksurv.ensemble import (
    ComponentwiseGradientBoostingSurvivalAnalysis,
    GradientBoostingSurvivalAnalysis,
    RandomSurvivalForest,
)
from sksurv.linear_model import CoxnetSurvivalAnalysis, CoxPHSurvivalAnalysis
from sksurv.metrics import concordance_index_ipcw, cumulative_dynamic_auc
from sksurv.util import Surv
from statsmodels.duration.hazard_regression import PHReg, PHRegResults
from tqdm import tqdm
from typing import List


@dataclass
class RSFParams:
    save_name: str = "default"
    year: int = 2017
    seed: int = 14
    target_col: str = "hours_to_complete"
    status_col: str = "resolution_class"
    datetime_col: str = "created_H"
    strata: list = field(default_factory=lambda: ["borough"])
    add_datetime_cols: bool = True
    keep_cols: List = field(default_factory=lambda: [])
    cpu_prop: float = 0.7
    ohe: bool = True
    max_depth: int = 3
    min_samples_leaf: int = 10
    n_estimators: int = 50
    remove_cols: set = field(
        default_factory=lambda: {
            "descriptor",
            "resolution_description",
            "resolution_action_updated_date",
            "incident_zip",
            "city",
            "bbl",
            "status",
            "closed_H",
            "created_date",
            "closed_date",
            "sector",
            "due_date",
            "created",
            "date_H",
            # "precinct",
            # 'created',
            # 'created_bo',
            # 'created_pr',
            # 'created_co',
            # 'created_bo_co',
            # 'created_pr_co',
            'created_bo_pr_co',
            # 'open',
            # 'open_bo',
            # 'open_pr',
            # 'open_co',
            # 'open_pr_co',
            # 'open_bo_co',
            'open_bo_pr_co',


            "precip_period_hrs",
            "precip_accumulation_mm",
            "direction_deg",
            "agency",
            "latitude",
            "longitude",
            "year",
            # 'temperature_c',
            # 'speed_mps',
            # 'dew_temperature_c',
        }
    )


class RSFHyperEncoder(json.JSONEncoder):
    def default(self, o):
        o_dict = o.__dict__
        for k in o_dict.keys():
            v = o_dict[k]
            if isinstance(v, str):
                o_dict[k] = v
            elif isinstance(v, Iterable):
                o_dict[k] = list(v)
        return o_dict


class DefaultDecoder:
    def __init__(self, cls):
        self.cls = cls
        self.annotations = getattr(cls, "__annotations__", {})

    def __call__(self, data):
        for key, value_type in self.annotations.items():
            if key not in data:
                continue
            value = data[key]
            try:
                if not isinstance(value, value_type):
                    data[key] = value_type(value)
            except (TypeError, ValueError):
                raise TypeError(
                    f"Value for '{key}' must be of type '{value_type.__name__}'"
                )
        return self.cls(data)
