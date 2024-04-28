import json
from dataclasses import dataclass, field
from typing import Iterable, List


@dataclass
class SurvParams:
    save_name: str = "default"
    year: int = 2017
    seed: int = 14
    target_col: str = "hours_to_complete"
    status_col: str = "resolution_class"
    datetime_col: str = "created_H"
    strata: List = field(default_factory=lambda: ["borough"])
    add_datetime_cols: bool = True
    keep_cols: List = field(default_factory=lambda: [])
    cpu_prop: float = 0.7
    ohe: bool = True
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
            'created_bo_pr_co', # redundant
            'created_bo_pr', # redundant
            'open_bo_pr', # redundant
            # 'open',
            # 'open_bo',
            # 'open_pr',
            # 'open_co',
            # 'open_pr_co',
            # 'open_bo_co',
            'open_bo_pr_co', # redundant
            "precip_period_hrs",
            "precip_accumulation_mm",
            "direction_deg",
            "agency",
            # "latitude",
            # "longitude",
            "year",
            # 'temperature_c',
            # 'speed_mps',
            # 'dew_temperature_c',
        }
    )


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


class DefaultEncoder(json.JSONEncoder):
    def default(self, o):
        o_dict = o.__dict__
        for k in o_dict.keys():
            v = o_dict[k]
            if isinstance(v, str):
                o_dict[k] = v
            elif isinstance(v, Iterable):
                o_dict[k] = list(v)
        return o_dict