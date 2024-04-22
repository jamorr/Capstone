import datetime
import pathlib

import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from lifelines import CoxPHFitter
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder


def process_categorical(
    df: pd.DataFrame,
    categorical_feats: list[str],
    strata: list[str],
    sparse_data: bool = False,
    random_state=12,
):
    df.loc[:, categorical_feats].fillna('',inplace=True)
    ohe = OneHotEncoder(drop="first", sparse_output=sparse_data).fit(
        df[categorical_feats]
    )

    if sparse_data is True:
        spdf = pd.DataFrame.sparse.from_spmatrix(
            ohe.transform(df[categorical_feats]),
            columns=ohe.get_feature_names_out(),
            index=df.index,
        )
        strata = ohe.get_feature_names_out(strata)
    elif sparse_data is False:
        spdf = pd.DataFrame(
            ohe.transform(df[categorical_feats]),
            columns=ohe.get_feature_names_out(),
            index=df.index,
        ).astype("float32[pyarrow]")
        df.loc[:, strata] = df.loc[:, strata].astype("string[pyarrow]")
    else:
        spdf = df[categorical_feats].astype("category")

    non_cat_feats = [col for col in df.columns if col not in categorical_feats]
    spdf[non_cat_feats] = df[non_cat_feats]

    if sparse_data is True:
        spdf = spdf.astype(pd.SparseDtype("float", 0))

    return spdf


def strata_threshold_remove(df:pd.DataFrame, strata:list[str], drop_strata_threshold:int=50):
    strata_sizes = (
        df[strata + ["hours_to_complete"]]
        .groupby(strata)
        .count()
        .sort_values(by="hours_to_complete")
        .rename({"hours_to_complete": "count"}, axis="columns")
    )
    # drop_strata_threshold = 50
    # print()
    dropped_strata_names = strata_sizes[
        strata_sizes["count"] < drop_strata_threshold
    ].index
    if len(strata) > 1:
        strata_cols = df[strata].apply(tuple, axis=1)
    else:
        strata_cols = df[strata]
    # dropped_strata_names
    drop_idxs = strata_cols.isin(set(dropped_strata_names))
    # df = df.loc[~drop_idxs]
    # df.(drop_idxs, axis=0, inplace=True)
    print(
        f"Dropped {drop_idxs.sum()} entries from strata below threshold {drop_strata_threshold}"
    )
    return dropped_strata_names, strata_sizes, drop_idxs
