import datetime

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sksurv.util import Surv


def process_categorical(
    df: pd.DataFrame,
    categorical_feats: list[str],
    strata: list[str],
    sparse_data: bool = False,
    ohe=None,
):
    df.loc[:, categorical_feats].fillna("", inplace=True)
    if ohe is None:
        ohe = OneHotEncoder(drop="first", sparse_output=sparse_data).fit(
            df[categorical_feats]
        )
    categorical_feats = ohe.feature_names_in_

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

    return spdf, ohe


def strata_threshold_remove(
    df: pd.DataFrame, strata: list[str], drop_strata_threshold: int = 50
):
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


def prep_data_for_surv_analysis(
    df: pd.DataFrame,
    remove_cols: set[str],
    strata: list[str],
    target_col: str,
    status_col: str,
    year: int,
    add_datetime_cols: bool,
    seed: int,
    datetime_col: str = "created_H",
    keep_cols: list[str] = None,
    **kwargs,
):
    sdf = df.copy(deep=True)[
        (df[datetime_col] >= datetime.date(year, 1, 1))
        & (df[datetime_col] < datetime.date(year + 1, 1, 1))
    ]
    *_, idxs_to_drop = strata_threshold_remove(sdf, strata, 50)
    sdf = sdf[[c for c in sdf.columns if c not in remove_cols]]
    sdf = sdf.loc[~idxs_to_drop.values]
    # convert status col to boolean
    sdf[status_col] = map_resolution_to_int(sdf[status_col])
    sdf = sdf[[c for c in sdf.columns if c not in remove_cols]]


    if add_datetime_cols:
        sdf["month"] = sdf[datetime_col].dt.month.astype("string[pyarrow]")
        sdf["day_of_week"] = sdf[datetime_col].dt.day_of_week.astype("string[pyarrow]")
        sdf["hour"] = df[datetime_col].dt.hour.astype("string[pyarrow]")
    sdf.drop(datetime_col, axis="columns", inplace=True)
    categorical_x = set(sdf.select_dtypes(exclude="number").columns)
    categorical_x = list(categorical_x)
    sdf.loc[:, categorical_x] = sdf.loc[:, categorical_x].fillna("")
    categorical_x = list(set(categorical_x).difference(set(strata)))
    numeric_x = list(set(sdf.select_dtypes(include="number").columns).difference(set(strata+[target_col, status_col])))

    sdf = sdf.ffill()
    # Drop where negative time to complete
    sdf = sdf[sdf[target_col] > 0]

    if categorical_x:
        ohe = kwargs['ohe_obj'] if 'ohe_obj' in kwargs else None
        sdf, ohe = process_categorical(sdf, categorical_feats=categorical_x, strata=strata, ohe=ohe)
    else:
        ohe = None
    scaler = StandardScaler()
    scaler.fit(sdf[numeric_x])
    if 'scaler_obj' in kwargs:
        scaler.mean_ = kwargs['scaler_obj'].mean_
        scaler.scale = kwargs['scaler_obj'].scale_

    sdf[numeric_x] = scaler.transform(sdf[numeric_x])
    sdf_train, sdf_test = train_test_split(
        sdf,
        train_size=0.8,
        test_size=0.2,
        stratify=sdf[status_col],
        random_state=seed,
    )
    X_train = sdf_train[
        [
            c
            for c in sdf_train.columns
            if c not in strata and c not in {target_col, status_col}
        ]
    ]
    X_test = sdf_test[
        [
            c
            for c in sdf_test.columns
            if c not in strata and c not in {target_col, status_col}
        ]
    ]
    y_train = Surv.from_arrays(
        event=sdf_train[status_col].to_numpy(), time=sdf_train[target_col].to_numpy()
    )
    y_test = Surv.from_arrays(
        event=sdf_test[status_col].to_numpy(), time=sdf_test[target_col].to_numpy()
    )
    if keep_cols:
        X_train = X_train[keep_cols["keep_cols"]]
        X_test = X_test[keep_cols["keep_cols"]]
    return X_train, X_test, y_train, y_test, ohe, scaler


def map_resolution_to_int(col: pd.Series):
    mapper = {
        "": 0,
        "failed to respond": 0,
        "resolution unknown": 0,
        "resolved before police": 1,
        "resolved by police": 1,
    }
    return col.map(mapper)
