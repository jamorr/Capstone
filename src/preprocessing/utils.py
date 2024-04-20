import datetime
import os
import pathlib
from typing import TypeAlias
from matplotlib import pyplot as plt
from itertools import chain, combinations

import numpy as np
import pandas as pd
import pyarrow as pa
from sodapy import Socrata
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

DateTime: TypeAlias = str | datetime.datetime | datetime.date | np.datetime64

# Unauthenticated client only works with public data sets. Note 'None'
# in place of application token, and no username or password:
def get_connection(api_key:str|pathlib.Path=pathlib.Path(__file__).parents[2]/"api_key.txt"):

    if (isinstance(api_key, str) and os.path.exists(api_key)) \
        or isinstance(api_key, pathlib.Path):
        try:
            with open(api_key, "r") as f:
                api_key = f.readline()
        except FileNotFoundError:
            api_key = None


    return Socrata("data.cityofnewyork.us", api_key, timeout=10000)

def parse_filename(filename:str):
    parts = filename.split('_')
    if len(parts) != 3:
        return None
    try:
        size = int(parts[0])
        offset = int(parts[1].split('.')[0])
        return (size, offset)
    except ValueError:
        return None


def get_311_data(
    connection:Socrata,
    start_date:DateTime,
    end_date: DateTime,
    size:int=200_000,
    offset:int=0,
    agency:str='',
    db_code:str="erm2-nwe9"
    ):
    ag_field = f"AND agency='{agency}'" if agency else ''
    query = (
        "SELECT\n"
            "created_date,\n"
            "closed_date,\n"
            "agency,\n"
            "complaint_type,\n"
            "descriptor,\n"
            "status,\n"
            "resolution_description,\n"
            "resolution_action_updated_date,\n"
            "due_date,\n"
            "borough,\n"
            # "incident_zip,\n"
            # "city,\n"
            # "bbl,\n"
            "latitude,\n"
            "longitude\n"
        "WHERE\n"
            f"(date_extract_y(created_date)={start_date} OR\n"
            f"date_extract_y(closed_date)={end_date})\n"
            f"{ag_field}\n"
        f"LIMIT {size}\n"
        f"OFFSET {offset}\n"
    )
    results = connection.get(db_code, content_type="json", query=query)
    df = pd.DataFrame.from_records(results)
    time_feats = [
        "created_date",
        "closed_date",
        "due_date",
        "resolution_action_updated_date"
    ]
    numeric_feats = [
        "latitude",
        "longitude"
    ]
    for f in time_feats:
        # if df[f].dtype
        try:
            df[f] = pd.to_datetime(df[f], format = "ISO8601")
            df[f] = df[f].astype("timestamp[ns][pyarrow]")
        except pd.errors.OutOfBoundsDatetime:
            continue
    for f in numeric_feats:
        df[f] = pd.to_numeric(df[f])

    return df.convert_dtypes(dtype_backend="pyarrow")


def get_crime_data(
    connection:Socrata,
    start_date:DateTime,
    size:int=200_000,
    offset:int=0,
    db_code:str="qgea-i56i"
    ) -> pd.DataFrame:
    # https://data.cityofnewyork.us/Public-Safety/NYPD-Complaint-Data-Historic/qgea-i56i/about_data
    # {
    #     "cmplnt_num": "27287587", Unique ID
    #     "cmplnt_fr_dt": "2007-03-07T00:00:00.000", Date Of crime
    #     "cmplnt_fr_tm": "16:10:00", Time of crime
    #     "cmplnt_to_dt":"2007-03-07T00:00:00.000" Date crime concluded
    #     "cmplnt_to_tm": "(null)", time crime concluded
    #     "addr_pct_cd": "14", precinct number
    #     "rpt_dt": "2007-03-07T00:00:00.000", Date reported
    #     "ky_cd": "105", Crime code general
    #     "ofns_desc": "ROBBERY", crime description
    #     "pd_cd": "361", Crime code specific
    #     "pd_desc": "ROBBERY,BANK", crime specific description
    #     "crm_atpt_cptd_cd": "ATTEMPTED", just attempt or crime committed sucessfully
    #     "law_cat_cd": "FELONY", crime degree
    #     "boro_nm": "MANHATTAN", borough name
    #     "loc_of_occur_desc": "INSIDE", where it happened relative
    #     "prem_typ_desc": "BANK",  type of location
    #     "juris_desc": "N.Y. POLICE DEPT", jurisdiction
    #     "jurisdiction_code": "0",
    #     "parks_nm": "(null)", park name if crime was in a park
    #     "hadevelopt": "(null)", housing authority development name
    #     "housing_psa": "(null)", housing authority development code
    #     "susp_age_group": "(null)", age group of suspect
    #     "susp_race": "BLACK", race of suspect
    #     "susp_sex": "M", sex of suspect
    #     "latitude": "40.75564",
    #     "longitude": "-73.990952",
    #     "patrol_boro": "PATROL BORO MAN SOUTH", patrol borough name
    #     "station_name": "(null)", station name
    #     "vic_age_group": "(null)", victim age
    #     "vic_race": "UNKNOWN", victim race
    #     "vic_sex": "D" victim sex
    # },

    query = (
    'SELECT\n'
        'cmplnt_num as idx,\n'
        'cmplnt_fr_dt as crime_date,\n'
        'cmplnt_fr_tm as crime_time,\n'
        'cmplnt_to_dt as crime_close_date,\n'
        'cmplnt_to_tm as crime_close_time,\n'
        'rpt_dt as report_date,\n'
        'addr_pct_cd as precinct,\n'
        'latitude,\n'
        'longitude,\n'
        # 'ky_cd as general_code,\n'
        # 'pd_cd as specific_code,\n'
        'crm_atpt_cptd_cd as attempted,\n'
        'law_cat_cd as crime_degree,\n'
        # 'loc_of_occur_desc as relative_loc,\n'
        # 'prem_typ_desc as loc_description,\n'
        # 'parks_nm as park_name,\n'
        # 'housing_psa as housing_code,\n'
        # 'susp_age_group as suspect_age,\n'
        # 'susp_race as suspect_race,\n'
        # 'susp_sex as suspect_sex,\n'
        # 'vic_age_group as victim_age,\n'
        # 'vic_race as victim_race,\n'
        # 'vic_sex as victim_sex\n'
        'boro_nm as borough\n'
    'WHERE\n'
        f'date_extract_y(report_date)={start_date}\n'
    f'LIMIT {size}\n'
    f'OFFSET {offset}\n')

    results = connection.get(db_code, content_type="json", query=query)
    df = pd.DataFrame.from_records(results)
    try:
        date_time = {
            "crime_close_date":[
                "crime_close_date",
                "crime_close_time"
            ],
            "crime_date":[
                "crime_date",
                "crime_time"
                ],
            "report_date":"report_date",

        }

        for k, f in date_time.items():
            if isinstance(f, list):
                p = df[f[0]]+' '+df[f[1]]
                df.drop(f, axis='columns', inplace=True)
            else:
                p = df[f]
            df[k] = pd.to_datetime(p, errors='coerce')

        numeric_feats = [
            "latitude",
            "longitude"
        ]

        for f in numeric_feats:
            df[f] = pd.to_numeric(df[f])
    except Exception:
        return df.convert_dtypes(dtype_backend='pyarrow')


    return df.convert_dtypes(dtype_backend="pyarrow")

def get_crime_codes(connection:Socrata) -> tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame]:
    db_code:str="qgea-i56i"
    query = """
    SELECT DISTINCT
        ky_cd as general_code,
        ofns_desc as general_description
    """
    res = connection.get(db_code, content_type="json", query=query)
    general = pd.DataFrame.from_records(res)

    query = """
    SELECT DISTINCT
        pd_cd as specific_code,
        pd_desc as specific_description
        """
    res = connection.get(db_code, content_type="json", query=query)
    specific = pd.DataFrame.from_records(res)
    query = """
    SELECT DISTINCT
        hadevelopt as housing_name,
        housing_psa as housing_code
    """
    res = connection.get(db_code, content_type="json", query=query)
    housing = pd.DataFrame.from_records(res)
    return general, specific, housing

def load_saved_data():
    return pd.read_feather("../data/2017_1_200k.feather")

def check_stationarity(series, verbose = False):
    # Copied from https://machinelearningmastery.com/time-series-data-stationary-python/

    result = adfuller(series.values)
    if verbose:
        print('ADF Statistic: %f' % result[0])
        print('p-value: %f' % result[1])
        print('Critical Values:')
        for key, value in result[4].items():
            print('\t%s: %.3f' % (key, value))

    if (result[1] <= 0.05) & (result[4]['5%'] > result[0]):
        if verbose: print("\u001b[32mStationary\u001b[0m")
        return True
    else:
        if verbose: print("\x1b[31mNon-stationary\x1b[0m")
        return False

def check_autocorr_ts(target_df:pd.DataFrame, lags:int=24, plots:bool=True):
    figures = []
    for c in target_df.columns:
        print(f"\n{c}")
        # stationarity check
        is_stationary = check_stationarity(target_df[c])
        print("Stationary" if is_stationary else "Non-stationary")
        print("Best Lag:",max(list(acorr_ljungbox(target_df[c], lags=lags, auto_lag=True).index)))
        if plots:
            f, ax = plt.subplots(nrows=2, ncols=1, figsize=(15, 9))
            plot_acf(target_df[c],lags=lags, ax=ax[0], zero=False, auto_ylims=True)
            ax[0].set_title(f'{c} ACF')
            plot_pacf(target_df[c],lags=lags, ax=ax[1], method='ols', zero=False, auto_ylims=True)
            ax[1].set_title(f'{c} PACF')
            figures.append(f)
    if plots:
        return figures

def powerset(s):
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

def make_rounded_time_column_depr(time_col:pd.Series, interval:str = 'H'):
    return pd.to_datetime(time_col.astype('int64')).dt.floor(interval)

def make_rounded_time_column(time_col:pd.Series, interval:str = 'H'):
    return pd.to_datetime(time_col).dt.floor(interval)