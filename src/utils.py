import datetime
import os
import pathlib
from typing import TypeAlias

import numpy as np
import pandas as pd
import pyarrow as pa
from sodapy import Socrata

DateTime: TypeAlias = str | datetime.datetime | datetime.date | np.datetime64

# Unauthenticated client only works with public data sets. Note 'None'
# in place of application token, and no username or password:
def get_connection(api_key:str|pathlib.Path = pathlib.Path("../api_key.txt")):

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


def get_data(
    connection:Socrata,
    start_date:DateTime,
    end_date: DateTime,
    size:int=200_000,
    offset:int=0,
    db_code:str="erm2-nwe9"
    ):
    query = f"""
    SELECT
        created_date,
        closed_date,
        agency,
        complaint_type,
        descriptor,
        status,
        resolution_description,
        resolution_action_updated_date,
        due_date,
        borough,
        incident_zip,
        city,
        bbl,
        latitude,
        longitude
    WHERE
        date_extract_y(created_date)={start_date} OR
        date_extract_y(closed_date)={end_date}
    LIMIT {size}
    OFFSET {offset}
    """
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
        df[f] = pd.to_datetime(df[f], format = "ISO8601")

    for f in numeric_feats:
        df[f] = pd.to_numeric(df[f])

    return df.convert_dtypes(dtype_backend="pyarrow")


def get_crime_data(
    connection:Socrata,
    start_date:DateTime,
    size:int=200_000,
    offset:int=0,
    db_code:str="qgea-i56i"
    ):

    query = f"""
    SELECT
        cmplnt_num
        cmplnt_fr_dt as crime_date
        cmplnt_fr_tm
        cmplnt_to_dt as crime_close_date
        cmplnt_to_tm
        rpt_dt as report_time
        ky_cd
        ofns_desc
        pd_cd
        pd_desc
        crm_atpt_cptd_cd
        law_cat_cd
        boro_nm
        loc_of_occur_desc
        prem_typ_desc
        juris_desc
        jurisdiction_code
        parks_nm
        hadevelopt
        housing_psa
        susp_age_group
        susp_race
        susp_sex
        patrol_boro
        station_name
        vic_age_group
        vic_race
        vic_sex
    WHERE
        date_extract_y(report_time)={start_date}
    LIMIT {size}
    OFFSET {offset}
    """
    results = connection.get(db_code, content_type="json", query=query)
    df = pd.DataFrame.from_records(results)
    time_feats = [
        "cmplnt_to_dt",
        "cmplnt_fr_dt",
        "rpt_dt",
    ]
    numeric_feats = [
        "latitude",
        "longitude"
    ]
    for f in time_feats:
        df[f] = pd.to_datetime(df[f], format = "ISO8601")

    for f in numeric_feats:
        df[f] = pd.to_numeric(df[f])

    return df.convert_dtypes(dtype_backend="pyarrow")

def load_saved_data():
    return pd.read_feather("../data/2017_1_200k.feather")