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

    query = f"""
    SELECT
        cmplnt_num as idx,
        cmplnt_fr_dt as crime_date,
        cmplnt_fr_tm as crime_time,
        cmplnt_to_dt as crime_close_date,
        cmplnt_to_tm as crime_close_time,
        rpt_dt as report_date,
        addr_pct_cd as precinct,
        latitude,
        longitude,
        ky_cd as general_code,
        pd_cd as specific_code,
        crm_atpt_cptd_cd as attempted,
        law_cat_cd as crime_degree,
        boro_nm as borough,
        loc_of_occur_desc as relative_loc,
        prem_typ_desc as loc_description,
        parks_nm as park_name,
        housing_psa as housing_code,
        susp_age_group as suspect_age,
        susp_race as suspect_race,
        susp_sex as suspect_sex,
        vic_age_group as victim_age,
        vic_race as victim_race,
        vic_sex as victim_sex
    WHERE
        date_extract_y(report_date)={start_date}
    LIMIT {size}
    OFFSET {offset}
    """
    results = connection.get(db_code, content_type="json", query=query)
    df = pd.DataFrame.from_records(results)
    try:
        time_feats = [
            'crime_date',
            'crime_time',
            'crime_close_date',
            'crime_close_time',
            'report_date'
        ]
        numeric_feats = [
            "idx",
            "latitude",
            "longitude"
        ]
        for f in time_feats:
            df[f] = pd.to_datetime(df[f], format = "ISO8601")

        for f in numeric_feats:
            df[f] = pd.to_numeric(df[f])
    except Exception:
        return df
    # for f in numeric_feats:
    #     df[f] = pd.to_numeric(df[f])

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