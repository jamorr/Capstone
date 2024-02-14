import datetime
from typing import TypeAlias
import pandas as pd
from sodapy import Socrata
import pyarrow as pa
import numpy as np
import pathlib
import os

DateTime: TypeAlias = str | datetime.datetime | datetime.date | np.datetime64

# Unauthenticated client only works with public data sets. Note 'None'
# in place of application token, and no username or password:
def get_connection(api_key:str|pathlib.Path = pathlib.Path("../api_key.txt")):
    if isinstance(api_key, str) and os.path.exists(api_key):
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
    offset:int,
    size:int
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
        due_date,
        resolution_action_updated_date,
        latitude,
        longitude
    WHERE
        date_extract_y(created_date)={start_date} OR
        date_extract_y(closed_date)={end_date}
    LIMIT {size}
    OFFSET {offset}
    """
    results = connection.get("erm2-nwe9", content_type="json", query=query)
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


def load_saved_data():
    return pd.read_feather("../data/2017_1_200k.feather")