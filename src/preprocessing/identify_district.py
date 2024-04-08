"""
Author:
    Jack Morris

Description:
    Collection of functions for labeling which NYPD precinct or sector
    took place in given latitude and longitude. These functions use
    Dask to speed up compute due to the O(nlog(d)+d) where n is the number
    of lat/lon points and d is number of districts.
"""

import copy
import datetime
import multiprocessing as mp
import pathlib

import dask.dataframe as dd
import geopandas as gpd
import joblib
import numpy as np
import pandas as pd
import pyarrow as pa
from dask.distributed import Client, LocalCluster
from shapely.geometry import Point
from shapely.strtree import STRtree

# print(pathlib.Path(__file__).parents[2])

def load_sectors():
    root_dir = pathlib.Path(__file__).parents[2]/"data"/"nypd"
    try:
        # data from https://data.cityofnewyork.us/Public-Safety/NYPD-Sectors/eizi-ujye
        sectors = joblib.load(root_dir/"sectors.pkl")
    except FileNotFoundError:
        # convert geometry column to shapely objects

        sectors = pd.read_csv(root_dir/"NYPD_Sectors_20240306.csv").convert_dtypes(dtype_backend='pyarrow')
        sectors['the_geom'] = gpd.GeoSeries.from_wkt(sectors['the_geom'])
        sectors = gpd.GeoDataFrame(sectors, geometry='the_geom')

        joblib.dump(sectors, root_dir/"sectors.pkl")
    return sectors

def load_precincts():
    root_dir = pathlib.Path(__file__).parents[2]/"data"/"nypd"
    try:
        # data from https://data.cityofnewyork.us/Public-Safety/Police-Precincts/78dh-3ptz
        precincts = joblib.load(root_dir/"precincts.pkl")
    except FileNotFoundError:
        # convert geometry column to shapely objects
        precincts = gpd.read_file(root_dir/"Police Precincts.geojson")
        joblib.dump(precincts, root_dir/"precincts.pkl")
    return precincts

def get_district(latitude:pa.float64, longitude:pa.float64, snames:np.ndarray, tree:STRtree)->str:
    """Fast function for finding which district a point lies inside

    Args:
        latitude (pa.float64): floating point latitude value
        longitude (pa.float64): floating point longitude value
        snames (np.ndarray): array of sector names indexed the same as the tree
        tree (STRtree): tree of geometries for quick lookup

    Returns:
        str: csv of sector names a point falls into
    """
    p = Point(longitude,latitude)
    idx = tree.query(p, 'within')
    if len(idx):
        return snames[idx[0]]
    return ''

def id_sector_precinct(
    df,
    sectors:gpd.GeoDataFrame|None = None,
    date_range:tuple[int,int]|None = None,
    date_col_name:str|None = 'created_date',
):
    if sectors is None:
        sectors = load_sectors()
    id_sector(df, sectors, date_range, date_col_name, inplace=True)
    id_precinct(df, 'sector', date_range, date_col_name, inplace=True)

def id_sector(
    df:pd.DataFrame,
    sectors:gpd.GeoDataFrame|None = None,
    # date_range:tuple[int,int]|None = None,
    # date_col_name:str|None = 'created_date',
    inplace:bool = False):

    if sectors is None:
        sectors = load_sectors()

    if not inplace:
        df = copy.deepcopy(df)
    df.dropna(how='any',subset=['latitude','longitude'], inplace=True)

    # if date_range:
    #     df = df[(df[date_col_name]>= datetime.date(date_range[0], 1, 1))&(df[date_col_name] < datetime.date(date_range[1], 1, 1))]
    # Create snames list for use in labeling points
    snames:np.ndarray = sectors['SCT_TEXT'].values
    # Build a tree of geometry for faster queries
    tree:STRtree = STRtree(sectors['the_geom'].values)

    sct = _distributed_district_compute(df, snames, tree, 'sector')

    if inplace:
        # add computed values to dataframe
        df['sector'] = sct
    else:
        return pd.Series(sct)


def id_precinct(
    df:pd.DataFrame,
    precinct:gpd.GeoDataFrame|None = None,
    sectors:str|None=None,
    date_range:tuple[int,int]|None = None,
    date_col_name:str|None = 'created_date',
    inplace:bool=False
    ):


    if date_range:
        adf:pd.DataFrame = df[(df[date_col_name]>= datetime.date(date_range[0], 1, 1))&(df[date_col_name] < datetime.date(date_range[1], 1, 1))]
    else:
        adf = df


    if isinstance(sectors, str):
        precinct = strip_sector_to_precinct(adf[sectors])
    else:
        if precinct is None:
            precinct = load_precincts()
        # Create snames list for use in labeling points
        snames:np.ndarray = precinct['precinct'].values
        tree:STRtree = STRtree(precinct.geometry.values)
        precinct = _distributed_district_compute(adf, snames, tree, 'precinct')

    if inplace:
        df['precinct'] = precinct
    else:
        return precinct

def strip_sector_to_precinct(sector):
    return sector.str.slice(0,3).str.lstrip("0").str.rstrip(r"ABCDEFGHIJKL")

def _distributed_district_compute(df, snames, tree, col_name):
    worker_count = int(0.9 * mp.cpu_count()) # leave a core or so for other use
    with LocalCluster(
        n_workers=worker_count,
        processes=True,
        threads_per_worker=1,
        memory_limit='1GB', # per worker memory limit
    ) as cluster, Client(cluster) as client:
        print("View progress here", client.dashboard_link)

        # convert to dask frame with 4 partitions
        ddf: dd.DataFrame = dd.from_pandas(df[['latitude','longitude']], npartitions=worker_count)
        # compute which district each complaint fall into
        sct = ddf.apply(lambda x: get_district(x['latitude'], x['longitude'], snames, tree), axis=1, meta=(col_name, str))
        sct.compute()
    return sct
