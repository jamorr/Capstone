import pandas as pd
from .identify_district import id_precinct, id_sector, id_sector_precinct
from .utils import (
    get_311_data,
    get_connection,
    get_crime_data,
)
from .volume_feature import (
    add_powerset_created,
    add_powerset_open,
)
from .weather_parse import weather_parse


def process_crime_data(conn, start_date, sector_id:bool=True):
    print("Querying crime data...")
    df_crime = get_crime_data(conn, start_date, size=5_000_000)
    if sector_id:
        print("Adding sector column...")
        id_sector(df_crime, date_range=[start_date, start_date+1], date_col_name='crime_date', inplace=True)
    print("Converting lat/long dtypes")
    df_crime[['latitude','longitude']] = df_crime[['latitude','longitude']].astype('float64')

    return df_crime.convert_dtypes(dtype_backend='pyarrow')

def combine_data_weather(df_311:pd.DataFrame, df_weather:pd.DataFrame):
    df_weather['date_H'] = df_weather['DATE'].dt.round('H')
    hourly_weather = df_weather.groupby('date_H').head(1).drop('DATE', axis='columns')
    df_311 = pd.merge(df_311, hourly_weather, how='left', left_on=['created_H'], right_on=['date_H'])
    del df_weather, hourly_weather

    return df_311


def combine_data_NYPD(df_311:pd.DataFrame, df_crime:pd.DataFrame, df_weather:pd.DataFrame):
    print("Preparing crime df for merge...")
    df_crime.rename({'crime_H':'created_H'}, axis='columns', inplace=True)
    # get all the counts
    hourly_crime = df_crime[['created_H','precinct', 'crime_degree','idx']].groupby(['created_H','precinct', 'crime_degree'], observed=False).count().unstack([ 'crime_degree'])
    hourly_crime.columns = hourly_crime.columns.droplevel().rename('')
    hourly_crime.reset_index(inplace=True)

    print("Combining weather data...")
    df_311 = combine_data_weather(df_311, df_weather)
    print("Combining crime data...")
    df_311['precinct'] = pd.Categorical(df_311['precinct'].replace('', pd.NA).astype('int64[pyarrow]'))
    df_311 = pd.merge(df_311, hourly_crime, how='left', left_on=['created_H', 'precinct'], right_on=['created_H', 'precinct'],)
    del hourly_crime, df_crime,
    df_311['precinct'] = pd.to_numeric(df_311['precinct']).astype('string[pyarrow]')

    return df_311

def preprocess_311(
    df_311,
    agency:str='NYPD',
    interval:str = 'H',
    opened_created_add:list[str]|None=None,
    remove_unclosed:bool=False,
    sectors:bool=True,
    precincts:bool=True
    ):
    try:
        print("Checking dates...")
        current_date = pd.Timestamp.now().normalize()
        df_311 = df_311[df_311['agency']==agency]
        # df_311 = df_311.drop(df_311[df_311['closed_date']>current_date].index)
        # (close before created)
        df_311.loc[(df_311['closed_date']>df_311['created_date'])|(df_311['closed_date']>current_date)]['closed_date'] = pd.NaT
        if remove_unclosed:
            # remove null closed dates
            df_311 = df_311.loc[df_311['closed_date'].notnull()]
        #TODO: Prevent this from becoming a duration or find a way to handle duration
        df_311['hours_to_complete'] = pd.to_datetime(df_311['closed_date']) - pd.to_datetime(df_311['created_date'])
        df_311['hours_to_complete'] = df_311.loc[:,'hours_to_complete'].astype('float64') / (3.6e12)

        print("Adding sectors...")
        if sectors and precincts:
            id_sector_precinct(df_311)
        elif sectors:
            id_sector(df_311, inplace=True)
        elif precincts:
            id_precinct(df_311, inplace=True)

        print("Adding open and created volume columns...")
        if opened_created_add is not None:
            add_powerset_created(df_311, opened_created_add, interval)
            add_powerset_open(df_311, opened_created_add, interval)

    except Exception as e:
        print(e)
        return df_311.convert_dtypes(dtype_backend="pyarrow")

    return df_311.convert_dtypes(dtype_backend="pyarrow")


def get_preprocessed_data(
    start_date:int,
    end_date:int,
    agency:str='NYPD',
    interval:str = 'H',
    opened_created_add:list[str]|None=None,
    remove_unclosed:bool=False,
    sectors:bool=True,
    precincts:bool=True
    ):

    with get_connection() as conn:
        print("Querying 311 data...")
        df_311 = get_311_data(conn, start_date, end_date,agency=agency, size=10_000_000)
        try:
            if 'NYPD' == agency:
                df_crime = process_crime_data(conn, start_date, sector_id=sectors)
        except Exception as e:
            print(e)
            return df_311

    try:
        print("Preprocessing 311 data...")
        df_311 = preprocess_311(df_311, remove_unclosed=remove_unclosed, agency=agency, sectors=sectors, precincts=precincts, opened_created_add=opened_created_add, interval=interval)
        print("Reading weather data...")
        df_weather = weather_parse(start_date=start_date, end_date=end_date)
        print("Combining data frames")
        if agency == 'NYPD':
            df_311 = combine_data_NYPD(df_311, df_crime, df_weather)
        else:
            df_311 = combine_data_weather(df_311, df_weather)
    except Exception as e:
        print(e)
        return df_311

    return df_311
