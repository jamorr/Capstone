import pandas as pd
from .identify_district import id_precinct, id_sector, id_sector_precinct
from .utils import (
    get_311_data,
    get_connection,
    get_crime_data,
    make_rounded_time_column
)
from .volume_feature import (
    add_powerset_created,
    add_powerset_open,
)
from .weather_parse import weather_parse
# from logging import Logger

# logger = Logger("pipe logger")
# resolution was completed by police or determined to be unnecessary by police
resolved_by_police = {
    'The Police Department responded to the complaint and took action to fix the condition.',
    'The Police Department responded to the complaint and determined that police action was not necessary.',
    'The Police Department issued a summons in response to the complaint.',
    'The Police Department responded to the complaint and a report was prepared.',
    'The Police Department made an arrest in response to the complaint.',
}

# resolved before police arrived
resolved_before_police = {
    'The Police Department responded to the complaint and with the information available observed no evidence of the violation at that time.',
    'The Police Department responded and upon arrival those responsible for the condition were gone.',
}
# police were unable to repond to complaint either because of legal constraints on their work or becuase of user error
failed_to_respond = {
    'The Police Department responded to the complaint but officers were unable to gain entry into the premises.',
    "This complaint does not fall under the Police Department's jurisdiction.",
    'Your request can not be processed at this time because of insufficient contact information. Please create a new Service Request on NYC.gov and provide more detailed contact information.',
}
# censored entries
resolution_unknown = {
    'The Police Department reviewed your complaint and provided additional information below.',
    'Your complaint has been forwarded to the New York Police Department for a non-emergency response. Your complaint will take priority over other non-emergency complaints. 311 will have additional information in 8 hours. Please note your service request number for future reference.',
    'Your complaint has been received by the Police Department and additional information will be available later.',
    "Your complaint has been forwarded to the New York Police Department for a non-emergency response. If the police determine the vehicle is illegally parked, they will ticket the vehicle and then you may either contact a private towing company to remove the vehicle or ask your local precinct to contact 'rotation tow'. Any fees charged for towing will have to be paid by the vehicle owner. 311 will have additional information in 8 hours. Please note your service request number for future reference."
}

def process_crime_data(df_crime:pd.DataFrame, sector_id:bool=True):
    if sector_id:
        print("Adding sector column to Crime Data...")
        df_crime.dropna(axis=0, how='any', subset=['latitude','longitude'], inplace=True)
        id_sector(df_crime, inplace=True)
    return df_crime.convert_dtypes(dtype_backend='pyarrow')

def combine_data_weather(df_311:pd.DataFrame, df_weather:pd.DataFrame):
    df_weather['date_H'] = df_weather['DATE'].dt.round('H')
    hourly_weather = df_weather.groupby('date_H').head(1).drop('DATE', axis='columns')
    df_311 = pd.merge(df_311, hourly_weather, how='left', left_on=['created_H'], right_on=['date_H'])
    del df_weather, hourly_weather

    return df_311


def combine_data_NYPD(df_311:pd.DataFrame, df_crime:pd.DataFrame, df_weather:pd.DataFrame, sectors:bool=False):
    print("Preparing crime df for merge...")
    if "crime_H" not in df_crime.index:
       df_crime['created_H'] = make_rounded_time_column(df_crime['report_date']).convert_dtypes(dtype_backend="pyarrow")
    else:
        df_crime.rename({'crime_H':'created_H'}, axis='columns', inplace=True)
    keep_cols = ['created_H', 'crime_degree','precinct','idx', ]
    if sectors:
        df_crime['sector'] = pd.Categorical(df_crime['sector'])
        keep_cols[2] = "sector"
    else:
        df_crime['precinct'] = pd.Categorical(df_crime['precinct'].replace('', pd.NA).astype('int64[pyarrow]'))


    hourly_crime = df_crime[keep_cols].groupby(keep_cols[:3], observed=False).count().unstack([ 'crime_degree'])
    hourly_crime.columns = hourly_crime.columns.droplevel().rename('')
    hourly_crime.reset_index(inplace=True)
    num_cols = hourly_crime.select_dtypes(include="number").columns
    hourly_crime[num_cols] = hourly_crime[num_cols].fillna(0)


    print("Combining weather data...")
    df_311 = combine_data_weather(df_311, df_weather)
    print("Combining crime data...")
    if sectors:
        df_311['sector'] = pd.Categorical(df_311['sector'])
        df_311 = pd.merge(df_311, hourly_crime, how='left', left_on=['created_H', 'sector'], right_on=['created_H', 'sector'],)
    else:
        df_311['precinct'] = pd.Categorical(df_311['precinct'].replace('', pd.NA).astype('int64[pyarrow]'))
        print("311 dtypes", df_311['precinct'].dtype, df_311['created_H'].dtype, )
        print("Crime dtypes",hourly_crime['precinct'].dtype, hourly_crime['created_H'].dtype, )
        df_311 = pd.merge(df_311, hourly_crime, how='left', left_on=['created_H', 'precinct'], right_on=['created_H', 'precinct'],)

    df_311[num_cols] = df_311[num_cols].fillna(0)
    df_311['precinct'] = pd.to_numeric(df_311['precinct']).astype('string[pyarrow]')

    return df_311

def add_text_classes(df_311:pd.DataFrame):
    s1 = {res:"resolved before police" for res in resolved_before_police}
    s2 = {res:"resolved by police" for res in resolved_by_police}
    s3 = {res:"resolution unknown" for res in resolution_unknown}
    s4 = {res:"failed to respond" for res in failed_to_respond}
    mapper = dict(**s1, **s2, **s3, **s4)
    df_311['resolution_class'] = df_311["resolution_description"].map(mapper)

def preprocess_311(
    df_311,
    agency:str='NYPD',
    interval:str = 'H',
    opened_created_add:list[str]|None=None,
    remove_unclosed:bool=False,
    sectors:bool=True,
    precincts:bool=True,
    text_classes:bool=True
    ):
    # try:
    print("Checking dates...")
    current_date = pd.Timestamp.now().normalize()
    df_311 = df_311[df_311['agency']==agency]
    # df_311 = df_311.drop(df_311[df_311['closed_date']>current_date].index)
    # (close before created)
    df_311.loc[(df_311['closed_date']<df_311['created_date'])|(df_311['closed_date']>current_date),'closed_date'] = pd.NaT
    if remove_unclosed:
        # remove null closed dates
        df_311 = df_311.loc[(df_311['closed_date'].notnull())|(df_311['status']!='Closed')]
    #TODO: Prevent this from becoming a duration or find a way to handle duration
    df_311['hours_to_complete'] = pd.to_datetime(df_311['closed_date']) - pd.to_datetime(df_311['created_date'])
    df_311['hours_to_complete'] = df_311.loc[:,'hours_to_complete'].astype('int64') / (3.6e12)

    if sectors or precincts:
        df_311.dropna(axis=0, how='any', subset=['latitude','longitude'], inplace=True)
    if sectors and precincts:
        print("Adding sectors and precincts...")
        id_sector_precinct(df_311)
    elif sectors:
        print("Adding sectors...")
        id_sector(df_311, inplace=True)
    elif precincts:
        print("Adding precincts...")
        id_precinct(df_311, inplace=True)

    if opened_created_add is not None:
        print("Adding open and created volume columns...")
        add_powerset_created(df_311, opened_created_add, interval)
        add_powerset_open(df_311, opened_created_add, interval)

    if text_classes:
        add_text_classes(df_311)

    return df_311.convert_dtypes(dtype_backend="pyarrow")


def query_data(
    start_date:int,
    end_date:int,
    agency:str='NYPD'
    ):

    with get_connection() as conn:
        print("Querying 311 data...")
        df_311 = get_311_data(conn, start_date, end_date,agency=agency, size=10_000_000)
        if 'NYPD' == agency:
            print("Querying crime data...")
            df_crime = get_crime_data(conn, start_date, size=5_000_000)
            return df_311, df_crime
        return df_311


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
    if agency == 'NYPD':
        df_311, df_crime = query_data(start_date, end_date, agency)
    else:
        df_311 =  query_data(start_date, end_date, agency)
    try:
        print("Preprocessing 311 data...")
        df_311 = preprocess_311(df_311, remove_unclosed=remove_unclosed, agency=agency, sectors=sectors, precincts=precincts, opened_created_add=opened_created_add, interval=interval)

        print("Reading weather data...")
        df_weather = weather_parse(start_date=start_date, end_date=end_date)
        print("Combining data frames")
        if agency == 'NYPD':
            df_crime = process_crime_data(df_crime, sectors)
            df_311 = combine_data_NYPD(df_311, df_crime, df_weather, sectors)
        else:
            df_311 = combine_data_weather(df_311, df_weather)
    except Exception as e:
        print(e)
        return df_311

    return df_311
