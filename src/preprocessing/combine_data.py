import datetime
import os
import pathlib

import pandas as pd
from .volume_feature import add_created_count_feat


def combine_data(combined_dir:pathlib.Path, file_name:str = 'cdf.parquet'):
    root_dir = pathlib.Path(__file__).parents[2]/"data"

    df = pd.read_parquet(root_dir/"nypd"/"precincts")
    df = df[df['created_H']>=datetime.date(2018,1,1)]

    weather = pd.read_parquet(root_dir/"weather"/"unprocessed"/"year=2018")
    weather['date_H'] = weather['DATE'].dt.round('H')
    hourly_weather = weather.groupby('date_H').head(1).drop('DATE', axis='columns')

    crime = pd.read_parquet(root_dir/"crime"/"processed")
    crime = crime[crime['crime_H']>=datetime.date(2018,1,1)]
    crime.rename({'crime_H':'created_H'}, axis='columns', inplace=True)
    crime = crime.convert_dtypes(dtype_backend='pyarrow')
    # get all the counts
    hourly_crime = crime[['created_H','precinct', 'crime_degree','idx']].groupby(['created_H','precinct', 'crime_degree'], observed=False).count().unstack([ 'crime_degree'])
    hourly_crime.columns = hourly_crime.columns.droplevel().rename('')
    hourly_crime.reset_index(inplace=True)

    df['precinct'] = pd.Categorical(df['precinct'].replace('', pd.NA).astype('int64[pyarrow]'))
    df = pd.merge(df, hourly_weather, how='left', left_on=['created_H'], right_on=['date_H'])
    df = pd.merge(df, hourly_crime, how='left', left_on=['created_H', 'precinct'], right_on=['created_H', 'precinct'],)
    del hourly_crime, hourly_weather, crime, weather
    df['precinct'] = pd.to_numeric(df['precinct']).astype('string[pyarrow]')

    add_created_count_feat(df, ['precinct','complaint_type'], inplace=True)

    if not combined_dir.exists():
        os.makedirs(combined_dir)

    if not (combined_dir/file_name).exists():
        df.to_parquet(combined_dir/file_name)

    return df