from .utils import powerset, make_rounded_time_column
from time import perf_counter
import pandas as pd


def add_created_count_feat(df:pd.DataFrame,
                           features:list[str],
                           inplace = False,
                           interval:str = 'H'):
    trunc_name = '_'.join(['created']+[name[:2] for name in features])
    created_by_date = df.groupby(
        by=[
            f'created_{interval}',
            *features
            ],
        observed=False
    )['created_date'].count()

    # ab['time'] = pd.to_datetime(ab['created_date'].astype(str)+' '+ab['hour'].astype(str)+':00:00')
    # ab.drop(['created_date', 'hour'], axis=1, inplace=True)
    # ab.set_index('time', inplace=True)
    if not inplace:
        return created_by_date.unstack(level=features).fillna(0)
    else:
        df.set_index(
            [
                f'created_{interval}',
                *features
            ],
            inplace=True)
        df[trunc_name] = created_by_date.fillna(0)
        df.reset_index(inplace=True)


def add_powerset_created(df:pd.DataFrame, s:set|list, interval:str = 'H'):
    if f'created_{interval}' not in df.columns:
        df[f'created_{interval}'] = make_rounded_time_column(df['created_date'], interval)
    if f'closed_{interval}' not in df.columns:
        df[f'closed_{interval}'] = make_rounded_time_column(df['closed_date'], interval)

    for features in powerset(s):
        start = perf_counter()

        if not features:
            print("general")
            add_created_count_feat(df, features=[], inplace=True, interval=interval)
        # each complaint type seems to be handled by a single agency
        # this reduces runtime a bit by removing this redundancy
        elif 'agency' in features and 'complaint_type' in features:
            continue
        else:
            print(*features)
            add_created_count_feat(df, list(features),inplace=True, interval=interval)
        print(f'{perf_counter()-start:.3f} seconds\n')


def add_open_count(df:pd.DataFrame, interval:str):
    df.set_index([f'closed_{interval}'], inplace=True)
    # sort by closed date
    df.sort_index(axis=0, level=f'closed_{interval}', ascending=True, inplace=True)
    t_num_closed = df['closed_date'].groupby(level=f'closed_{interval}',observed=False).count()
    t_num_closed:pd.Series = t_num_closed[
        t_num_closed.index.get_level_values(f'closed_{interval}') < df[f'created_{interval}'].max()
        ].cumsum()

    # remove closed date from index
    df.reset_index(f'closed_{interval}', drop=False, inplace=True)
    df.set_index(f'created_{interval}', inplace=True)
    df.sort_index(level=f'created_{interval}', ascending=True, inplace=True)

    # count number created per hour
    t_num_created = df['created_date'].groupby(level=f'created_{interval}').count().cumsum()

    # change index of closed to be more like created
    t_num_closed.index.rename(f'created_{interval}', inplace=True)
    missing_indices = t_num_closed.index.union(t_num_created.index)
    t_num_closed = t_num_closed.reindex(missing_indices, method='ffill')
    t_num_closed.rename(t_num_created.name, inplace=True)

    # display(t_num_closed)
    # display(t_num_created)
    # display(df)
    df["open"] = t_num_created - t_num_closed
    df.reset_index(inplace=True)


def add_open_count_feat(df:pd.DataFrame, features:list[str], interval:str):
    # col name for new feature
    trunc_name = '_'.join(['open']+[name[:2] for name in features])

    # set the index to features and date closed
    df.set_index([f'closed_{interval}']+features, inplace=True)

    # sort by closed date
    df.sort_index(axis=0, level=f'closed_{interval}', ascending=True, inplace=True)

    # count number closed per hour
    t_num_closed = df['closed_date'].groupby(level=list(range(df.index.nlevels))).count()
    t_num_closed:pd.Series = t_num_closed[t_num_closed.index.get_level_values(f'closed_{interval}') < df[f'created_{interval}'].max()]
    t_num_closed = t_num_closed.unstack(level=features, fill_value=0).cumsum()

    # remove closed date from index
    df.reset_index(f'closed_{interval}', drop=False, inplace=True)
    # df.reset_index(features, drop=False, inplace=True)

    # Add created date to index
    df.set_index(f'created_{interval}', append=True, inplace=True)
    # if features:
    #     df = df.reorder_levels(['created_{interval}']+features)
    df.sort_index(level=f'created_{interval}', ascending=True, inplace=True)

    # count number created per hour
    t_num_created = df['created_date'].groupby(level=list(range(df.index.nlevels))).count()

    # unstack to get columns for all features
    # get cumsum over the features then restack into a series
    t_num_created = t_num_created.unstack(level=features, fill_value=0).cumsum()

    # change index of closed to be more like created
    t_num_closed.index.rename(f'created_{interval}', inplace=True)
    missing_indices = t_num_closed.index.union(t_num_created.index)
    t_num_closed = t_num_closed.reindex(missing_indices, method='ffill').stack(level=features)

    # stack back into a series
    t_num_created = t_num_created.stack(level=features)
    t_num_closed.rename(t_num_created.name, inplace=True)

    # difference to get number open at a given hour
    diff = t_num_created - t_num_closed
    # diff:pd.Series = t_num_created.combine(t_num_closed, lambda x,y: x-y, fill_value=0)
    diff.fillna(0, inplace=True)
    # reorder axis for merger if needed
    if not all([a == b for a, b in zip(diff.index.names, df.index.names)]):
        diff = diff.reorder_levels(df.index.names)

    # add to the dataframe and reset the index
    df[trunc_name] = diff
    df.reset_index(inplace=True)

def add_powerset_open(df:pd.DataFrame, s:set|list, interval:str = 'H'):
    if f'created_{interval}' not in df.columns:
        df[f'created_{interval}'] = make_rounded_time_column(df['created_date'], interval)
    if f'closed_{interval}' not in df.columns:
        df[f'closed_{interval}'] = make_rounded_time_column(df['closed_date'], interval)

    for features in powerset(s):
        start = perf_counter()

        if not features:
            print("general")
            add_open_count(df, interval)
        # each complaint type seems to be handled by a single agency
        # this reduces runtime a bit by removing this redundancy
        elif 'agency' in features and 'complaint_type' in features:
            continue
        else:
            print(*features)
            add_open_count_feat(df, list(features), interval)
        print(f'{perf_counter()-start:.3f} seconds\n')