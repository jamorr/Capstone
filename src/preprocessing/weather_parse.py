import pandas as pd
import numpy as np
import pathlib
import datetime


def parse_temperature_c(x):
    sign = x[0]
    num = x[1:].lstrip("0")
    if num == "9999":
        return np.NAN
    elif num:
        num = int(num) / 10
    else:
        num = 0.0
    return num if sign == "+" else -num


# https://www.visualcrossing.com/resources/documentation/weather-data/how-we-process-integrated-surface-database-historical-weather-data/
def explode_temperature(df):
    temperature_exp = pd.DataFrame(
        df["TMP"].str.split(",").tolist(),
        columns=["temperature_c", "temp_obs_quality"],
        index=df.index,
    )
    return temperature_exp["temperature_c"].apply(parse_temperature_c)


def explode_wind(df):
    wind_exp = pd.DataFrame(
        df["WND"].str.split(",").tolist(),
        columns=[
            "direction_deg",
            "direction_quality",
            "observation_type",
            "speed_mps",
            "speed_quality",
        ],
        index=df.index,
    )
    wind_exp.drop(
        ["speed_quality", "direction_quality", "observation_type"], axis=1, inplace=True
    )
    wind_exp["speed_mps"] = wind_exp["speed_mps"].astype("float64") / 10
    wind_exp["direction_deg"] = wind_exp["direction_deg"].astype("uint16")
    return wind_exp


def explode_precip(df):
    df["AA1"] = df["AA1"].fillna("0,-1,-1,-1")
    precip_exp = pd.DataFrame(
        df["AA1"].str.split(",").tolist(),
        columns=[
            "precip_period_hrs",
            "precip_accumulation_mm",
            "precip_condition",
            "precip_quality",
        ],
        index=df.index,
    )
    precip_exp = precip_exp[precip_exp["precip_quality"] != "-1"]
    precip_exp.drop(["precip_condition", "precip_quality"], axis=1, inplace=True)
    precip_exp["precip_accumulation_mm"] = (
        precip_exp["precip_accumulation_mm"].astype("float64") / 10
    )
    precip_exp["precip_period_hrs"] = precip_exp["precip_period_hrs"].astype("uint8")
    return precip_exp


def explode_dew(df):
    dew_exp = pd.DataFrame(
        df["DEW"].str.split(",").tolist(),
        columns=["dew_temperature_c", "dew_obs_quality"],
        index=df.index,
    )
    return dew_exp["dew_temperature_c"].apply(parse_temperature_c)


def explode_vis(df):
    vis_exp = pd.DataFrame(
        df["VIS"].str.split(",").tolist(),
        columns=[
            "vis_distance_km",
            "vis_dist_quality",
            "vis_variablitiy",
            "vis_quality",
        ],
        index=df.index,
    )
    return vis_exp["vis_distance_km"].astype("float64") / 1000


def explode_weather(df: pd.DataFrame):
    weather_df = pd.DataFrame(explode_temperature(df))
    weather_df = weather_df.join(explode_precip(df), how="outer")
    weather_df = weather_df.join(explode_wind(df), how="outer")
    weather_df = weather_df.join(explode_dew(df), how="outer")
    weather_df.reset_index(inplace=True)
    weather_df["year"] = weather_df["DATE"].dt.strftime("%Y").astype(int)
    weather_df = weather_df.convert_dtypes(dtype_backend="pyarrow")
    return weather_df


def weather_parse(
    weather_dir:pathlib.Path, start_date:int|None=None, end_date:int|None=None
):
    if isinstance(start_date, int):
        start_date = datetime.date(start_date, 1, 1)
    if isinstance(end_date, int):
        end_date = datetime.date(end_date, 1, 1)
    print("Reading weather data...")
    try:
        weather_df = pd.read_parquet(weather_dir/"unprocessed")
        assert len(weather_df) != 0

    except (FileNotFoundError, AssertionError):
        temp = pd.read_csv(
            weather_dir / "hourly_temp_laguardia.csv", dtype_backend="pyarrow"
        )
        print("Processing raw weather data...")

        # all data subjected to quality control since no data in category V01
        # Exclude Daily/Monthly summary reports
        temp = temp[~temp["REPORT_TYPE"].isin(["SOD  ", "SOM  "])]
        temp["DATE"] = pd.to_datetime(temp["DATE"])
        temp.set_index("DATE", inplace=True)
        weather_df = explode_weather(temp)
        del temp
    if start_date is not None:
        weather_df = weather_df[weather_df["DATE"] >= start_date]
    if end_date is not None:
        weather_df = weather_df[weather_df["DATE"] < end_date]
    return weather_df


if __name__ == "__main__":
    root = pathlib.Path(__file__).parents[2]
    weather_dir = root/"data"/"weather"
    # df =  temp = pd.read_csv(
    #         weather_dir / "hourly_temp_laguardia.csv", dtype_backend="pyarrow"
    # )
    # pre = explode_precip(df)
    # print(pre)

    df = weather_parse(weather_dir)
    print(df.head(5))