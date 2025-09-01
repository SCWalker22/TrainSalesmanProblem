import polars as pl
import requests
from requests.auth import HTTPBasicAuth
import datetime
from typing import Type

def location_detail_structure(
        
    ) -> dict[str, Type]:
    """
    
    """
    location_detail: dict[str, Type] = {
        "realtimeActivated": bool, # Activated - always true?
        "tiploc": str, # Longer station code (tiploc)
        "crs": str, # 3 Letter station code
        "description": str, # Full station name
        "gbttBookedArrival": int | str, # Planned arrival (marked as string, but can be an int (might remove leading 0))
        "gbttBookedDeparture": int | str, # Planned departure (marked as string, but can be an int (might remove leading 0s))
        "origin": list[dict[str, str]], # List of details about origin station: Tiploc, description, workingTime (Time accurate to 1/4 of a minute (hhmmss)), publicTime (hhmm)
        "destination": list[dict[str, str]], # Ditto
        "isCall": bool, # Is calling at this station
        "isPublicCall": bool, # Can passengers (board/alight)?
        "realtimeArrival": int | str | pl.Null, # Expected arrival
        "realtimeArrivalActual": bool | pl.Null, # Has arrived
        "realtimeDeparture": int | str, # Expected Departure
        "realtimeDepartureActual": bool | pl.Null, # Has departed
        "realtimeDepartureNextDay": bool | pl.Null, # Departs in next day
        "platform": int | str | pl.Null, # Platform number (stored as str)
        "platformChanged": bool | pl.Null, # Change to platform?
        "displayAs": str # Call/Origin/Destination
    }
    return location_detail

def format_df(
        df: pl.DataFrame,

    ) -> pl.DataFrame:
    """
    
    """
    location_detail = location_detail_structure()
    new_columns = []
    for i, key in enumerate(location_detail.keys()):
        if key not in ["origin", "destination"]:
            new_columns.append(pl.col("locationDetail")[i].alias(key)) # Cast to correct type?
        else:
            special_columns = [pl.col("locationDetail")[i][col_name] for col_name in pl.col("locationDetail")[i].items()]
            new_columns += special_columns

    # new_columns = [pl.col("Location")[i].alias(key) if key not in ["origin", "destination"] else  for i, key, val in enumerate(location_detail.items())]
    df = df.with_columns(
        new_columns
    )
    print(df)
    return df

def make_request(
        url: str,
        auth: HTTPBasicAuth

    ) -> dict:
    response = requests.get(url, auth=auth)

    if response.ok:
        return response.json()
    else:
        raise requests.HTTPError(f'Request to {url} failed ({response.status_code}, {response.reason})')

def load_data(
    base_url: str,
    username: str,
    password: str,
    mappingDF: pl.DataFrame,
    date: datetime.datetime | str = ""

    ) -> pl.DataFrame:
    """
    
    """
    stationMap = dict(zip(mappingDF["TLC"], mappingDF["Station"]))

    if date != "":
        date = f'/{date.strftime("%Y/%m/%d")}'

    df_list: list[pl.DataFrame] = []

    for key in stationMap.keys():
        print(f"Loading {key}")
        request = f"{base_url}/json/search/{key}{date}"
        trains = make_request(request, HTTPBasicAuth(username, password))
        if "services" in trains.keys():
            df =  pl.DataFrame(trains["services"])
            df = df.with_columns(pl.lit(key).alias("Station"))
            # print(df)
            # print(df["locationDetail"])
            # print(df.columns)
            df_list.append(df)
        else:
            print(f"Unable to find services for {key}")

    dfs = pl.concat(df_list)
    return dfs