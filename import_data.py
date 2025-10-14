import polars as pl
import requests
from requests.auth import HTTPBasicAuth
import datetime
from typing import Type
import numpy as np

USERNAME = ""
PASSWORD = ""

# def location_detail_structure(
        
#     ) -> dict[str, Type]:
#     """
    
#     """
#     location_detail: dict[str, Type] = {
#         "realtimeActivated": bool, # Activated - always true?
#         "tiploc": str, # Longer station code (tiploc)
#         "crs": str, # 3 Letter station code
#         "description": str, # Full station name
#         "gbttBookedArrival": int | str, # Planned arrival (marked as string, but can be an int (might remove leading 0))
#         "gbttBookedDeparture": int | str, # Planned departure (marked as string, but can be an int (might remove leading 0s))
#         "origin": list[dict[str, str]], # List of details about origin station: Tiploc, description, workingTime (Time accurate to 1/4 of a minute (hhmmss)), publicTime (hhmm)
#         "destination": list[dict[str, str]], # Ditto
#         "isCall": bool, # Is calling at this station
#         "isPublicCall": bool, # Can passengers (board/alight)?
#         "realtimeArrival": int | str | pl.Null, # Expected arrival
#         "realtimeArrivalActual": bool | pl.Null, # Has arrived
#         "realtimeDeparture": int | str, # Expected Departure
#         "realtimeDepartureActual": bool | pl.Null, # Has departed
#         "realtimeDepartureNextDay": bool | pl.Null, # Departs in next day
#         "platform": int | str | pl.Null, # Platform number (stored as str)
#         "platformChanged": bool | pl.Null, # Change to platform?
#         "displayAs": str # Call/Origin/Destination
#     }
#     return location_detail

def format_df(
        df: pl.DataFrame,

    ) -> pl.DataFrame:
    """
    
    """
    # location_detail = location_detail_structure()
    # new_columns = []
    # for i, key in enumerate(location_detail.keys()):
    #     if key not in ["origin", "destination"]:
    #         print(f"{df["locationDetail"][0]=}, \n\n\n {df["locationDetail"][0].keys()=}\n\n\n\n\n")
    #         new_columns.append(pl.col("locationDetail").alias(key)) # Cast to correct type?
    #     else:
    #         special_columns = [pl.col("locationDetail")[col_name] for col_name in pl.col("locationDetail").items()]
    #         new_columns += special_columns

    # # new_columns = [pl.col("Location")[i].alias(key) if key not in ["origin", "destination"] else  for i, key, val in enumerate(location_detail.items())]
    if "origin" in df.columns:
        df = df.drop("origin")
    if "destination" in df.columns:
        df = df.drop("destination")
    df = df.unnest("locationDetail")
    df = df.with_columns([
        pl.col("origin").list.first().struct.rename_fields([f"origin_{key.name}" for key in df.schema["origin"].inner.fields]).alias("origin"),
        pl.col("destination").list.first().struct.rename_fields([f"destination_{key.name}" for key in df.schema["destination"].inner.fields]).alias("destination")
    ])
    df = df.unnest("origin")
    df = df.unnest("destination")
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
    Collect data all at once, then collect and store
    """
    stationMap = dict(zip(mappingDF["TLC"], mappingDF["Station"]))

    if date != "":
        date = f'/{date.strftime("%Y/%m/%d")}'

    df_list: list[pl.DataFrame] = []

    for key in stationMap.keys():
        print(f"Loading {key}")
        request = f"{base_url}/json/search/{key}{date}"
        print(request)
        trains = make_request(request, HTTPBasicAuth(username, password))
        if "services" in trains.keys():
            df = pl.DataFrame(trains["services"])
            df = df.with_columns(pl.lit(key).alias("Station"))
            # print(df)
            # print(df["locationDetail"])
            # print(df.columns)
            df_list.append(df)
        else:
            print(f"Unable to find services for {key}")

    dfs = pl.concat(df_list)
    dfs.write_csv("Services.csv")
    return dfs

def stream_data(
    base_url: str,
    username: str,
    password: str,
    mappingDF: pl.DataFrame,
    date: datetime.datetime | str = "",
    start_station: str = ""

    ) -> pl.DataFrame:
    """
    Allows data to be streamed, and picked up later if an error occurs
    Can add code to allow auto resuming on error
    """
    stationMap = dict(zip(mappingDF["TLC"], mappingDF["Station"]))

    if date != "":
        date = f'/{date.strftime("%Y/%m/%d")}'

    # time_cols: list[str] = ["gbttBookedArrival, gbttBookedDeparture", "origin_workingTime", "origin_publicTime", "destination_workingTime", "destination_publicTime", "realtimeArrival", "realtimeDeparture"]

    if not start_station:
        dfs: pl.DataFrame = pl.DataFrame()
        reached_first = True
    else:
        dfs = pl.read_csv("Services.csv", infer_schema_length = None)
        dfs = dfs.with_columns([
            pl.col(col).cast(pl.String).alias(col) for col in dfs.columns if dfs[col].dtype == pl.Int64
        ])
        reached_first = False


    for key in stationMap.keys():
        if key == start_station:
            reached_first = True
        if reached_first:
            print(f"Loading {key}")
            request = f"{base_url}/json/search/{key}{date}"
            trains = make_request(request, auth=HTTPBasicAuth(username, password))
            if "services" in trains.keys() and trains["services"] is not None:
                df = pl.DataFrame(trains["services"])
                df = df.with_columns(pl.lit(key).alias("Station")) # Not needed ??? ======== crs
                # print(f"{df},\n{df.columns},\n{df.dtypes}")
                # print(f"{dfs},\n{dfs.columns},\n{dfs.dtypes}")
                df = format_df(df)
                # print(f"{df},\n{df.columns},\n{df.dtypes}")
                if dfs.is_empty():
                    dfs = df
                else:
                    dfs = pl.concat([dfs, df], how="diagonal")
            if "associations" in dfs.columns:
                dfs = dfs.drop("associations")
            # print(f"{dfs},\n{dfs.dtypes},\n{dfs.columns},\n{dfs["associations"]}")
            dfs.write_csv("Services.csv")
    return dfs

def rename_cols(
        df: pl.DataFrame
    
    ) -> pl.DataFrame:
    """
    Renames some columns with more simple names
    Maybe add try except column not found error
    """
    col_names = {
        "realtimeActivated": "activated",
        "gbttBookedArrival": "arrival",
        "gbttBookedArrivalNextDay": "nextDay",
        "gbttBookedDeparture": "departure",
        "origin_publicTime": "startDeparture",
    }
    df = df.rename(col_names)
    return df

def convert_cols_to_numeric(
        df: pl.DataFrame,
        cols: list[str] = [
            "arrival",
            "departure"
        ]
    ) -> pl.DataFrame:
    """
    
    """
    df = df.with_columns([pl.col(col).cast(pl.Int64) for col in cols])
    return df

def get_times_connections(
        services: pl.DataFrame

    ) -> pl.DataFrame:
    """
    Get the shortest time for each connection
    This method fills a square df with optimal static times between services sparsely (directional)
    This only has the direct services, we then later fill the table based on this
    Might need some cleaning up
    """
    stations = services["crs"].unique() # Create a series of all stations

    timed_connections = pl.DataFrame(
                {"startStation": stations},
    ).with_columns([pl.lit(None).alias(col) for col in stations])

    rows: list[pl.DataFrame] = []

    for start_station in stations: # Loop through all stations, try to find optimal route to all other stations
        connections: dict[str, int] = {}
        start_station_service_uids = services.filter(pl.col("crs") == start_station).select("serviceUid").unique()
        station_services = services.join(start_station_service_uids, on="serviceUid", how="inner") # Get services starting from this station

        departure_times = services.filter(pl.col("crs") == start_station).select(["serviceUid", "departure"]).group_by("serviceUid").agg(
            pl.first("departure").alias("departPrevious")
        )

        station_services = station_services.join(departure_times, on="serviceUid", how="left")#.filter(pl.col("arrival") > pl.col("departPrevious")) # Add column for the depart time - here select arrival after departure?
        if not services.filter(pl.col("crs") == start_station).is_empty():
            times_df = station_services.with_columns((60*(pl.col("arrival").cast(pl.Int64)//100 - pl.col("departPrevious").cast(pl.Int64)//100) +
                                                      (pl.col("arrival").cast(pl.Int64)%100 - pl.col("departPrevious").cast(pl.Int64))).alias("time"))
            # Find the connection time
            times_df = times_df.filter(pl.col("time") > 0) # Ensure it is positive, we could instead use the filter a few line sup
            fastest = times_df.group_by("crs").agg(pl.min("time").alias("fastestConnectionTime"))
            if not fastest.is_empty():
                row = fastest.with_columns(pl.lit(start_station).alias("stationStart")).pivot(on="crs", index="stationStart", values="fastestConnectionTime", aggregate_function="first")
                rows.append(row)
    connection_times = pl.concat(rows, how="diagonal")
    return connection_times

def get_full_connection_times(
        connection_time: pl.DataFrame,
        penalty: int = 5

    ) -> pl.DataFrame:
    """
    Fill all (possible) null values in distance matrix
    """
    rows = set(connection_time["stationStart"].to_list())
    cols = set(connection_time.columns) - {"stationStart"}

    common_stations = sorted(list(rows & cols))
    connection_time = connection_time.filter(pl.col("stationStart").is_in(common_stations)).select(["stationStart"] + common_stations)

    nodes = connection_time["stationStart"].to_list()
    n = len(nodes)

    # Adjacency matric
    adj = np.array(connection_time.select(nodes).to_numpy(), dtype=float)
    adj[np.isnan(adj)] = np.inf

    # Floyd-Warshall algorithm
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if i != j:
                    if adj[i, k] < np.inf and adj[k, j] < np.inf:
                        new_dist = adj[i, k] + adj[k, j] + penalty
                        if new_dist < adj[i, j]:
                            adj[i, j] = new_dist
    adj = np.where(np.isinf(adj), None, adj) # Fill with Null

    dm_dict = {"stationStart": nodes}
    for i, station in enumerate(common_stations):
        dm_dict[station] = adj[:, i].tolist()

    dm_filled = pl.DataFrame(dm_dict).with_columns([pl.col(col).cast(pl.Int64) for col in common_stations]) # Maybe use int16???
    return dm_filled

if __name__ == "__main__":
    services = load_data("https://api.rtt.io/api/v1", USERNAME, PASSWORD, pl.read_csv("StationMap.csv"), datetime.datetime.now())
    services.write_csv("Services.csv")
    dm = get_full_connection_times(get_times_connections(services))
    dm.write_csv("ConnectionTimes.csv")