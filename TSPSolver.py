import polars as pl
from datetime import datetime, timedelta
from import_data import rename_cols
import time
import numpy as np
from scipy.sparse import csr_array
from scipy.sparse.csgraph import minimum_spanning_tree
import networkx as nx

def get_times_connections(
        services: pl.DataFrame

    ) -> pl.DataFrame:
    """
    Get the shortest time for each connection
    """
    stations = services["crs"].unique()
    timed_connections = pl.DataFrame(
                {"startStation": stations},
    ).with_columns([pl.lit(None).alias(col) for col in stations])
    rows: list[pl.DataFrame] = []
    for start_station in stations:
        connections: dict[str, int] = {}
        start_station_service_uids = services.filter(pl.col("crs") == start_station).select("serviceUid").unique()
        station_services = services.join(start_station_service_uids, on="serviceUid", how="inner")
        departure_times = services.filter(pl.col("crs") == start_station).select(["serviceUid", "departure"]).group_by("serviceUid").agg(
            pl.first("departure").alias("departPrevious")
        )
        station_services = station_services.join(departure_times, on="serviceUid", how="left")
        if not services.filter(pl.col("crs") == start_station).is_empty():
            times_df = station_services.with_columns((60*(pl.col("arrival").cast(pl.Int64)//100 - pl.col("departPrevious").cast(pl.Int64)//100) +
                                                      (pl.col("arrival").cast(pl.Int64)%100 - pl.col("departPrevious").cast(pl.Int64))).alias("time"))
            times_df = times_df.filter(pl.col("time") > 0)
            fastest = times_df.group_by("crs").agg(pl.min("time").alias("fastestConnectionTime"))
            if not fastest.is_empty():
                row = fastest.with_columns(pl.lit(start_station).alias("stationStart")).pivot(on="crs", index="stationStart", values="fastestConnectionTime", aggregate_function="first")
                rows.append(row)
    connection_times = pl.concat(rows, how="diagonal")
    return connection_times

def get_service_times(
        services: pl.DataFrame,
        route: list[str],
        start_time: datetime,
        change_time: int = 5

    ) -> list[dict[str, int]]:
    """
    """
    timings: list[dict[str, int]] = []
    for index, station in enumerate(route[:-1]):
        next_station = route[index + 1]
        starting_from_station_after_time = services.filter(pl.col("crs") == station).filter(pl.col("departure").cast(pl.Int64) > int(start_time.strftime("%H%M")))["serviceUid"]
        services_from_station_after_time = services.filter(pl.col("serviceUid").is_in(starting_from_station_after_time))
        condensed = services_from_station_after_time.group_by("serviceUid").agg([
                pl.when(pl.col("crs") == next_station)
                .then(pl.col("arrival").cast(pl.Int64))
                .otherwise(None)
            .alias("arrivalNext"),
                pl.when(pl.col("crs") == station)
                .then(pl.col("departure").cast(pl.Int64))
                .otherwise(None)
            .alias("departurePrevious"),
        ])
        condensed = condensed.with_columns([
            pl.col("arrivalNext").list.max(),
            pl.col("departurePrevious").list.max()
        ])
        correct_direction_uids = condensed.filter(pl.col("arrivalNext") > pl.col("departurePrevious"))["serviceUid"]
        services_correct_direction = services_from_station_after_time.filter(pl.col("serviceUid").is_in(correct_direction_uids))
        services_stopping_at_station = services_correct_direction.filter(pl.col("crs") == next_station)
        quickest_service = services_stopping_at_station.with_columns(pl.col("arrival").cast(pl.Int64)).sort(by="arrival", nulls_last = True)[0]
        arrival_time = quickest_service["arrival"].to_list()[0]
        quickest_uid = quickest_service["serviceUid"].to_list()[0]
        departure_time = services.filter(pl.col("crs") == station).filter(pl.col("serviceUid") == quickest_uid)["departure"].to_list()[0]
        hours = int(str(arrival_time)[:2])
        minutes = int(str(arrival_time)[2:])
        start_time = datetime(datetime.today().year, datetime.today().month, datetime.today().day,
                hours, minutes, 0) + timedelta(minutes=change_time)
        timings.append({"departs": station, "arrives": next_station,
                "departure_time": departure_time, "arrival_time": arrival_time})

    return timings

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
    

def find_minimum_spanning_tree(
        dm: pl.DataFrame,
        stations: list[str]

    ) -> tuple[pl.DataFrame, list[str]]:
    """
    Generate a minimum spanning tree based on the shortest connection time between stations
    """
    for station in stations:
        if station not in dm.columns:
            print(f"Removing {station}, as it was not in data")
    stations = [station for station in stations if station in dm.columns]
    square_dm = dm.filter(pl.col("stationStart").is_in(stations)).select(stations)
    adjacency_matrix = np.array(square_dm.to_numpy(), dtype=float)
    adjacency_matrix[np.isnan(adjacency_matrix)] = np.inf

    min_spanning_tree_sparse = minimum_spanning_tree(adjacency_matrix)
    min_spanning_tree_dense = min_spanning_tree_sparse.toarray()

    rows: dict[str, str | int] = {"stationStart": stations}
    for i, col in enumerate(stations):
        rows[col] = min_spanning_tree_dense[:, i].tolist()
    # rows = [[dm["stationStart"][i]] + list(min_spanning_tree_dense[i]) for i in range(len(stations))]

    min_span_tree = pl.DataFrame(rows)

    min_span_tree = min_span_tree.with_columns([pl.col(col).cast(pl.Int64) for col in stations])
    return min_span_tree, stations

def christofides(
    services: pl.DataFrame,
    stations: list[str],
    start_time: datetime | None = None

    ) -> list[str]:
    """
    Use Christofides algorithm to get an initial sorting
    """
    services = rename_cols(services)
    connection_times = get_times_connections(services)
    dm = get_full_connection_times(connection_times)
    min_span_tree, valid_stations = find_minimum_spanning_tree(dm, stations)

    nodes = [col for col in min_span_tree.columns if col != "stationStart"]

    degree = min_span_tree.with_columns([
        pl.sum_horizontal([(pl.col(col).cast(pl.Int64) > 0) for col in nodes]).alias("degree")
        # (pl.sum([(pl.col(col) > 0) for col in nodes])).alias("degree")
    ])

    odd_vertices = degree.filter(pl.col("degree") % 2 == 1).select("stationStart").to_series().to_list()
    print(odd_vertices)

    # graph = nx.Graph()

    # for i, vertex_1 in enumerate(odd_vertices):
    #     for j, vertex_2 in enumerate(odd_vertices):
    #         if i < j:
    #             dist = dm[dm["stationStart"] == vertex_1].select(vertex_2).item()
    #             graph.add_edge(vertex_1, vertex_2, weight=dist)
    # matching = nx.algorithms.matching.min_weight_matching(graph, maxcardinality=True)
    # print(matching)
    ...


if __name__ == "__main__":
    services = rename_cols(pl.read_csv("Services.csv", infer_schema=None))
    # services = rename_cols(services)
    # print(services)
    # connection_times = get_times_connections(services)
    # print(connection_times)
    # time_fill_start = time.time()
    # dm = get_full_connection_times(connection_times)
    # time_fill_end = time.time()
    # print(dm)
    # dm.write_csv("DistanceMatrix.csv")
    # dm = pl.read_csv("DistanceMatrix.csv", infer_schema_length=None)
    # time_span_start = time.time()
    # min_span_tree = find_minimum_spanning_tree(dm, ['MYB', 'BDS', 'CST', 'CHX', 'CTK', 'EPH', 'EUS', 'ZFD', 'FST', 'HOX', 'KGX', 'LST', 'LBG', 'MOG', 'OLD', 'PAD', 'SDC', 'STP', 'TCR', 'VXH', 'VIC', 'BFR', 'WAT', 'WAE'])
    # time_span_end = time.time()
    # print(min_span_tree)
    # stations = ['MYB', 'BDS', 'CST', 'CHX', 'CTK', 'EPH', 'EUS', 'ZFD', 'FST', 'HOX', 'KGX', 'LST', 'LBG', 'MOG', 'OLD', 'PAD', 'SDC', 'STP', 'TCR', 'VXH', 'VIC', 'BFR', 'WAT', 'WAE']

    # time_christofides_start = time.time()
    # stations_to_travel = christofides(services, stations)
    # time_christofides_end = time.time()
    # print(f"took {time_christofides_end - time_christofides_start} seconds")
    print(get_service_times(services, ["EXD", "EXC", "EXD", "SJP", "EXD", "TAU", "PLY"], datetime(2025, 9, 2, 12, 0, 0)))
    # print(services.filter(pl.col("crs") == "EXC"))