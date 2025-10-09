import polars as pl
from datetime import datetime, timedelta
from import_data import rename_cols, convert_cols_to_numeric
import time
import numpy as np
from scipy.sparse import csr_array
from scipy.sparse.csgraph import minimum_spanning_tree
import networkx as nx


"""
Progress: We find an initial fastest route using the theoretical travel time between stations,
then find the corresponding route using actual services.
We now need to optimise this route, and try to improve it
Data set input is not perfect, so only includes national rail services, not tube (i think) - so might need to find better data
Once the backend is working, I will add a front-end GUI
"""

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

def normalise_time(time: int) -> int:
    """
    Ensure time is of the form hhmm and a valid time
    """
    if time % 100 >= 60:
        time += 40
    if time//100 >= 24:
        time = time - 2400
    return time

def find_route(
        arrivals: pl.DataFrame,
        station_start: str,
        station_end: str

    ) -> list[str]:
    """
    
    """
    route_reverse: list[str] = [station_end]
    while station_start not in route_reverse:
        route_reverse.append(arrivals.filter(pl.col("crs") == route_reverse[-1])["serviceFrom"][0])
    route = route_reverse[::-1] # Reverse route
    return route

def fill_first_arrival_table(
        arrivals: pl.DataFrame,
        services_calling: pl.DataFrame,
        station: str

    ) -> pl.DataFrame:
    """
    
    """
    quickest_arrival_per_station = services_calling.group_by("crs").agg(pl.col("arrival").min().alias("firstArrival"))
    arrivals = (arrivals.join(quickest_arrival_per_station, on="crs", how="left")
                .with_columns([
                    pl.when(
                        (pl.col("arrival").is_null()) | (pl.col("arrival") > pl.col("firstArrival"))
                )
                .then(pl.col("firstArrival"))
                .otherwise(pl.col("arrival"))
                .alias("arrival"),
                    pl.when(
                        (pl.col("arrival").is_null()) | (pl.col("arrival") > pl.col("firstArrival"))
                    )
                .then(pl.lit(station))
                .otherwise(pl.col("serviceFrom"))
                .alias("serviceFrom")
            ]
                ).select(["crs", "arrival", "serviceFrom"])
    )
    return arrivals

def find_timed_stations(
        services: pl.DataFrame,
        station_start: str,
        arrivals: pl.DataFrame,
        start_time: datetime = datetime(2025, 9, 1, 8, 0, 0),

    ) -> pl.DataFrame:
    """
    
    """
    valid_services_out = services.filter((pl.col("crs") == station_start) & (pl.col("departure") > int(start_time.strftime("%H%M"))))
    valid_services_out = valid_services_out.with_columns(pl.col("departure").alias("time")).select(["serviceUid", "time"])
    services_via_start = (services.join(valid_services_out, on="serviceUid")
                        .filter((pl.col("crs") != station_start) & (pl.col("time") < pl.col("arrival")))
                        )
    services_via_start = services_via_start.with_columns(pl.when(pl.col("nextDay") == "true").then(pl.col("arrival").cast(pl.Int64) + 2400).otherwise(pl.col("arrival")).alias("arrival")).sort(by="arrival")
    arrivals = fill_first_arrival_table(arrivals, services_via_start, station_start)
    return arrivals

def find_first_arrival(
        services: pl.DataFrame,
        station_start: str,
        station_end: str,
        start_time: datetime = datetime(2025, 9, 1, 8, 0, 0),
        change_time: int = 5,
        extra_check_depth: int = 0

    ) -> tuple[datetime, list[str]]:
    """
    Might need to prevent double counting - checked stations - but check again if earlier arrival
    """
    # if is_direct_path(station_start, station_end, services):
    #     path: list[str] = [station_start, station_end]
    # else:
    if True:
        # Generate path by finding earliest arrival
        stations = services["crs"].unique()
        arrivals: pl.DataFrame = pl.DataFrame({"crs": stations, "arrival": None, "serviceFrom": None})
        arrivals = find_timed_stations(services, station_start, arrivals, start_time)
        arrival_time: int | None = arrivals.filter(pl.col("crs") == station_end)["arrival"][0]
        # count = 0
        extra_checks: int = 0
        checked_stations: list[str] = []
        
        while arrival_time is None or extra_checks < extra_check_depth:
            stations_reachable = arrivals.filter(pl.col("arrival").is_not_null())["crs"].unique().to_list()
            for station in stations_reachable:
                # arrive_at_start_time = str(services.filter(pl.col("crs") == station & pl.col("departure") > )["arrival"][0]).zfill(4) # Enforce 4 digit
                departure_time = normalise_time(arrivals.filter(pl.col("crs") == station)["arrival"][0] + change_time)
                # services_from_station = services.filter(pl.col("crs") == station & pl.col("departure") > departure_time)
                new_start_time = datetime(2025, 9, 1, departure_time//100, departure_time%100, 0)
                arrivals = find_timed_stations(services, station, arrivals, new_start_time)
            arrival_time: int | None = arrivals.filter(pl.col("crs") == station_end)["arrival"][0]

            if arrival_time is not None:
                extra_checks += 1

            checked_stations += stations_reachable

        route = find_route(arrivals, station_start, station_end)
        departure_time = normalise_time(arrivals.filter(pl.col("crs") == station_end)["arrival"][0])
        start_time = datetime(2025, 9, 1, departure_time//100, departure_time%100, 0)
    return start_time, route

def get_full_route_timed(
        services: pl.DataFrame,
        disconnected_route: list[str],
        start_time: datetime = datetime(2025, 9, 1, 6, 0, 0),
        change_time: int = 5

    ) -> tuple[datetime, list[str]]:
    """
    
    """
    route: list[str] = [disconnected_route[0]]
    for index in range(len(disconnected_route) - 1):
        station_start = disconnected_route[index]
        station_end = disconnected_route[index + 1]
        print(f"finding route for {station_start} - {station_end} after {start_time}")
        start_time, partial_route = find_first_arrival(services, station_start, station_end, start_time, change_time=change_time)
        print(f"route found for {station_start} - {station_end}:\n{partial_route}")
        route += partial_route[1:]
    return start_time, route

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

    ) -> list[str]:
    """
    Use Christofides algorithm to get an initial sorting
    """
    # services = rename_cols(services)
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
    return odd_vertices

def simple_route_finder(
        services: pl.DataFrame,
        stations: list[str],
        start_time: datetime,
        change_time: int = 5

    ) -> tuple[list[str], int]:
    """
    
    """
    end_time, route = get_full_route_timed(services, christofides(services, stations), start_time=start_time, change_time=change_time)
    mins_taken = (end_time - start_time).total_seconds() / 60
    return route, mins_taken

def a_star(
        services: pl.DataFrame,
        stations: list[str],
        start_time: datetime,
        change_time: int = 5

    ) -> tuple[list[str], int]:
    """
    
    """
    pass

def simple_improved(
        services: pl.DataFrame,
        stations: list[str],
        start_time: datetime,
        change_time: int = 5

    ) -> tuple[list[str], int]:
    """
    
    """
    pass

def full_graph_traversal(
        services: pl.DataFrame,
        stations: list[str],
        start_time: datetime,
        change_time: int = 5
    
    ) -> tuple[list[list[str]], list[int]]:
    """

    """
    pass

if __name__ == "__main__":
    services = convert_cols_to_numeric(rename_cols(pl.read_csv("Services.csv", infer_schema=None)))
    stations = ['MYB', 'BDS', 'CST', 'CHX', 'CTK', 'EPH', 'EUS', 'ZFD', 'FST', 'HOX', 'KGX', 'LST', 'LBG', 'MOG', 'OLD', 'PAD', 'SDC', 'STP', 'TCR', 'VXH', 'VIC', 'BFR', 'WAT', 'WAE']
    stations_to_travel = christofides(services, stations)
    print(f"initial route {stations_to_travel}")
    print(get_full_route_timed(services, stations_to_travel))