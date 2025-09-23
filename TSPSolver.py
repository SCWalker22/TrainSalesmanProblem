import polars as pl
from datetime import datetime, timedelta
from import_data import rename_cols, convert_cols_to_numeric
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
        try:
            quickest_service = services_stopping_at_station.with_columns(pl.col("arrival").cast(pl.Int64)).sort(by="arrival", nulls_last = True)[0]
            arrival_time = int(quickest_service["arrival"].to_list()[0])
            quickest_uid = quickest_service["serviceUid"].to_list()[0]
            departure_time = int(services.filter(pl.col("crs") == station).filter(pl.col("serviceUid") == quickest_uid)["departure"].to_list()[0])
            hours = int(str(arrival_time)[:2])
            minutes = int(str(arrival_time)[2:])
            start_time = datetime(datetime.today().year, datetime.today().month, datetime.today().day,
                    hours, minutes, 0) + timedelta(minutes=change_time)
            timings.append({"departs": station, "arrives": next_station,
                    "departure_time": departure_time, "arrival_time": arrival_time})
        except IndexError:
            print(f"Unable to connect {station}, {next_station} after {start_time}")
            return []            

    return timings

def get_connection_time(
    stations: list[str],
    services: pl.DataFrame,
    start_time: datetime = datetime(2025, 9, 1, 8, 0, 0),
    change_time: int = 5

    ) -> int:
    """
    """
    timings = get_service_times(services, stations, start_time, change_time)
    if timings != []:
        first_depart = timings[0]["departure_time"]
        last_arrive = timings[-1]["arrival_time"]
        print(f"{first_depart=}, {last_arrive=}")
        time_taken = last_arrive - first_depart
        return time_taken
    return 0

def is_direct_path(
    station_start: str,
    station_end: str,
    services: pl.DataFrame
    
    ) -> bool:
    services_from_station_uids = services.filter(pl.col("crs") == station_start)["serviceUid"]
    services_to_station = services.filter(pl.col("serviceUid").is_in(services_from_station_uids)).filter(pl.col("crs") == station_end)
    if services_to_station.is_empty():
        return False
    return True

def is_all_direct_path(
    services: pl.DataFrame,
    stations: list[str]

    ) -> bool:
    """
    """
    for index, station in enumerate(stations[:-1]):
        next_station = stations[index + 1]
        if not is_direct_path(station, next_station, services):
            return False
    return True

def extract_possible_routes(
    connection_dict: dict[str, list[str]],
    station_start: str,
    max_length: int = 10

    ) -> list[list[str]]:
    """
    """
    output: list[list[str]] = []
    # for station in connection_dict[station_start]:
    #     route: list[str] = [station]
    def extend_route(station, route):
        route.append(station)
        print(f"appending {station} to {route}")
        if connection_dict.get(station, []) == [] or len(route) >= max_length:
            output.append(route[:])
        else:
            for next_station in connection_dict.get(station, []):
                if next_station not in route:
                    already_found: bool = False
                    for each_route in output:
                        if each_route != []:
                            if each_route[-1] == station:
                                already_found = True
                    if not already_found:
                        extend_route(next_station, route)
        # route.pop()
        output.append(route)
        try:
            route.pop()#  - should this be uncommented
        except IndexError:
            print(f"{route}, {station}")
        # next_station = station
        # # while connection_dict.get(station, []) != []:
        # for next_station in connection_dict.get(next_station, []):
        #     route.append(next_station)
        #     print(f"appended {next_station} to {route}")
        # output.append(route)
    extend_route(station_start, [])
    output = [route for route in output if route != []]
    return output

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
    # if station_start == "EXD":
    #     print(services_via_start.filter(pl.col("crs") == "EXM"))
    #     print(services.filter(pl.col("crs") == "EXM").select(["arrival", "departure", "serviceUid"]))
    #     print(f"{valid_services_out=}, {services_via_start=}")
    arrivals = fill_first_arrival_table(arrivals, services_via_start, station_start)
    return arrivals

def find_first_arrival(
        services: pl.DataFrame,
        station_start: str,
        station_end: str,
        start_time: datetime = datetime(2025, 9, 1, 8, 0, 0),
        change_time: int = 5,
        extra_check_depth: int = 3

    ) -> tuple[datetime, list[str]]:
    """
    Might need to prevent double counting - checked stations - but check again if earlier arrival
    """
    if is_direct_path(station_start, station_end, services):
        path: list[str] = [station_start, station_end]
    else:
        # Generate path by finding earliest arrival
        stations = services["crs"].unique()
        arrivals: pl.DataFrame = pl.DataFrame({"crs": stations, "arrival": None, "serviceFrom": None})
        arrivals = find_timed_stations(services, station_start, arrivals, start_time)
        arrival_time: int | None = arrivals.filter(pl.col("crs") == station_end)["arrival"][0]
        # count = 0
        extra_checks: list[bool] = [False for _ in range(extra_check_depth)]
        checked_stations: list[str] = []
        
        while arrival_time is None or not all(extra_checks):
            stations_reachable = arrivals.filter(pl.col("arrival").is_not_null())["crs"].unique().to_list()
            for station in stations_reachable:
                arrive_at_start_time = str(arrivals.filter(pl.col("crs") == station)["arrival"][0]).zfill(4) # Enforce 4 digit
                new_start_time = datetime(2025, 9, 1, int(arrive_at_start_time[:-2]), int(arrive_at_start_time[-2:]), 0) + timedelta(minutes=change_time)
                arrivals = find_timed_stations(services, station, arrivals, new_start_time)
                arrival_time: int | None = arrivals.filter(pl.col("crs") == station_end)["arrival"][0]

            if arrival_time is not None:
                for index, bool_val in enumerate(extra_checks):
                    if not bool_val:
                        extra_checks[index] = True
                        break

            checked_stations += stations_reachable
        #     count += 1
        #     print(f"{count=}, {arrivals.filter(pl.col('crs') == station_end)}")
        # print(f"{arrivals}, {arrival_time}") # Need to then find which route gave this time
        route = find_route(arrivals, station_start, station_end)
        arrival_time_str = str(arrival_time).zfill(4) # Enforce 4 digit
        start_time = datetime(2025, 9, 1, int(arrive_at_start_time[:-2]), int(arrive_at_start_time[-2:]), 0) + timedelta(minutes=change_time)
    return start_time, route

def get_full_route_timed(
        services: pl.DataFrame,
        disconnected_route: list[str],
        start_time: datetime = datetime(2025, 9, 1, 8, 0, 0),
        change_time: int = 5

    ) -> tuple[datetime, list[str]]:
    """
    
    """
    route: list[str] = [disconnected_route[0]]
    for index in range(len(disconnected_route) - 1):
        station_start = disconnected_route[index]
        station_end = disconnected_route[index + 1]
        start_time, partial_route = find_first_arrival(services, station_start, station_end, start_time, change_time=change_time)
        route += partial_route[1:]
    return start_time, route


# def find_path_between_stations(
#     station_start: str,
#     station_end: str,
#     services: pl.DataFrame

#     ) -> list[str]:
#     """
#     """
#     full_path = [station_start, station_end]
#     best_connection = []
#     if not is_all_direct_path(services, full_path):
#         print("No full direct route")
#         # while not is_direct_path(station, next_station, services):
#         fastest_connection: int = 0
#         services_from_station = services.filter(pl.col("crs") == station_start)["serviceUid"]
#         possible_next_stations = services.filter(pl.col("serviceUid").is_in(services_from_station))["crs"].unique().to_list()
#         print(f"{station_start} to {station_end} can go via {possible_next_stations}")
#         found_route: bool = False
#         for possible_intermediary in possible_next_stations:
#             full_route = [station_start, possible_intermediary, station_end]
#             if is_all_direct_path(services, full_route):
#                 connection_time = get_connection_time(services, full_route)
#                 if connection_time < fastest_connection or fastest_connection == 0:
#                     if connection_time != 0:
#                         print(f"Success for {station_start}, {possible_intermediary}, {station_end}: {connection_time} vs {fastest_connection}")
#                         fastest_connection = connection_time
#                         found_route = True
#                         best_connection = full_route
#         possible_multiple_intermediaries: list[dict[str, list[str]]] = {station_start: possible_next_stations}
#         while not found_route:
#             for intermediary in possible_multiple_intermediaries[station_start]:
#                 possible_next_stations: list[str] = []
#                 services_from_intermediary = services.filter(pl.col("crs") == intermediary)["serviceUid"]
#                 possible_next = services.filter(pl.col("serviceUid").is_in(services_from_intermediary))["crs"].unique().to_list()
#                 if station_start in possible_next: # Filter out any unwanted - this needs to be remade
#                     possible_next.remove(station_start)
#                 for station_list in possible_multiple_intermediaries:
#                     for station in station_list:
#                         if station in possible_next:
#                             possible_next.remove(station)
#                 for station in possible_next_stations:
#                     if station in possible_next:
#                         possible_next.remove(station)
#                 possible_next_stations += possible_next
#                 possible_multiple_intermediaries[intermediary] = possible_next_stations
#             print(possible_multiple_intermediaries)
#             # Now build subroutes from all combinations of routes
#             fastest_connection: int = 0
#             # print(possible_multiple_intermediaries)
#             # for combo in itertools.product(*possible_multiple_intermediaries):
#             #     print(list(combo))
#             possible_routes = extract_possible_routes(possible_multiple_intermediaries, station_start)
#             for route in possible_routes:
#                 full_route: list[str] = route + [station_end]
#                 if is_all_direct_path(services, full_route):
#                     print(f"Success for {full_route}")
#                     connection_time = get_connection_time(full_route, services)
#                     # If this was in invalid route, we will just fill with 0 again
#                     if connection_time < fastest_connection or fastest_connection == 0:
#                         if connection_time != 0:
#                             fastest_connection = connection_time
#                             best_connection = full_route
#                             found_route = True
#         return best_connection
#     return full_path

# def get_full_service_stops(
#         services: pl.DataFrame,
#         stations: list[str]
    
#     ) -> list[str]:
#     """
#     """
#     for index, station in enumerate(stations[:-1]):
#         next_station = stations[index+1]
#         print(f"checking {station} to {next_station}")
#         # Loop through stations, an try to find next station
#         if is_direct_path(station, next_station, services):
#             print(f"Direct between {station}, {next_station}")
#             continue
#         else:
#             # No direct path, so we need to find a (the quickest ideally) path
#             sub_path = find_path_between_stations(station, next_station, services)

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


if __name__ == "__main__":
    services = convert_cols_to_numeric(rename_cols(pl.read_csv("Services.csv", infer_schema=None)))
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
    stations = ['MYB', 'BDS', 'CST', 'CHX', 'CTK', 'EPH', 'EUS', 'ZFD', 'FST', 'HOX', 'KGX', 'LST', 'LBG', 'MOG', 'OLD', 'PAD', 'SDC', 'STP', 'TCR', 'VXH', 'VIC', 'BFR', 'WAT', 'WAE']
    # print(extract_possible_routes({"EXD": ["EXC", "SJP", "PLY"], "EXC": ["SJP", "WAT"], "SJP": ["WAT"]}, "EXD"))
    # get_full_service_stops(services, ["EXD", "IPS", "EXC", "EXD", "SJP", "EXD", "TAU", "PLY"])
    print(get_full_route_timed(services, ["EXD", "IPS", "EXC", "EXD", "SJP", "EXD", "TAU", "PLY"]))
    # print(find_first_arrival(services, "PLY", "HON"))
    # time_christofides_start = time.time()
    # stations_to_travel = christofides(services, stations)
    # time_christofides_end = time.time()
    # print(f"took {time_christofides_end - time_christofides_start} seconds")
    # print(stations_to_travel)
    # print(get_service_times(services, ["EXD", "EXC", "EXD", "SJP", "EXD", "TAU", "PLY"], datetime(2025, 9, 2, 12, 0, 0)))
    # print(services.filter(pl.col("crs") == "EXC"))