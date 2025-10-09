from TSPSolver import simple_route_finder, simple_improved, a_star, full_graph_traversal
from import_data import rename_cols, convert_cols_to_numeric
import polars as pl
import matplotlib.pyplot as plt
# Import 3D plot
import time
import random
from datetime import datetime

num_samples = 5
num_stations_range = [i for i in range(5, 15)]
stations_list = []
start_time = datetime(2025, 9, 1, 6, 0, 0)
change_time = 5
services = convert_cols_to_numeric(rename_cols(pl.read_csv("Services.csv", infer_schema=None)))
timings: dict[int, dict[str, list[tuple[int, int]]]] = {i: {"Simple": [], "A Star": [], "Simple Improved": []} for i in num_stations_range} 
# Form {num_stations: {"method_1": [(time_run_1, time_route_1), ..., (time_run_num_samples, time_route_num_samples)]}}
graph_traversal: dict[int, list[tuple[int, list[int]]]] = {i: [] for i in num_stations_range}

def get_n_correct_stations(stations_list: list[str], n: int) -> list[str]:
    stations = []
    for i in range(n):
        next_station = random.choice(stations_list) # Add logic to prevent repeats?
        # Add logic to ensure stations are reachable
        stations.append(next_station)
    return stations

for num_stations in num_stations_range:
    for i in range(num_samples):
        stations = get_n_correct_stations(stations_list, num_stations)
        # vvv Full Graph Traversal vvv
        start_time = time.time()
        _, route_times = full_graph_traversal(services, stations, start_time, change_time=change_time)
        end_time = time.time()
        method_time_taken = end_time - start_time
        graph_traversal[num_stations].append((method_time_taken, route_times))
        # ^^^ Full Graph Traversal ^^^

        # vvv Simple vvv
        start_time = time.time()
        _, route_time_taken = simple_route_finder(services, stations, start_time, change_time=change_time)
        end_time = time.time()
        method_time_taken = end_time - start_time
        timings[num_stations]["Simple"].append((method_time_taken, route_time_taken))
        # ^^^ Simple ^^^

        # vvv A Star vvv
        start_time = time.time()
        _, route_time_taken = a_star(services, stations, start_time, change_time=change_time)
        end_time = time.time()
        method_time_taken = end_time - start_time
        timings[num_stations]["A Star"].append((method_time_taken, route_time_taken))
        # ^^^ A Star ^^^

        # vvv Simple Improved vvv
        start_time = time.time()
        _, route_time_taken = simple_improved(services, stations, start_time, change_time=change_time)
        end_time = time.time()
        method_time_taken = end_time - start_time
        timings[num_stations]["Simple Improved"].append((method_time_taken, route_time_taken))
        # ^^^ Simple Improved ^^^

print(timings)
# Now we have the timings, we want to find the average time as a percentage or multiple of the optimal time
# Will show a histogram showing the distributions of route times found by route traversal, then vertical lines showing the
# Route time for each method, with the key showing the average processing time for each method
# Times on x axis will be displayed as a multiple of the optimal time (as found by the full graph traversal method)