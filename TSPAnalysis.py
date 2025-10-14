from TSPSolver import simple_route_finder, simple_improved, a_star, full_graph_traversal
from import_data import rename_cols, convert_cols_to_numeric
import polars as pl
import matplotlib.pyplot as plt
# Import 3D plot
import time
import random
from datetime import datetime

num_samples = 5
num_stations_range = [i for i in range(5, 6)]
depart_time = datetime(2025, 9, 1, 0, 0, 0)
change_time = 5
services = convert_cols_to_numeric(rename_cols(pl.read_csv("Services.csv", infer_schema=None)))
timings: dict[int, dict[str, list[tuple[float, float]]]] = {i: {"Simple": [], "A Star": [], "Simple Improved": []} for i in num_stations_range} 
# Form {num_stations: {"method_1": [(time_run_1, time_route_1), ..., (time_run_num_samples, time_route_num_samples)]}}
graph_traversal: dict[int, list[tuple[int, list[int]]]] = {i: [] for i in num_stations_range}
dm = pl.read_csv("ConnectionTimes.csv", infer_schema=None)
stations_list = dm.columns

def get_n_correct_stations(stations_list: list[str], n: int) -> list[str]:
    stations = []
    for i in range(n):
        next_station = random.choice(stations_list) # Add logic to prevent repeats?
        # Add logic to ensure stations are reachable
        while next_station in stations:
            next_station = random.choice(stations_list)
        stations.append(next_station)
    return stations

for num_stations in num_stations_range:
    for i in range(num_samples):
        stations = get_n_correct_stations(stations_list, num_stations)
        print(f"{num_stations=}, {i=}, {stations=}")

        # vvv Full Graph Traversal vvv
        start_time = time.time()
        _, route_times = full_graph_traversal(services, stations, depart_time, change_time=change_time)
        end_time = time.time()
        method_time_taken = end_time - start_time
        graph_traversal[num_stations].append((method_time_taken, route_times))
        # ^^^ Full Graph Traversal ^^^

        # vvv Simple vvv
        start_time = time.time()
        _, route_time_taken = simple_route_finder(services, stations, depart_time, dm, change_time=change_time)
        end_time = time.time()
        method_time_taken = end_time - start_time
        timings[num_stations]["Simple"].append((method_time_taken, route_time_taken))
        # ^^^ Simple ^^^

        # vvv A Star vvv
        start_time = time.time()
        _, route_time_taken = a_star(services, stations, depart_time, change_time=change_time)
        end_time = time.time()
        method_time_taken = end_time - start_time
        timings[num_stations]["A Star"].append((method_time_taken, route_time_taken))
        # ^^^ A Star ^^^

        # vvv Simple Improved vvv
        start_time = time.time()
        _, route_time_taken = simple_improved(services, stations, depart_time, change_time=change_time)
        end_time = time.time()
        method_time_taken = end_time - start_time
        timings[num_stations]["Simple Improved"].append((method_time_taken, route_time_taken))
        # ^^^ Simple Improved ^^^

print(timings)
# Now we have the timings, we want to find the average time as a percentage or multiple of the optimal time
# Will show a histogram showing the distributions of route times found by route traversal, then vertical lines showing the
# Route time for each method, with the key showing the average processing time for each method
# Times on x axis will be displayed as a multiple of the optimal time (as found by the full graph traversal method)

for num_stations, methods in timings.items():
    plt.figure(figsize=(16,9))
    max_y = 1
    for method, results in methods.items():
        means = [sum(x)/len(x) for x in zip(*results)]
        mean_route_time = means[1]
        mean_calc_time = means[0]
        plt.vlines(mean_route_time, ymin=0, ymax=max_y, label=f"{method}: {mean_calc_time:.3f}")
    plt.legend(loc="upper right")
    plt.xlabel("Route time (AVG - Mins)")
    plt.ylabel("Count")
    plt.show()