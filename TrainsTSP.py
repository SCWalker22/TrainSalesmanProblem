from import_data import load_data, format_df
import polars as pl
from datetime import datetime
import os
import time

username: str = ""
password: str = ""

def run_all(

    ):
    """
    
    """
    current_directory = os.getcwd()
    print(current_directory)
    time_start_load = time.time()
    df = load_data(
        "https://api.rtt.io/api/v1",
        username,
        password,
        pl.read_csv(os.path.join(current_directory, "TrainSalesmanProblem", "StationMap.csv")),
        datetime.today()
    )
    time_end_load = time.time()
    print(f"Time taken to load {time_end_load - time_start_load}")
    print(df)
    time_start_format = time.time()
    df = format_df(df)
    time_end_format = time.time()
    print(f"Time taken to format {time_end_format - time_start_format}")
    print(df)

if __name__ == "__main__":
    run_all()