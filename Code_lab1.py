import polars as pl
import pandas as pd
from geopy.distance import geodesic
from datetime import datetime
import time
import concurrent.futures
import os
import math
from geopy.distance import geodesic
import matplotlib.pyplot as plt
from collections import defaultdict

# A. Identifying Location Anomalies
def detect_location_anomalies(data):
    anomalies = []
    vessel_data = {} 
    DISTANCE_THRESHOLD = 50
    SPEED_THRESHOLD = 100

    # Iterate over each row in the input data
    for row in data.iter_rows(named=True):
        mmsi, curr_lat, curr_lon, _, timestamp = row["MMSI"], row["Latitude"], row["Longitude"], row["SOG"], row["timestamp"]
    
        # If this is the first record for a specific vessel (MMSI), initialize its data
        if mmsi not in vessel_data:
            vessel_data[mmsi] = {"prev_lat": None, "prev_lon": None, "prev_time": None}

        # Retrieve the previous latitude, longitude, and timestamp for the current vessel (if available)
        prev_lat, prev_lon, prev_time = vessel_data[mmsi].values()

        # If there is previous data for this vessel, calculate distance and speed
        if prev_lat is not None and prev_lon is not None:
            distance = geodesic((prev_lat, prev_lon), (curr_lat, curr_lon)).nm 
            if isinstance(timestamp, datetime) and isinstance(prev_time, datetime):
                time_delta = (timestamp - prev_time).total_seconds() / 3600
            else:
                time_delta = (float(timestamp) - float(prev_time)) / 3600

            # Only consider comparisons if the time difference is positive
            if time_delta > 0:
                speed = distance / time_delta
                if distance > DISTANCE_THRESHOLD or speed > SPEED_THRESHOLD:
                    anomalies.append(row) 
        vessel_data[mmsi] = {"prev_lat": curr_lat, "prev_lon": curr_lon, "prev_time": timestamp}

    return anomalies 

# B. Analyzing Speed and Course Consistency
def detect_speed_anomalies(data, speed_threshold=30):
    anomalies = [] 
    vessel_data = {}
    
    # Iterate over each row 
    for row in data.iter_rows(named=True):
        mmsi, sog, timestamp = row["MMSI"], row["SOG"], row["timestamp"]
        
        if sog is None:
            continue
        
        # If this is the first record for a specific vessel (MMSI), initialize its data
        if mmsi not in vessel_data:
            vessel_data[mmsi] = {"prev_sog": None, "prev_time": None}
        
        # Retrieve the previous SOG and timestamp for the current vessel (if available)
        prev_sog, prev_time = vessel_data[mmsi].values()
        
        # If there is previous data for this vessel, compare the current and previous records
        if prev_sog is not None:
            if isinstance(timestamp, datetime) and isinstance(prev_time, datetime):
                time_delta = (timestamp - prev_time).total_seconds() / 3600
            else:
                time_delta = (float(timestamp) - float(prev_time)) / 3600

            # Only consider the comparison if the time difference is positive
            if time_delta > 0:
                # Calculate the change in speed
                speed_change = abs(sog - prev_sog)
                # If the change in speed exceeds the specified threshold, flag it as an anomaly
                if speed_change > speed_threshold:
                    anomalies.append(row)
        vessel_data[mmsi] = {"prev_sog": sog, "prev_time": timestamp}
    
    return anomalies

# C. Comparing Neighboring Vessel Data
# Function to detect neighboring anomalies (vessels too close at the same timestamp)
def detect_neighboring_anomalies(df, decimals=3, time_window='min', min_vessels=2):
    # Convert to pandas
    df = df.to_pandas()

    # Ensure timestamp is in datetime format
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    print("1")
    # Round timestamp to the nearest minute
    df["timestamp_rounded"] = df["timestamp"].dt.floor(time_window)
    print("2")
    # Round latitude and longitude
    df["lat_rounded"] = df["Latitude"].round(decimals)
    df["lon_rounded"] = df["Longitude"].round(decimals)

    # Group by timestamp_rounded, lat_rounded, and lon_rounded
    grouped = df.groupby(["timestamp_rounded", "lat_rounded", "lon_rounded"]) \
                .agg(vessel_count=("MMSI", "count")) \
                .reset_index()
    print("3")
    # Filter anomalies where vessel count is >= min_vessels
    anomalies_df = grouped[grouped["vessel_count"] >= min_vessels]
    anomalies = list(anomalies_df.itertuples(index=False, name=None))

    return anomalies

# Running code in parallel (or not)
def run_parallel(df_pandas, num_cpus=None):
    max_cpus = os.cpu_count()
    num_cpus = num_cpus if num_cpus is not None else max_cpus
    num_cpus = min(num_cpus, max_cpus)

    # Calculate the chunk size: divide the dataframe into chunks based on available CPUs
    chunk_size = math.ceil(len(df_pandas) / (2 * num_cpus))
    
    # Create a list of chunks of the dataframe
    chunks = [df_pandas[i:i + chunk_size] for i in range(0, len(df_pandas), chunk_size)]

    start_time = time.time()
    
    # Using ProcessPoolExecutor for better parallelism in CPU-bound tasks
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_cpus) as executor:
        # Submit tasks for each chunk
        future1 = executor.submit(detect_location_anomalies, chunks[0])
        future2 = executor.submit(detect_speed_anomalies, chunks[1] if len(chunks) > 1 else chunks[0])
        future3 = executor.submit(detect_neighboring_anomalies, chunks[2] if len(chunks) > 2 else chunks[0])
        # Get the results
        loc_anomalies = future1.result()
        speed_anomalies = future2.result()
        neighbor_anomalies = future3.result()
    
    end_time = time.time()
    
    print("\nParallel Execution:")
    print(f"Using {num_cpus} CPU(s)")
    print(f"Location Anomalies: {len(loc_anomalies)}")
    print(f"Speed Anomalies: {len(speed_anomalies)}")
    print(f"Neighboring Anomalies: {len(neighbor_anomalies)}")
    print(f"Total Time: {end_time - start_time:.4f} seconds")
    
    return end_time - start_time

if __name__ == '__main__':
    df_lazy = pl.scan_csv(
        "/scratch/lustre/home/lusa7563/big_data/Results/BD_1/aisdk-2025-01-22.csv",
        schema_overrides={"# Timestamp": pl.Utf8, "MMSI": pl.Int64, "Latitude": pl.Float64, "Longitude": pl.Float64, "SOG": pl.Float64}
    )

    df = (
        df_lazy.with_columns(
            pl.col("# Timestamp").str.strptime(pl.Datetime, "%d/%m/%Y %H:%M:%S").alias("timestamp")
        )
        .select(["MMSI", "Latitude", "Longitude", "SOG", "timestamp"])
    )

    # Filter out invalid rows based on latitude and longitude before running the parallel functions
    df = df.filter(
        (pl.col('Latitude').is_not_null()) & 
        (pl.col('Longitude').is_not_null()) &
        (pl.col('Latitude').is_between(-90, 90)) & 
        (pl.col('Longitude').is_between(-180, 180))
    )

    df = df.collect().head(3000000)
    print("Data is loaded")
    parallel_time = run_parallel(df,48)
    print("Parallel done")
    sequential_time = run_parallel(df, 1)
    
    speedup = sequential_time
    print(f"\nSpeedup (Sequential Time / Parallel Time): {speedup:.2f}x")
