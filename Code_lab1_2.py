import polars as pl
import pandas as pd
from geopy.distance import geodesic
from datetime import datetime
import time
import concurrent.futures
import os
import math
from geopy.distance import geodesic
import psutil
import matplotlib.pyplot as plt
import threading 
from collections import defaultdict
import requests
import zipfile
import io
import pyarrow

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

    # Round timestamp to the nearest minute
    df["timestamp_rounded"] = df["timestamp"].dt.floor(time_window)

    # Round latitude and longitude
    df["lat_rounded"] = df["Latitude"].round(decimals)
    df["lon_rounded"] = df["Longitude"].round(decimals)

    # Group by timestamp_rounded, lat_rounded, and lon_rounded
    grouped = df.groupby(["timestamp_rounded", "lat_rounded", "lon_rounded"]) \
                .agg(vessel_count=("MMSI", "count")) \
                .reset_index()

    # Filter anomalies where vessel count is >= min_vessels
    anomalies_df = grouped[grouped["vessel_count"] >= min_vessels]
    anomalies = list(anomalies_df.itertuples(index=False, name=None))

    return anomalies

##############################################################
# Function to get CPU and memory usage
def get_system_usage():
    return psutil.cpu_percent(interval=0.5), psutil.virtual_memory().percent

# Function to continuously monitor system usage
def monitor_usage(cpu_data, memory_data, stop_event):
    while not stop_event.is_set():
        cpu, mem = get_system_usage()
        cpu_data.append(cpu)
        memory_data.append(mem)
        time.sleep(0.5)  # Adjust the interval as needed

# Function to run parallel tasks
def run_parallel_with_monitoring(df_pandas, num_cpus=None, chunk_size=None):
    max_cpus = os.cpu_count()
    num_cpus = num_cpus if num_cpus else max_cpus
    num_cpus = min(num_cpus, max_cpus)
    
    chunk_size = chunk_size if chunk_size else math.ceil(len(df_pandas) / (2 * num_cpus))
    chunks = [df_pandas[i:i + chunk_size] for i in range(0, len(df_pandas), chunk_size)]
    
    cpu_usage_data = []
    memory_usage_data = []
    stop_event = threading.Event()
    
    # Start monitoring thread
    monitor_thread = threading.Thread(target=monitor_usage, args=(cpu_usage_data, memory_usage_data, stop_event))
    monitor_thread.start()
    
    start_time = time.time()
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_cpus) as executor:
        futures = [executor.submit(run_task, chunk) for chunk in chunks]
        for future in futures:
            future.result()  # Ensure tasks complete
    
    end_time = time.time()
    stop_event.set()  # Stop monitoring
    monitor_thread.join()

    return end_time - start_time, cpu_usage_data, memory_usage_data

# Function to run a single task
def run_task(chunk):
    detect_location_anomalies(chunk)
    detect_speed_anomalies(chunk)
    detect_neighboring_anomalies(chunk)

# Function to test configurations
def test_various_configurations(df_pandas):
    cpu_configs =[24,36, 48]
    chunk_configs = [50000, 150000, 250000]
    results = []

    for num_cpus in cpu_configs:
        for chunk_size in chunk_configs:
            print(f"\nTesting {num_cpus} CPUs, chunk size {chunk_size}...")
            execution_time, cpu_data, mem_data = run_parallel_with_monitoring(df_pandas, num_cpus, chunk_size)
            results.append({
                "num_cpus": num_cpus,
                "chunk_size": chunk_size,
                "execution_time": execution_time,
                "cpu_usage_data": cpu_data,
                "memory_usage_data": mem_data
            })
 
    return results

# Function to plot results
def plot_results(results, output_dir="."):
    df_results = pd.DataFrame(results)

    # Execution Time Plot
    plt.figure(figsize=(10, 6))
    for cpu_count in df_results['num_cpus'].unique():  # Loop through unique CPU counts
        subset = df_results[df_results['num_cpus'] == cpu_count]  # Filter rows with the same CPU count
        
        # Ensure the columns are numeric before plotting
        subset.loc[:, 'chunk_size'] = pd.to_numeric(subset['chunk_size'], errors='coerce')
        subset.loc[:, 'execution_time'] = pd.to_numeric(subset['execution_time'], errors='coerce')
        
        # Plotting: Ensure the columns are 1D arrays
        plt.plot(subset['chunk_size'].values, subset['execution_time'].values, label=f"{cpu_count} CPU(s)", marker='o')


    plt.title("Execution Time vs Chunk Size")
    plt.xlabel("Chunk Size")
    plt.ylabel("Execution Time (s)")
    plt.legend(title="Number of CPUs")
    plt.grid()
    plt.savefig(f"{output_dir}/execution_time_vs_chunk_size.png")  # Save the plot as a PNG file
    plt.close()  # Close the figure to free up memory

    # CPU Usage Plot
    plt.figure(figsize=(10, 6))
    for _, row in df_results.iterrows():
        plt.plot(row['cpu_usage_data'], label=f"CPU {row['num_cpus']}, Chunk {row['chunk_size']}")
    plt.title("CPU Usage Over Time")
    plt.xlabel("Time Step")
    plt.ylabel("CPU Usage (%)")
    plt.legend()
    plt.grid()
    plt.savefig(f"{output_dir}/cpu_usage_over_time.png")  # Save the plot as a PNG file
    plt.close()

    # Memory Usage Plot
    plt.figure(figsize=(10, 6))
    for _, row in df_results.iterrows():
        plt.plot(row['memory_usage_data'], label=f"CPU {row['num_cpus']}, Chunk {row['chunk_size']}")
    plt.title("Memory Usage Over Time")
    plt.xlabel("Time Step")
    plt.ylabel("Memory Usage (%)")
    plt.legend()
    plt.grid()
    plt.savefig(f"{output_dir}/memory_usage_over_time.png")  # Save the plot as a PNG file
    plt.close()

    # Summary
    print("\nPerformance Summary:")
    print(df_results.drop(columns=['cpu_usage_data', 'memory_usage_data']))

##############################################################
if __name__ == '__main__':
    # Load data
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

    # Filter out invalid rows based on latitude and longitude
    df = df.filter(
        (pl.col('Latitude').is_not_null()) & 
        (pl.col('Longitude').is_not_null()) & 
        (pl.col('Latitude').is_between(-90, 90)) & 
        (pl.col('Longitude').is_between(-180, 180))
    )

    df = df.collect()
    print("Data loaded successfully")
    
    # Test different configurations and plot results
    results = test_various_configurations(df)
    plot_results(results)
