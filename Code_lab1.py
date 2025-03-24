import polars as pl
from geopy.distance import geodesic
import numpy as np
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor
import time
import psutil
import matplotlib.pyplot as plt
import copy
from scipy.spatial import KDTree
from geopy.distance import geodesic

# A. Identifying Location Anomalies
def detect_location_anomalies(data):
    anomalies = []
    vessel_data = {}

    DISTANCE_THRESHOLD = 50  
    SPEED_THRESHOLD = 100  

    for row in data.to_numpy():
        mmsi, curr_lat, curr_lon, _, timestamp = row  

        if not (-90 <= curr_lat <= 90) or not (-180 <= curr_lon <= 180):
            continue

        if mmsi not in vessel_data:
            vessel_data[mmsi] = {"prev_lat": None, "prev_lon": None, "prev_time": None}
        
        prev_lat, prev_lon, prev_time = vessel_data[mmsi].values()

        if prev_lat is not None and prev_lon is not None:
            distance = geodesic((prev_lat, prev_lon), (curr_lat, curr_lon)).nm  
            if isinstance(timestamp, (int, float)) and isinstance(prev_time, (int, float)):
                time_delta = (timestamp - prev_time) / 3600  # Correct for numerical timestamps
            else:
                timestamp = datetime.fromtimestamp(timestamp)
                prev_time = datetime.fromtimestamp(prev_time)
                time_delta = (timestamp - prev_time).total_seconds() / 3600 

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
    
    for row in data.to_numpy():
        mmsi, _, _, sog, timestamp = row  

        if sog is None:
            continue
        
        if mmsi not in vessel_data:
            vessel_data[mmsi] = {"prev_sog": None, "prev_time": None}
        
        prev_sog, prev_time = vessel_data[mmsi].values()
        
        if prev_sog is not None:
            if isinstance(timestamp, (int, float)) and isinstance(prev_time, (int, float)):
                time_delta = (timestamp - prev_time) / 3600  # Correct for numerical timestamps
            else:
                timestamp = datetime.fromtimestamp(timestamp)
                prev_time = datetime.fromtimestamp(prev_time)
                time_delta = (timestamp - prev_time).total_seconds() / 3600  
            if time_delta > 0:
                speed_change = abs(sog - prev_sog)  
                if speed_change > speed_threshold:  
                    anomalies.append(row)
        
        vessel_data[mmsi] = {"prev_sog": sog, "prev_time": timestamp}
    
    return anomalies

# C. Comparing Neighboring Vessel Data
def detect_neighboring_anomalies(data, radius_nm=50):
    anomalies = []
    vessel_positions = {}

    for row in data.to_numpy():
        mmsi, lat, lon, _, timestamp, _ = row  
        if lat is None or lon is None or not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
            continue

        vessel_positions.setdefault(mmsi, []).append((lat, lon, timestamp))
    
    vessel_keys = list(vessel_positions.keys())
    coords = np.array([(lat, lon) for mmsi in vessel_keys for lat, lon, _ in vessel_positions[mmsi]])
    if len(coords) == 0:
        return anomalies

    tree = KDTree(coords)

    for i, mmsi1 in enumerate(vessel_keys):
        for lat1, lon1, time1 in vessel_positions[mmsi1]:
            indices = tree.query_ball_point([lat1, lon1], radius_nm * 1.852)
            for idx in indices:
                mmsi2_idx = idx // len(vessel_positions[mmsi1])  # Ensure proper indexing for mmsi2
                if mmsi2_idx < len(vessel_keys):
                    mmsi2 = vessel_keys[mmsi2_idx]
                    if mmsi1 == mmsi2:
                        continue
                    lat2, lon2, time2 = vessel_positions[mmsi2][idx % len(vessel_positions[mmsi2])]
                    time_delta = abs(time1 - time2) / 3600 if isinstance(time1, (int, float)) else abs((datetime.fromtimestamp(time1) - datetime.fromtimestamp(time2)).total_seconds()) / 3600   
                    if time_delta < 1:  
                        anomalies.append({"MMSI_1": mmsi1, "MMSI_2": mmsi2, "Issue": "Unusual proximity detected"})
    
    return anomalies

# Runs anomaly detection in parallel
def process_anomalies_parallel(detect_func, vessel_groups):
    start_time = time.time()
    anomalies = []
    with ProcessPoolExecutor(max_workers=psutil.cpu_count(logical=False)) as executor:
        results = executor.map(detect_func, (group[1] for group in vessel_groups), chunksize=10000)
    for result in results:
        anomalies.extend(result)
    return anomalies, time.time() - start_time

# Runs anomaly detection sequentially
def process_anomalies_non_parallel(detect_func, vessel_groups):
    start_time = time.time()
    anomalies = [result for _, group in vessel_groups for result in detect_func(group)]
    return anomalies, time.time() - start_time

# Performance analysis
def measure_performance(parallel_time, non_parallel_time):
    speedup = non_parallel_time / parallel_time if parallel_time > 0 else 0
    cpu_usage = psutil.cpu_percent(interval=1)
    mem_usage = psutil.virtual_memory().percent
    return speedup, cpu_usage, mem_usage

# Plots speedup analysis
def plot_speedup(parallel_times, non_parallel_times, labels):
    speedups = [n / p for n, p in zip(non_parallel_times, parallel_times)]
    plt.figure(figsize=(10, 5))
    plt.bar(labels, speedups, color='skyblue')
    plt.xlabel("Anomaly Type")
    plt.ylabel("Speedup Factor")
    plt.title("Speedup Analysis")
    plt.show()

if __name__ == '__main__':
    df_lazy = pl.scan_csv(
        "C:/Users/37068/Desktop/UNIVERSITETAS/Magistras/2 kursas/Didžiųjų duomenų analizė/Lab1/aisdk-2025-01-22.csv",
        schema_overrides={"# Timestamp": pl.Utf8, "MMSI": pl.Int64, "Latitude": pl.Float64, "Longitude": pl.Float64, "SOG": pl.Float64}
    )
    df = (
        df_lazy.with_columns(
            pl.col("# Timestamp").str.strptime(pl.Datetime, "%d/%m/%Y %H:%M:%S").alias("timestamp")
        )
        .select(["MMSI", "Latitude", "Longitude", "SOG", "timestamp"])
    )

    df_pandas = df.collect().head(400000)

    vessel_groups = df_pandas.group_by('MMSI')

    # Process location anomalies
    location_anomalies_parallel, parallel_A = process_anomalies_parallel(detect_location_anomalies, vessel_groups)
    location_anomalies_non_parallel, non_parallel_A = process_anomalies_non_parallel(detect_location_anomalies, vessel_groups)
    speedup_A, cpu_A, mem_A = measure_performance(parallel_A, non_parallel_A)

    # Process speed anomalies
    speed_anomalies_parallel, parallel_B = process_anomalies_parallel(detect_speed_anomalies, vessel_groups)
    speed_anomalies_non_parallel, non_parallel_B = process_anomalies_non_parallel(detect_speed_anomalies, vessel_groups)
    speedup_B, cpu_B, mem_B = measure_performance(parallel_B, non_parallel_B)

    df_pandas = df_pandas.with_columns(pl.col("timestamp").dt.hour().alias("hour"))
    hourly_groups = df_pandas.group_by('hour')

    # Process neighbor anomalies
    neighbor_anomalies_parallel, parallel_C = process_anomalies_parallel(detect_neighboring_anomalies, hourly_groups)
    neighbor_anomalies_non_parallel, non_parallel_C = process_anomalies_non_parallel(detect_neighboring_anomalies, hourly_groups)
    speedup_C, cpu_C, mem_C = measure_performance(parallel_C, non_parallel_C)

    # Print anomalies
    print(f"Location Anomalies count: {len(location_anomalies_parallel):.2f}")
    print(f"Speed Anomalies count: {len(speed_anomalies_parallel):.2f}")
    print(f"Neighboring Anomalies count: {len(neighbor_anomalies_parallel):.2f}")

    # Print results
    print(f"Location Anomalies: Speedup (sequential/parallel) {speedup_A:.2f}, CPU {cpu_A}%, Memory {mem_A}%")
    print(f"Speed Anomalies: Speedup (sequential/parallel) {speedup_B:.2f}, CPU {cpu_B}%, Memory {mem_B}%")
    print(f"Neighbor Anomalies: Speedup (sequential/parallel) {speedup_C:.2f}, CPU {cpu_C}%, Memory {mem_C}%")


    # Plot speedup analysis
    # plot_speedup([parallel_A, parallel_B], [non_parallel_A, non_parallel_B], ["Location", "Speed"])
    plot_speedup([parallel_A, parallel_B, parallel_C], [non_parallel_A, non_parallel_B, non_parallel_C], ["Location", "Speed", "Neighbor"])