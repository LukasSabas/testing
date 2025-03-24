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
# Haversine formula to calculate the distance between two points on the Earth
def detect_neighboring_anomalies(data, radius_nm=1, time_threshold=1):  
    anomalies = []  
    all_points = []  
    vessel_ids = []  
    timestamps = []  
      
    # Earth's radius in nautical miles  
    R_nm = 3440.065  
      
    # Gather valid points and associated vessel info  
    for row in data.to_numpy():  
        mmsi, lat, lon, _, ts, _ = row  
        if lat is None or lon is None or not (-90 <= lat <= 90) or not (-180 <= lon <= 180):  
            continue  
        all_points.append((lat, lon))  
        vessel_ids.append(mmsi)  
        timestamps.append(ts)  
      
    if not all_points:  
        return anomalies  
      
    # Convert lat/lon from degrees to radians  
    points_rad = np.radians(np.array(all_points))  
    lats = points_rad[:, 0]  
    lons = points_rad[:, 1]  
      
    # Convert spherical coordinates to 3D Cartesian coordinates: x, y, z, scaled by Earth's radius in nm.  
    x = R_nm * np.cos(lats) * np.cos(lons)  
    y = R_nm * np.cos(lats) * np.sin(lons)  
    z = R_nm * np.sin(lats)  
    coords_3d = np.column_stack((x, y, z))  
      
    # Determine equivalent chord distance for the given arc distance (radius_nm)  
    # For small angles, chord_length ~ 2 * R_nm * sin(angle/2) where angle = radius_nm / R_nm  
    # Simplify using: chord_distance = 2 * R_nm * sin(radius_nm/(2*R_nm))  
    chord_threshold = 2 * R_nm * np.sin(radius_nm / (2 * R_nm))  
      
    # Build the KDTree for fast spatial queries  
    tree = KDTree(coords_3d)  
      
    # Query neighbors within chord_threshold for each point  
    for idx, (point, vessel_id, ts) in enumerate(zip(coords_3d, vessel_ids, timestamps)):  
        # Get indices of neighbors (including the point itself)  
        indices = tree.query_ball_point(point, r=chord_threshold)  
        for j in indices:  
            # Avoid self-comparison and intra-vessel comparisons  
            if j <= idx or vessel_ids[j] == vessel_id:  
                continue  
            # Check time difference. Allow both numeric timestamp or datetime conversion.  
            ts2 = timestamps[j]  
            try:  
                if isinstance(ts, (int, float)) and isinstance(ts2, (int, float)):  
                    time_diff = abs(ts - ts2) / 3600  # seconds to hours  
                else:  
                    time_diff = abs((datetime.fromtimestamp(ts) - datetime.fromtimestamp(ts2)).total_seconds()) / 3600  
            except Exception:  
                continue  
              
            if time_diff < time_threshold:  
                anomaly = {"MMSI_1": vessel_id, "MMSI_2": vessel_ids[j], "Issue": "Unusual proximity detected"}  
                anomalies.append(anomaly)  
    return anomalies  

# Runs anomaly detection in parallel
def process_anomalies_parallel(detect_func, vessel_groups, n_rows):
    anomalies = []

    total_rows = n_rows
    cpu_count = psutil.cpu_count(logical=False)
    chunksize = max(1, total_rows // cpu_count)

    start_time = time.time()
    # Run the anomaly detection in parallel
    with ProcessPoolExecutor(max_workers=cpu_count) as executor:
        results = executor.map(detect_func, (group[1] for group in vessel_groups), chunksize=chunksize)

    # Collect the results from all workers
    for result in results:
        anomalies.extend(result)

    return anomalies, time.time() - start_time


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

    df_pandas = df.collect().head(40000)

    vessel_groups = df_pandas.group_by('MMSI')

    # Process location anomalies
    location_anomalies_parallel, parallel_A = process_anomalies_parallel(detect_location_anomalies, vessel_groups)

    # Process speed anomalies
    speed_anomalies_parallel, parallel_B = process_anomalies_parallel(detect_speed_anomalies, vessel_groups)


    df_pandas = df_pandas.with_columns(pl.col("timestamp").dt.hour().alias("hour"))
    hourly_groups = df_pandas.group_by('hour')

    # Process neighbor anomalies
    neighbor_anomalies_parallel, parallel_C = process_anomalies_parallel(detect_neighboring_anomalies, hourly_groups)

