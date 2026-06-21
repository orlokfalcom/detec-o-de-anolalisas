import os
import yaml
import numpy as np

def load_yaml(file_path):
    """
    Safely loads a YAML configuration file.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Configuration file not found: {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance in kilometers between two points 
    on the earth (specified in decimal degrees).
    """
    # Convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    c = 2.0 * np.arcsin(np.sqrt(a))
    r = 6371.0  # Radius of earth in kilometers
    
    return float(c * r)

def calculate_speed_kph(dist_km, time_diff_seconds):
    """
    Calculates speed in km/h given distance and time difference.
    """
    if time_diff_seconds <= 0:
        return 0.0
    hours = time_diff_seconds / 3600.0
    return float(dist_km / hours)
