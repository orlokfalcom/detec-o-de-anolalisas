import pandas as pd
import numpy as np
from datetime import datetime
from src.utils.helpers import haversine_distance, calculate_speed_kph
from src.utils.logger import logger

class FeatureEngineer:
    def __init__(self):
        pass

    def transform_batch(self, df):
        """
        Processes a raw pandas DataFrame of transactions.
        Computes velocity, behavioral, geographic, and device features.
        
        Input df columns:
        [transaction_id, user_id, timestamp, amount, latitude, longitude, device_id]
        """
        logger.info("Computing batch feature engineering...")
        df = df.copy()
        
        # Ensure timestamp is datetime and sort
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values(by=["user_id", "timestamp"]).reset_index(drop=True)
        
        # 1. Geographic & Velocity Features (Distance & Speed)
        df["prev_lat"] = df.groupby("user_id")["latitude"].shift(1)
        df["prev_lon"] = df.groupby("user_id")["longitude"].shift(1)
        df["prev_timestamp"] = df.groupby("user_id")["timestamp"].shift(1)
        df["prev_device"] = df.groupby("user_id")["device_id"].shift(1)
        
        # Calculate distances
        distances = []
        time_diffs = []
        speeds = []
        
        for idx, row in df.iterrows():
            if pd.isna(row["prev_lat"]) or pd.isna(row["prev_lon"]):
                distances.append(0.0)
                time_diffs.append(0.0)
                speeds.append(0.0)
            else:
                dist = haversine_distance(row["prev_lat"], row["prev_lon"], row["latitude"], row["longitude"])
                tdiff = (row["timestamp"] - row["prev_timestamp"]).total_seconds()
                speed = calculate_speed_kph(dist, tdiff)
                
                distances.append(dist)
                time_diffs.append(tdiff)
                speeds.append(speed)
                
        df["dist_from_prev_km"] = distances
        df["time_since_prev_sec"] = time_diffs
        df["speed_kph"] = speeds
        
        # Clean temporary shift columns
        df.drop(columns=["prev_lat", "prev_lon", "prev_timestamp", "prev_device"], inplace=True)
        
        # 2. Rolling Velocity Features (using sliding windows per user)
        # We index by timestamp to use pandas rolling on time offsets
        df_time_indexed = df.set_index("timestamp")
        
        # Group by user and compute rolling counts/sums
        # 1 min window
        r1m = df_time_indexed.groupby("user_id")["amount"].rolling("60s")
        df["velocity_count_1m"] = r1m.count().values
        df["velocity_sum_1m"] = r1m.sum().values
        
        # 5 min window
        r5m = df_time_indexed.groupby("user_id")["amount"].rolling("300s")
        df["velocity_count_5m"] = r5m.count().values
        df["velocity_sum_5m"] = r5m.sum().values
        
        # 1 hour window
        r1h = df_time_indexed.groupby("user_id")["amount"].rolling("3600s")
        df["velocity_count_1h"] = r1h.count().values
        df["velocity_sum_1h"] = r1h.sum().values
        
        # 3. Behavioral Features (Deviance from User Mean and Std)
        # We calculate running cumulative means and standard deviations to avoid lookahead bias
        # For simplicity in this demo batch, we can use expanding window or group averages. 
        # Expanding window is cleaner as it only uses past transactions for the user.
        expanding_stats = df.groupby("user_id")["amount"].expanding()
        user_cum_mean = expanding_stats.mean().reset_index(level=0, drop=True)
        user_cum_std = expanding_stats.std().reset_index(level=0, drop=True).fillna(0.0)
        
        df["user_mean_amount"] = user_cum_mean
        df["user_std_amount"] = user_cum_std
        
        # Z-score of current transaction amount
        df["amount_z_score"] = (df["amount"] - df["user_mean_amount"]) / (df["user_std_amount"] + 1e-5)
        
        # 4. Device Features
        # Number of unique devices in the last 24h
        df["device_count_24h"] = (
            df_time_indexed.groupby("user_id")["device_id"]
            .rolling("86400s")
            .apply(lambda x: len(np.unique(x)), raw=False)
            .values
        )
        
        # Flag indicating if this device is different from the last device
        df["device_changed"] = (df["device_id"] != df.groupby("user_id")["device_id"].shift(1)).astype(int)
        
        # Fill NaNs created by shifts or divisions
        df.fillna(0.0, inplace=True)
        return df

    def transform_online(self, tx_dict, user_history_list):
        """
        Generates engineered features for a single incoming transaction
        based on the user's historical transactions (list of dicts).
        
        tx_dict keys:
        [transaction_id, user_id, timestamp, amount, latitude, longitude, device_id]
        
        user_history_list: List of dictionaries of past transactions for this user.
        """
        curr_time = pd.to_datetime(tx_dict["timestamp"])
        curr_amount = float(tx_dict["amount"])
        curr_lat = float(tx_dict["latitude"])
        curr_lon = float(tx_dict["longitude"])
        curr_dev = tx_dict["device_id"]
        
        # Parse history
        history = pd.DataFrame(user_history_list)
        
        if history.empty:
            # First transaction for the user
            return {
                **tx_dict,
                "dist_from_prev_km": 0.0,
                "time_since_prev_sec": 0.0,
                "speed_kph": 0.0,
                "velocity_count_1m": 1.0,
                "velocity_sum_1m": curr_amount,
                "velocity_count_5m": 1.0,
                "velocity_sum_5m": curr_amount,
                "velocity_count_1h": 1.0,
                "velocity_sum_1h": curr_amount,
                "user_mean_amount": curr_amount,
                "user_std_amount": 0.0,
                "amount_z_score": 0.0,
                "device_count_24h": 1.0,
                "device_changed": 0
            }
            
        # Ensure sorted history
        history["timestamp"] = pd.to_datetime(history["timestamp"])
        history = history.sort_values(by="timestamp").reset_index(drop=True)
        
        # Get last transaction
        last_tx = history.iloc[-1]
        
        # 1. Geographic Features
        dist = haversine_distance(last_tx["latitude"], last_tx["longitude"], curr_lat, curr_lon)
        time_diff = (curr_time - last_tx["timestamp"]).total_seconds()
        speed = calculate_speed_kph(dist, time_diff)
        
        # 2. Rolling Velocity Features
        # Include current transaction in calculation
        window_1m = history[history["timestamp"] >= (curr_time - pd.Timedelta(seconds=60))]
        v_cnt_1m = len(window_1m) + 1
        v_sum_1m = float(window_1m["amount"].sum()) + curr_amount
        
        window_5m = history[history["timestamp"] >= (curr_time - pd.Timedelta(seconds=300))]
        v_cnt_5m = len(window_5m) + 1
        v_sum_5m = float(window_5m["amount"].sum()) + curr_amount
        
        window_1h = history[history["timestamp"] >= (curr_time - pd.Timedelta(seconds=3600))]
        v_cnt_1h = len(window_1h) + 1
        v_sum_1h = float(window_1h["amount"].sum()) + curr_amount
        
        # 3. Behavioral Features
        mean_amt = float(history["amount"].mean())
        std_amt = float(history["amount"].std())
        if pd.isna(std_amt):
            std_amt = 0.0
            
        z_score = 0.0 if std_amt == 0.0 else (curr_amount - mean_amt) / std_amt
        
        # 4. Device Features
        window_24h = history[history["timestamp"] >= (curr_time - pd.Timedelta(seconds=86400))]
        unique_devices = set(window_24h["device_id"].tolist())
        unique_devices.add(curr_dev)
        dev_cnt_24h = len(unique_devices)
        
        dev_changed = int(curr_dev != last_tx["device_id"])
        
        return {
            **tx_dict,
            "dist_from_prev_km": dist,
            "time_since_prev_sec": time_diff,
            "speed_kph": speed,
            "velocity_count_1m": v_cnt_1m,
            "velocity_sum_1m": v_sum_1m,
            "velocity_count_5m": v_cnt_5m,
            "velocity_sum_5m": v_sum_5m,
            "velocity_count_1h": v_cnt_1h,
            "velocity_sum_1h": v_sum_1h,
            "user_mean_amount": mean_amt,
            "user_std_amount": std_amt,
            "amount_z_score": z_score,
            "device_count_24h": float(dev_cnt_24h),
            "device_changed": dev_changed
        }
