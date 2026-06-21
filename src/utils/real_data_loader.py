import os
import urllib.request
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.utils.logger import logger

def auto_detect_columns(cols, custom_mapping=None):
    """
    Auto-detects target columns in the dataset using predefined matching groups.
    Supports a custom mapping to override detection.
    """
    mapping = {
        "amount": ["amount", "valor", "amt", "price", "value", "tx_amount"],
        "timestamp": ["time", "timestamp", "date", "datetime", "data", "dt", "epoch"],
        "user_id": ["user", "user_id", "customer", "customer_id", "sender", "sender_id", "client", "client_id", "src", "source", "orig"],
        "recipient_id": ["recipient", "recipient_id", "receiver", "receiver_id", "dest", "destination", "to", "target"],
        "device_id": ["device", "device_id", "dev", "dev_id", "ip", "ip_address", "fingerprint"],
        "latitude": ["lat", "latitude", "coord_lat", "y"],
        "longitude": ["lon", "lng", "longitude", "coord_lon", "x"],
        "is_fraud": ["fraud", "class", "label", "target", "is_fraud"]
    }
    
    result = {}
    if custom_mapping:
        for target, raw_col in custom_mapping.items():
            if raw_col in cols:
                result[target] = raw_col
                
    for target, candidates in mapping.items():
        if target in result:
            continue
        # Find first matching candidate (exact case-insensitive)
        for c in candidates:
            for col in cols:
                if col.lower() == c.lower():
                    result[target] = col
                    break
            if target in result:
                break
        
        # If still not found, try substring matching
        if target not in result:
            for col in cols:
                for c in candidates:
                    if c.lower() in col.lower():
                        result[target] = col
                        break
                if target in result:
                    break
                    
    return result

def download_and_map_real_dataset(save_path="data/real_transactions.csv", limit_samples=15000, dataset_url=None, column_mapping=None):
    """
    Downloads a transaction dataset from the web and dynamically maps its schema
    to the Fraud Intelligence schema (transaction_id, user_id, timestamp, amount, lat/lon, device_id, recipient_id, is_fraud).
    """
    default_url = "https://raw.githubusercontent.com/nsethi31/Kaggle-Data-Credit-Card-Fraud-Detection/master/creditcard.csv"
    url = dataset_url or default_url
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    temp_path = "data/temp_downloaded_dataset.csv"
    
    try:
        # Download dataset if file doesn't exist
        if not os.path.exists(save_path):
            logger.info(f"Downloading real dataset from URL: {url}...")
            urllib.request.urlretrieve(url, temp_path)
            logger.info("Dataset download completed.")
            
            # Read first N rows or full dataset
            logger.info(f"Reading downloaded dataset (limiting to {limit_samples} rows)...")
            df_raw = pd.read_csv(temp_path, nrows=limit_samples)
            
            # Detect columns
            detected = auto_detect_columns(df_raw.columns, column_mapping)
            logger.info(f"Auto-detected column mapping: {detected}")
            
            # 1. Map transaction_id
            if "transaction_id" in detected:
                transaction_ids = df_raw[detected["transaction_id"]].astype(str).tolist()
            else:
                transaction_ids = [f"T{i:06d}" for i in range(len(df_raw))]
            
            # 2. Map amount
            if "amount" in detected:
                amounts = df_raw[detected["amount"]].astype(float).round(2).tolist()
            else:
                amounts = [100.0] * len(df_raw)
                
            # 3. Map is_fraud
            if "is_fraud" in detected:
                # Convert potential string labels to 0/1
                raw_labels = df_raw[detected["is_fraud"]]
                if raw_labels.dtype == object:
                    is_frauds = raw_labels.apply(lambda x: 1 if str(x).strip().lower() in ['1', 'true', 'yes', 'fraud'] else 0).tolist()
                else:
                    is_frauds = raw_labels.astype(int).tolist()
            else:
                is_frauds = [0] * len(df_raw)
                
            # 4. Map timestamp
            base_time = datetime.now() - timedelta(days=30)
            if "timestamp" in detected:
                time_col = df_raw[detected["timestamp"]]
                if pd.api.types.is_numeric_dtype(time_col):
                    # Numeric column like seconds from start
                    timestamps = [ (base_time + timedelta(seconds=float(s))).isoformat() for s in time_col ]
                else:
                    timestamps = pd.to_datetime(time_col).dt.strftime('%Y-%m-%dT%H:%M:%S').tolist()
            else:
                timestamps = [ (base_time + timedelta(seconds=i*60)).isoformat() for i in range(len(df_raw)) ]
                
            # 5. Map user_id (fallback to PCA V1 column or sequential IDs if missing)
            if "user_id" in detected:
                user_ids = df_raw[detected["user_id"]].astype(str).tolist()
            elif "V1" in df_raw.columns:
                # Backward-compatibility for standard Kaggle Credit Card dataset
                user_ids = [ f"U{int(abs(v1) * 10) % 200:03d}" for v1 in df_raw["V1"] ]
            else:
                user_ids = [ f"U{i % 200:03d}" for i in range(len(df_raw)) ]
                
            # 6. Map latitude & longitude (fallback to PCA V2/V3 or São Paulo centered noise)
            if "latitude" in detected and "longitude" in detected:
                latitudes = df_raw[detected["latitude"]].astype(float).tolist()
                longitudes = df_raw[detected["longitude"]].astype(float).tolist()
            elif "V2" in df_raw.columns and "V3" in df_raw.columns:
                # Backward-compatibility for standard Kaggle Credit Card dataset
                latitudes = [ round(-23.5505 + float(v2) * 0.1, 5) for v2 in df_raw["V2"] ]
                longitudes = [ round(-46.6333 + float(v3) * 0.1, 5) for v3 in df_raw["V3"] ]
            else:
                # Generate random points centered around São Paulo, Brazil
                np.random.seed(42)
                latitudes = [ round(-23.5505 + np.random.normal(0, 0.05), 5) for _ in range(len(df_raw)) ]
                longitudes = [ round(-46.6333 + np.random.normal(0, 0.05), 5) for _ in range(len(df_raw)) ]
                
            # 7. Map device_id (fallback to PCA V4 or sequential devices)
            if "device_id" in detected:
                device_ids = df_raw[detected["device_id"]].astype(str).tolist()
            elif "V4" in df_raw.columns:
                # Backward-compatibility for standard Kaggle Credit Card dataset
                device_ids = [ f"DEV{int(abs(v4) * 100) % 100:04d}" for v4 in df_raw["V4"] ]
            else:
                device_ids = [ f"DEV{i % 100:04d}" for i in range(len(df_raw)) ]
                
            # 8. Map recipient_id
            if "recipient_id" in detected:
                recipient_ids = df_raw[detected["recipient_id"]].astype(str).tolist()
            else:
                recipient_ids = [ f"REC_{dev[3:]}" if dev.startswith("DEV") else f"REC_{dev}" for dev in device_ids ]
                
            # Create final mapped DataFrame
            df_mapped = pd.DataFrame({
                "transaction_id": transaction_ids,
                "user_id": user_ids,
                "timestamp": timestamps,
                "amount": amounts,
                "latitude": latitudes,
                "longitude": longitudes,
                "device_id": device_ids,
                "recipient_id": recipient_ids,
                "is_fraud": is_frauds
            })
            
            # Save the mapped dataset
            df_mapped.to_csv(save_path, index=False)
            logger.info(f"Mapped dataset successfully saved to {save_path}. Total frauds: {df_mapped['is_fraud'].sum()}/{len(df_mapped)}")
            
            # Clean up the raw file
            if os.path.exists(temp_path):
                os.remove(temp_path)
        else:
            logger.info(f"Using existing mapped dataset at {save_path}")
            df_mapped = pd.read_csv(save_path)
            
        return df_mapped
        
    except Exception as e:
        logger.exception("Failed to download or map transaction dataset.")
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise e
