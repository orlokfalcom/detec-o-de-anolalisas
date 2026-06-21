import os
import urllib.request
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.utils.logger import logger

def download_and_map_real_dataset(save_path="data/real_transactions.csv", limit_samples=15000):
    """
    Downloads the real Kaggle Credit Card Fraud Detection dataset from a public GitHub mirror
    and maps its PCA-transformed columns to our schema (user_id, timestamp, lat/lon, device_id, etc.).
    """
    raw_url = "https://raw.githubusercontent.com/nsethi31/Kaggle-Data-Credit-Card-Fraud-Detection/master/creditcard.csv"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # 1. Download dataset if not exists
    if not os.path.exists(save_path):
        logger.info(f"Downloading real fraud dataset from {raw_url}...")
        try:
            # We download to a temp file first, then parse it
            temp_path = "data/creditcard_raw.csv"
            if not os.path.exists(temp_path):
                urllib.request.urlretrieve(raw_url, temp_path)
                logger.info("Raw dataset download completed.")
            
            # Read first N rows or full dataset. 
            # We read a subset or full dataset, but subset is much faster for verification.
            logger.info("Reading raw dataset...")
            # We want to make sure we include enough fraud cases.
            # In the first 15,000 transactions, there are ~76 frauds (Class == 1).
            # This is a solid sample to train/test.
            df_raw = pd.read_csv(temp_path, nrows=limit_samples)
            
            # If the user wants to test on more data, we can read up to limit_samples.
            # Let's verify we have frauds in the dataset
            num_frauds = df_raw["Class"].sum()
            logger.info(f"Loaded {len(df_raw)} transactions, including {num_frauds} fraud cases.")
            
            # 2. Map schema to our project's schema
            logger.info("Mapping raw features to Fraud Intelligence schema...")
            
            # Time is in seconds from the first transaction in the dataset.
            # Let's map it to real timestamps starting from 30 days ago.
            base_time = datetime.now() - timedelta(days=30)
            timestamps = [ (base_time + timedelta(seconds=float(s))).isoformat() for s in df_raw["Time"] ]
            
            # Map user_id using PCA column V1 (deterministic group mapping)
            # We map users based on V1 values into 200 distinct users
            user_ids = [ f"U{int(abs(v1) * 10) % 200:03d}" for v1 in df_raw["V1"] ]
            
            # Map latitude and longitude using V2 and V3 (centered around São Paulo, Brazil)
            latitudes = [ round(-23.5505 + float(v2) * 0.1, 5) for v2 in df_raw["V2"] ]
            longitudes = [ round(-46.6333 + float(v3) * 0.1, 5) for v3 in df_raw["V3"] ]
            
            # Map device_id using V4 into 100 devices
            device_ids = [ f"DEV{int(abs(v4) * 100) % 100:04d}" for v4 in df_raw["V4"] ]
            
            # Create the final mapped dataframe
            df_mapped = pd.DataFrame({
                "transaction_id": [f"T{i:06d}" for i in range(len(df_raw))],
                "user_id": user_ids,
                "timestamp": timestamps,
                "amount": df_raw["Amount"].round(2),
                "latitude": latitudes,
                "longitude": longitudes,
                "device_id": device_ids,
                "is_fraud": df_raw["Class"]
            })
            
            # Save the mapped dataset
            df_mapped.to_csv(save_path, index=False)
            logger.info(f"Mapped real dataset saved to {save_path}")
            
            # Clean up the raw file to save disk space
            if os.path.exists(temp_path):
                os.remove(temp_path)
                
        except Exception as e:
            logger.exception("Failed to download or map real credit card dataset.")
            raise e
    else:
        logger.info(f"Using existing mapped dataset at {save_path}")
        df_mapped = pd.read_csv(save_path)
        
    return df_mapped
