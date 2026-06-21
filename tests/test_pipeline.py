import pytest
import pandas as pd
import numpy as np
from src.pipeline.ingestion import DataIngestion
from src.pipeline.feature_engineering import FeatureEngineer
from src.pipeline.preprocessing import DataPreprocessor

@pytest.fixture
def sample_raw_df():
    ingestion = DataIngestion(sample_path="data/test_sample_transactions.csv")
    df = ingestion.generate_synthetic_data(num_samples=100, fraud_ratio=0.1, seed=42)
    yield df
    import os
    if os.path.exists("data/test_sample_transactions.csv"):
        os.remove("data/test_sample_transactions.csv")

def test_ingestion(sample_raw_df):
    assert len(sample_raw_df) == 100
    assert "transaction_id" in sample_raw_df.columns
    assert "user_id" in sample_raw_df.columns
    assert "amount" in sample_raw_df.columns
    assert "is_fraud" in sample_raw_df.columns
    assert "recipient_id" in sample_raw_df.columns

def test_feature_engineering_batch(sample_raw_df):
    fe = FeatureEngineer()
    df_feat = fe.transform_batch(sample_raw_df)
    
    assert "dist_from_prev_km" in df_feat.columns
    assert "speed_kph" in df_feat.columns
    assert "velocity_count_1h" in df_feat.columns
    assert "amount_z_score" in df_feat.columns
    assert len(df_feat) == len(sample_raw_df)

def test_feature_engineering_online():
    fe = FeatureEngineer()
    history = [
        {"transaction_id": "T0", "user_id": "U1", "timestamp": "2026-06-21T10:00:00", "amount": 100.0, "latitude": -23.5, "longitude": -46.6, "device_id": "DEV1", "recipient_id": "REC1"}
    ]
    current = {
        "transaction_id": "T1", "user_id": "U1", "timestamp": "2026-06-21T10:05:00", "amount": 150.0, "latitude": -23.501, "longitude": -46.601, "device_id": "DEV1", "recipient_id": "REC1"
    }
    
    feats = fe.transform_online(current, history)
    
    assert feats["velocity_count_1h"] == 2
    assert feats["velocity_sum_1h"] == 250.0
    assert feats["device_changed"] == 0
    assert feats["dist_from_prev_km"] > 0.0

def test_preprocessing(sample_raw_df):
    fe = FeatureEngineer()
    df_feat = fe.transform_batch(sample_raw_df)
    
    preprocessor = DataPreprocessor(
        scaler_path="models/test_scaler.pkl", 
        target_encoder_path="models/test_target_encoder.pkl"
    )
    preprocessor.fit_scaler(df_feat)
    
    df_scaled = preprocessor.scale_features(df_feat)
    assert "scaled_amount" in df_scaled.columns
    assert "scaled_device_id_encoded" in df_scaled.columns
    
    seqs = preprocessor.create_sequences_for_lstm(df_scaled, seq_length=3)
    assert seqs.ndim == 3
    assert seqs.shape[1] == 3
    assert seqs.shape[2] == len(preprocessor.feature_cols)
    
    import os
    for f in ["models/test_scaler.pkl", "models/test_target_encoder.pkl"]:
        if os.path.exists(f):
            os.remove(f)
