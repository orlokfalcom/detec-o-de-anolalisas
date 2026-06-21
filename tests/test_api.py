import os
import pytest
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient
from api.app import app
from src.pipeline.pipeline import FraudPipeline
from src.detectors.isolation_forest import IsolationForestDetector
from src.detectors.lstm_autoencoder import LSTMAutoencoderDetector
from src.detectors.xgboost_detector import XGBoostDetector
from src.detectors.graph_triangulation import GraphTriangulationDetector
from src.ensemble.meta_learner import EnsembleMetaLearner
from src.utils.helpers import load_yaml

@pytest.fixture(scope="module")
def trained_models_setup():
    """
    Fits and saves models on a small set of records so the API doesn't use untrained stubs.
    """
    config = load_yaml("config/config.yaml")
    
    # Set low limits for quick training
    config["data"]["num_samples"] = 200
    config["models"]["lstm_autoencoder"]["epochs"] = 1
    
    pipeline = FraudPipeline(config)
    df_scaled = pipeline.run_training_pipeline()
    
    # Train and save models
    feat_matrix = pipeline.preprocessor.get_scaled_feature_matrix(df_scaled)
    y_true = df_scaled["is_fraud"].values
    
    iforest = IsolationForestDetector(model_path="models/isolation_forest.pkl")
    iforest.fit(feat_matrix)
    
    seq_len = config["models"]["lstm_autoencoder"]["sequence_length"]
    lstm_seq_data = pipeline.preprocessor.create_sequences_for_lstm(df_scaled, seq_length=seq_len)
    
    lstm_ae = LSTMAutoencoderDetector(
        model_path="models/lstm_autoencoder.pth",
        input_dim=feat_matrix.shape[1],
        sequence_length=seq_len
    )
    lstm_ae.fit(lstm_seq_data, epochs=1)
    
    # XGBoost
    xgb_det = XGBoostDetector(model_path="models/xgboost_model.pkl")
    xgb_det.fit(feat_matrix, y_true)
    
    # NetworkX
    graph_det = GraphTriangulationDetector(model_path="models/graph_network.pkl")
    graph_det.fit(df_scaled)
    
    # Meta Learner
    scores_matrix = np.random.uniform(0, 100, (200, 6))
    targets = np.random.randint(0, 2, 200)
    meta = EnsembleMetaLearner(meta_model_path="models/meta_model.pkl")
    meta.fit_meta_learner(scores_matrix, targets)
    
    # Reload model states in app to register trained files
    app.state.iforest_detector.load_model()
    app.state.lstm_detector.load_model()
    app.state.xgb_detector.load_model()
    app.state.graph_detector.load_model()
    app.state.meta_learner.load_meta_learner()
    app.state.pipeline.preprocessor.load_scaler()
    
    yield
    
    # Clean up test artifacts
    files = [
        "models/isolation_forest.pkl", 
        "models/lstm_autoencoder.pth", 
        "models/xgboost_model.pkl",
        "models/graph_network.pkl",
        "models/meta_model.pkl", 
        "models/scaler.pkl",
        "models/target_encoder.pkl"
    ]
    for f in files:
        if os.path.exists(f):
            try:
                os.remove(f)
            except Exception:
                pass

def test_api_predict(trained_models_setup):
    client = TestClient(app)
    
    payload = {
        "transaction_id": "T_TEST_1001",
        "user_id": "U999",
        "amount": 120.50,
        "timestamp": "2026-06-21T14:00:00",
        "latitude": -23.5505,
        "longitude": -46.6333,
        "device_id": "DEV9999",
        "recipient_id": "REC9999"
    }
    
    response = client.post("/predict", json=payload)
    
    assert response.status_code == 200
    data = response.json()
    assert data["transaction_id"] == "T_TEST_1001"
    assert "risk_score" in data
    assert "decision" in data
    assert "llm_explanation" in data
    assert "all_probabilities" in data
    assert "xgboost_probability" in data["all_probabilities"]
    assert "statistical_z_score" in data["all_probabilities"]
    assert "latency_ms" in data
