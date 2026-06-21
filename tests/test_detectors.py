import pytest
import numpy as np
import networkx as nx
from src.detectors.statistical import StatisticalDetector
from src.detectors.rules import RulesDetector
from src.detectors.isolation_forest import IsolationForestDetector
from src.detectors.lstm_autoencoder import LSTMAutoencoderDetector
from src.detectors.xgboost_detector import XGBoostDetector
from src.detectors.graph_triangulation import GraphTriangulationDetector
from src.ensemble.meta_learner import EnsembleMetaLearner
from src.decision.risk_engine import RiskDecisionEngine

def test_statistical_detector():
    detector = StatisticalDetector(z_score_threshold=2.0)
    history = [
        {"amount": 90.0},
        {"amount": 100.0},
        {"amount": 110.0}
    ]
    
    tx_normal = {"amount": 100.0}
    score_normal, _ = detector.predict_score(tx_normal, history)
    assert score_normal <= 30.0
    
    tx_anomalous = {"amount": 200.0}
    score_anomalous, details_anomalous = detector.predict_score(tx_anomalous, history)
    assert score_anomalous >= 60.0
    assert details_anomalous["z_score_triggered"] is True

def test_rules_detector():
    detector = RulesDetector("config/rules.yaml")
    tx_normal = {
        "amount": 100.0,
        "timestamp": "2026-06-21T12:00:00",
        "velocity_count_1m": 1,
        "speed_kph": 50.0,
        "device_count_24h": 1
    }
    score, details = detector.predict_score(tx_normal)
    assert score == 0.0
    assert len(details["triggered_rules"]) == 0
    
    tx_night = {
        "amount": 2000.0,
        "timestamp": "2026-06-21T23:00:00",
        "velocity_count_1m": 1,
        "speed_kph": 50.0,
        "device_count_24h": 1
    }
    score_night, details_night = detector.predict_score(tx_night)
    assert score_night > 0.0
    assert "bacen_night_limit" in details_night["triggered_rules"]

def test_decision_engine():
    engine = RiskDecisionEngine()
    
    dec, _ = engine.evaluate_decision(15)
    assert dec == "APPROVE"
    dec, _ = engine.evaluate_decision(45)
    assert dec == "MONITOR"
    dec, _ = engine.evaluate_decision(75)
    assert dec == "REVIEW"
    dec, _ = engine.evaluate_decision(95)
    assert dec == "BLOCK"

def test_graph_detector():
    detector = GraphTriangulationDetector()
    
    # Empty graph
    tx = {"user_id": "U1", "recipient_id": "U2", "amount": 100.0, "timestamp": "2026-06-21T12:00:00"}
    score, details = detector.predict_score(tx)
    assert score == 0.0
    assert details["cycle_detected"] is False
    
    # Populate a triangulation sequence: U1 -> U2, U2 -> U3
    detector.G.add_edge("U1", "U2", amount=100.0, timestamp="2026-06-21T12:00:00")
    detector.G.add_edge("U2", "U3", amount=100.0, timestamp="2026-06-21T12:01:00")
    
    # Incoming Tx: U3 -> U1 (completes cycle U1 -> U2 -> U3 -> U1)
    tx_cycle = {"user_id": "U3", "recipient_id": "U1", "amount": 100.0, "timestamp": "2026-06-21T12:02:00"}
    score, details = detector.predict_score(tx_cycle)
    assert score == 90.0
    assert details["cycle_detected"] is True
    assert details["cycle_type"] == "Length-3 Cycle (Account Triangulation)"

def test_ensemble_meta_learner():
    weights = {
        "statistical": 0.15,
        "rules": 0.20,
        "isolation_forest": 0.15,
        "lstm": 0.15,
        "xgboost": 0.25,
        "graph_triangulation": 0.10
    }
    learner = EnsembleMetaLearner(weights=weights)
    
    score, details = learner.predict_score(100.0, 50.0, 0.0, 0.0, 0.0, 0.0)
    expected = (0.15 * 100.0) + (0.20 * 50.0)
    assert score == expected
    assert details["method"] == "weighted_average"
