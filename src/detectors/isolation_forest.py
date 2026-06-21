import os
import joblib
import numpy as np
from sklearn.ensemble import IsolationForest
from src.utils.logger import logger

class IsolationForestDetector:
    def __init__(self, model_path="models/isolation_forest.pkl", contamination=0.05, n_estimators=100, random_state=42):
        self.model_path = model_path
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.model = IsolationForest(
            contamination=self.contamination,
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            n_jobs=-1
        )
        self.feature_cols = [
            "scaled_amount", 
            "scaled_dist_from_prev_km", 
            "scaled_speed_kph", 
            "scaled_velocity_count_1m", 
            "scaled_velocity_sum_1m", 
            "scaled_velocity_count_5m", 
            "scaled_velocity_sum_5m", 
            "scaled_velocity_count_1h", 
            "scaled_velocity_sum_1h", 
            "scaled_amount_z_score", 
            "scaled_device_count_24h", 
            "scaled_device_changed",
            "scaled_device_id_encoded"
        ]

    def fit(self, X_train):
        """
        Fits the Isolation Forest on the scaled feature matrix X_train.
        """
        logger.info("Training Isolation Forest model...")
        self.model.fit(X_train)
        
        # Save model
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        joblib.dump(self.model, self.model_path)
        logger.info(f"Isolation Forest model saved to {self.model_path}")
        return self

    def load_model(self):
        """
        Loads model from disk.
        """
        if os.path.exists(self.model_path):
            self.model = joblib.load(self.model_path)
            logger.info(f"Isolation Forest model loaded from {self.model_path}")
        else:
            logger.warning(f"Isolation Forest model file not found at {self.model_path}. Fitting is required.")
        return self

    def predict_score(self, tx_scaled):
        """
        Scores a single scaled transaction.
        Returns a risk score from 0 to 100.
        
        tx_scaled: dictionary containing the scaled features, 
                   or a numpy array of shape (1, num_features)
        """
        if isinstance(tx_scaled, dict):
            X = np.array([[tx_scaled[col] for col in self.feature_cols]])
        else:
            X = np.array(tx_scaled)
            if X.ndim == 1:
                X = X.reshape(1, -1)

        # score_samples returns the anomaly score of the input samples.
        # The lower the score, the more anomalous.
        # Typically values are between -0.8 (highly anomalous) and -0.4 (very normal).
        score_sample = self.model.score_samples(X)[0]
        
        # IsolationForest decision_function returns: offset_ - score_samples
        # Normal samples have positive values, anomalies have negative values.
        decision = self.model.decision_function(X)[0]
        
        # Map decision score to 0-100 risk score.
        # If decision is negative, it's anomalous (risk should be high, e.g. >50).
        # We can map it using a sigmoid or a simple linear piece-wise scaling.
        # A simple linear piece-wise formula:
        if decision < 0:
            # Anomaly region. Scale from 50 to 100 based on severity.
            # Normal range of negative decisions is 0 to -0.35.
            val = abs(decision)
            risk = 50.0 + min((val / 0.35) * 50.0, 50.0)
        else:
            # Normal region. Scale from 0 to 50.
            # Normal range of positive decisions is 0 to 0.25.
            val = decision
            risk = max(50.0 - (val / 0.25) * 50.0, 0.0)

        details = {
            "raw_score": float(score_sample),
            "decision_function_val": float(decision),
            "isolation_forest_score": float(risk),
            "is_anomaly": bool(decision < 0)
        }
        
        return float(risk), details
