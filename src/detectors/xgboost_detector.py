import os
import joblib
import numpy as np
import xgboost as xgb
from src.utils.logger import logger

class XGBoostDetector:
    def __init__(self, model_path="models/xgboost_model.pkl", max_depth=5, learning_rate=0.1, n_estimators=100, random_state=42):
        self.model_path = model_path
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.model = xgb.XGBClassifier(
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            eval_metric="logloss",
            use_label_encoder=False,
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

    def fit(self, X_train, y_train):
        """
        Trains the XGBoost classifier on feature matrix X_train and targets y_train.
        """
        logger.info("Training XGBoost supervised classifier...")
        self.model.fit(X_train, y_train)
        
        # Save model
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        joblib.dump(self.model, self.model_path)
        logger.info(f"XGBoost model saved to {self.model_path}")
        return self

    def load_model(self):
        """
        Loads XGBoost model from file.
        """
        if os.path.exists(self.model_path):
            self.model = joblib.load(self.model_path)
            logger.info(f"XGBoost model loaded from {self.model_path}")
        else:
            logger.warning(f"XGBoost model not found at {self.model_path}. Fitting is required.")
        return self

    def predict_score(self, tx_scaled):
        """
        Predicts fraud probability score (0-100) for a single scaled transaction.
        
        tx_scaled: dictionary containing scaled features, 
                   or a numpy array of shape (1, num_features)
        """
        if isinstance(tx_scaled, dict):
            X = np.array([[tx_scaled[col] for col in self.feature_cols]])
        else:
            X = np.array(tx_scaled)
            if X.ndim == 1:
                X = X.reshape(1, -1)

        # Get probability of fraud (class 1)
        prob = self.model.predict_proba(X)[0, 1]
        risk = float(prob * 100.0)

        details = {
            "xgboost_probability": float(prob),
            "xgboost_score": risk
        }

        return risk, details
