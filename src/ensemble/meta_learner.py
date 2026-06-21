import os
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from src.utils.logger import logger

class EnsembleMetaLearner:
    def __init__(self, weights=None, meta_model_path="models/meta_model.pkl"):
        self.meta_model_path = meta_model_path
        
        if weights is None:
            self.weights = {
                "statistical": 0.15,
                "rules": 0.20,
                "isolation_forest": 0.15,
                "lstm": 0.15,
                "xgboost": 0.25,
                "graph_triangulation": 0.10
            }
        else:
            self.weights = weights
            
        self.meta_model = None

    def fit_meta_learner(self, X_scores, y_true):
        """
        Trains a Logistic Regression meta-learner using output scores of individual detectors.
        X_scores shape: (num_samples, 6)
        """
        logger.info("Training Ensemble Logistic Regression meta-learner...")
        self.meta_model = LogisticRegression(class_weight="balanced", random_state=42)
        self.meta_model.fit(X_scores, y_true)
        
        # Save model
        os.makedirs(os.path.dirname(self.meta_model_path), exist_ok=True)
        joblib.dump(self.meta_model, self.meta_model_path)
        logger.info(f"Meta-learner saved to {self.meta_model_path}")
        return self

    def load_meta_learner(self):
        """
        Loads the trained Logistic Regression meta-model from disk.
        """
        if os.path.exists(self.meta_model_path):
            self.meta_model = joblib.load(self.meta_model_path)
            logger.info(f"Meta-learner loaded from {self.meta_model_path}")
        else:
            logger.warning(f"Meta-learner not found at {self.meta_model_path}. Using weighted average formula instead.")
            self.meta_model = None
        return self

    def predict_score(self, stat_score, rules_score, iforest_score, lstm_score, xgb_score, graph_score):
        """
        Combines model scores.
        If a meta-learner is loaded, uses its predict_proba (probability * 100).
        Otherwise, falls back to the weighted formula.
        """
        scores_arr = np.array([[stat_score, rules_score, iforest_score, lstm_score, xgb_score, graph_score]])
        
        if self.meta_model is not None:
            # Predict probability of fraud (class 1)
            prob = self.meta_model.predict_proba(scores_arr)[0, 1]
            ensemble_score = float(prob * 100.0)
            method = "meta_learner_lr"
        else:
            # Weighted average fallback
            ensemble_score = float(
                self.weights.get("statistical", 0.15) * stat_score +
                self.weights.get("rules", 0.20) * rules_score +
                self.weights.get("isolation_forest", 0.15) * iforest_score +
                self.weights.get("lstm", 0.15) * lstm_score +
                self.weights.get("xgboost", 0.25) * xgb_score +
                self.weights.get("graph_triangulation", 0.10) * graph_score
            )
            method = "weighted_average"
            
        details = {
            "method": method,
            "inputs": {
                "statistical": float(stat_score),
                "rules": float(rules_score),
                "isolation_forest": float(iforest_score),
                "lstm": float(lstm_score),
                "xgboost": float(xgb_score),
                "graph_triangulation": float(graph_score)
            },
            "ensemble_score": ensemble_score
        }
        
        return ensemble_score, details
