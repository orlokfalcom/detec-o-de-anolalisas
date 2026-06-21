import numpy as np
from src.utils.logger import logger

class StatisticalDetector:
    def __init__(self, z_score_threshold=3.0, iqr_multiplier=1.5):
        self.z_score_threshold = z_score_threshold
        self.iqr_multiplier = iqr_multiplier

    def predict_score(self, tx_dict, user_history=None):
        """
        Calculates a statistical anomaly risk score (0-100) based on
        how much the transaction amount deviates from historical patterns.
        """
        # If user has no history or very little, return low score
        if not user_history or len(user_history) < 3:
            return 0.0, {"z_score_triggered": False, "iqr_triggered": False}

        amounts = [float(h["amount"]) for h in user_history]
        curr_amount = float(tx_dict["amount"])

        # 1. Z-Score Calculation
        mean = np.mean(amounts)
        std = np.std(amounts)
        
        z_score = 0.0 if std == 0.0 else abs(curr_amount - mean) / std
        z_triggered = z_score > self.z_score_threshold

        # 2. IQR Calculation
        q25, q75 = np.percentile(amounts, [25, 75])
        iqr = q75 - q25
        lower_bound = q25 - (self.iqr_multiplier * iqr)
        upper_bound = q75 + (self.iqr_multiplier * iqr)
        
        iqr_triggered = curr_amount > upper_bound or curr_amount < lower_bound

        # Calculate final combined score (0 - 100)
        # Z-score contribution: max score at z=5
        z_score_normalized = min((z_score / 5.0) * 100.0, 100.0)
        
        # IQR contribution: how far past the upper bound the amount is, relative to upper bound
        if curr_amount > upper_bound and upper_bound > 0:
            iqr_excess_pct = ((curr_amount - upper_bound) / upper_bound) * 100.0
            iqr_score = min(iqr_excess_pct, 100.0)
        else:
            iqr_score = 0.0

        # Ensemble statistical score
        final_score = float(max(z_score_normalized, iqr_score))
        
        # Cap rules
        if not (z_triggered or iqr_triggered):
            # If neither threshold is triggered, limit risk score contribution
            final_score = min(final_score, 30.0)
        else:
            # Ensure it is significant if triggered
            final_score = max(final_score, 60.0)

        details = {
            "z_score": float(z_score),
            "z_score_threshold": self.z_score_threshold,
            "z_score_triggered": bool(z_triggered),
            "user_mean": float(mean),
            "user_std": float(std),
            "iqr_lower_bound": float(lower_bound),
            "iqr_upper_bound": float(upper_bound),
            "iqr_triggered": bool(iqr_triggered),
            "statistical_score": final_score
        }

        return final_score, details
