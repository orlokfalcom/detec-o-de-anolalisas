import pandas as pd
from datetime import datetime
from src.utils.helpers import load_yaml
from src.utils.logger import logger

class RulesDetector:
    def __init__(self, rules_path="config/rules.yaml"):
        config = load_yaml(rules_path)
        self.rules = config.get("rules", {})

    def predict_score(self, tx_dict, user_history=None):
        """
        Processes business/compliance rules on the incoming transaction.
        Accumulates risk scores based on rule triggers, capped at 100.
        """
        risk_score = 0.0
        triggered_rules = []
        details = {}

        amount = float(tx_dict["amount"])
        timestamp = pd.to_datetime(tx_dict["timestamp"])
        hour = timestamp.hour

        # 1. BACEN Night Limit Rule
        bacen = self.rules.get("bacen_night_limit", {})
        if bacen.get("enabled", False):
            start = bacen["start_hour"]
            end = bacen["end_hour"]
            
            # Check if time is within night hours
            is_night = False
            if start > end:  # e.g., 20:00 to 06:00
                is_night = hour >= start or hour < end
            else:  # e.g., 22:00 to 24:00 (less common but possible)
                is_night = start <= hour < end
                
            if is_night and amount > bacen["max_amount"]:
                triggered_rules.append("bacen_night_limit")
                risk_score += bacen["severity"]
                details["bacen_night_limit_triggered"] = True
                details["bacen_night_limit_amount"] = amount

        # 2. Single Transaction Limit Rule
        tx_limits = self.rules.get("transaction_limits", {})
        if tx_limits:
            max_limit = tx_limits["single_max_amount"]
            if amount > max_limit:
                triggered_rules.append("single_max_amount")
                risk_score += tx_limits["severity"]
                details["single_max_amount_triggered"] = True

        # 3. High Frequency Velocity Rule (using pre-engineered features)
        velocity = self.rules.get("velocity", {})
        if velocity:
            # We check the pre-engineered count if present in tx_dict
            # or fallback if user_history is provided
            count_1m = tx_dict.get("velocity_count_1m")
            if count_1m is None and user_history:
                # Calculate manually
                cutoff = timestamp - pd.Timedelta(seconds=velocity["time_window_seconds"])
                past_txs = [h for h in user_history if pd.to_datetime(h["timestamp"]) >= cutoff]
                count_1m = len(past_txs) + 1 # Include current one

            if count_1m is not None and count_1m > velocity["max_transactions"]:
                triggered_rules.append("high_frequency_velocity")
                risk_score += velocity["severity"]
                details["velocity_count_1m"] = count_1m

        # 4. Impossible Travel/Geographic Speed Rule
        travel = self.rules.get("impossible_travel", {})
        if travel:
            speed = tx_dict.get("speed_kph")
            # If speed not calculated but user_history is provided, we can fallback, 
            # though feature engineering pipeline already provides speed_kph.
            if speed is not None and speed > travel["max_speed_kph"]:
                triggered_rules.append("impossible_travel")
                risk_score += travel["severity"]
                details["speed_kph"] = speed

        # 5. Device Swap Rule
        dev_swap = self.rules.get("device_switch", {})
        if dev_swap:
            dev_count = tx_dict.get("device_count_24h")
            if dev_count is not None and dev_count > dev_swap["max_devices_24h"]:
                triggered_rules.append("device_fingerprint_limit")
                risk_score += dev_swap["severity"]
                details["device_count_24h"] = dev_count

        # Cap score at 100
        final_score = float(min(risk_score, 100.0))
        details["rules_score"] = final_score
        details["triggered_rules"] = triggered_rules

        return final_score, details
