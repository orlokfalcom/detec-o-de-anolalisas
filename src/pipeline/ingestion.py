import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
from src.utils.logger import logger

class DataIngestion:
    def __init__(self, sample_path="data/sample_transactions.csv"):
        self.sample_path = sample_path

    def generate_synthetic_data(self, num_samples=10000, fraud_ratio=0.05, seed=42):
        """
        Generates synthetic transaction history with realistic fraud patterns:
        - Velocity attacks (many fast transactions)
        - Nighttime BACEN Pix violations
        - Geographic leaps (impossible speed)
        - Large single transaction amounts
        - Device fingerprint switches
        - Money Laundering cycle triangulation (P2P Pix transfers)
        """
        logger.info(f"Generating {num_samples} synthetic transactions...")
        np.random.seed(seed)
        
        # User base setup
        num_users = int(num_samples * 0.05)  # 500 users
        user_ids = [f"U{i:03d}" for i in range(1, num_users + 1)]
        
        # Recipient base (merchants + P2P)
        merchants = [f"M{i:03d}" for i in range(1, 101)]
        
        # Device base setup
        num_devices = int(num_samples * 0.08)
        devices = [f"DEV{i:04d}" for i in range(1, num_devices + 1)]
        
        # Base cities for user patterns
        cities = [
            ("São Paulo", -23.5505, -46.6333),
            ("Rio de Janeiro", -22.9068, -43.1729),
            ("Belo Horizonte", -19.9173, -43.9345),
            ("Curitiba", -25.4284, -49.2733),
            ("Salvador", -12.9714, -38.5014),
            ("Brasília", -15.7801, -47.9292),
            ("Porto Alegre", -30.0346, -51.2177)
        ]
        
        # Generate base profiles for users
        user_profiles = {}
        for uid in user_ids:
            city_idx = np.random.choice(len(cities))
            user_profiles[uid] = {
                "city": cities[city_idx][0],
                "lat": cities[city_idx][1],
                "lon": cities[city_idx][2],
                "device": np.random.choice(devices),
                "avg_amount": np.random.exponential(150.0) + 10.0
            }
            
        data = []
        base_time = datetime.now() - timedelta(days=30)
        
        # Generate time series
        current_times = {uid: base_time + timedelta(minutes=float(np.random.randint(1, 100))) for uid in user_ids}
        
        # Helper to generate recipients
        def get_recipient(sender):
            if np.random.rand() < 0.7:
                # 70% go to merchants
                return np.random.choice(merchants)
            else:
                # 30% are P2P transfers to other users
                return np.random.choice([u for u in user_ids if u != sender])

        i = 0
        while i < num_samples:
            # Pick user
            uid = np.random.choice(user_ids)
            profile = user_profiles[uid]
            
            # Step time forward
            time_step = timedelta(minutes=float(np.random.exponential(120.0) + 1.0))
            current_times[uid] += time_step
            timestamp = current_times[uid]
            
            # Normal values
            amount = np.random.exponential(profile["avg_amount"]) + 1.0
            lat = profile["lat"] + np.random.normal(0, 0.05)
            lon = profile["lon"] + np.random.normal(0, 0.05)
            device = profile["device"]
            recipient = get_recipient(uid)
            is_fraud = 0
            
            # Inject explicit fraud patterns based on fraud ratio
            if np.random.rand() < fraud_ratio:
                is_fraud = 1
                fraud_type = np.random.choice(["limit", "night", "velocity", "impossible_travel", "device_swap", "triangulation"])
                
                if fraud_type == "limit":
                    amount = float(np.random.uniform(55000, 100000))
                elif fraud_type == "night":
                    amount = float(np.random.uniform(1500, 5000))
                    night_hour = np.random.choice([20, 21, 22, 23, 0, 1, 2, 3, 4, 5])
                    timestamp = timestamp.replace(hour=night_hour, minute=np.random.randint(0, 59))
                elif fraud_type == "velocity":
                    amount = float(np.random.uniform(100, 500))
                elif fraud_type == "impossible_travel":
                    other_city = cities[np.random.choice([idx for idx in range(len(cities)) if cities[idx][0] != profile["city"]])]
                    lat = other_city[1]
                    lon = other_city[2]
                    time_step = timedelta(minutes=float(np.random.uniform(1, 5)))
                    timestamp = current_times[uid] + time_step
                    current_times[uid] = timestamp
                elif fraud_type == "device_swap":
                    device = np.random.choice([d for d in devices if d != profile["device"]])
                    amount = profile["avg_amount"] * 5.0
                elif fraud_type == "triangulation":
                    # Money laundering loop: U_A -> U_B, U_B -> U_C, U_C -> U_A
                    # We inject these three sequential transactions directly
                    u_a = uid
                    u_b = np.random.choice([u for u in user_ids if u != u_a])
                    u_c = np.random.choice([u for u in user_ids if u not in [u_a, u_b]])
                    
                    t1 = timestamp
                    t2 = t1 + timedelta(seconds=10)
                    t3 = t2 + timedelta(seconds=10)
                    
                    # Tx 1: A -> B
                    data.append({
                        "transaction_id": f"T{i:06d}",
                        "user_id": u_a,
                        "timestamp": t1.isoformat(),
                        "amount": 5000.0,
                        "latitude": round(lat, 5),
                        "longitude": round(lon, 5),
                        "device_id": device,
                        "recipient_id": u_b,
                        "is_fraud": 1
                    })
                    i += 1
                    
                    # Tx 2: B -> C
                    data.append({
                        "transaction_id": f"T{i:06d}",
                        "user_id": u_b,
                        "timestamp": t2.isoformat(),
                        "amount": 4950.0,
                        "latitude": round(user_profiles[u_b]["lat"], 5),
                        "longitude": round(user_profiles[u_b]["lon"], 5),
                        "device_id": user_profiles[u_b]["device"],
                        "recipient_id": u_c,
                        "is_fraud": 1
                    })
                    i += 1
                    
                    # Tx 3: C -> A (completes loop)
                    data.append({
                        "transaction_id": f"T{i:06d}",
                        "user_id": u_c,
                        "timestamp": t3.isoformat(),
                        "amount": 4900.0,
                        "latitude": round(user_profiles[u_c]["lat"], 5),
                        "longitude": round(user_profiles[u_c]["lon"], 5),
                        "device_id": user_profiles[u_c]["device"],
                        "recipient_id": u_a,
                        "is_fraud": 1
                    })
                    i += 1
                    
                    # Update times
                    current_times[u_a] = t1
                    current_times[u_b] = t2
                    current_times[u_c] = t3
                    continue
            
            data.append({
                "transaction_id": f"T{i:06d}",
                "user_id": uid,
                "timestamp": timestamp.isoformat(),
                "amount": round(amount, 2),
                "latitude": round(lat, 5),
                "longitude": round(lon, 5),
                "device_id": device,
                "recipient_id": recipient,
                "is_fraud": is_fraud
            })
            i += 1
            
        df = pd.DataFrame(data[:num_samples])
        # Ensure directory exists
        os.makedirs(os.path.dirname(self.sample_path), exist_ok=True)
        df.to_csv(self.sample_path, index=False)
        logger.info(f"Synthetic data saved to {self.sample_path}")
        return df

    def load_batch_data(self):
        """
        Loads batch data from file. Generates it if missing.
        """
        if not os.path.exists(self.sample_path):
            logger.warning(f"{self.sample_path} not found. Generating a new batch.")
            return self.generate_synthetic_data()
        
        logger.info(f"Loading transaction dataset from {self.sample_path}")
        df = pd.read_csv(self.sample_path)
        return df
