import numpy as np
import pandas as pd
import joblib
import os
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from src.utils.logger import logger

# We use joblib, let's make sure it is imported. We will import joblib directly.
# Wait, let's write standard python imports.
import joblib

class DataPreprocessor:
    def __init__(self, scaler_path="models/scaler.pkl", target_encoder_path="models/target_encoder.pkl"):
        self.scaler_path = scaler_path
        self.target_encoder_path = target_encoder_path
        self.scaler = StandardScaler()
        self.device_encoding_map = {}
        self.global_fraud_mean = 0.0
        self.feature_cols = [
            "amount", 
            "dist_from_prev_km", 
            "speed_kph", 
            "velocity_count_1m", 
            "velocity_sum_1m", 
            "velocity_count_5m", 
            "velocity_sum_5m", 
            "velocity_count_1h", 
            "velocity_sum_1h", 
            "amount_z_score", 
            "device_count_24h", 
            "device_changed",
            "device_id_encoded" # Added target encoded feature
        ]

    def fit_scaler(self, df):
        """
        Fits target encoding for categorical device_id and fits standard scaler on numerical features.
        """
        logger.info("Fitting target encoder for device_id...")
        # Fit custom target encoder with smoothing
        self.global_fraud_mean = float(df["is_fraud"].mean()) if "is_fraud" in df.columns else 0.0
        
        if "device_id" in df.columns and "is_fraud" in df.columns:
            smoothing = 10
            grouped = df.groupby("device_id")["is_fraud"]
            counts = grouped.count()
            means = grouped.mean()
            
            # Smoothed target encoding formula
            smoothed_val = (counts * means + smoothing * self.global_fraud_mean) / (counts + smoothing)
            self.device_encoding_map = smoothed_val.to_dict()
        else:
            self.device_encoding_map = {}

        # Save target encoder mappings
        os.makedirs(os.path.dirname(self.target_encoder_path), exist_ok=True)
        joblib.dump({
            "mapping": self.device_encoding_map,
            "global_mean": self.global_fraud_mean
        }, self.target_encoder_path)
        logger.info(f"Target encoder saved to {self.target_encoder_path}")

        # Compute encoded values temporarily to fit the scaler
        df_temp = df.copy()
        if "device_id" in df_temp.columns:
            df_temp["device_id_encoded"] = df_temp["device_id"].map(self.device_encoding_map).fillna(self.global_fraud_mean)
        else:
            df_temp["device_id_encoded"] = self.global_fraud_mean
            
        logger.info("Fitting feature scaler...")
        self.scaler.fit(df_temp[self.feature_cols])
        
        # Save scaler
        os.makedirs(os.path.dirname(self.scaler_path), exist_ok=True)
        joblib.dump(self.scaler, self.scaler_path)
        logger.info(f"Scaler saved to {self.scaler_path}")
        return self

    def load_scaler(self):
        """
        Loads fitted scaler and target encoder from file.
        """
        if os.path.exists(self.scaler_path):
            self.scaler = joblib.load(self.scaler_path)
            logger.info(f"Scaler loaded from {self.scaler_path}")
        else:
            logger.warning(f"Scaler not found at {self.scaler_path}. Fitting is required.")
            
        if os.path.exists(self.target_encoder_path):
            te_data = joblib.load(self.target_encoder_path)
            self.device_encoding_map = te_data.get("mapping", {})
            self.global_fraud_mean = te_data.get("global_mean", 0.0)
            logger.info(f"Target encoder loaded from {self.target_encoder_path}")
        else:
            logger.warning(f"Target encoder not found at {self.target_encoder_path}. Fitting is required.")
        return self

    def scale_features(self, df):
        """
        Transforms features using fitted scaler. Supports DataFrame or dict.
        """
        if isinstance(df, dict):
            # Single transaction dictionary mapping
            df_temp = pd.DataFrame([df])
            # Apply target encoder
            dev_id = df_temp["device_id"].values[0] if "device_id" in df_temp.columns else None
            encoded_val = self.device_encoding_map.get(dev_id, self.global_fraud_mean)
            df_temp["device_id_encoded"] = encoded_val
            
            scaled_vals = self.scaler.transform(df_temp[self.feature_cols])
            
            # Return as dictionary
            scaled_dict = df.copy()
            scaled_dict["device_id_encoded"] = float(encoded_val)
            for i, col in enumerate(self.feature_cols):
                scaled_dict[f"scaled_{col}"] = float(scaled_vals[0, i])
            return scaled_dict
        
        # DataFrame mapping
        df_scaled = df.copy()
        # Apply target encoder
        if "device_id" in df_scaled.columns:
            df_scaled["device_id_encoded"] = df_scaled["device_id"].map(self.device_encoding_map).fillna(self.global_fraud_mean)
        else:
            df_scaled["device_id_encoded"] = self.global_fraud_mean
            
        scaled_vals = self.scaler.transform(df_scaled[self.feature_cols])
        for i, col in enumerate(self.feature_cols):
            df_scaled[f"scaled_{col}"] = scaled_vals[:, i]
        return df_scaled

    def get_scaled_feature_matrix(self, df):
        """
        Extracts only the scaled features as a numpy array.
        """
        scaled_cols = [f"scaled_{col}" for col in self.feature_cols]
        # If scaled columns do not exist, scale them first
        if not all(col in df.columns for col in scaled_cols):
            df = self.scale_features(df)
        return df[scaled_cols].values

    def resample_training_data(self, X, y, method="smote"):
        """
        Applies SMOTE to balance fraud class (is_fraud = 1).
        """
        logger.info(f"Resampling training data using {method}...")
        if method == "smote":
            smote = SMOTE(random_state=42)
            X_res, y_res = smote.fit_resample(X, y)
            logger.info(f"Original shape: {X.shape}, Resampled shape: {X_res.shape}")
            return X_res, y_res
        else:
            return X, y

    def create_sequences_for_lstm(self, df, seq_length=5):
        """
        Prepares sequences of scaled features grouped by user for the LSTM Autoencoder.
        Output shape: (num_sequences, seq_length, num_features)
        
        We construct sequences where each sequence represents a user's chronological transaction steps.
        If a user has fewer transactions than seq_length, we zero-pad the sequence at the beginning.
        """
        scaled_cols = [f"scaled_{col}" for col in self.feature_cols]
        # Make sure columns exist
        if not all(col in df.columns for col in scaled_cols):
            df = self.scale_features(df)
            
        logger.info(f"Creating LSTM sequence windows (length={seq_length})...")
        
        sequences = []
        # Group by user_id and sort by timestamp
        grouped = df.sort_values("timestamp").groupby("user_id")
        
        for user_id, group in grouped:
            user_features = group[scaled_cols].values
            
            # If user has fewer transactions than sequence length, pad with zeros
            if len(user_features) < seq_length:
                padding = np.zeros((seq_length - len(user_features), len(self.feature_cols)))
                padded_seq = np.vstack([padding, user_features])
                sequences.append(padded_seq)
            else:
                # Slide window across user transaction history
                for i in range(len(user_features) - seq_length + 1):
                    sequences.append(user_features[i:i + seq_length])
                    
        return np.array(sequences)

    def get_user_sequence(self, user_history_list, current_tx_scaled, seq_length=5):
        """
        Constructs a single sequence of shape (1, seq_length, num_features) 
        for online inference with the LSTM Autoencoder.
        
        Includes the current transaction (already scaled) at the end.
        """
        scaled_cols = [f"scaled_{col}" for col in self.feature_cols]
        
        # Parse history
        if not user_history_list:
            # First transaction: pad everything
            seq = np.zeros((seq_length, len(self.feature_cols)))
            # Set last item as current transaction
            for idx, col in enumerate(self.feature_cols):
                seq[-1, idx] = current_tx_scaled[f"scaled_{col}"]
            return np.expand_dims(seq, axis=0)

        # Fast path optimization: if history is already scaled, extract directly
        first_item = user_history_list[0]
        if f"scaled_{self.feature_cols[0]}" in first_item:
            recent_hist = user_history_list[-(seq_length - 1):]
            hist_features = []
            for tx in recent_hist:
                hist_features.append([tx.get(f"scaled_{col}", 0.0) for col in self.feature_cols])
            
            curr_features = [current_tx_scaled[f"scaled_{col}"] for col in self.feature_cols]
            hist_features.append(curr_features)
            
            while len(hist_features) < seq_length:
                hist_features.insert(0, [0.0] * len(self.feature_cols))
                
            return np.expand_dims(np.array(hist_features), axis=0)
            
        # Slow path (for raw API inputs): construct dataframe and scale
        history = pd.DataFrame(user_history_list)
        history_scaled = self.scale_features(history)
        
        hist_features = history_scaled[scaled_cols].values
        curr_features = np.array([current_tx_scaled[f"scaled_{col}"] for col in self.feature_cols])
        
        # Combine
        combined = np.vstack([hist_features, curr_features])
        
        if len(combined) < seq_length:
            padding = np.zeros((seq_length - len(combined), len(self.feature_cols)))
            seq = np.vstack([padding, combined])
        else:
            seq = combined[-seq_length:]
            
        return np.expand_dims(seq, axis=0)
