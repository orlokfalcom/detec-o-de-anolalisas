import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from src.utils.logger import logger

class LSTMAutoencoderModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=16, latent_dim=8):
        super(LSTMAutoencoderModel, self).__init__()
        # Encoder
        self.encoder_lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.encoder_linear = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder
        self.decoder_lstm = nn.LSTM(latent_dim, hidden_dim, batch_first=True)
        self.decoder_linear = nn.Linear(hidden_dim, input_dim)
        
        self.latent_dim = latent_dim

    def forward(self, x):
        # x shape: (batch, seq_len, input_dim)
        seq_len = x.size(1)
        
        # Encode
        _, (h_n, _) = self.encoder_lstm(x)  # h_n shape: (1, batch, hidden_dim)
        latent = torch.relu(self.encoder_linear(h_n.squeeze(0)))  # shape: (batch, latent_dim)
        
        # Repeat latent vector seq_len times
        latent_repeated = latent.unsqueeze(1).repeat(1, seq_len, 1)  # shape: (batch, seq_len, latent_dim)
        
        # Decode
        decoder_out, _ = self.decoder_lstm(latent_repeated)  # shape: (batch, seq_len, hidden_dim)
        reconstructed = self.decoder_linear(decoder_out)  # shape: (batch, seq_len, input_dim)
        
        return reconstructed

class LSTMAutoencoderDetector:
    def __init__(self, model_path="models/lstm_autoencoder.pth", input_dim=12, hidden_dim=16, latent_dim=8, sequence_length=5):
        self.model_path = model_path
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.seq_len = sequence_length
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model = LSTMAutoencoderModel(input_dim, hidden_dim, latent_dim).to(self.device)
        self.threshold = 0.5  # Default, set during training

    def fit(self, X_train_seq, epochs=10, batch_size=64, lr=0.001, threshold_percentile=95):
        """
        Trains the LSTM Autoencoder model on sequence dataset X_train_seq.
        X_train_seq shape: (num_samples, seq_len, input_dim)
        """
        logger.info(f"Training LSTM Autoencoder on device: {self.device}...")
        self.model.train()
        
        tensor_x = torch.tensor(X_train_seq, dtype=torch.float32)
        dataset = TensorDataset(tensor_x)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.MSELoss()
        
        for epoch in range(1, epochs + 1):
            epoch_loss = 0.0
            for batch in loader:
                inputs = batch[0].to(self.device)
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, inputs)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * inputs.size(0)
            
            epoch_loss /= len(loader.dataset)
            if epoch % 2 == 0 or epoch == epochs:
                logger.info(f"Epoch {epoch}/{epochs} - Reconstruction Loss: {epoch_loss:.5f}")
                
        # Calculate thresholds on training data
        self.model.eval()
        with torch.no_grad():
            inputs = tensor_x.to(self.device)
            reconstructed = self.model(inputs)
            # Compute MSE per sample (average over sequence and features)
            errors = torch.mean((inputs - reconstructed)**2, dim=(1, 2)).cpu().numpy()
            self.threshold = float(np.percentile(errors, threshold_percentile))
            logger.info(f"LSTM Autoencoder reconstruction error threshold set at: {self.threshold:.5f} (percentile={threshold_percentile})")
            
        # Save model weights & metadata
        self.save_model()
        return self

    def save_model(self):
        """
        Saves weights and configuration variables.
        """
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        state = {
            "model_state": self.model.state_dict(),
            "threshold": self.threshold,
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "latent_dim": self.latent_dim,
            "seq_len": self.seq_len
        }
        torch.save(state, self.model_path)
        logger.info(f"LSTM Autoencoder model weights saved to {self.model_path}")

    def load_model(self):
        """
        Loads weights and variables.
        """
        if os.path.exists(self.model_path):
            state = torch.load(self.model_path, map_location=self.device)
            self.input_dim = state.get("input_dim", self.input_dim)
            self.hidden_dim = state.get("hidden_dim", self.hidden_dim)
            self.latent_dim = state.get("latent_dim", self.latent_dim)
            self.seq_len = state.get("seq_len", self.seq_len)
            self.threshold = state.get("threshold", 0.5)
            
            # Recreate model if dims changed
            self.model = LSTMAutoencoderModel(self.input_dim, self.hidden_dim, self.latent_dim).to(self.device)
            self.model.load_state_dict(state["model_state"])
            self.model.eval()
            logger.info(f"LSTM Autoencoder loaded from {self.model_path} with threshold {self.threshold:.5f}")
        else:
            logger.warning(f"LSTM Autoencoder weights not found at {self.model_path}. Fitting required.")
        return self

    def predict_score(self, sequence_matrix):
        """
        Scores a single sequence.
        sequence_matrix: shape (1, seq_len, input_dim) or (seq_len, input_dim)
        Returns anomaly risk score (0-100) based on reconstruction error.
        """
        self.model.eval()
        
        if isinstance(sequence_matrix, np.ndarray):
            X = torch.tensor(sequence_matrix, dtype=torch.float32)
        else:
            X = sequence_matrix
            
        if X.ndim == 2:
            X = X.unsqueeze(0)  # Add batch dimension
            
        X = X.to(self.device)
        
        with torch.no_grad():
            reconstructed = self.model(X)
            # MSE error of the last sequence step or average over sequence
            # Usually, average reconstruction error over entire sequence is standard:
            error = float(torch.mean((X - reconstructed)**2).item())
            
        # Map reconstruction error to 0-100 risk score.
        # If error = threshold, score = 50.
        # If error > threshold, score scales to 100.
        # If error < threshold, score scales from 0 to 50.
        if error > self.threshold:
            # Scale from 50 to 100. Let's cap maximum excess at 3x threshold.
            excess = min((error - self.threshold) / (2.0 * self.threshold), 1.0)
            risk = 50.0 + (excess * 50.0)
        else:
            # Scale from 0 to 50
            risk = (error / self.threshold) * 50.0
            
        details = {
            "reconstruction_error": error,
            "reconstruction_threshold": self.threshold,
            "lstm_score": risk,
            "is_anomaly": bool(error > self.threshold)
        }
        
        return float(risk), details
