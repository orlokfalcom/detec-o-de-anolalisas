import pandas as pd
from src.pipeline.ingestion import DataIngestion
from src.pipeline.feature_engineering import FeatureEngineer
from src.pipeline.preprocessing import DataPreprocessor
from src.utils.logger import logger

class FraudPipeline:
    def __init__(self, config):
        self.config = config
        self.ingestion = DataIngestion(sample_path=config["data"]["sample_path"])
        self.feature_engineer = FeatureEngineer()
        self.preprocessor = DataPreprocessor(scaler_path=f"{config['models']['save_dir']}/scaler.pkl")

    def run_training_pipeline(self):
        """
        Orchestrates full batch pipeline:
        1. Ingest/generate batch data
        2. Engineer features
        3. Fit scaler
        4. Apply scaling
        5. Return structured dataframe
        """
        logger.info("Running training pipeline...")
        # 1. Ingestion
        df = self.ingestion.load_batch_data()
        
        # 2. Feature Engineering
        df_feat = self.feature_engineer.transform_batch(df)
        
        # 3 & 4. Fitting Scaler & Scaling
        self.preprocessor.fit_scaler(df_feat)
        df_scaled = self.preprocessor.scale_features(df_feat)
        
        logger.info("Training pipeline completed successfully.")
        return df_scaled

    def run_inference_pipeline(self, tx_dict, user_history):
        """
        Processes a single transaction event in real-time.
        
        tx_dict: incoming transaction dictionary.
        user_history: list of past transaction dictionaries for this user.
        """
        # Load scaler (if not already loaded)
        self.preprocessor.load_scaler()
        
        # 1. Feature Engineering (Online)
        tx_feat = self.feature_engineer.transform_online(tx_dict, user_history)
        
        # 2. Scaling
        tx_scaled = self.preprocessor.scale_features(tx_feat)
        
        return tx_scaled
