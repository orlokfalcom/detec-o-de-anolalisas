import os
import sys
import yaml
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
import mlflow.pytorch
from src.utils.helpers import load_yaml
from src.utils.logger import logger
from src.utils.real_data_loader import download_and_map_real_dataset
from src.pipeline.pipeline import FraudPipeline
from src.detectors.statistical import StatisticalDetector
from src.detectors.rules import RulesDetector
from src.detectors.isolation_forest import IsolationForestDetector
from src.detectors.lstm_autoencoder import LSTMAutoencoderDetector
from src.detectors.xgboost_detector import XGBoostDetector
from src.detectors.graph_triangulation import GraphTriangulationDetector
from src.ensemble.meta_learner import EnsembleMetaLearner
from src.decision.risk_engine import RiskDecisionEngine
from src.llm.fraud_explainer import FraudExplainer
from src.utils.metrics import calculate_fraud_metrics, calculate_financial_impact

def main():
    # Opt-out of MLflow file store deprecation exception
    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
    
    logger.info("==================================================")
    logger.info("STARTING FRAUD INTELLIGENCE AI SYSTEM")
    logger.info("==================================================")
    
    # 1. Load Configurations
    config = load_yaml("config/config.yaml")
    
    # 2. Check if we should use the real dataset (check CLI arguments or force True since requested)
    use_real = "--real" in sys.argv or True  # Default to True since user asked for real dataset
    
    if use_real:
        logger.info("System is configured to use the REAL Credit Card Fraud dataset.")
        # Load and map Kaggle Credit Card Fraud dataset (first 15,000 rows to keep it lightweight)
        df_raw = download_and_map_real_dataset(config["data"]["real_path"], limit_samples=15000)
    else:
        logger.info("System is configured to use SYNTHETIC dataset.")
        df_raw = None # Pipeline will generate synthetic data

    # 3. Setup MLflow Experiment
    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(config["mlflow"]["experiment_name"])
    
    # 4. Initialize Pipeline
    pipeline = FraudPipeline(config)
    
    # If using real data, override pipeline's ingestion load to use the downloaded CSV
    if use_real:
        pipeline.ingestion.sample_path = config["data"]["real_path"]
        
    # Run the pipeline (generates features, fits scaler, target encodes, and returns scaled df)
    df_scaled = pipeline.run_training_pipeline()
    y_true = df_scaled["is_fraud"].values
    
    # 5. Extract feature matrix
    feature_matrix = pipeline.preprocessor.get_scaled_feature_matrix(df_scaled)
    
    # Start MLflow run
    with mlflow.start_run() as run:
        logger.info(f"MLflow Run ID: {run.info.run_id}")
        
        # Log config params
        mlflow.log_params({
            "num_samples": len(df_scaled),
            "iforest_contamination": config["models"]["isolation_forest"]["contamination"],
            "iforest_n_estimators": config["models"]["isolation_forest"]["n_estimators"],
            "lstm_epochs": config["models"]["lstm_autoencoder"]["epochs"],
            "lstm_learning_rate": config["models"]["lstm_autoencoder"]["learning_rate"],
            "xgb_max_depth": config["models"]["xgboost"]["max_depth"],
            "xgb_n_estimators": config["models"]["xgboost"]["n_estimators"],
            "dataset_type": "real_kaggle_mapped" if use_real else "synthetic"
        })
        
        # --- Model 1: Isolation Forest ---
        iforest = IsolationForestDetector(
            model_path=f"{config['models']['save_dir']}/isolation_forest.pkl",
            contamination=config["models"]["isolation_forest"]["contamination"],
            n_estimators=config["models"]["isolation_forest"]["n_estimators"]
        )
        iforest.fit(feature_matrix)
        mlflow.log_artifact(iforest.model_path, "isolation_forest")
        
        # --- Model 2: LSTM Autoencoder ---
        seq_len = config["models"]["lstm_autoencoder"]["sequence_length"]
        lstm_seq_data = pipeline.preprocessor.create_sequences_for_lstm(df_scaled, seq_length=seq_len)
        
        lstm_ae = LSTMAutoencoderDetector(
            model_path=f"{config['models']['save_dir']}/lstm_autoencoder.pth",
            input_dim=feature_matrix.shape[1],
            hidden_dim=config["models"]["lstm_autoencoder"]["hidden_dim"],
            latent_dim=config["models"]["lstm_autoencoder"]["latent_dim"],
            sequence_length=seq_len
        )
        lstm_ae.fit(
            lstm_seq_data, 
            epochs=config["models"]["lstm_autoencoder"]["epochs"],
            batch_size=config["models"]["lstm_autoencoder"]["batch_size"],
            lr=config["models"]["lstm_autoencoder"]["learning_rate"],
            threshold_percentile=config["models"]["lstm_autoencoder"]["threshold_percentile"]
        )
        mlflow.log_artifact(lstm_ae.model_path, "lstm_autoencoder")
        
        # --- Model 3: Supervised XGBoost (Trained on balanced data via SMOTE) ---
        # Apply SMOTE to handle imbalance
        X_res, y_res = pipeline.preprocessor.resample_training_data(feature_matrix, y_true, method="smote")
        
        xgb_det = XGBoostDetector(
            model_path=f"{config['models']['save_dir']}/xgboost_model.pkl",
            max_depth=config["models"]["xgboost"]["max_depth"],
            learning_rate=config["models"]["xgboost"]["learning_rate"],
            n_estimators=config["models"]["xgboost"]["n_estimators"]
        )
        xgb_det.fit(X_res, y_res)
        mlflow.log_artifact(xgb_det.model_path, "xgboost")
        
        # --- Model 4: NetworkX Graph ---
        graph_det = GraphTriangulationDetector(
            model_path=f"{config['models']['save_dir']}/graph_network.pkl"
        )
        graph_det.fit(df_scaled)
        # Log graph file
        mlflow.log_artifact(graph_det.model_path, "graph_network")
        
        # Log scaler and target encoder
        mlflow.log_artifact(pipeline.preprocessor.scaler_path, "pipeline_scaler")
        mlflow.log_artifact(pipeline.preprocessor.target_encoder_path, "pipeline_target_encoder")
        
        # Instantiate remaining detectors
        stat_detector = StatisticalDetector(
            z_score_threshold=config["models"]["statistical"]["z_score_threshold"],
            iqr_multiplier=config["models"]["statistical"]["iqr_multiplier"]
        )
        rules_detector = RulesDetector("config/rules.yaml")
        
        # Configure terminal output to support UTF-8 emojis on Windows
        try:
            sys.stdout.reconfigure(encoding='utf-8')
        except Exception:
            pass

        # --- Ensemble Meta-Learner Training ---
        logger.info("Selecting balanced transactions to train Ensemble Meta-Learner...")
        # To handle severe imbalance, include all fraud cases and a sample of normal cases
        fraud_indices = df_scaled[df_scaled["is_fraud"] == 1].index.tolist()
        normal_indices = df_scaled[df_scaled["is_fraud"] == 0].index.tolist()
        
        np.random.seed(42)
        sampled_normal_indices = list(np.random.choice(
            normal_indices, 
            size=min(2000, len(normal_indices)), 
            replace=False
        ))
        
        meta_train_indices = set(fraud_indices + sampled_normal_indices)
        
        scores_dataset = []
        user_histories = {}
        y_true_meta = []
        
        for idx, row in df_scaled.iterrows():
            user_id = row["user_id"]
            tx_dict = row.to_dict()
            hist = user_histories.get(user_id, [])
            
            if idx in meta_train_indices:
                s_score, _ = stat_detector.predict_score(tx_dict, hist)
                r_score, _ = rules_detector.predict_score(tx_dict, hist)
                
                scaled_feat = np.array([[row[f"scaled_{col}"] for col in pipeline.preprocessor.feature_cols]])
                i_score, _ = iforest.predict_score(scaled_feat)
                
                seq = pipeline.preprocessor.get_user_sequence(hist, row, seq_length=seq_len)
                l_score, _ = lstm_ae.predict_score(seq)
                
                x_score, _ = xgb_det.predict_score(scaled_feat)
                
                g_score, _ = graph_det.predict_score(tx_dict)
                
                scores_dataset.append([s_score, r_score, i_score, l_score, x_score, g_score])
                y_true_meta.append(row["is_fraud"])
            else:
                # Build graph edges to preserve state
                recipient = row.get("recipient_id")
                if not recipient:
                    recipient = f"REC_{row['device_id'][3:]}" if str(row['device_id']).startswith("DEV") else "REC_UNKNOWN"
                graph_det.G.add_edge(user_id, recipient, amount=float(row["amount"]), timestamp=str(row["timestamp"]))
            
            # Keep history updated
            hist.append(tx_dict)
            user_histories[user_id] = hist
            
        scores_matrix = np.array(scores_dataset)
        y_true_meta = np.array(y_true_meta)
        
        # Apply SMOTE to perfectly balance meta-learner inputs
        from imblearn.over_sampling import SMOTE
        smote = SMOTE(random_state=42)
        scores_matrix_res, y_true_meta_res = smote.fit_resample(scores_matrix, y_true_meta)
        
        meta_learner = EnsembleMetaLearner(
            weights=config["ensemble"]["weights"],
            meta_model_path=f"{config['models']['save_dir']}/meta_model.pkl"
        )
        meta_learner.fit_meta_learner(scores_matrix_res, y_true_meta_res)
        mlflow.log_artifact(meta_learner.meta_model_path, "meta_learner")
        
        # --- Evaluate Performance metrics ---
        logger.info("Computing metrics for logging...")
        y_pred_scores = []
        
        for scores in scores_matrix:
            final_risk, _ = meta_learner.predict_score(*scores)
            y_pred_scores.append(final_risk)
            
        y_pred = [1 if risk > 50.0 else 0 for risk in y_pred_scores]
        
        metrics = calculate_fraud_metrics(y_true_meta, y_pred, y_prob=np.array(y_pred_scores)/100.0)
        
        # Extract corresponding transaction amounts for the scored items
        meta_amounts = df_scaled.loc[list(meta_train_indices), "amount"].values
        financial = calculate_financial_impact(y_true_meta, y_pred, meta_amounts)
        
        # Log metrics to MLflow
        mlflow.log_metrics({
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1_score": metrics["f1_score"],
            "accuracy": metrics["accuracy"],
            "saved_fraud_value": financial["saved_fraud_value"],
            "lost_fraud_value": financial["lost_fraud_value"],
            "net_savings": financial["net_savings"]
        })
        
        logger.info("==================================================")
        logger.info("TRAINING COMPLETED AND METRICS LOGGED TO MLFLOW")
        logger.info("==================================================")
        logger.info(f"Precision: {metrics['precision']:.4f}")
        logger.info(f"Recall: {metrics['recall']:.4f}")
        logger.info(f"F1-Score: {metrics['f1_score']:.4f}")
        logger.info(f"Net Financial Savings: R$ {financial['net_savings']:.2f}")
        logger.info("==================================================")
        
        # --- Run Single Simulation (Fraud Case) ---
        decision_engine = RiskDecisionEngine()
        
        # Select a user that had transaction histories
        # Let's inspect user histories and pick one with > 3 transactions
        test_user = None
        for uid, hist in user_histories.items():
            if len(hist) >= 4:
                test_user = uid
                break
        
        if not test_user:
            test_user = list(user_histories.keys())[0]
            
        history_records = user_histories[test_user]
        last_tx = history_records[-1]
        
        # Create a suspicious transaction (Velocity + large amount + impossible travel)
        suspect_tx = {
            "transaction_id": "T_SIMUL_9999",
            "user_id": test_user,
            "timestamp": (pd.to_datetime(last_tx["timestamp"]) + pd.Timedelta(seconds=5)).isoformat(),
            "amount": 2500.00,  # High relative to standard Kaggle cards
            "latitude": last_tx["latitude"] + 6.0,  # Extreme jump
            "longitude": last_tx["longitude"] - 4.5,
            "device_id": "DEV_ROUGE_99",  # Device change
            "recipient_id": test_user  # Complete a 2-length direct laundering loop!
        }
        
        # Pre-seed the graph with the inverse transaction to ensure a cycle triggers
        # (recipient transfers to user)
        graph_det.G.add_edge(suspect_tx["recipient_id"], suspect_tx["user_id"], amount=1000.0, timestamp=last_tx["timestamp"])
        
        tx_scaled = pipeline.run_inference_pipeline(suspect_tx, history_records)
        
        s_score, s_det = stat_detector.predict_score(tx_scaled, history_records)
        r_score, r_det = rules_detector.predict_score(tx_scaled, history_records)
        i_score, i_det = iforest.predict_score(tx_scaled)
        
        seq = pipeline.preprocessor.get_user_sequence(history_records, tx_scaled, seq_length=seq_len)
        l_score, l_det = lstm_ae.predict_score(seq)
        
        x_score, x_det = xgb_det.predict_score(tx_scaled)
        g_score, g_det = graph_det.predict_score(tx_scaled)
        
        final_risk, ensemble_det = meta_learner.predict_score(s_score, r_score, i_score, l_score, x_score, g_score)
        decision, decision_det = decision_engine.evaluate_decision(final_risk)
        
        eval_details = {
            "ensemble_score": final_risk,
            "decision": decision,
            "inputs": ensemble_det["inputs"],
            "statistical_details": s_det,
            "rules_details": r_det,
            "isolation_forest_details": i_det,
            "lstm_details": l_det,
            "xgboost_details": x_det,
            "graph_details": g_det
        }
        
        explainer = FraudExplainer(provider=config["api"]["llm"]["provider"])
        explanation = explainer.generate_explanation(tx_scaled, eval_details)
        
        print("\n=== SINGLE SIMULATION EVALUATION ===")
        print(f"Transaction ID: {suspect_tx['transaction_id']}")
        print(f"User: {suspect_tx['user_id']} | Recipient: {suspect_tx['recipient_id']}")
        print(f"Amount: R$ {suspect_tx['amount']:.2f}")
        print("\n--- Model Outliers Scores (0-100) ---")
        print(f"- Statistical Outlier: {s_score:.1f}")
        print(f"- Business compliance rules: {r_score:.1f} (Disparados: {r_det['triggered_rules']})")
        print(f"- Isolation Forest Spatial anomaly: {i_score:.1f}")
        print(f"- Sequential LSTM Autoencoder: {l_score:.1f}")
        print(f"- Supervised Card Fraud (XGBoost): {x_score:.1f} (Prob: {x_det['xgboost_probability']:.2f})")
        print(f"- NetworkX Graph triangulation: {g_score:.1f} (Ciclo: {g_det['cycle_type']})")
        print("\n--- Consolidated Decision Output ---")
        print(f"- Final Ensemble Risk Score: {final_risk:.1f}/100")
        print(f"- Recommendation: {decision} - {decision_det['description']}")
        print("\n--- Explanatory GenAI Audit Report ---")
        print(explanation)
        
        # Save output to a report file and log to mlflow
        report_path = "reports/fraud_report.json"
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        import json
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump({
                "transaction": suspect_tx,
                "scores": ensemble_det["inputs"],
                "consolidated": {
                    "risk_score": final_risk,
                    "decision": decision,
                    "reasons": r_det["triggered_rules"]
                },
                "explanation": explanation
            }, f, indent=2, ensure_ascii=False)
            
        mlflow.log_artifact(report_path, "simulation_reports")

if __name__ == "__main__":
    main()
