import os
from fastapi import FastAPI
from src.utils.helpers import load_yaml
from src.utils.logger import logger
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

class CacheManager:
    """
    Manages user transaction histories.
    Falls back to a standard Python dictionary if Redis is not configured or fails.
    """
    def __init__(self, use_redis=False, redis_url=None):
        self.use_redis = use_redis
        self.local_cache = {}
        self.redis_client = None
        
        if use_redis:
            try:
                import redis
                import json
                self.redis_client = redis.from_url(redis_url)
                self.redis_client.ping()
                logger.info("Successfully connected to Redis cache.")
            except Exception as e:
                logger.error(f"Failed to connect to Redis: {e}. Falling back to In-Memory Cache.")
                self.use_redis = False

    def get_user_history(self, user_id):
        if self.use_redis:
            try:
                import json
                data = self.redis_client.get(f"user_hist:{user_id}")
                if data:
                    return json.loads(data)
            except Exception as e:
                logger.error(f"Error fetching from Redis: {e}")
        return self.local_cache.get(user_id, [])

    def add_transaction_to_history(self, user_id, tx_dict):
        history = self.get_user_history(user_id)
        history.append(tx_dict)
        history = history[-20:]
        
        if self.use_redis:
            try:
                import json
                self.redis_client.set(f"user_hist:{user_id}", json.dumps(history))
                return
            except Exception as e:
                logger.error(f"Error saving to Redis: {e}")
                
        self.local_cache[user_id] = history


# Create the FastAPI app instance
app = FastAPI(
    title="Fraud Intelligence AI System API",
    description="Real-time hybrid system for financial fraud detection, decision orchestration, and explanation.",
    version="1.0.0"
)

# Load configuration and models
config_path = os.environ.get("CONFIG_PATH", "config/config.yaml")
config = load_yaml(config_path)

# Initialize system engines
logger.info("Initializing Fraud Pipeline and ML/DL Detectors...")
app.state.pipeline = FraudPipeline(config)

app.state.stat_detector = StatisticalDetector(
    z_score_threshold=config["models"]["statistical"]["z_score_threshold"],
    iqr_multiplier=config["models"]["statistical"]["iqr_multiplier"]
)
app.state.rules_detector = RulesDetector("config/rules.yaml")

# Load pre-trained models
app.state.iforest_detector = IsolationForestDetector(
    model_path=f"{config['models']['save_dir']}/isolation_forest.pkl"
).load_model()

app.state.lstm_detector = LSTMAutoencoderDetector(
    model_path=f"{config['models']['save_dir']}/lstm_autoencoder.pth"
).load_model()

app.state.xgb_detector = XGBoostDetector(
    model_path=f"{config['models']['save_dir']}/xgboost_model.pkl"
).load_model()

app.state.graph_detector = GraphTriangulationDetector(
    model_path=f"{config['models']['save_dir']}/graph_network.pkl"
).load_model()

app.state.meta_learner = EnsembleMetaLearner(
    weights=config["ensemble"]["weights"],
    meta_model_path=f"{config['models']['save_dir']}/meta_model.pkl"
).load_meta_learner()

app.state.decision_engine = RiskDecisionEngine()
app.state.explainer = FraudExplainer(provider=config["api"]["llm"]["provider"])

# Initialize cache manager
app.state.cache = CacheManager(
    use_redis=config["api"]["use_redis"],
    redis_url=config["api"]["redis_url"]
)

# Include routes
from api.routes import router
app.include_router(router)

@app.on_event("startup")
async def startup_event():
    logger.info("Fraud Intelligence AI API is online and fully loaded.")
