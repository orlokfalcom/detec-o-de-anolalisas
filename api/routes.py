import time
from datetime import datetime
from typing import List, Optional
from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel, Field
from src.utils.logger import logger

router = APIRouter()

class TransactionRequest(BaseModel):
    transaction_id: str = Field(..., example="TX10293")
    user_id: str = Field(..., example="U001")
    amount: float = Field(..., example=250.50)
    timestamp: Optional[str] = Field(None, example="2026-06-21T12:00:00")
    latitude: Optional[float] = Field(None, example=-23.5505)
    longitude: Optional[float] = Field(None, example=-46.6333)
    device_id: Optional[str] = Field(None, example="DEV8899")
    recipient_id: Optional[str] = Field(None, example="REC9900")

class PredictionResponse(BaseModel):
    transaction_id: str
    user_id: str
    risk_score: float
    decision: str
    llm_explanation: str
    reasons: List[str]
    latency_ms: float

@router.post("/predict", response_model=PredictionResponse)
async def predict_fraud(payload: TransactionRequest, request: Request):
    start_time = time.time()
    
    # 1. Fetch parameters from state
    app = request.app
    pipeline = app.state.pipeline
    stat_detector = app.state.stat_detector
    rules_detector = app.state.rules_detector
    iforest_detector = app.state.iforest_detector
    lstm_detector = app.state.lstm_detector
    xgb_detector = app.state.xgb_detector
    graph_detector = app.state.graph_detector
    meta_learner = app.state.meta_learner
    decision_engine = app.state.decision_engine
    explainer = app.state.explainer
    cache = app.state.cache

    # 2. Extract values and populate defaults if missing
    tx_dict = payload.dict()
    if not tx_dict["timestamp"]:
        tx_dict["timestamp"] = datetime.now().isoformat()
    if tx_dict["latitude"] is None:
        tx_dict["latitude"] = -23.5505
    if tx_dict["longitude"] is None:
        tx_dict["longitude"] = -46.6333
    if not tx_dict["device_id"]:
        tx_dict["device_id"] = "DEV_UNKNOWN"
    if not tx_dict["recipient_id"]:
        tx_dict["recipient_id"] = f"REC_{tx_dict['device_id'][3:]}" if str(tx_dict['device_id']).startswith("DEV") else "REC_UNKNOWN"

    try:
        user_id = tx_dict["user_id"]
        
        # 3. Retrieve User History
        history = cache.get_user_history(user_id)
        
        # 4. Feature Engineering & Preprocessing
        tx_scaled = pipeline.run_inference_pipeline(tx_dict, history)
        
        # 5. Run detectors
        # Statistical outlier score
        stat_score, stat_details = stat_detector.predict_score(tx_scaled, history)
        
        # Business compliance rules
        rules_score, rules_details = rules_detector.predict_score(tx_scaled, history)
        
        # Unsupervised spatial Isolation Forest
        iforest_score, iforest_details = iforest_detector.predict_score(tx_scaled)
        
        # Behavior recurrent LSTM sequence error
        seq_len = lstm_detector.seq_len
        lstm_sequence = pipeline.preprocessor.get_user_sequence(history, tx_scaled, seq_length=seq_len)
        lstm_score, lstm_details = lstm_detector.predict_score(lstm_sequence)
        
        # Supervised XGBoost card fraud probability
        xgb_score, xgb_details = xgb_detector.predict_score(tx_scaled)
        
        # NetworkX Graph Cycle triangulation
        graph_score, graph_details = graph_detector.predict_score(tx_scaled)
        
        # 6. Ensemble Meta-Learner Consolidated Risk Score
        risk_score, ensemble_details = meta_learner.predict_score(
            stat_score, rules_score, iforest_score, lstm_score, xgb_score, graph_score
        )
        
        # 7. Final Decision Mapping
        decision, decision_details = decision_engine.evaluate_decision(risk_score)
        
        # 8. Explain Decision using LLM (with mock fallback)
        evaluation_details = {
            "ensemble_score": risk_score,
            "decision": decision,
            "inputs": ensemble_details["inputs"],
            "statistical_details": stat_details,
            "rules_details": rules_details,
            "isolation_forest_details": iforest_details,
            "lstm_details": lstm_details,
            "xgboost_details": xgb_details,
            "graph_details": graph_details
        }
        explanation = explainer.generate_explanation(tx_scaled, evaluation_details)
        
        # 9. Update Cache with current transaction
        cache.add_transaction_to_history(user_id, tx_dict)
        
        # Calculate Latency
        latency_ms = (time.time() - start_time) * 1000.0
        
        # Log latency
        logger.info(f"Evaluated TX {tx_dict['transaction_id']} in {latency_ms:.2f}ms. Risk Score: {risk_score:.2f} -> {decision}")
        
        return PredictionResponse(
            transaction_id=tx_dict["transaction_id"],
            user_id=user_id,
            risk_score=round(risk_score, 2),
            decision=decision,
            llm_explanation=explanation,
            reasons=rules_details.get("triggered_rules", []),
            latency_ms=round(latency_ms, 2)
        )

    except Exception as e:
        logger.exception("An error occurred during real-time transaction scoring.")
        raise HTTPException(status_code=500, detail=str(e))
