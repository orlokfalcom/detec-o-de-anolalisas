from src.utils.logger import logger

class RiskDecisionEngine:
    def __init__(self):
        pass

    def evaluate_decision(self, risk_score):
        """
        Maps a risk score (0-100) to an action decision:
        - 0–30: APPROVE (Low Risk, instant release)
        - 31–60: MONITOR (Medium Risk, pass but flag for observation / telemetry)
        - 61–80: REVIEW (High Risk, route to human fraud analysts queue)
        - 81–100: BLOCK (Critical Risk, block instantly)
        """
        score = float(risk_score)
        
        if score < 0:
            score = 0.0
        elif score > 100:
            score = 100.0
            
        if score <= 30.0:
            decision = "APPROVE"
            description = "Transação de baixo risco aprovada automaticamente."
        elif score <= 60.0:
            decision = "MONITOR"
            description = "Transação aprovada com monitoramento adicional ativo."
        elif score <= 80.0:
            decision = "REVIEW"
            description = "Transação enviada para análise manual da equipe antifraude."
        else:
            decision = "BLOCK"
            description = "Transação bloqueada devido a altíssimo risco de fraude detectado."
            
        details = {
            "risk_score": score,
            "decision": decision,
            "description": description
        }
        
        return decision, details
