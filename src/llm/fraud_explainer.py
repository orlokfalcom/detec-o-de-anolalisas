import os
from src.utils.logger import logger
from src.llm.prompts import FRAUD_EXPLAINER_SYSTEM_PROMPT, FRAUD_EXPLAINER_USER_TEMPLATE

class FraudExplainer:
    def __init__(self, provider="mock", api_key=None, model_name=None):
        self.provider = provider
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY") or os.environ.get("GEMINI_API_KEY")
        self.model_name = model_name
        
        if self.api_key and self.provider == "mock":
            if os.environ.get("OPENAI_API_KEY"):
                self.provider = "openai"
                self.model_name = model_name or "gpt-4o-mini"
            elif os.environ.get("GEMINI_API_KEY"):
                self.provider = "gemini"
                self.model_name = model_name or "gemini-2.5-flash"

        logger.info(f"Initialized FraudExplainer with provider: {self.provider}")

    def generate_explanation(self, transaction_data, evaluation_details):
        """
        Generates an audit explanation for the risk engine decision.
        """
        fmt_vars = self._prepare_variables(transaction_data, evaluation_details)
        prompt = FRAUD_EXPLAINER_USER_TEMPLATE.format(**fmt_vars)
        
        if self.provider == "openai" and self.api_key:
            return self._call_openai(prompt)
        elif self.provider == "gemini" and self.api_key:
            return self._call_gemini(prompt)
        else:
            return self._generate_mock_explanation(fmt_vars)

    def _prepare_variables(self, tx, eval_details):
        risk_score = eval_details.get("ensemble_score", 0.0)
        decision = eval_details.get("decision", "REVIEW")
        
        inputs = eval_details.get("inputs", {})
        stat_details = eval_details.get("statistical_details", {})
        rules_details = eval_details.get("rules_details", {})
        iforest_details = eval_details.get("isolation_forest_details", {})
        lstm_details = eval_details.get("lstm_details", {})
        xgb_details = eval_details.get("xgboost_details", {})
        graph_details = eval_details.get("graph_details", {})

        recipient = tx.get("recipient_id")
        if not recipient:
            recipient = f"REC_{tx.get('device_id', 'UNKNOWN')[3:]}" if str(tx.get('device_id')).startswith("DEV") else "REC_UNKNOWN"

        return {
            "transaction_id": tx.get("transaction_id", "N/A"),
            "user_id": tx.get("user_id", "N/A"),
            "recipient_id": recipient,
            "amount": float(tx.get("amount", 0.0)),
            "timestamp": tx.get("timestamp", "N/A"),
            "latitude": float(tx.get("latitude", 0.0)),
            "longitude": float(tx.get("longitude", 0.0)),
            "device_id": tx.get("device_id", "N/A"),
            
            "risk_score": float(risk_score),
            "decision": decision,
            
            "rules_score": float(inputs.get("rules", 0.0)),
            "triggered_rules": ", ".join(rules_details.get("triggered_rules", ["Nenhuma"])),
            
            "statistical_score": float(inputs.get("statistical", 0.0)),
            "z_score": float(stat_details.get("z_score", 0.0)),
            "z_score_threshold": float(stat_details.get("z_score_threshold", 3.0)),
            "user_mean": float(stat_details.get("user_mean", 0.0)),
            "user_std": float(stat_details.get("user_std", 0.0)),
            
            "xgboost_score": float(inputs.get("xgboost", 0.0)),
            "xgboost_probability": float(xgb_details.get("xgboost_probability", 0.0)),
            
            "isolation_forest_score": float(inputs.get("isolation_forest", 0.0)),
            "iforest_anomaly": "ALERTA: Anomalia Multidimensional" if iforest_details.get("is_anomaly", False) else "Normal",
            
            "lstm_score": float(inputs.get("lstm", 0.0)),
            "lstm_anomaly": "ALERTA: Desvio Sequencial Comportamental" if lstm_details.get("is_anomaly", False) else "Normal",
            
            "graph_score": float(inputs.get("graph_triangulation", 0.0)),
            "graph_anomaly": f"ALERTA: Ciclo detectado ({graph_details.get('cycle_type', 'N/A')})" if graph_details.get("cycle_detected", False) else "Sem ciclo",
            
            "speed_kph": float(tx.get("speed_kph", 0.0)),
            "dist_from_prev_km": float(tx.get("dist_from_prev_km", 0.0)),
            "velocity_count_1m": float(tx.get("velocity_count_1m", 1.0)),
            "velocity_sum_1m": float(tx.get("velocity_sum_1m", 0.0)),
            "velocity_count_5m": float(tx.get("velocity_count_5m", 1.0)),
            "velocity_sum_5m": float(tx.get("velocity_sum_5m", 0.0)),
            "velocity_count_1h": float(tx.get("velocity_count_1h", 1.0)),
            "velocity_sum_1h": float(tx.get("velocity_sum_1h", 0.0)),
            "device_count_24h": float(tx.get("device_count_24h", 1.0)),
            "device_changed": "Sim" if tx.get("device_changed", 0) == 1 else "Não"
        }

    def _call_openai(self, prompt):
        try:
            import openai
            client = openai.OpenAI(api_key=self.api_key)
            response = client.chat.completions.create(
                model=self.model_name or "gpt-4o-mini",
                messages=[
                    {"role": "system", "content": FRAUD_EXPLAINER_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=300
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"OpenAI API call failed: {e}. Falling back to mock explanation.")
            return self._generate_mock_explanation(self._last_fmt_vars)

    def _call_gemini(self, prompt):
        try:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            model = genai.GenerativeModel(
                model_name=self.model_name or "gemini-1.5-flash",
                system_instruction=FRAUD_EXPLAINER_SYSTEM_PROMPT
            )
            response = model.generate_content(
                prompt,
                generation_config={"temperature": 0.2, "max_output_tokens": 300}
            )
            return response.text
        except Exception as e:
            logger.error(f"Gemini API call failed: {e}. Falling back to mock explanation.")
            return self._generate_mock_explanation(self._last_fmt_vars)

    def _generate_mock_explanation(self, fmt_vars):
        self._last_fmt_vars = fmt_vars
        
        user_id = fmt_vars["user_id"]
        recipient_id = fmt_vars["recipient_id"]
        amount = fmt_vars["amount"]
        risk_score = fmt_vars["risk_score"]
        decision = fmt_vars["decision"]
        triggered = fmt_vars["triggered_rules"]
        speed = fmt_vars["speed_kph"]
        dev_changed = fmt_vars["device_changed"]
        dev_cnt = fmt_vars["device_count_24h"]
        
        is_night_limit = "bacen_night_limit" in triggered
        is_limit = "single_max_amount" in triggered
        is_velocity = "high_frequency_velocity" in triggered
        is_travel = "impossible_travel" in triggered
        is_dev_swap = "device_fingerprint_limit" in triggered
        
        is_graph_cycle = fmt_vars["graph_anomaly"] != "Sem ciclo"
        
        if decision == "BLOCK":
            decision_exp = f"A transação foi BLOQUEADA preventivamente por suspeita grave de fraude/lavagem. Score de risco consolidado em {risk_score:.1f}/100."
        elif decision == "REVIEW":
            decision_exp = f"A transação foi enviada para FILA DE ANÁLISE MANUAL devido a desvios estatísticos médios e violação de regras operacionais secundárias."
        elif decision == "MONITOR":
            decision_exp = f"A transação foi liberada no limite do risco, sob MONITORAMENTO adicional por desvio sutil comportamental."
        else:
            decision_exp = f"A transação foi APROVADA automaticamente. O comportamento é condizente com o histórico do cliente {user_id}."

        signals = []
        if is_night_limit:
            signals.append(f"Disparo de Pix Noturno (BACEN) para valor de R$ {amount:.2f}.")
        if is_limit:
            signals.append(f"Gargalo de limite individual ultrapassado (R$ {amount:.2f}).")
        if is_velocity:
            signals.append(f"Frequência anômala de transações ({fmt_vars['velocity_count_1m']:.0f} no último minuto).")
        if is_travel:
            signals.append(f"Velocidade incompatível fisicamente: {speed:.1f} km/h entre cidades.")
        if is_dev_swap:
            signals.append(f"Uso de {dev_cnt:.0f} dispositivos nas últimas 24h.")
        if is_graph_cycle:
            signals.append(f"ALERTA FINANCEIRO: O grafo de transferências detectou triangulação de contas Pix ({fmt_vars['graph_anomaly']}) direcionado a {recipient_id}.")

        if not signals:
            if risk_score > 30:
                signals.append("Desvio acumulado nos detectores probabilísticos e de rede neural sequencial.")
            else:
                signals.append("Nenhum sinal suspeito foi disparado.")

        signals_exp = "\n".join([f"- {s}" for s in signals])

        user_mean = fmt_vars["user_mean"]
        z_score = fmt_vars["z_score"]
        
        if user_mean > 0:
            dev_pct = ((amount - user_mean) / user_mean) * 100.0
            if dev_pct > 200.0:
                behavior = f"O valor de R$ {amount:.2f} representa um aumento anômalo de {dev_pct:.1f}% em relação ao gasto médio histórico (R$ {user_mean:.2f}). Z-score: {z_score:.2f}."
            else:
                behavior = f"Gasto compatível com a média histórica do cliente (R$ {user_mean:.2f}). Z-score: {z_score:.2f}."
        else:
            behavior = "Histórico insuficiente para cálculo de base de comparação."

        if fmt_vars["lstm_anomaly"] == "ALERTA: Desvio Sequencial Comportamental":
            behavior += " O modelo recorrente LSTM acionou desvio de padrão sequencial recente."
        if fmt_vars["xgboost_score"] > 60:
            behavior += f" O classificador XGBoost reportou probabilidade de {fmt_vars['xgboost_probability']:.2f} de fraude no cartão."

        if decision == "BLOCK":
            conclusion = "Manter conta sob bloqueio. Se houver ciclo de triangulação, notificar os bancos envolvidos (mecanismo especial de devolução MED - Pix)."
        elif decision == "REVIEW":
            conclusion = f"Recomenda-se contatar o portador para verificação cadastral. Confirmar se o beneficiário {recipient_id} é legítimo."
        elif decision == "MONITOR":
            conclusion = "Manter observação analítica sobre a telemetria do usuário."
        else:
            conclusion = "Nenhuma ação necessária."

        report = f"""### 🧠 RELATÓRIO ANTIFRAUDE (MOCK GENERATIVE AI RESPONSE)

**1. 📌 Explicação da decisão:**
{decision_exp}

**2. 📊 Sinais principais:**
{signals_exp}

**3. 🧬 Análise comportamental:**
{behavior}

**4. 🧠 Conclusão:**
{conclusion}"""
        return report
