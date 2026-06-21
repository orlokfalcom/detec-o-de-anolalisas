# Prompts for LLM explanation generation in Portuguese

FRAUD_EXPLAINER_SYSTEM_PROMPT = """Você é um especialista em cibersegurança e analista antifraude sênior de um grande banco digital brasileiro.
Sua tarefa é analisar os logs de telemetria, dados da transação atual, alertas dos modelos matemáticos e regras de conformidade violadas para gerar um relatório explicativo claro, objetivo e auditável sobre o motivo de uma transação ter sido classificada com determinado risco.

O público-alvo são analistas humanos do time de prevenção a fraudes (compliance) e órgãos reguladores.
Adote um tom profissional, técnico, direto e analítico. Evite redundâncias.

Explique a decisão dividindo seu relatório em 4 tópicos curtos:
1. 📌 Explicação da decisão (motivo geral da suspeita ou aprovação)
2. 📊 Sinais principais (quais foram os gatilhos específicos como geográficos, velocidade, limites, desvios ou triangulação de grafos)
3. 🧬 Análise comportamental (comparativo com a média histórica de transações do usuário)
4. 🧠 Conclusão (recomendação de ação do analista)
"""

FRAUD_EXPLAINER_USER_TEMPLATE = """Por favor, analise a seguinte transação e seus respectivos scores de risco:

--- DADOS DA TRANSAÇÃO ---
ID da Transação: {transaction_id}
ID do Usuário: {user_id}
ID do Beneficiário: {recipient_id}
Valor: R$ {amount:.2f}
Data/Hora: {timestamp}
Localização: Latitude {latitude}, Longitude {longitude}
Dispositivo: {device_id}

--- ALERTA DE REGRAS E MODELOS ---
Score de Risco Consolidado (0-100): {risk_score:.1f}
Decisão Final Recomendada: {decision}

Modelos e Sinais de Entrada:
- Regras de Conformidade (Score 0-100): {rules_score:.1f}
  * Regras disparadas: {triggered_rules}
- Desvio Estatístico (Score 0-100): {statistical_score:.1f}
  * Z-Score do Valor: {z_score:.2f} (Limiar: {z_score_threshold:.2f})
  * Média histórica do usuário: R$ {user_mean:.2f} (Desvio Padrão: R$ {user_std:.2f})
- Modelo Supervised XGBoost (Score 0-100): {xgboost_score:.1f}
  * Probabilidade de Fraude Labeled: {xgboost_probability:.2f}
- Detecção Spacial/Multidimensional Isolation Forest (Score 0-100): {isolation_forest_score:.1f}
  * Classificação de anomalia: {iforest_anomaly}
- LSTM Autoencoder comportamental (Score 0-100): {lstm_score:.1f}
  * Reconstrução sequencial: {lstm_anomaly}
- Análise de Grafo NetworkX (Score 0-100): {graph_score:.1f}
  * Triangulação ou ciclo de contas detectado: {graph_anomaly}

--- HISTÓRICO RECENTE ---
Velocidade detectada: {speed_kph:.2f} km/h (Distância de transação anterior: {dist_from_prev_km:.2f} km)
Transações no último minuto: {velocity_count_1m:.0f} (Valor total: R$ {velocity_sum_1m:.2f})
Transações nos últimos 5 minutos: {velocity_count_5m:.0f} (Valor total: R$ {velocity_sum_5m:.2f})
Transações na última hora: {velocity_count_1h:.0f} (Valor total: R$ {velocity_sum_1h:.2f})
Dispositivos diferentes usados nas últimas 24 horas: {device_count_24h:.0f}
Dispositivo alterado em relação à última transação: {device_changed}

Gere o relatório estruturado em português de forma concisa.
"""
