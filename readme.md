🧠 📦 REPOSITÓRIO PROFISSIONAL — FRAUD INTELLIGENCE SYSTEM
===========================================================

🚀 Nome do Projeto
------------------
**fraud-intelligence-ai**

📁 Estrutura do Repositório
---------------------------
```
fraud-intelligence-ai/
│
├── README.md                     # Este guia completo do sistema
├── requirements.txt              # Dependências em pacotes Python
├── setup.py                      # Script de instalação do pacote local
├── .gitignore                    # Regras de exclusão do git para binários e logs
│
├── config/
│   ├── config.yaml               # Parâmetros de rede, MLflow, e limites de modelos
│   └── rules.yaml                # Regras regulatórias e limites de compliance (BACEN)
│
├── data/
│   ├── raw/
│   ├── processed/
│   ├── sample_transactions.csv   # Dataset sintético de transações com fraudes acopladas
│   └── real_transactions.csv     # Dataset real Kaggle mapped no esquema de telemetria
│
├── notebooks/
│   ├── 01_exploratory_analysis.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_model_training.ipynb
│
├── src/
│   ├── main.py                   # Script de treinamento, carregamento real e simulação
│   │
│   ├── pipeline/
│   │   ├── ingestion.py          # Simulador de ingestão batch/streaming
│   │   ├── preprocessing.py      # Imputação, StandardScaler e Target Encoding (Dispositivos)
│   │   ├── feature_engineering.py# Extração de velocidade, geografia e desvio cumulativo
│   │   └── pipeline.py           # Orquestração do processamento raw-to-scaled
│   │
│   ├── detectors/
│   │   ├── statistical.py        # Anomalia de desvios via Z-score e IQR
│   │   ├── rules.py              # Motor de regras heurísticas regulatórias (Pix Noturno)
│   │   ├── isolation_forest.py   # Anomalia multidimensional espacial não supervisionada
│   │   ├── lstm_autoencoder.py   # Reconstrução de sequências temporais normais (PyTorch)
│   │   ├── xgboost_detector.py   # Classificação supervisionada de fraude de cartões (SMOTE)
│   │   └── graph_triangulation.py# Análise de grafo em rede de triangulação Pix (NetworkX)
│   │
│   ├── ensemble/
│   │   └── meta_learner.py       # Combinador de scores (Regressão Logística + Veto em Cascata)
│   │
│   ├── llm/
│   │   ├── fraud_explainer.py    # Explicador gerador de auditoria (Gemini/OpenAI/Mock)
│   │   └── prompts.py            # Prompts estruturados em Markdown para a IA Generativa
│   │
│   ├── decision/
│   │   └── risk_engine.py        # Conversor de Score consolidado para Ações
│   │
│   └── utils/
│       ├── logger.py             # Log estruturado em JSON
│       ├── metrics.py            # Avaliador de Acurácia, F1 e Impacto Financeiro (Economia)
│       └── helpers.py            # Distância Haversine, cálculo de velocidades
│
├── api/
│   ├── app.py                    # Bootstrap FastAPI com Cache-Aside (In-Memory/Redis)
│   └── routes.py                 # Endpoint POST /predict
│
├── models/                       # Binários serializados dos scalers, encoders e modelos
│   ├── scaler.pkl
│   ├── target_encoder.pkl
│   ├── isolation_forest.pkl
│   ├── lstm_autoencoder.pth
│   ├── xgboost_model.pkl
│   └── meta_model.pkl
│
├── reports/
│   └── fraud_report.json         # Artefato de simulação com predição e explicação gerada
│
└── tests/                        # Conjunto completo de testes pytest
    ├── test_pipeline.py
    ├── test_detectors.py
    └── test_api.py
```

🧱 Arquitetura e Engenharia do Sistema
--------------------------------------
O sistema foi arquitetado de forma modular sob conceitos de barreira de latência rígida e tratamento estatístico robusto.

### 1. Ingestão e Processamento Stateful
*   **Feature Engineering**: Transforma as transações computando dinamicamente a velocidade temporal (janelas rolantes de 1 min, 5 min e 1 hora), velocidade de movimentação geográfica física (alerta de impossibilidade espacial utilizando a fórmula de distância Haversine) e desvio estatístico de consumo.
*   **Target Encoding**: Transforma categorias de IDs de Dispositivos móveis (`device_id`) em um mapeamento contínuo baseado no target de fraude histórico, suavizado por um fator global para evitar overfitting.

### 2. Barreira Multicamadas de Detectores (6 Modelos)
O motor antifraude executa simultaneamente seis estratégias complementares:
1.  **Motor Heurístico de Regras**: Processa de forma síncrona validações regulatórias, como o limite Pix Noturno imposto pelo Banco Central do Brasil (BACEN) e limites de transações repetitivas.
2.  **Detector Estatístico (Z-Score & IQR)**: Identifica anomalias volumétricas individuais comparadas diretamente com a média e desvio padrão acumulado das movimentações do próprio cliente.
3.  **Isolation Forest (ML)**: Algoritmo não supervisionado que localiza e isola anomalias no espaço multidimensional de coordenadas geográficas e quantidades movimentadas.
4.  **LSTM Autoencoder (Deep Learning)**: Rede Neural Recorrente em PyTorch que aprende a sequência comportamental temporal normal de uso da conta do usuário. Anomalias são detectadas pelo erro de reconstrução na saída.
5.  **Supervised XGBoost**: Classificador supervisionado com alta precisão e recall para fraude clássica de cartões de crédito. O modelo é treinado sobre base balanceada através do algoritmo SMOTE (`imbalanced-learn`).
6.  **Grafo de Triangulação (NetworkX)**: Constrói uma representação em rede dirigida das transferências da conta do remetente e do destinatário (`recipient_id`) para acusar ciclos fraudulentos e triangulação de Pix (ex: A -> B -> C -> A) em tempo de execução.

### 3. Voto de Veto em Cascata (Decision & Ensemble)
Para ponderar as opiniões dos modelos, o sistema emprega uma regressão logística treinada (`meta_learner`). No entanto, para fins de compliance bancário, a decisão do modelo de ML é envelopada em uma **Arquitetura de Veto em Cascata**: caso o motor de regras da conformidade acuse uma violação direta ou o grafoNetworkX identifique uma triangulação clara de lavagem Pix, a transação é bloqueada preventivamente (`BLOCK`), sobrepondo o classificador probabilístico.

### 4. IA Generativa Explicável (LLM)
As transações de alto risco encaminhadas para fila humana (`REVIEW` ou `BLOCK`) passam pela camada explicadora que formula um prompt complexo traduzindo estatísticas e alertas técnicos de rede para um relatório humanamente legível, no formato ideal para analistas de compliance seniores. 
*   *Fallback*: Caso nenhuma API key de IA (`GEMINI_API_KEY` ou `OPENAI_API_KEY`) esteja ativa na sessão, o sistema gera deterministicamente o mesmo relatório estruturado localmente via templates sem quebrar o fluxo da API.

### 5. Análise Fina de Riscos e Mitigação de Falsos Positivos
O sistema rastreia e analisa todas as probabilidades e anomalias de fraude detectadas, por mais irrelevantes ou sutis que sejam. Todas as saídas dos 6 detectores individuais são expostas em um bloco de resposta granular (`all_probabilities`). Para mitigar falsos positivos e evitar o bloqueio de usuários legítimos por desvios mínimos de comportamento ou valores, o motor de decisão e o gerador de relatórios toleram estes desvios sutis, mantendo a transação como `APPROVE` ou `MONITOR` e justificando no relatório de auditoria por que o bloqueio não foi acionado.

🌐 Especificações da API (FastAPI)
----------------------------------
O serviço opera em baixa latência (<50ms).

### Endpoint de Predição Antifraude
**POST `/predict`**

*   **Request Payload**:
    ```json
    {
      "transaction_id": "TX_990182",
      "user_id": "U015",
      "amount": 2500.00,
      "timestamp": "2026-06-21T22:30:00",
      "latitude": -23.5505,
      "longitude": -46.6333,
      "device_id": "DEV0088",
      "recipient_id": "U002"
    }
    ```

*   **Response Payload**:
    ```json
    {
      "transaction_id": "TX_990182",
      "user_id": "U015",
      "risk_score": 12.5,
      "decision": "APPROVE",
      "llm_explanation": "### 🧠 RELATÓRIO ANTIFRAUDE...\n\n**1. 📌 Explicação da decisão:**\nTransação aprovada... Nota: Foram analisadas todas as probabilidades irrelevantes encontradas (Z-score leve de 1.15), mas optou-se pela liberação...",
      "reasons": [],
      "all_probabilities": {
        "xgboost_probability": 0.02,
        "statistical_z_score": 1.15,
        "isolation_forest_anomaly_score": 0.08,
        "lstm_reconstruction_error": 0.12,
        "network_cycle_detected": false,
        "rules_triggered_count": 0
      },
      "latency_ms": 12.4
    }
    ```

🧪 Execução do Ambiente de Treinamento e Testes
-----------------------------------------------
1.  **Criação do ambiente virtual e instalação**:
    ```bash
    python -m venv .venv
    .venv\Scripts\pip install -r requirements.txt
    ```

2.  **Treinamento completo e Simulação**:
    O script faz o download do dataset real de fraudes no GitHub (ou de qualquer link customizado), executa o mapeamento dinâmico inteligente das colunas e gera dados sintéticos de fallback caso faltem parâmetros como geolocalização ou destinatários.
    
    *Executar treinamento padrão (Kaggle mirror):*
    ```bash
    .venv\Scripts\python.exe -m src.main
    ```

    *Executar com dataset externo customizado e mapeamento explícito:*
    ```bash
    .venv\Scripts\python.exe -m src.main --dataset-url "https://sua-url-publica/creditcard.csv" --column-map "{\"amount\": \"Amount\", \"is_fraud\": \"Class\", \"timestamp\": \"Time\"}"
    ```

3.  **Execução da Suíte de Testes**:
    ```bash
    .venv\Scripts\python.exe -m pytest
    ```

4.  **Iniciar a API de Produção**:
    ```bash
    .venv\Scripts\uvicorn api.app:app --reload
    ```
