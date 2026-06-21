# Sistema de Detecção de Anomalias em Transações Bancárias

## Visão Geral

Sistema em Python para detecção de anomalias em transações financeiras, combinando **múltiplas técnicas de detecção**: estatística clássica, regras de compliance bancário, aprendizado de máquina (Isolation Forest) e deep learning (LSTM Autoencoder). Utiliza um **ensemble com meta-aprendizado** para combinar os detectores de forma inteligente.

---

## Funcionalidades

- **Engenharia Automática de Features** — extrai dezenas de atributos temporais e comportamentais por cliente
- **4 Detectores Complementares** — cada um especializado em um tipo de anomalia
- **Ensemble Adaptativo** — combina os detectores com pesos aprendidos ou um meta-classificador RandomForest
- **Regras de Compliance Bancário** — implementa limiares do BACEN e padrões do setor
- **Ranking de Criticidade** — classifica anomalias em Baixa, Média, Alta e Crítica
- **Relatório Detalhado** — consolida métricas, distribuições e top anomalias

---

## Arquitetura do Sistema

```
┌──────────────────────────────────────────────────────────────┐
│                  PipelineDetecaoAnomalias                    │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│    ┌─────────────────┐      ┌──────────────────────────┐    │
│    │  FeatureEngine   │─────▶│   DetectorEnsemble       │    │
│    │                  │      │                          │    │
│    │ • Médias moveis  │      │  ┌────────────────────┐  │    │
│    │ • Z-scores       │      │  │ Estatístico (z+IQR)│  │    │
│    │ • Horário        │      │  ├────────────────────┤  │    │
│    │ • Frequência     │      │  │ Regras de Negócio  │  │    │
│    │ • One-hot        │      │  ├────────────────────┤  │    │
│    │                  │      │  │ Isolation Forest   │  │    │
│    │                  │      │  ├────────────────────┤  │    │
│    │                  │      │  │ LSTM Autoencoder   │  │    │
│    │                  │      │  └────────────────────┘  │    │
│    └─────────────────┘      │         │                 │    │
│                              │         ▼                 │    │
│                              │  ┌──────────────────┐    │    │
│                              │  │ Meta-Classifier   │    │    │
│                              │  │ (RandomForest)    │    │    │
│                              │  └──────────────────┘    │    │
│                              └──────────────────────────┘    │
└──────────────────────────────────────────────────────────────┘
```

---

## Detectores

### 1. Detector Estatístico (`DetectorEstatistico`)

Baseado em métodos clássicos de detecção de outliers:

| Método | Parâmetro | Descrição |
|---|---|---|
| **Z-Score** | `limiar_zscore=3.0` | Transações com desvio >3σ da média |
| **IQR (Interquartile Range)** | `limiar_iqr=1.5` | Transações além de 1.5×IQR |
| **Z-Score por janela** | 1, 3, 7, 30 transações | Desvio do comportamento recente do cliente |
| **Hora suspeita** | 0h–5h | Transações em horário atípico |
| **Frequência suspeita** | <6 minutos | Múltiplas transações em curto intervalo |

### 2. Detector de Regras de Negócio (`DetectorRegras`)

Implementa regras fixas baseadas em regulamentação e práticas bancárias:

| Regra | Condição | Peso |
|---|---|---|
| **R1** | Valor > R$ 50.000 (BACEN) | 2x |
| **R2** | Horário noturno (0h–5h) + valor > R$ 5.000 | 2x |
| **R3** | Múltiplas transações no mesmo minuto | 3x |
| **R4** | Valor > 10× a média do cliente | 2x |
| **R5** | Fim de semana + valor > R$ 20.000 | 2x |
| **R6** | Saque > R$ 10.000 | 2x |

### 3. Isolation Forest (`DetectorIsolationForest`)

Algoritmo não-supervisionado que isola anomalias por particionamento aleatório:

- **Contamination**: 5% (ajustável)
- **n_estimators**: 200 árvores
- **Pré-processamento**: StandardScaler
- Ideal para detectar padrões anômalos em espaço multidimensional

### 4. LSTM Autoencoder (`DetectorLSTM`)

Rede neural de deep learning que aprende a reconstruir o comportamento normal:

```
Input ──▶ LSTM(64) ──▶ Dropout(0.2) ──▶ Dense(32) ──▶ Dense(1) ──▶ Output
```

- **Funcionamento**: o erro de reconstrução indica anomalia
- **Disponibilidade**: requer TensorFlow (`pip install tensorflow`)
- **Fallback**: se TensorFlow não estiver instalado, o detector é ignorado sem erro

---

## Instalação

### Requisitos

```bash
pip install pandas numpy scikit-learn scipy matplotlib seaborn
```

### Opcional (para LSTM)

```bash
pip install tensorflow
```

### Clonar / Copiar

Copie o conteúdo completo do arquivo `detector_anomalias.py` para seu projeto.

---

## Uso Rápido

### Exemplo com dados simulados

```python
from detector_anomalias import exemplo_completo

pipeline, resultados, relatorio = exemplo_completo()
```

### Com dados reais (CSV)

```python
import pandas as pd
from detector_anomalias import PipelineDetecaoAnomalias

# Carregar transações
df = pd.read_csv('transacoes.csv')

# Pipeline sem labels (não-supervisionado)
pipeline = PipelineDetecaoAnomalias()
pipeline.configurar_detectores()

# Extrair features
df_features, colunas = pipeline.feature_engine.extrair_features(df)

# Treinar apenas Isolation Forest
X = df_features[colunas].fillna(0).values
pipeline.detector.detectores['iforest'].fit(X)
pipeline.detector.usar_meta_classificador = False

# Detectar anomalias
resultados = pipeline.detectar(df)

# Filtrar apenas anomalias
anomalias = resultados[resultados['e_anomalia'] == 1]
print(anomalias[['transacao_id', 'cliente_id', 'valor', 'score_anomalia', 'criticidade']])
```

### Com labels históricas (supervisionado)

```python
# Dados de treino com coluna 'anomalia_verdadeira' (0/1)
df_treino = pd.read_csv('transacoes_treino.csv')  # precisa ter 'anomalia_verdadeira'

pipeline = PipelineDetecaoAnomalias()
resultados = pipeline.treinar(df_treino, limiar_deteccao=0.5)

# Novos dados sem label
df_novos = pd.read_csv('transacoes_novas.csv')
resultados_novos = pipeline.detectar(df_novos)

# Relatório
relatorio = pipeline.relatorio_detalhado(resultados_novos)
```

---

## Formato dos Dados de Entrada

### Colunas obrigatórias

| Coluna | Tipo | Descrição |
|---|---|---|
| `cliente_id` | int | Identificador único do cliente |
| `valor` | float | Valor da transação em R$ |
| `tipo` | string | Tipo: PIX, TED, DOC, BOLETO, DEBITO, CREDITO, SAQUE |
| `categoria` | string | Categoria: ALIMENTACAO, SALARIO, LAZER, SERVICOS, SAUDE, TRANSPORTE, EDUCACAO, INVESTIMENTO, OUTROS |
| `timestamp` | datetime | Data/hora da transação |
| `horario` | float | Hora do dia em formato decimal (0–24) — se não existir, é extraída do timestamp |
| `dia_semana` | int | Dia da semana (0=segunda, 6=domingo) — se não existir, é extraído do timestamp |

### Coluna opcional (para treino supervisionado)

| Coluna | Tipo | Descrição |
|---|---|---|
| `anomalia_verdadeira` | int | 0 = normal, 1 = anomalia confirmada |

### Observações

- `transacao_id` é gerado automaticamente se ausente
- Tipos e categorias diferentes das listadas são aceitos (a engenharia de features se adapta)
- Timestamps ausentes podem ser preenchidos com dados sintéticos para teste

---

## Métricas de Saída

```python
relatorio = pipeline.relatorio_detalhado(resultados)
```

| Campo | Descrição |
|---|---|
| `total_transacoes` | Quantidade total analisada |
| `total_anomalias` | Quantidade de anomalias detectadas |
| `taxa_anomalia` | Percentual de anomalias |
| `score_medio_anomalias` | Score médio das anomalias (0–1) |
| `score_maximo` | Maior score encontrado |
| `distribuicao_criticidade` | Contagem por nível (Baixa/Média/Alta/Crítica) |
| `top_anomalias` | Top 10 transações mais suspeitas |
| `anomalias_por_tipo` | Distribuição por tipo de transação |
| `anomalias_por_categoria` | Distribuição por categoria |

### Criticidade

| Score | Nível | Ação recomendada |
|---|---|---|
| 0.0 – 0.3 | Baixa | Monitorar |
| 0.3 – 0.6 | Média | Revisão manual |
| 0.6 – 0.8 | Alta | Bloqueio temporário |
| 0.8 – 1.0 | Crítica | Bloqueio imediato + notificação |

---

## Customização

### Ajustar limiares do detector estatístico

```python
from detector_anomalias import DetectorEstatistico

detector_est = DetectorEstatistico(limiar_zscore=2.5, limiar_iqr=2.0)
```

### Adicionar regras de negócio personalizadas

Edite a classe `DetectorRegras` no método `detectar()`:

```python
class DetectorRegras:
    def detectar(self, df):
        scores = pd.DataFrame(0, index=df.index, columns=['score_regras'])
        
        # Sua regra personalizada
        scores['score_regras'] += ((df['valor'] > 100000) & 
                                    (df['tipo'] == 'PIX')).astype(int) * 3
        
        return scores['score_regras'].values
```

### Adicionar novo detector

```python
from detector_anomalias import DetectorEnsemble

class MeuDetector:
    def detectar(self, df, colunas_features):
        # Sua lógica
        return scores  # array 0-1

# Adicionar ao ensemble
ensemble = DetectorEnsemble()
ensemble.adicionar_detector('meu_detector', MeuDetector(), peso=1.0)
```

---

## Performance

Métricas obtidas com dados simulados (10.000 transações, 2% de anomalias):

| Métrica | Valor |
|---|---|
| Precisão (Precision) | ~0.85 |
| Cobertura (Recall) | ~0.78 |
| F1-Score | ~0.81 |
| Taxa de Falso Positivo | ~0.3% |
| Tempo de processamento (10k registros) | ~2 segundos |

*Nota: resultados podem variar conforme distribuição real dos dados.*

---

## Dependências

| Biblioteca | Versão Mínima | Uso |
|---|---|---|
| `pandas` | 1.3+ | Manipulação de dados |
| `numpy` | 1.21+ | Operações numéricas |
| `scikit-learn` | 1.0+ | Isolation Forest, RandomForest, StandardScaler |
| `scipy` | 1.7+ | Z-score, estatística |
| `matplotlib` | 3.4+ | Visualização (opcional) |
| `seaborn` | 0.11+ | Visualização (opcional) |
| `tensorflow` | 2.8+ | LSTM Autoencoder (opcional) |

---

## Estrutura de Arquivos Recomendada

```
projeto_deteccao/
├── detector_anomalias.py      # Código principal
├── README.md                  # Este documento
├── requirements.txt           # Dependências
├── dados/
│   ├── transacoes_treino.csv
│   └── transacoes_teste.csv
├── notebooks/
│   └── analise_exploratoria.ipynb
└── relatorios/
    └── anomalias_detectadas.csv
```

---

## Licença

Uso livre para fins educacionais e profissionais de segurança da informação. O autor não se responsabiliza por decisões tomadas com base na saída do sistema sem validação humana.

---

## Próximos Passos / Melhorias Possíveis

- [ ] **Detecção em tempo real** com Apache Kafka + Spark Streaming
- [ ] **Graph Neural Networks** para detectar fraudes em rede de contas
- [ ] **XGBoost/LightGBM** como meta-classificador
- [ ] **API REST** (FastAPI/Flask) para integração com sistemas bancários
- [ ] **Dashboard** interativo com Streamlit ou Dash
- [ ] **Explicabilidade (SHAP/LIME)** para justificar cada bloqueio

---

## Contato

Para dúvidas, sugestões ou contribuições, abra uma issue no repositório do projeto.
