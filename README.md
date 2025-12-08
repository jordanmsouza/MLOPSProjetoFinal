# 🧠 Projeto Final – MLE / MLOps Nível 0  
# **Análise de Sentimentos em Reviews da Amazon**

Este projeto implementa um pipeline completo de **Machine Learning + MLOps nível 0**, passando por:

- ingestão e preparação dos dados  
- análise exploratória  
- treinamento e versionamento do modelo (MLflow)  
- serviço via API FastAPI  
- logging e monitoramento básico  
- execução local e via Docker  

O objetivo foi responder ao desafio proposto, criando um fluxo **reproduzível, escalável e alinhado a boas práticas de MLOps**.

---

## ⚙️ Stack utilizada

### Linguagem e bibliotecas
- Python 3.9+
- pandas  
- scikit-learn  
- joblib  

### MLOps
- MLflow (tracking + model registry)

### Serviço
- FastAPI  
- Uvicorn  

### Infraestrutura
- Docker  
- docker-compose  

---

## 📂 Estrutura do Projeto

```bash
MLOPSProjetoFinal/
├── data/
│   ├── raw/
│   │   ├── amazon_reviews_train_sample.csv
│   │   └── amazon_reviews_test_sample.csv
│   └── processed/
│       ├── train.csv
│       └── test.csv
├── logs/
│   ├── predictions_log.csv
│   └── feedback_log.csv
├── mlruns/                     # MLflow local (tracking + registry)
│   ├── <experiments>...
│   └── models/
│       └── sentiment-logreg-tfidf/
│           ├── version-1/
│           └── meta.yaml
├── mlruns_backup/              # Backup do registry antigo
├── models/
│   └── sentiment_model.joblib  # fallback local
├── notebook/
│   └── EDA.ipynb
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── ingest_data.py
│   ├── data_prep.py
│   ├── train.py
│   └── serve.py
├── Dockerfile
├── docker-compose.yml
├── README.md
└── requirements.txt
```

---

## 📊 Fonte de Dados

Dataset público do Kaggle:

- **Amazon Reviews**  
- ID: `kritanjalijain/amazon-reviews`

Colunas originais:

| Coluna | Descrição |
|--------|-----------|
| `label` | 1 = negativo, 2 = positivo |
| `title` | título da review |
| `text` | texto completo |

Para modelagem, usamos apenas `text` e o target `sentiment` convertido para 0/1.

---

# 🔁 Pipeline Completo

---

## **1. Ingestão do Dataset — `src/ingest_data.py`**

O dataset completo é grande (~1.5 GB), então utilizamos:

- download automático via `kagglehub`
- leitura em chunks para evitar estouro de memória
- amostragem controlada para desenvolvimento mais rápido

Saídas:

```
data/raw/amazon_reviews_train_sample.csv
data/raw/amazon_reviews_test_sample.csv
```

**Executar:**

```bash
python -m src.ingest_data
```

---

## **2. Preparação dos Dados — `src/data_prep.py`**

Processos aplicados:

- uso das colunas `label` e `text`
- mapeamento de `label → sentiment`  
  - 1 → 0 (negativo)  
  - 2 → 1 (positivo)
- remoção da coluna `title` (redundante)
- limpeza de linhas inconsistentes

Saídas:

```
data/processed/train.csv
data/processed/test.csv
```

**Executar:**

```bash
python -m src.data_prep
```

---

## **3. EDA — `notebook/EDA.ipynb`**

Análises realizadas:

- distribuição das classes  
- comprimento dos textos  
- amostras de textos positivos e negativos  
- contagem de tokens por classe  
- estimativa de memória para o dataset completo  
- justificativa da redução do dataset  

Conclusões:

- O dataset reduzido mantém representatividade  
- TF-IDF é apropriado  
- Logistic Regression funciona muito bem como baseline  

---

## **4. Treinamento + Registro do Modelo — `src/train.py`**

Modelo utilizado:

### **TF-IDF**
- `max_features=40000`
- `ngram_range=(1, 2)`
- `stop_words="english"`

### **Logistic Regression**
- `max_iter=1000`
- `n_jobs=-1`

Métricas:

- Accuracy  
- Precision  
- Recall  
- F1-score  

### Registro no MLflow

```python
mlflow.register_model(
    model_uri=f"runs:/{run_id}/model",
    name="sentiment-logreg-tfidf"
)
```

Saída local (fallback):

```
models/sentiment_model.joblib
```

**Executar:**

```bash
python -m src.train
```

---

## **5. Servindo o Modelo — `src/serve.py`**

A API tenta carregar:

1. Modelo do **MLflow Registry** (alias `latest`)  
2. Se falhar → fallback para `sentiment_model.joblib`

### **Endpoints**

#### `GET /health`
Checa se o serviço está no ar.

#### `POST /predict`
Entrada:
```json
{"text": "This product is amazing!"}
```

Resposta:
```json
{
  "sentiment": 1,
  "label": "positivo",
  "confidence": 0.94
}
```

#### `POST /feedback`
Armazena feedback do usuário:

```
logs/feedback_log.csv
logs/predictions_log.csv
```

**Executar API localmente:**

```bash
uvicorn src.serve:app --reload
```

Swagger:
```
http://localhost:8000/docs
```

---

# 🐳 Execução com Docker

### Subir API + MLflow UI

```bash
docker compose up --build
```

### Acessos

- API → http://localhost:8000  
- Swagger → http://localhost:8000/docs  
- MLflow UI → http://localhost:5000  

---

# 📈 Monitoramento

### Prediction Log → drift básico

Arquivo:
```
logs/predictions_log.csv
```

Campos:
- timestamp  
- text_length  
- model_sentiment  
- confidence  

### Feedback Loop → qualidade real em produção

Arquivo:
```
logs/feedback_log.csv
```

Campos:
- user_sentiment  
- model_sentiment  
- is_correct  

Permite medir:

- acurácia de produção  
- divergência entre offline x online  

---

# 🧠 Decisões de Modelagem

- chunking para otimizar ingestão  
- remoção de `title` por redundância  
- TF-IDF + Logistic Regression = baseline robusto  
- MLflow como registry + tracking  
- logs estruturados para monitoramento  
- API FastAPI para servir o modelo  

---

# 🚀 Melhorias Futuras

- Testes unitários e integração  
- EvidentlyAI para monitoramento avançado de drift  
- CI/CD com pipelines automáticos  
- Retreino automático baseado em feedback do usuário  
- Orquestração com Airflow ou Prefect  

---
