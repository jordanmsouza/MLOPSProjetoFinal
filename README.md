# Projeto Final – MLE / MLOps Nível 0  
Análise de Sentimentos em Reviews da Amazon

Este projeto implementa um pipeline de **Machine Learning + MLOps nível 0** para análise de sentimentos em reviews da Amazon, desde a **ingestão dos dados** até o **serviço do modelo via API** e **monitoramento básico** em produção.

O objetivo é responder às perguntas do case proposto, mostrando um fluxo completo e reproduzível.

---

## 🧱 Stack utilizada

- Python 3.9+
- pandas
- scikit-learn
- joblib
- FastAPI
- Uvicorn

---

## 📂 Estrutura do projeto

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
├── models/
│   └── sentiment_model.joblib
├── notebook/
│   └── EDA.ipynb
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── ingest_data.py
│   ├── data_prep.py
│   ├── train.py
│   └── serve.py
├── README.md
└── requirements.txt
```

---

## 📊 Fonte de dados

- Dataset: **Amazon Reviews**
- Origem: Kaggle  
- ID usado no código: `kritanjalijain/amazon-reviews`

O dataset já vem com as colunas:

- `label` – rótulo numérico (1 ou 2, já binário)
- `title` – título curto da review
- `text` – texto completo da review

Neste projeto, usei o `label` para construir a variável alvo de sentimento.

---

## 🔁 Pipeline de ponta a ponta

### 1. Ingestão e redução do dataset – `src/ingest_data.py`

Objetivo: baixar o dataset do Kaggle e criar uma **amostra reduzida** para facilitar o desenvolvimento local.

Principais pontos:

- Download automático via `kagglehub`.
- Leitura em **chunks** (`chunksize`) para não estourar memória.
- Amostragem do conjunto de treino e teste com limite de linhas (`target_size`).
- Padronização das colunas para: `["label", "title", "text"]`.

Saídas:

- `data/raw/amazon_reviews_train_sample.csv`
- `data/raw/amazon_reviews_test_sample.csv`

**Comando:**

```bash
python -m src.ingest_data
```

---

### 2. Preparação dos dados – `src/data_prep.py`

Objetivo: transformar os dados crus em um formato pronto para modelagem.

Passos principais:

1. Leitura dos arquivos reduzidos (`raw`).
2. Seleção das colunas relevantes: `label` e `text`.
3. Conversão de `label` em `sentiment` binário:
   - `label == 2` → `sentiment = 1` (positivo)
   - `label == 1` → `sentiment = 0` (negativo)
4. Remoção de linhas inválidas / nulas.
5. **Remoção da coluna `title`** por ser redundante:
   - o título é muito curto e costuma repetir o sentimento já expresso em `text`;
   - manter apenas `text` simplifica o modelo, reduz sparsidade e evita features redundantes.

Saídas:

- `data/processed/train.csv` – colunas: `text`, `sentiment`
- `data/processed/test.csv` – colunas: `text`, `sentiment`

**Comando:**

```bash
python -m src.data_prep
```

---

### 3. Análise Exploratória – `notebook/EDA.ipynb`

No notebook foram feitas análises como:

- Visualização das primeiras linhas do dataset.
- Distribuição da variável `label` / `sentiment`.
- Exemplos de reviews positivas e negativas.
- Verificação de balanceamento de classes.

Principais conclusões:

- O dataset é binário (labels 1 e 2).
- Há predominância de reviews positivas.
- Os textos são longos, favorecendo TF-IDF em n-grams.

---

### 4. Treinamento do modelo – `src/train.py`

Modelo utilizado:

- Pipeline:
  - `TfidfVectorizer`
    - `max_features=40000`
    - `ngram_range=(1, 2)`
    - `stop_words="english"`
  - `LogisticRegression`
    - `max_iter=1000`
    - `n_jobs=-1`

Motivação da escolha:

- **TF-IDF**: representação clássica e eficiente para texto.
- **Logistic Regression**: simples, robusta e ideal como baseline.

Métricas calculadas:

- Accuracy  
- F1-score  
- Precision / Recall  

Saída:

- `models/sentiment_model.joblib`

**Comando:**

```bash
python -m src.train
```

---

### 5. Serviço do modelo – API FastAPI (`src/serve.py`)

Endpoints:

#### `GET /health`
Verifica se o serviço está ativo.

#### `POST /predict`

Entrada:
```json
{
  "text": "This product is amazing!"
}
```

Saída:
```json
{
  "sentiment": 1,
  "label": "positivo",
  "confidence": 0.94
}
```

#### `POST /feedback`

Entrada:
```json
{
  "text": "This product is amazing!",
  "user_sentiment": 1
}
```

Saída:
```json
{
  "model_sentiment": 1,
  "model_label": "positivo",
  "model_confidence": 0.94,
  "user_sentiment": 1,
  "is_correct": true,
  "message": "Feedback registrado com sucesso."
}
```

**Comando para subir a API:**
```bash
uvicorn src.serve:app --reload
```

Docs automáticas:
http://127.0.0.1:8000/docs

---

## 📈 Monitoramento do modelo

O monitoramento está dividido em três camadas:

### 1. Saúde do serviço (API)
- Endpoint `/health`
- Logs do servidor com status codes e tempos de resposta

### 2. Monitoramento das previsões (prediction drift)
Cada chamada ao `/predict` gera um registro em:

```
logs/predictions_log.csv
```

Campos:

- timestamp  
- text_length  
- sentiment  
- confidence  

Isso permite acompanhar:
- distribuição das previsões ao longo do tempo  
- mudanças no padrão dos textos (ex.: textos muito curtos)  
- possíveis sinais de drift

### 3. Qualidade do modelo em produção (feedback)

Cada chamada a `/feedback` gera:

```
logs/feedback_log.csv
```

Campos:

- timestamp  
- text_length  
- model_sentiment  
- model_confidence  
- user_sentiment  
- is_correct  

Permite calcular uma **acurácia em produção** usando:

```
mean(is_correct)
```

E comparar com resultados offline.

---

## 🧪 Como reproduzir o pipeline completo

```bash
# 1. Ingestão + amostragem
python -m src.ingest_data

# 2. Preparação dos dados
python -m src.data_prep

# 3. Treinamento do modelo
python -m src.train

# 4. Subir API
uvicorn src.serve:app --reload
```

---

## 🧠 Decisões de modelagem (resumo)

- Uso de amostragem por chunks para processar datasets grandes.
- labels 1 e 2 transformados em sentimento (0/1).
- Remoção da coluna `title` por redundância.
- TF-IDF + Logistic Regression como baseline simples e eficaz.
- Serviço via FastAPI.
- Log de previsões + log de feedback para monitoramento contínuo.

---

## 🚀 Melhorias futuras

- Testes unitários e de integração
- Containerização com Docker
- Pipeline CI/CD
- Re-treino automático com base em feedback
- Monitoramento avançado com EvidentlyAI

---
