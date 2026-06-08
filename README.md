# Prologis Financial Assistant

End-to-end AI-powered Financial Assistant Web Application for **Prologis (NYSE: PLD)**, a global industrial real estate investment trust. The platform combines structured data querying, classic machine learning (regression and classification), and generative AI to retrieve financial and property insights, perform predictive analytics, and provide natural-language summaries.

Built as a multi-cloud system: Postgres (Supabase) for structured data, **AWS SageMaker** for ML model hosting, **Google Cloud Vertex AI** (Gemini 2.5 Flash with function calling) for the conversational agent, and **AWS Bedrock** (Claude Haiku 4.5) for summarization.

---

## Architecture

```
                  ┌─────────────────────────────────────────────┐
                  │        Streamlit Web App (Frontend)         │
                  │  💬 Chat   📊 Data Browser   🤖 ML Tab      │
                  └──────┬──────────┬──────────────┬────────────┘
                         │          │              │
          ┌──────────────┴──┐    ┌──┴──────┐   ┌───┴──────────────┐
          │  Vertex AI      │    │Postgres │   │ AWS SageMaker    │
          │  Gemini 2.5     │    │(props + │   │  • RF Regressor  │
          │  + function     │    │ financs)│   │  • LR Classifier │
          │  calling (ADK)  │    │         │   │                  │
          └────┬───┬───┬────┘    └─────────┘   └──────────────────┘
               │   │   │
  ┌────────────┘   │   └──────────────┐
  ▼                ▼                  ▼
┌──────────┐  ┌──────────┐   ┌────────────────┐
│ SEC EDGAR│  │  Press   │   │  AWS Bedrock   │
│ (10-K,   │  │ Releases │   │ (Claude Haiku, │
│  10-Q)   │  │  (JSON)  │   │ summarization) │
└──────────┘  └──────────┘   └────────────────┘
```

**Cloud services:**
- **GCP Vertex AI** — Gemini 2.5 Flash agent using function calling (Vertex AI ADK primitive)
- **AWS SageMaker** — hosted endpoints for regression and classification models
- **AWS Bedrock** — Claude Haiku 4.5 for press-release summarization (multi-cloud integration)
- **Supabase Postgres** — properties + financials structured database
- **Streamlit Community Cloud** — public web hosting

---

## Data Sources

| Source | Format | Records | Purpose |
|---|---|---|---|
| **SEC EDGAR** | JSON (XBRL Company Facts API) | Latest 10-K / 10-Q | Revenue, net income, operating expenses, total assets/liabilities |
| **Postgres** | `properties` + `financials` tables | 20 properties, 11 US metros | Property-level financial queries |
| **Press Releases** | JSON | 10 mocked releases | Acquisitions, expansions, earnings, sustainability |

---

## Repository Layout

```
prologis-financial-assistant/
├── agent/
│   ├── tools.py            # 3 tool functions: postgres, sec, press releases
│   ├── bedrock.py          # AWS Bedrock summarization (multi-cloud)
│   └── agent.py            # Vertex AI Gemini agent with function calling
├── app/
│   └── streamlit_app.py    # 3-tab Streamlit frontend
├── data/
│   ├── press_releases.json # 10 mocked Prologis press releases
│   └── sec/
│       └── prologis_financials.json  # cached SEC EDGAR data
├── db/
│   ├── schema.sql          # properties + financials tables
│   └── seed.sql            # 20 sample properties + financials
├── ml/
│   ├── regression/
│   │   ├── train.py        # California Housing → Random Forest
│   │   ├── inference.py    # SageMaker inference handler
│   │   └── metrics.json    # generated after training
│   ├── classification/
│   │   ├── train.py        # UCI Bank Marketing → Logistic Regression
│   │   ├── inference.py    # SageMaker inference handler
│   │   └── metrics.json    # generated after training
│   └── plots/              # generated evaluation plots
├── notebooks/
│   ├── regression_training.ipynb
│   └── classification_training.ipynb
├── scripts/
│   ├── fetch_sec.py        # pull SEC EDGAR data
│   ├── deploy_sagemaker.py # deploy both endpoints to SageMaker
│   ├── delete_endpoints.py # cleanup endpoints after demo
│   └── generate_plots.py   # render evaluation plots
├── requirements.txt
└── .env.example
```

---

## Setup

### Prerequisites

- Python 3.9 (required — SageMaker sklearn 1.2-1 container uses Python 3.9)
- Postgres 15+ (or a Supabase project)
- AWS account with Bedrock and SageMaker access
- Google Cloud project with Vertex AI enabled (Express Mode)

### 1. Clone and Install

```bash
git clone <your-repo-url>
cd prologis-financial-assistant

# Use Python 3.9 to match SageMaker container
conda create -n smpy39 python=3.9 -y
conda activate smpy39

pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env with your real credentials
```

Required environment variables:
```
GOOGLE_API_KEY=...          # From https://aistudio.google.com/app/apikey
GOOGLE_GENAI_USE_VERTEXAI=True
POSTGRES_HOST=...           # Supabase or local Postgres
POSTGRES_USER=...
POSTGRES_PASSWORD=...
POSTGRES_DB=financial_assistant
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
AWS_REGION=us-east-1
SAGEMAKER_BUCKET=...
SAGEMAKER_ROLE_ARN=...
```

### 3. Set Up Postgres

**Local:**
```bash
createdb financial_assistant
psql -U postgres -d financial_assistant -f db/schema.sql
psql -U postgres -d financial_assistant -f db/seed.sql
```

**Supabase:**
1. Create a free project at https://supabase.com
2. Open SQL Editor → run `db/schema.sql` then `db/seed.sql`
3. Use the Session Pooler connection string in `.env`

Verify: `psql ... -c "SELECT COUNT(*) FROM properties;"` → should return `20`

### 4. Fetch SEC Data

```bash
# Edit scripts/fetch_sec.py — add your email to USER_AGENT
python scripts/fetch_sec.py
```

This populates `data/sec/prologis_financials.json`.

---

## Machine Learning Models

### Regression — Random Forest on California Housing

**Dataset:** California Housing (sklearn) — 20,640 samples, 8 features. Target: median house value (100k USD).

**Pipeline:** `StandardScaler → RandomForestRegressor(n_estimators=100, max_depth=15)`

```bash
python ml/regression/train.py
```

**Test Set Metrics:**

| Metric | Value |
|---|---|
| RMSE | $51,024 |
| MAE | $33,270 |
| R² | 0.8014 |

Top features: median income (54%), average occupancy (14%), lat/long (~17%).

### Classification — Logistic Regression on UCI Bank Marketing

**Dataset:** UCI Bank Marketing — 45,211 records, 16 features. Binary target: will customer subscribe?

**Pipeline:** `ColumnTransformer(StandardScaler + OneHotEncoder) → LogisticRegression(class_weight="balanced")`

```bash
python ml/classification/train.py
```

**Test Set Metrics:**

| Metric | Value |
|---|---|
| Accuracy | 0.846 |
| Precision | 0.419 |
| Recall | 0.819 |
| F1 | 0.555 |

The model prioritizes recall (high-value for a marketing use case — capturing real subscribers matters more than false positives).

### Generate Evaluation Plots

```bash
python scripts/generate_plots.py
```

Plots saved to `ml/plots/`.

### Deploy to SageMaker

```bash
python scripts/deploy_sagemaker.py
```

- Packages each `model.joblib` → S3
- Creates `SKLearnModel` with `inference.py` as the entry point
- Deploys to `ml.t2.medium` instances using the `1.2-1` sklearn framework container

**After your demo, delete endpoints to stop charges:**
```bash
python scripts/delete_endpoints.py
```

---

## Conversational Agent

The agent uses the **unified `google-genai` SDK** routed through **Google Cloud Vertex AI** Express Mode. Gemini 2.5 Flash decides which tools to call — this is the function-calling primitive the Vertex AI ADK is built on.

### Tools

| Tool | Purpose | Backend |
|---|---|---|
| `query_postgres` | Filter properties + financials | Postgres (Supabase) |
| `query_sec_edgar` | Look up revenue, net income, etc. | SEC EDGAR JSON cache |
| `query_press_releases` | Search press releases by keyword/category | JSON store |
| `summarize_with_bedrock` | Condense text to N words | **AWS Bedrock (Claude Haiku 4.5)** |

### Query Routing

| Query | Tools Called |
|---|---|
| *"What was net income last year?"* | `query_sec_edgar(metric="net_income", period="annual")` |
| *"Show industrial properties in Chicago"* | `query_postgres(metro_area="Chicago", property_type="Industrial")` |
| *"Any recent acquisitions?"* | `query_press_releases(category="acquisition")` |
| *"Summarize the latest earnings"* | `query_press_releases` → `summarize_with_bedrock` |
| *"Compare Dallas vs Phoenix revenues"* | `query_postgres(metro="Dallas")` → `query_postgres(metro="Phoenix")` |

The last example chains Vertex AI (GCP) → AWS Bedrock — demonstrating the multi-cloud design.

---

## Running the App

```bash
streamlit run app/streamlit_app.py
```

**Tabs:**
- **💬 Chat** — Natural-language Q&A backed by Vertex AI agent
- **📊 Data** — Properties (Postgres), SEC Filings (EDGAR JSON), Press Releases (JSON)
- **🤖 ML Predictions** — Sliders that POST to live SageMaker endpoints

---

## Multi-Cloud Summary

| Cloud | Service | Component |
|---|---|---|
| **GCP** | Vertex AI (Gemini 2.5 Flash) | Conversational agent / function-calling orchestrator |
| **AWS** | SageMaker | Two hosted ML endpoints |
| **AWS** | Bedrock (Claude Haiku 4.5) | Press-release summarization |
| **AWS** | S3 | Model artifact storage |
| **AWS** | IAM | SageMaker execution role |
| **Supabase** | Managed Postgres | Properties + financials |

The multi-cloud design is functional: press release summaries go through **Bedrock** (AWS), ML predictions through **SageMaker** (AWS), agent reasoning through **Vertex AI** (GCP).
