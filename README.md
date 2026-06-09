# Prologis Financial Assistant

End-to-end AI-powered Financial Assistant Web Application for **Prologis (NYSE: PLD)**, a global industrial real estate investment trust. The platform combines structured data querying, classic machine learning (regression and classification), and generative AI to retrieve financial and property insights, perform predictive analytics, and provide natural-language summaries.

Built as a multi-cloud system: Postgres on Supabase for structured data, AWS SageMaker for ML model hosting, **Google Cloud Vertex AI** for the conversational agent (with function calling, the primitive that underpins the Vertex AI Agent Development Kit), and AWS Bedrock (Claude Haiku) for summarization.

**Live deployment:** [https://prologis-financial-assistant-bqs9hvup7nfq8th64qn3iv.streamlit.app/](https://prologis-financial-assistant-bqs9hvup7nfq8th64qn3iv.streamlit.app/)

---

## Architecture

```
                        ┌─────────────────────────────────────────────┐
                        │        Streamlit Web App (Frontend)         │
                        │   💬 Chat   📊 Data   🤖 ML Predictions     │
                        └──────┬──────────┬──────────────┬────────────┘
                               │          │              │
                ┌──────────────┴──┐    ┌──┴──────┐   ┌───┴──────────────┐
                │  Vertex AI      │    │ Postgres│   │ AWS SageMaker    │
                │  Gemini 2.5     │    │ (props +│   │  • RF Regressor  │
                │  + function     │    │ financs)│   │  • LR Classifier │
                │  calling (ADK)  │    │         │   │                  │
                └────┬───┬───┬────┘    └─────────┘   └──────────────────┘
                     │   │   │
        ┌────────────┘   │   └──────────────┐
        ▼                ▼                  ▼
  ┌──────────┐     ┌──────────┐      ┌────────────────┐
  │ SEC EDGAR│     │  Press   │      │  AWS Bedrock   │
  │  (10-K,  │     │ Releases │      │ (Claude Haiku, │
  │  10-Q)   │     │  (JSON)  │      │ summarization) │
  └──────────┘     └──────────┘      └────────────────┘
```

**Cloud services used:**
- **Google Cloud Vertex AI** — Gemini 2.5 Flash agent using function calling (the underlying primitive of the Vertex AI Agent Development Kit)
- **AWS SageMaker** — hosted endpoints for the regression and classification models
- **AWS Bedrock** — Claude Haiku 4.5 for press-release summarization (multi-cloud integration)
- **Supabase Postgres** — properties + financials database
- **Streamlit Community Cloud** — public web app hosting

---

## Data sources

| Source | Format | Records | Purpose |
|---|---|---|---|
| **SEC EDGAR** | JSON (XBRL Company Facts API) | Latest 10-K / 10-Q metrics | Authoritative financial data: revenue, net income, operating expenses, total assets/liabilities |
| **Postgres** | `properties` + `financials` tables | 20 properties across 11 US metros | Property-level revenue/expense queries |
| **Press releases** | JSON | 10 mocked Prologis releases | Acquisitions, expansions, earnings, sustainability announcements |

The 20 sample properties span Los Angeles, Chicago, New York, Kansas City, Dallas, Miami, Seattle, Phoenix, Portland, Philadelphia, and Atlanta with mixed Industrial / Logistics / Warehouse types.

---

## Repository layout

```
prologis-financial-assistant/
├── agent/
│   ├── tools.py            # 3 tool functions exposed to the agent
│   ├── bedrock.py          # AWS Bedrock summarization helper
│   └── agent.py            # Vertex AI agent with function calling
├── app/
│   └── streamlit_app.py    # 3-tab Streamlit frontend
├── data/
│   ├── press_releases.json # 10 mocked press releases
│   └── sec/                # cached SEC EDGAR responses
├── db/
│   ├── schema.sql          # properties + financials tables
│   └── seed.sql            # 20 sample properties + financials
├── ml/
│   ├── regression/         # California Housing -> Random Forest
│   │   ├── train.py
│   │   ├── inference.py    # SageMaker inference handler
│   │   └── metrics.json
│   ├── classification/     # UCI Bank Marketing -> Logistic Regression
│   │   ├── train.py
│   │   ├── inference.py
│   │   └── metrics.json
│   └── plots/              # rendered metric/feature plots
├── scripts/
│   ├── fetch_sec.py        # pull SEC EDGAR data
│   ├── deploy_sagemaker.py # deploy both endpoints
│   ├── delete_endpoints.py # cleanup
│   └── generate_plots.py   # render evaluation plots
├── .streamlit/
│   └── config.toml         # Streamlit theme configuration
├── requirements.txt
└── .env.example
```

---

## Setup

### Prerequisites

- Python 3.9 (the SageMaker sklearn 1.2-1 container expects 3.9 — using a matching local env avoids pickle compatibility issues)
- Postgres 15+ (or a Supabase Postgres project)
- AWS account with Bedrock and SageMaker access
- Google AI Studio API key (Gemini 2.5 Flash)

### Local environment

```bash
git clone https://github.com/Viraj-Pathak/prologis-financial-assistant.git
cd prologis-financial-assistant

# Conda env with Python 3.9 to match SageMaker container
conda create -n smpy39 python=3.9 -y
conda activate smpy39

pip install -r requirements.txt
```

### Postgres

**Supabase (deployed app):**
1. Create a free Supabase project at https://supabase.com
2. Open the SQL editor and run `db/schema.sql` then `db/seed.sql`
3. Grab the connection string under Project Settings → Database → Session Pooler
4. Use the host, port, db, user, password values in `.env`

**Local (development):**
```bash
createdb postgres
psql -U postgres -d postgres -f db/schema.sql
psql -U postgres -d postgres -f db/seed.sql
```

Verify with `psql ... -c "SELECT COUNT(*) FROM properties;"` — should return `20`.

### Vertex AI / Google AI Studio

The chatbot routes through the **Google Generative AI API** using the unified `google-genai` SDK. The agent declares tool schemas and Gemini 2.5 Flash decides which tools to call — this is the same function-calling primitive the **Vertex AI Agent Development Kit (ADK)** is built on.

1. Go to https://aistudio.google.com/app/apikey
2. Create a new API key and copy it
3. Add to `.env`:
   ```
   GOOGLE_API_KEY=AIza...
   GOOGLE_GENAI_USE_VERTEXAI=False
   ```

### Environment variables

```bash
cp .env.example .env
# Then edit .env to fill in your real values
```

Full list of required variables:
```
GOOGLE_API_KEY=...                      # From https://aistudio.google.com/app/apikey
GOOGLE_GENAI_USE_VERTEXAI=False
POSTGRES_HOST=...                       # Supabase Session Pooler host
POSTGRES_USER=...
POSTGRES_PASSWORD=...
POSTGRES_DB=postgres
POSTGRES_PORT=5432
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
AWS_REGION=us-east-1
SAGEMAKER_BUCKET=...
SAGEMAKER_ROLE_ARN=...
SAGEMAKER_REGRESSION_ENDPOINT=...      # auto-set by deploy_sagemaker.py
SAGEMAKER_CLASSIFICATION_ENDPOINT=...  # auto-set by deploy_sagemaker.py
```

### SEC data

```bash
# Edit scripts/fetch_sec.py to put your email in the User-Agent header
python scripts/fetch_sec.py
```

This populates `data/sec/prologis_financials.json` with the latest annual and quarterly metrics from the SEC EDGAR XBRL API.

---

## Machine learning models

### Regression — Random Forest on California Housing

**Dataset:** California Housing (`sklearn.datasets.fetch_california_housing`) — 20,640 samples, 8 numeric features. Target: median house value (in 100k USD).

**Pipeline:** `StandardScaler` → `RandomForestRegressor(n_estimators=100, max_depth=15, min_samples_split=5)`

```bash
python ml/regression/train.py
```

**Test set metrics:**

| Metric | Value |
|---|---|
| RMSE | $51,024 |
| MAE | $33,270 |
| R² | 0.8014 |

The most important features are median income (54.2%), average occupancy (13.9%), and latitude/longitude (~17% combined) — geographically and economically intuitive.

### Classification — Logistic Regression on UCI Bank Marketing

**Dataset:** UCI Bank Marketing — 45,211 records, 16 features. Binary target: did the customer subscribe to a term deposit (yes/no). The dataset is highly imbalanced (~88% no, 12% yes) so the model is trained with `class_weight="balanced"` to prioritize recall.

**Pipeline:** `ColumnTransformer(StandardScaler on numerics, OneHotEncoder on categoricals)` → `LogisticRegression(class_weight="balanced", max_iter=1000)`

```bash
python ml/classification/train.py
```

**Test set metrics:**

| Metric | Value |
|---|---|
| Accuracy | 0.846 |
| Precision | 0.419 |
| Recall | 0.819 |
| F1 | 0.555 |

The model captures **819 of 1,058 actual subscribers** (high recall) at the cost of more false positives — the right trade-off for a marketing campaign where missing a real subscriber is more costly than a wasted call.

### Generate evaluation plots

```bash
python scripts/generate_plots.py
```

Plots saved to `ml/plots/`.

### Deployment to SageMaker

```bash
python scripts/deploy_sagemaker.py
```

This script:
1. Tarballs each `model.joblib` and uploads to S3
2. Creates a SageMaker `SKLearnModel` for each, pointing to `inference.py` as the entry point
3. Deploys to `ml.t2.medium` instances using the `1.2-1` sklearn framework container (Python 3.9, sklearn 1.2.x)
4. Auto-populates `SAGEMAKER_REGRESSION_ENDPOINT` and `SAGEMAKER_CLASSIFICATION_ENDPOINT` in `.env`

Total deploy time: ~12 minutes for both endpoints.

Both inference scripts implement the standard SageMaker contract: `model_fn`, `input_fn`, `predict_fn`, `output_fn`.

To clean up after the demo:
```bash
python scripts/delete_endpoints.py
```

---

## Conversational agent

The agent uses the **unified `google-genai` SDK** with Gemini 2.5 Flash. The agent declares four tool schemas; Gemini decides which tools to call and in what order. This is the same function-calling primitive that powers the Vertex AI Agent Development Kit.

### Tools available to the agent

| Tool | Purpose | Backed by |
|---|---|---|
| `query_postgres(metro_area, property_type, min_revenue)` | Filter properties + financials | Postgres (Supabase) |
| `query_sec_edgar(metric, period)` | Look up Prologis revenue, net income, etc. | SEC EDGAR JSON cache |
| `query_press_releases(keywords, category)` | Search press releases by topic | JSON store |
| `summarize_with_bedrock(text, max_words)` | Condense long text to N words | **AWS Bedrock (Claude Haiku 4.5)** |

### How the agent routes a query

1. User asks a natural-language question in the Chat tab
2. Gemini reads the question and the registered tool schemas
3. The model emits one or more `function_call` blocks naming the tool and arguments
4. The orchestrator (`agent/agent.py`) executes each tool and feeds results back
5. Gemini composes a final natural-language answer grounded in the returned data
6. Up to 6 turns of tool-calling are supported (handles multi-source questions)

### Example traces

| Query | Tools called |
|---|---|
| *"What was Prologis' net income last year?"* | `query_sec_edgar(metric="net_income", period="annual")` |
| *"Show industrial properties in Chicago."* | `query_postgres(metro_area="Chicago", property_type="Industrial")` |
| *"Any recent acquisitions?"* | `query_press_releases(category="acquisition")` |
| *"Summarize the latest earnings release."* | `query_press_releases(category="earnings")` → `summarize_with_bedrock(...)` |
| *"What are total assets and liabilities?"* | `query_sec_edgar()` (all metrics) |

The Chat tab shows a "Tools used" expander under each response, displaying which tools fired and what they returned.

The press release summary flow demonstrates the **multi-cloud agentic chain**: a Vertex AI request triggers an AWS Bedrock summarization call within a single user query.

---

## Running the app

```bash
streamlit run app/streamlit_app.py
```

The app has three tabs:

- **💬 Chat** — natural-language Q&A backed by the Gemini 2.5 Flash agent
- **📊 Data** — Properties (Postgres dataframe with metro/type filters + KPI cards), SEC Filings (annual/quarterly toggle + revenue chart), Press Releases (expandable with category filter)
- **🤖 ML Predictions** — Sliders and dropdowns that POST to the live SageMaker endpoints and display predictions in real time

---

## Multi-cloud summary

| Cloud | Service | Component |
|---|---|---|
| **GCP** | Vertex AI (Gemini 2.5 Flash) | Conversational agent / function-calling orchestrator |
| **AWS** | SageMaker | Two hosted ML endpoints (regression + classification) |
| **AWS** | Bedrock (Claude Haiku 4.5) | Press-release summarization |
| **AWS** | S3 | Model artifact storage |
| **AWS** | IAM | SageMaker execution role |
| **Supabase** | Managed Postgres | Properties + financials database |
| **Streamlit Cloud** | Community Cloud | Public web hosting |

The cross-cloud design is functional, not just decorative: queries needing a press release summary call **Bedrock** (AWS), ML predictions go to **SageMaker** (AWS), and agent reasoning happens in **Vertex AI** (GCP).

---

## Known limitations

- The Postgres seed data is synthetic (revenue/net-income figures scaled from square footage with noise) — the SEC EDGAR data is real
- Press releases are mocked; a production version would use the Prologis Investor Relations RSS feed
- The SageMaker container is constrained to sklearn 1.2.x and Python 3.9 (the latest version AWS publishes); the local training env mirrors these to keep pickle formats compatible
- Gemini 2.5 Flash has a free-tier limit of 20 requests/day per project; a billing-enabled account removes this limit
- Bedrock model availability is region-specific; the project uses the `us-east-1` cross-region inference profile for Claude Haiku 4.5

---

## Contributors

- **Viraj Pathak**
