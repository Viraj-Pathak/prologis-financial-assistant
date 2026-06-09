import json
import os
import sys
from pathlib import Path

import boto3
import pandas as pd
import streamlit as st
from dotenv import load_dotenv, dotenv_values as _dotenv_values
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL as _SAURL

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

load_dotenv(ROOT / ".env", override=True)
_ENV = _dotenv_values(ROOT / ".env")

st.set_page_config(
    page_title="Prologis Financial Assistant",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    font-size: 16px;
}

p, div, span, label { font-size: 1rem; }

h2, .stSubheader { font-size: 1.4rem !important; }

.stButton > button,
.stButton > button p,
.stButton > button div,
button[kind="secondary"],
button[kind="primary"] {
    font-size: 0.95rem !important;
    font-weight: 700 !important;
}

.stSelectbox label, .stSlider label, .stRadio label,
.stNumberInput label { font-size: 1rem !important; }

.stDataFrame, .stTable { font-size: 0.95rem; }

.app-header {
    background: white;
    border: 1px solid #e2e8f0;
    border-radius: 14px;
    padding: 20px 28px;
    margin-bottom: 16px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
}
.app-header h1 {
    font-size: 1.85rem;
    font-weight: 700;
    color: #1d4ed8;
    margin: 0 0 6px;
}
.app-header .subtitle {
    color: #64748b;
    font-size: 1rem;
    margin-bottom: 12px;
    line-height: 1.5;
}
.badge {
    display: inline-block;
    font-size: 0.78rem;
    font-weight: 600;
    padding: 4px 12px;
    border-radius: 20px;
    margin-right: 8px;
    margin-bottom: 4px;
}
.badge-gcp { background: #dbeafe; color: #1e40af; }
.badge-aws { background: #fef3c7; color: #92400e; }
.badge-pg  { background: #d1fae5; color: #065f46; }

.section-info {
    background: #f8fafc;
    border-left: 4px solid #1d4ed8;
    border-radius: 0 8px 8px 0;
    padding: 8px 14px;
    margin-bottom: 12px;
    font-size: 0.95rem;
    color: #475569;
}

.pred-result {
    background: white;
    border-left: 5px solid #1d4ed8;
    border-radius: 0 12px 12px 0;
    padding: 20px 28px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.07);
    margin: 16px 0;
}
.pred-result.yes { border-left-color: #059669; }
.pred-result.no  { border-left-color: #dc2626; }
.pred-label {
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    color: #94a3b8;
    margin-bottom: 6px;
}
.pred-value {
    font-size: 2.4rem;
    font-weight: 700;
    color: #1d4ed8;
    font-family: monospace;
    line-height: 1.1;
}
.pred-value.yes { color: #059669; }
.pred-value.no  { color: #dc2626; }
.pred-sub { font-size: 0.8rem; color: #94a3b8; margin-top: 8px; }

.cat-badge {
    display: inline-block;
    font-size: 0.72rem;
    font-weight: 700;
    padding: 3px 10px;
    border-radius: 4px;
    color: white;
    margin-bottom: 10px;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="app-header">
  <h1>🏭 Prologis Financial Assistant</h1>
  <div class="subtitle">
    AI-powered financial intelligence platform — conversational agent with function calling,
    real-time property and SEC data, and predictive machine learning.
  </div>
  <span class="badge badge-gcp">GCP · Vertex AI · Gemini 2.5 Flash</span>
  <span class="badge badge-aws">AWS · SageMaker · Bedrock (Claude Haiku)</span>
  <span class="badge badge-pg">Supabase · Postgres</span>
</div>
""", unsafe_allow_html=True)


def get_db_engine():
    url = _SAURL.create(
        "postgresql+psycopg2",
        username=_ENV.get("POSTGRES_USER") or os.getenv("POSTGRES_USER", "postgres"),
        password=_ENV.get("POSTGRES_PASSWORD") or os.getenv("POSTGRES_PASSWORD", "postgres"),
        host=_ENV.get("POSTGRES_HOST") or os.getenv("POSTGRES_HOST", "localhost"),
        port=int(_ENV.get("POSTGRES_PORT") or os.getenv("POSTGRES_PORT", "5432")),
        database=_ENV.get("POSTGRES_DB") or os.getenv("POSTGRES_DB", "postgres"),
    )
    return create_engine(url)


@st.cache_data(ttl=300)
def load_properties(metro_filter=None, type_filter=None):
    sql = """
        SELECT p.property_id, p.address, p.metro_area, p.sq_footage,
               p.property_type, f.revenue, f.net_income, f.expenses
        FROM properties p
        LEFT JOIN financials f ON p.property_id = f.property_id
        WHERE 1=1
    """
    params = {}
    if metro_filter:
        sql += " AND p.metro_area = :metro"
        params["metro"] = metro_filter
    if type_filter:
        sql += " AND p.property_type = :ptype"
        params["ptype"] = type_filter
    sql += " ORDER BY f.revenue DESC NULLS LAST"
    try:
        with get_db_engine().connect() as conn:
            df = pd.read_sql(text(sql), conn, params=params)
        for col in ["revenue", "net_income", "expenses"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        return df
    except Exception as e:
        st.error(f"Database error: {e}")
        return pd.DataFrame()


@st.cache_data
def load_sec_data():
    path = ROOT / "data" / "sec" / "prologis_financials.json"
    return json.loads(path.read_text()) if path.exists() else {}


@st.cache_data
def load_press_releases():
    path = ROOT / "data" / "press_releases.json"
    return json.loads(path.read_text()) if path.exists() else []


def call_sagemaker(endpoint_name: str, payload: dict):
    try:
        runtime = boto3.client(
            "sagemaker-runtime",
            region_name=_ENV.get("AWS_REGION") or os.getenv("AWS_REGION", "us-east-1"),
            aws_access_key_id=_ENV.get("AWS_ACCESS_KEY_ID") or os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=_ENV.get("AWS_SECRET_ACCESS_KEY") or os.getenv("AWS_SECRET_ACCESS_KEY"),
        )
        resp = runtime.invoke_endpoint(
            EndpointName=endpoint_name,
            ContentType="application/json",
            Accept="application/json",
            Body=json.dumps([payload]),
        )
        return json.loads(resp["Body"].read())[0]
    except Exception as e:
        st.error(f"SageMaker error: {e}")
        return None


tab_chat, tab_data, tab_ml = st.tabs(["💬  Chat", "📊  Data", "🤖  ML Predictions"])


# ──────────────────────────────────────────────────────────────────
# CHAT TAB
# ──────────────────────────────────────────────────────────────────
with tab_chat:
    st.subheader("Ask anything about Prologis")
    st.caption("🤖 Gemini 2.5 Flash via Google Vertex AI — automatically routes each question to Postgres, SEC EDGAR, or press releases.")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    st.markdown("**Try one of these questions:**")
    suggestions = [
        "What was Prologis' net income last year?",
        "Show industrial properties in Chicago",
        "Any recent acquisitions?",
        "Show Dallas properties with revenue over $8M",
        "Summarize the latest earnings release",
        "What are total assets and liabilities?",
    ]
    row1 = st.columns(3)
    row2 = st.columns(3)
    for i, sug in enumerate(suggestions):
        cols = row1 if i < 3 else row2
        if cols[i % 3].button(sug, key=f"sug_{i}", use_container_width=True):
            st.session_state.pending_query = sug

    user_input = st.chat_input("Type your question about Prologis financials, properties, or news…")
    if "pending_query" in st.session_state:
        user_input = st.session_state.pop("pending_query")

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.spinner("Agent thinking…"):
            try:
                for mod in [k for k in sys.modules if k.startswith("agent")]:
                    del sys.modules[mod]
                from agent.agent import ask
                result = ask(user_input)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result.get("answer", "No answer returned."),
                    "tool_calls": result.get("tool_calls", []),
                })
            except Exception as e:
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"⚠️ Error: {e}",
                    "tool_calls": [],
                })

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("tool_calls"):
                with st.expander(f"🔧 Tools used — {len(msg['tool_calls'])} call(s)"):
                    for tc in msg["tool_calls"]:
                        st.markdown(f"**`{tc['tool']}`**")
                        st.json({"args": tc["args"], "result": tc["result"]})

    if st.session_state.messages:
        if st.button("🗑 Clear conversation", key="clear_chat"):
            st.session_state.messages = []
            st.rerun()


# ──────────────────────────────────────────────────────────────────
# DATA BROWSER TAB
# ──────────────────────────────────────────────────────────────────
with tab_data:
    sub_props, sub_sec, sub_press = st.tabs(["🏢  Properties", "📈  SEC Filings", "📰  Press Releases"])

    with sub_props:
        st.subheader("Property Portfolio")
        st.markdown(
            '<div class="section-info">'
            '🗄️ Live data from <strong>Supabase Postgres</strong> — 20 Prologis properties '
            'across 11 US metro areas with revenue, net income, and expense data.'
            '</div>',
            unsafe_allow_html=True,
        )

        c1, c2 = st.columns(2)
        metro_filter = c1.selectbox("Filter by Metro Area", [
            "All", "Los Angeles", "Chicago", "New York", "Kansas City",
            "Dallas", "Miami", "Seattle", "Phoenix", "Portland", "Philadelphia", "Atlanta",
        ])
        type_filter = c2.selectbox("Filter by Property Type", ["All", "Industrial", "Logistics", "Warehouse"])

        df = load_properties(
            metro_filter if metro_filter != "All" else None,
            type_filter  if type_filter  != "All" else None,
        )

        if not df.empty:
            st.markdown("")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Properties", len(df))
            m2.metric("Total Revenue",    f"${df['revenue'].sum()/1e6:.1f}M")
            m3.metric("Total Net Income", f"${df['net_income'].sum()/1e6:.1f}M")
            m4.metric("Avg Sq Footage",   f"{df['sq_footage'].mean():,.0f} ft²")
            st.markdown("")

            display = df.copy()
            display["revenue"]    = display["revenue"].apply(lambda x: f"${x:,.0f}")
            display["net_income"] = display["net_income"].apply(lambda x: f"${x:,.0f}")
            display["expenses"]   = display["expenses"].apply(lambda x: f"${x:,.0f}")
            display["sq_footage"] = display["sq_footage"].apply(lambda x: f"{x:,} ft²")
            st.dataframe(
                display.rename(columns={
                    "property_id": "ID", "address": "Address", "metro_area": "Metro",
                    "sq_footage": "Sq Footage", "property_type": "Type",
                    "revenue": "Revenue", "net_income": "Net Income", "expenses": "Expenses",
                }),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("No properties match the selected filters, or the database is not connected.")

    with sub_sec:
        st.subheader("SEC EDGAR Financial Data — Prologis (PLD)")
        st.markdown(
            '<div class="section-info">'
            '📋 Real Prologis filings fetched from the <strong>SEC EDGAR XBRL API</strong>. '
            'Covers revenue, net income, operating expenses, total assets, and total liabilities.'
            '</div>',
            unsafe_allow_html=True,
        )

        sec_data = load_sec_data()
        if sec_data:
            period = st.radio("Filing period", ["Annual (10-K)", "Quarterly (10-Q)"], horizontal=True)
            is_annual = period.startswith("Annual")
            metrics = sec_data.get("metrics", {})

            rows = []
            for key, data in metrics.items():
                entry = data.get("latest_annual" if is_annual else "latest_quarterly")
                if entry:
                    rows.append({
                        "Metric":      key.replace("_", " ").title(),
                        "Value (USD)": f"${entry.get('val', 0):,.0f}",
                        "Period End":  entry.get("end", "—"),
                        "Form":        entry.get("form", "—"),
                        "Fiscal Year": entry.get("fy", "—"),
                        "Fiscal Period": entry.get("fp", "—"),
                    })
            if rows:
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

            st.markdown("#### Revenue History")
            rev_history = metrics.get("revenue", {}).get(
                "annual_history" if is_annual else "quarterly_history", []
            )
            if rev_history:
                hist_df = pd.DataFrame(rev_history)
                hist_df["val_bn"] = hist_df["val"] / 1e9
                st.bar_chart(hist_df.sort_values("end").set_index("end")["val_bn"], height=260)
                st.caption("Revenue in $B — source: SEC EDGAR XBRL API")
        else:
            st.info("Run `python scripts/fetch_sec.py` to populate SEC data.")

    with sub_press:
        st.subheader("Press Releases")
        st.markdown(
            '<div class="section-info">'
            '📰 10 Prologis press releases covering earnings, acquisitions, expansions, and sustainability. '
            'Searchable by category — these same releases are queried by the Chat agent.'
            '</div>',
            unsafe_allow_html=True,
        )

        releases = load_press_releases()
        if releases:
            CAT_COLOR = {
                "earnings":       "#3b82f6",
                "acquisition":    "#7c3aed",
                "expansion":      "#0891b2",
                "sustainability": "#059669",
            }
            categories = ["All"] + sorted({r.get("category", "") for r in releases})
            cat_filter = st.selectbox("Filter by category", categories)
            filtered = sorted(
                [r for r in releases if cat_filter == "All" or r.get("category") == cat_filter],
                key=lambda r: r.get("date", ""),
                reverse=True,
            )
            st.markdown(f"**{len(filtered)} release(s) shown**")
            for r in filtered:
                color = CAT_COLOR.get(r.get("category", ""), "#64748b")
                with st.expander(f"📄  {r.get('date')}  —  {r.get('title')}"):
                    st.markdown(
                        f'<span class="cat-badge" style="background:{color}">'
                        f'{r.get("category", "").upper()}</span>',
                        unsafe_allow_html=True,
                    )
                    st.write(r.get("content", ""))
        else:
            st.info("No press releases found.")


# ──────────────────────────────────────────────────────────────────
# ML PREDICTIONS TAB
# ──────────────────────────────────────────────────────────────────
with tab_ml:
    ml_reg, ml_clf = st.tabs(["📐  Housing Price — Regression", "📊  Term Deposit — Classification"])

    with ml_reg:
        st.subheader("California Housing Price Prediction")
        st.markdown(
            '<div class="section-info">'
            '🌲 <strong>Random Forest Regressor</strong> trained on the California Housing dataset '
            '(20,640 samples, 8 features). Deployed as a live endpoint on <strong>Amazon SageMaker</strong>. '
            'Test set: R² = 0.801 · RMSE = $51,024 · MAE = $33,270.'
            '</div>',
            unsafe_allow_html=True,
        )

        reg_ep = _ENV.get("SAGEMAKER_REGRESSION_ENDPOINT") or os.getenv("SAGEMAKER_REGRESSION_ENDPOINT", "")
        if reg_ep:
            st.markdown(
                f'<div style="background:#f0fdf4;border:1px solid #bbf7d0;border-radius:8px;'
                f'padding:8px 14px;margin-bottom:12px;font-size:0.88rem;color:#065f46">'
                f'🟢 <strong>SageMaker Endpoint Active</strong> &nbsp;·&nbsp; '
                f'<code style="background:#dcfce7;padding:2px 6px;border-radius:4px">{reg_ep}</code>'
                f'&nbsp;·&nbsp; Region: us-east-1</div>',
                unsafe_allow_html=True,
            )
        else:
            st.warning("Set `SAGEMAKER_REGRESSION_ENDPOINT` in `.env` to enable live SageMaker predictions.")

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**📍 Location & Population**")
            latitude   = st.slider("Latitude",  32.5, 42.0, 35.6, 0.1)
            longitude  = st.slider("Longitude", -124.5, -114.3, -119.0, 0.1)
            population = st.slider("Block Population", 3, 35000, 1500)
            med_inc    = st.slider("Median Income (×$10K)", 0.5, 15.0, 5.0, 0.1)
        with c2:
            st.markdown("**🏠 Housing Characteristics**")
            house_age  = st.slider("Housing Median Age (years)", 1, 52, 20)
            ave_rooms  = st.slider("Avg Rooms per Household", 1.0, 15.0, 5.5, 0.1)
            ave_bedrms = st.slider("Avg Bedrooms per Household", 0.5, 5.0, 1.05, 0.05)
            ave_occup  = st.slider("Avg Occupants per Household", 1.0, 10.0, 2.8, 0.1)

        if st.button("🔮  Predict House Value", use_container_width=True, key="reg_predict"):
            payload = {
                "MedInc": med_inc, "HouseAge": house_age,
                "AveRooms": ave_rooms, "AveBedrms": ave_bedrms,
                "Population": population, "AveOccup": ave_occup,
                "Latitude": latitude, "Longitude": longitude,
            }
            if reg_ep:
                with st.spinner("Calling SageMaker endpoint…"):
                    result = call_sagemaker(reg_ep, payload)
                if result:
                    val = result.get("predicted_value_usd", 0)
                    st.markdown(f"""
                    <div class="pred-result">
                        <div class="pred-label">Predicted Median House Value</div>
                        <div class="pred-value">${val:,.0f}</div>
                        <div class="pred-sub">Random Forest Regressor · Amazon SageMaker · sklearn 1.2-1</div>
                    </div>""", unsafe_allow_html=True)
            else:
                try:
                    import joblib
                    model_path = ROOT / "ml" / "regression" / "model.joblib"
                    if model_path.exists():
                        model = joblib.load(model_path)
                        pred = model.predict([[med_inc, house_age, ave_rooms, ave_bedrms,
                                               population, ave_occup, latitude, longitude]])[0] * 100000
                        st.markdown(f"""
                        <div class="pred-result">
                            <div class="pred-label">Predicted Median House Value (local model)</div>
                            <div class="pred-value">${pred:,.0f}</div>
                        </div>""", unsafe_allow_html=True)
                    else:
                        st.warning("Run `python ml/regression/train.py` to generate a local model.")
                except Exception as e:
                    st.error(f"Local prediction error: {e}")

        metrics_path = ROOT / "ml" / "regression" / "metrics.json"
        if metrics_path.exists():
            with st.expander("📊 Model Performance Metrics"):
                m = json.loads(metrics_path.read_text())
                mc1, mc2, mc3 = st.columns(3)
                mc1.metric("RMSE", f"${m['rmse']*100000:,.0f}")
                mc2.metric("MAE",  f"${m['mae']*100000:,.0f}")
                mc3.metric("R²",   f"{m['r2']:.4f}")
                for p in [
                    ROOT / "ml" / "plots" / "regression_metrics.png",
                    ROOT / "ml" / "plots" / "regression_feature_importance.png",
                ]:
                    if p.exists():
                        st.image(str(p))

    with ml_clf:
        st.subheader("Term Deposit Subscription Prediction")
        st.markdown(
            '<div class="section-info">'
            '📈 <strong>Logistic Regression</strong> trained on the UCI Bank Marketing dataset '
            '(45,211 records, 16 features). Deployed on <strong>Amazon SageMaker</strong>. '
            'Test set: Accuracy = 84.6% · Recall = 81.9% · F1 = 0.555. '
            'Class weights balanced to prioritise recall (catching real subscribers).'
            '</div>',
            unsafe_allow_html=True,
        )

        clf_ep = _ENV.get("SAGEMAKER_CLASSIFICATION_ENDPOINT") or os.getenv("SAGEMAKER_CLASSIFICATION_ENDPOINT", "")
        if clf_ep:
            st.markdown(
                f'<div style="background:#f0fdf4;border:1px solid #bbf7d0;border-radius:8px;'
                f'padding:8px 14px;margin-bottom:12px;font-size:0.88rem;color:#065f46">'
                f'🟢 <strong>SageMaker Endpoint Active</strong> &nbsp;·&nbsp; '
                f'<code style="background:#dcfce7;padding:2px 6px;border-radius:4px">{clf_ep}</code>'
                f'&nbsp;·&nbsp; Region: us-east-1</div>',
                unsafe_allow_html=True,
            )
        else:
            st.warning("Set `SAGEMAKER_CLASSIFICATION_ENDPOINT` in `.env` to enable live SageMaker predictions.")

        c3, c4 = st.columns(2)
        with c3:
            st.markdown("**👤 Customer Profile**")
            age       = st.slider("Age", 18, 95, 40, key="clf_age")
            balance   = st.number_input("Account Balance ($)", value=1500, step=100)
            job       = st.selectbox("Job", ["management", "technician", "blue-collar", "admin.",
                                             "services", "retired", "self-employed", "unemployed",
                                             "entrepreneur", "housemaid", "student", "unknown"])
            marital   = st.selectbox("Marital Status", ["married", "single", "divorced"])
            education = st.selectbox("Education Level", ["secondary", "tertiary", "primary", "unknown"])
            housing   = st.selectbox("Has Housing Loan?", ["yes", "no"])
            loan      = st.selectbox("Has Personal Loan?", ["no", "yes"])
        with c4:
            st.markdown("**📞 Campaign Details**")
            duration  = st.slider("Last Contact Duration (sec)", 0, 3000, 250, key="clf_dur")
            campaign  = st.slider("Contacts This Campaign", 1, 50, 2)
            pdays     = st.slider("Days Since Last Contact (−1 = never)", -1, 999, -1)
            previous  = st.slider("Previous Campaign Contacts", 0, 275, 0)
            contact   = st.selectbox("Contact Method", ["cellular", "telephone", "unknown"])
            month     = st.selectbox("Last Contact Month", [
                "jan","feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec"
            ])
            poutcome  = st.selectbox("Previous Outcome", ["unknown", "failure", "success", "other"])

        if st.button("🔮  Predict Subscription", use_container_width=True, key="clf_predict"):
            payload = {
                "age": age, "balance": balance, "duration": duration,
                "campaign": campaign, "pdays": pdays, "previous": previous,
                "job": job, "marital": marital, "education": education,
                "default": "no", "housing": housing, "loan": loan,
                "contact": contact, "month": month, "poutcome": poutcome,
            }
            if clf_ep:
                with st.spinner("Calling SageMaker endpoint…"):
                    result = call_sagemaker(clf_ep, payload)
                if result:
                    label = result.get("label", "—")
                    prob  = result.get("probability", 0.0)
                    cls   = "yes" if label == "yes" else "no"
                    st.markdown(f"""
                    <div class="pred-result {cls}">
                        <div class="pred-label">Subscription Prediction</div>
                        <div class="pred-value {cls}">{label.upper()}</div>
                        <div class="pred-sub">
                            Confidence: {prob:.1%} · Logistic Regression · Amazon SageMaker · sklearn 1.2-1
                        </div>
                    </div>""", unsafe_allow_html=True)
            else:
                try:
                    import joblib
                    model_path = ROOT / "ml" / "classification" / "model.joblib"
                    if model_path.exists():
                        model = joblib.load(model_path)
                        row = pd.DataFrame([payload])
                        pred_label = model.predict(row)[0]
                        pred_prob  = model.predict_proba(row)[0, 1]
                        label = "yes" if pred_label else "no"
                        cls   = "yes" if pred_label else "no"
                        st.markdown(f"""
                        <div class="pred-result {cls}">
                            <div class="pred-label">Subscription Prediction (local model)</div>
                            <div class="pred-value {cls}">{label.upper()}</div>
                            <div class="pred-sub">Confidence: {pred_prob:.1%}</div>
                        </div>""", unsafe_allow_html=True)
                    else:
                        st.warning("Run `python ml/classification/train.py` to generate a local model.")
                except Exception as e:
                    st.error(f"Local prediction error: {e}")

        clf_metrics_path = ROOT / "ml" / "classification" / "metrics.json"
        if clf_metrics_path.exists():
            with st.expander("📊 Model Performance Metrics"):
                cm = json.loads(clf_metrics_path.read_text())
                mc1, mc2, mc3, mc4 = st.columns(4)
                mc1.metric("Accuracy",  f"{cm['accuracy']:.3f}")
                mc2.metric("Precision", f"{cm['precision']:.3f}")
                mc3.metric("Recall",    f"{cm['recall']:.3f}")
                mc4.metric("F1 Score",  f"{cm['f1']:.3f}")
                for p in [
                    ROOT / "ml" / "plots" / "classification_metrics.png",
                    ROOT / "ml" / "plots" / "classification_confusion_matrix.png",
                ]:
                    if p.exists():
                        st.image(str(p))


