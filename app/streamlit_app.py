"""
Prologis Financial Assistant — Streamlit Web Application

Three-tab interface:
  💬 Chat       — Vertex AI (Gemini 2.5 Flash) conversational agent
  📊 Data       — Properties (Postgres), SEC Filings, Press Releases
  🤖 ML Predictions — Live SageMaker endpoints (regression + classification)
"""
import json
import os
import sys
from pathlib import Path

import boto3
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL as _SAURL

# Add project root to path so agent imports work
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

load_dotenv(ROOT / ".env", override=True)

# Read .env directly so credentials are always fresh regardless of os.environ state
from dotenv import dotenv_values as _dotenv_values
_ENV = _dotenv_values(ROOT / ".env")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Prologis Financial Assistant",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif;
}

.stApp {
    background: #f0f4f8;
    color: #1e293b;
}

/* Header */
.main-header {
    background: linear-gradient(135deg, #ffffff 0%, #eff6ff 100%);
    border: 1px solid #bfdbfe;
    border-radius: 16px;
    padding: 24px 32px;
    margin-bottom: 24px;
    box-shadow: 0 4px 24px rgba(59,130,246,0.10);
}
.main-header h1 {
    font-size: 2rem;
    font-weight: 700;
    background: linear-gradient(90deg, #1d4ed8, #7c3aed);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
}
.main-header p {
    color: #64748b;
    margin: 8px 0 0;
    font-size: 0.95rem;
}

/* Chat messages */
.chat-user {
    background: #eff6ff;
    border: 1px solid #bfdbfe;
    border-radius: 12px 12px 4px 12px;
    padding: 12px 16px;
    margin: 8px 0;
    font-size: 0.9rem;
    color: #1e3a8a;
}
.chat-assistant {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 12px 12px 12px 4px;
    padding: 12px 16px;
    margin: 8px 0;
    font-size: 0.9rem;
    color: #1e293b;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
}

/* Metric cards */
.metric-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 16px;
    text-align: center;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
}
.metric-card h3 {
    color: #64748b;
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin: 0 0 8px;
}
.metric-card .value {
    color: #1d4ed8;
    font-size: 1.5rem;
    font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: #e2e8f0;
    border-radius: 12px;
    padding: 4px;
    gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    font-weight: 500;
    padding: 8px 20px;
    color: #475569;
}
.stTabs [aria-selected="true"] {
    background: #ffffff !important;
    color: #1d4ed8 !important;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #1d4ed8, #4f46e5);
    color: white;
    border: none;
    border-radius: 8px;
    font-weight: 600;
    padding: 8px 20px;
    transition: all 0.2s;
}
.stButton > button:hover {
    opacity: 0.88;
    transform: translateY(-1px);
    box-shadow: 0 4px 14px rgba(29,78,216,0.30);
}

/* Code / mono */
code {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.85em;
    background: #f1f5f9;
    color: #4f46e5;
    padding: 2px 6px;
    border-radius: 4px;
}

/* Prediction result box */
.prediction-box {
    background: #f0fdf4;
    border: 1px solid #bbf7d0;
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    margin: 16px 0;
    box-shadow: 0 2px 10px rgba(16,185,129,0.10);
}
.prediction-box .pred-value {
    font-size: 2rem;
    font-weight: 700;
    color: #059669;
    font-family: 'JetBrains Mono', monospace;
}

/* Expander headers */
.streamlit-expanderHeader {
    background: #f8fafc;
    border-radius: 8px;
}
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
  <h1>🏭 Prologis Financial Assistant</h1>
  <p>AI-powered platform for financial intelligence, property analytics, and predictive insights</p>
</div>
""", unsafe_allow_html=True)

# ── DB connection (cached) ────────────────────────────────────────────────────
def get_db_engine():
    user = _ENV.get("POSTGRES_USER") or os.getenv("POSTGRES_USER", "postgres")
    pw   = _ENV.get("POSTGRES_PASSWORD") or os.getenv("POSTGRES_PASSWORD", "postgres")
    host = _ENV.get("POSTGRES_HOST") or os.getenv("POSTGRES_HOST", "localhost")
    port = _ENV.get("POSTGRES_PORT") or os.getenv("POSTGRES_PORT", "5432")
    db   = _ENV.get("POSTGRES_DB") or os.getenv("POSTGRES_DB", "financial_assistant")
    url  = _SAURL.create("postgresql+psycopg2", username=user, password=pw,
                         host=host, port=int(port), database=db)
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
    if metro_filter and metro_filter != "All":
        sql += " AND p.metro_area = :metro"
        params["metro"] = metro_filter
    if type_filter and type_filter != "All":
        sql += " AND p.property_type = :ptype"
        params["ptype"] = type_filter
    sql += " ORDER BY f.revenue DESC NULLS LAST"
    try:
        engine = get_db_engine()
        with engine.connect() as conn:
            df = pd.read_sql(text(sql), conn, params=params)
        for col in ["revenue", "net_income", "expenses"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        return df
    except Exception as e:
        st.error(f"Database error: {e}")
        return pd.DataFrame()


# ── SEC data loader ───────────────────────────────────────────────────────────
@st.cache_data
def load_sec_data():
    sec_path = ROOT / "data" / "sec" / "prologis_financials.json"
    if sec_path.exists():
        return json.loads(sec_path.read_text())
    return {}


# ── Press releases loader ─────────────────────────────────────────────────────
@st.cache_data
def load_press_releases():
    press_path = ROOT / "data" / "press_releases.json"
    if press_path.exists():
        return json.loads(press_path.read_text())
    return []


# ── SageMaker inference ───────────────────────────────────────────────────────
def call_sagemaker(endpoint_name: str, payload: dict) -> dict | None:
    try:
        runtime = boto3.client(
            "sagemaker-runtime",
            region_name=_ENV.get("AWS_REGION") or os.getenv("AWS_REGION", "us-east-1"),
            aws_access_key_id=_ENV.get("AWS_ACCESS_KEY_ID") or os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=_ENV.get("AWS_SECRET_ACCESS_KEY") or os.getenv("AWS_SECRET_ACCESS_KEY"),
        )
        response = runtime.invoke_endpoint(
            EndpointName=endpoint_name,
            ContentType="application/json",
            Accept="application/json",
            Body=json.dumps([payload]),
        )
        return json.loads(response["Body"].read())[0]
    except Exception as e:
        st.error(f"SageMaker error: {e}")
        return None


# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_chat, tab_data, tab_ml = st.tabs(["💬 Chat", "📊 Data", "🤖 ML Predictions"])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1: CHAT
# ═══════════════════════════════════════════════════════════════════════════════
with tab_chat:
    st.subheader("Ask anything about Prologis")

    # Session state for chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Suggested queries
    st.markdown("**Try asking:**")
    cols = st.columns(3)
    suggestions = [
        "What was Prologis' net income last year?",
        "Show industrial properties in Chicago",
        "Any recent acquisitions?",
        "Show Dallas properties with revenue over $8M",
        "Summarize the latest earnings release",
        "What are total assets and liabilities?",
    ]
    for i, col in enumerate(cols):
        with col:
            if st.button(suggestions[i*2], key=f"sug_{i*2}", use_container_width=True):
                st.session_state.pending_query = suggestions[i*2]
            if st.button(suggestions[i*2+1], key=f"sug_{i*2+1}", use_container_width=True):
                st.session_state.pending_query = suggestions[i*2+1]

    st.divider()

    # Chat input
    user_input = st.chat_input("Ask a question about Prologis financials, properties, or press releases...")
    if "pending_query" in st.session_state:
        user_input = st.session_state.pop("pending_query")

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.spinner("Thinking..."):
            try:
                import sys
                for _m in [k for k in sys.modules if k.startswith("agent")]:
                    del sys.modules[_m]
                from agent.agent import ask
                result = ask(user_input)
                answer = result.get("answer", "No answer returned.")
                tool_calls = result.get("tool_calls", [])
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "tool_calls": tool_calls,
                })
            except Exception as e:
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"Error: {e}",
                    "tool_calls": [],
                })

    # Display messages (newest first)
    for msg in reversed(st.session_state.messages):
        if msg["role"] == "user":
            st.markdown(f'<div class="chat-user">🧑 {msg["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-assistant">🤖 {msg["content"]}</div>', unsafe_allow_html=True)
            tool_calls = msg.get("tool_calls", [])
            if tool_calls:
                with st.expander(f"🔧 Tools used ({len(tool_calls)})"):
                    for tc in tool_calls:
                        st.markdown(f"**`{tc['tool']}`**")
                        st.json({"args": tc["args"], "result": tc["result"]})

    if st.session_state.messages:
        if st.button("🗑 Clear chat"):
            st.session_state.messages = []
            st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2: DATA BROWSER
# ═══════════════════════════════════════════════════════════════════════════════
with tab_data:
    sub_props, sub_sec, sub_press = st.tabs(["🏢 Properties", "📈 SEC Filings", "📰 Press Releases"])

    # ── Properties sub-tab ───────────────────────────────────────────────────
    with sub_props:
        st.subheader("Property Portfolio")

        col_filter1, col_filter2 = st.columns(2)
        with col_filter1:
            metro_options = ["All", "Los Angeles", "Chicago", "New York", "Kansas City",
                             "Dallas", "Miami", "Seattle", "Phoenix", "Portland",
                             "Philadelphia", "Atlanta"]
            metro_filter = st.selectbox("Metro Area", metro_options)
        with col_filter2:
            type_options = ["All", "Industrial", "Logistics", "Warehouse"]
            type_filter = st.selectbox("Property Type", type_options)

        df = load_properties(
            metro_filter if metro_filter != "All" else None,
            type_filter  if type_filter  != "All" else None,
        )

        if not df.empty:
            # Summary metrics
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.markdown(f"""<div class="metric-card">
                    <h3>Properties</h3><div class="value">{len(df)}</div></div>""",
                    unsafe_allow_html=True)
            with m2:
                total_rev = df["revenue"].sum()
                st.markdown(f"""<div class="metric-card">
                    <h3>Total Revenue</h3><div class="value">${total_rev/1e6:.1f}M</div></div>""",
                    unsafe_allow_html=True)
            with m3:
                total_ni = df["net_income"].sum()
                st.markdown(f"""<div class="metric-card">
                    <h3>Total Net Income</h3><div class="value">${total_ni/1e6:.1f}M</div></div>""",
                    unsafe_allow_html=True)
            with m4:
                avg_sf = df["sq_footage"].mean()
                st.markdown(f"""<div class="metric-card">
                    <h3>Avg Sq Footage</h3><div class="value">{avg_sf:,.0f}</div></div>""",
                    unsafe_allow_html=True)

            st.markdown("")

            # Format the dataframe
            display_df = df.copy()
            display_df["revenue"]    = display_df["revenue"].apply(lambda x: f"${x:,.0f}")
            display_df["net_income"] = display_df["net_income"].apply(lambda x: f"${x:,.0f}")
            display_df["expenses"]   = display_df["expenses"].apply(lambda x: f"${x:,.0f}")
            display_df["sq_footage"] = display_df["sq_footage"].apply(lambda x: f"{x:,}")
            display_df = display_df.rename(columns={
                "property_id": "ID", "address": "Address", "metro_area": "Metro",
                "sq_footage": "Sq Ft", "property_type": "Type",
                "revenue": "Revenue", "net_income": "Net Income", "expenses": "Expenses",
            })
            st.dataframe(display_df, use_container_width=True, hide_index=True)
        else:
            st.info("No properties found or database not connected.")

    # ── SEC Filings sub-tab ──────────────────────────────────────────────────
    with sub_sec:
        st.subheader("SEC EDGAR Financial Data — Prologis (PLD)")

        sec_data = load_sec_data()
        if sec_data:
            period = st.radio("Period", ["Annual (10-K)", "Quarterly (10-Q)"], horizontal=True)
            is_annual = period.startswith("Annual")

            metrics = sec_data.get("metrics", {})
            rows = []
            for key, data in metrics.items():
                entry = data.get("latest_annual") if is_annual else data.get("latest_quarterly")
                if entry:
                    rows.append({
                        "Metric": key.replace("_", " ").title(),
                        "Value (USD)": f"${entry.get('val', 0):,.0f}",
                        "Period End": entry.get("end", "—"),
                        "Form": entry.get("form", "—"),
                        "Fiscal Year": entry.get("fy", "—"),
                        "Fiscal Period": entry.get("fp", "—"),
                    })

            if rows:
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

                # Historical trend for revenue
                st.markdown("#### Revenue History")
                rev_history = metrics.get("revenue", {}).get(
                    "annual_history" if is_annual else "quarterly_history", []
                )
                if rev_history:
                    hist_df = pd.DataFrame(rev_history)
                    hist_df["val_bn"] = hist_df["val"] / 1e9
                    hist_df = hist_df.sort_values("end")
                    st.bar_chart(hist_df.set_index("end")["val_bn"], height=250)
                    st.caption("Revenue in $B")
        else:
            st.info("SEC data not loaded. Run `python scripts/fetch_sec.py` or check `data/sec/prologis_financials.json`.")

    # ── Press Releases sub-tab ───────────────────────────────────────────────
    with sub_press:
        st.subheader("Press Releases")

        releases = load_press_releases()
        if releases:
            categories = ["All"] + sorted({r.get("category", "") for r in releases})
            cat_filter = st.selectbox("Category", categories)
            filtered = [r for r in releases
                        if cat_filter == "All" or r.get("category") == cat_filter]
            filtered.sort(key=lambda r: r.get("date", ""), reverse=True)

            for r in filtered:
                cat_color = {
                    "earnings": "#16a34a",
                    "acquisition": "#3b82f6",
                    "expansion": "#a855f7",
                    "sustainability": "#10b981",
                }.get(r.get("category", ""), "#64748b")

                with st.expander(f"**{r.get('date', '')}** — {r.get('title', '')}"):
                    st.markdown(
                        f"<span style='background:{cat_color};color:white;"
                        f"padding:2px 8px;border-radius:4px;font-size:0.8em'>"
                        f"{r.get('category','').upper()}</span>",
                        unsafe_allow_html=True,
                    )
                    st.markdown(r.get("content", ""))
        else:
            st.info("No press releases found.")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3: ML PREDICTIONS
# ═══════════════════════════════════════════════════════════════════════════════
with tab_ml:
    ml_reg, ml_clf = st.tabs(["📐 Housing Price (Regression)", "📊 Subscription (Classification)"])

    # ── Regression tab ───────────────────────────────────────────────────────
    with ml_reg:
        st.subheader("California Housing Price Prediction")
        st.markdown(
            "Predict median house value using a **Random Forest Regressor** "
            "deployed on Amazon SageMaker."
        )

        reg_endpoint = os.getenv("SAGEMAKER_REGRESSION_ENDPOINT", "")
        if not reg_endpoint:
            st.warning("Set `SAGEMAKER_REGRESSION_ENDPOINT` in .env to enable live predictions.")

        col1, col2 = st.columns(2)
        with col1:
            med_inc    = st.slider("Median Income (tens of thousands)", 0.5, 15.0, 5.0, 0.1)
            house_age  = st.slider("Housing Median Age (years)", 1, 52, 20)
            ave_rooms  = st.slider("Average Rooms per Household", 1.0, 15.0, 5.5, 0.1)
            ave_bedrms = st.slider("Average Bedrooms per Household", 0.5, 5.0, 1.05, 0.05)
        with col2:
            population = st.slider("Block Population", 3, 35000, 1500)
            ave_occup  = st.slider("Average Occupants per Household", 1.0, 10.0, 2.8, 0.1)
            latitude   = st.slider("Latitude", 32.5, 42.0, 35.6, 0.1)
            longitude  = st.slider("Longitude", -124.5, -114.3, -119.0, 0.1)

        if st.button("🔮 Predict Housing Value", use_container_width=True):
            payload = {
                "MedInc": med_inc, "HouseAge": house_age,
                "AveRooms": ave_rooms, "AveBedrms": ave_bedrms,
                "Population": population, "AveOccup": ave_occup,
                "Latitude": latitude, "Longitude": longitude,
            }
            if reg_endpoint:
                with st.spinner("Calling SageMaker endpoint..."):
                    result = call_sagemaker(reg_endpoint, payload)
                if result:
                    pred_val = result.get("predicted_value_usd", 0)
                    st.markdown(f"""
                    <div class="prediction-box">
                        <p style="color:#64748b;margin-bottom:8px">Predicted Median House Value</p>
                        <div class="pred-value">${pred_val:,.0f}</div>
                        <p style="color:#64748b;margin-top:8px;font-size:0.85rem">
                            Random Forest Regressor · SageMaker sklearn 1.2-1
                        </p>
                    </div>""", unsafe_allow_html=True)
            else:
                st.info("SageMaker endpoint not configured. Showing local prediction.")
                try:
                    import joblib
                    model_path = ROOT / "ml" / "regression" / "model.joblib"
                    if model_path.exists():
                        model = joblib.load(model_path)
                        import numpy as np
                        features = [[
                            med_inc, house_age, ave_rooms, ave_bedrms,
                            population, ave_occup, latitude, longitude
                        ]]
                        pred = model.predict(features)[0] * 100000
                        st.markdown(f"""
                        <div class="prediction-box">
                            <p style="color:#64748b;margin-bottom:8px">Predicted Median House Value (Local)</p>
                            <div class="pred-value">${pred:,.0f}</div>
                        </div>""", unsafe_allow_html=True)
                    else:
                        st.warning("Run `python ml/regression/train.py` to generate a local model.")
                except Exception as e:
                    st.error(f"Local prediction error: {e}")

        # Show metrics if available
        metrics_path = ROOT / "ml" / "regression" / "metrics.json"
        if metrics_path.exists():
            with st.expander("📊 Model Performance Metrics"):
                m = json.loads(metrics_path.read_text())
                mc1, mc2, mc3 = st.columns(3)
                mc1.metric("RMSE", f"${m['rmse']*100000:,.0f}")
                mc2.metric("MAE",  f"${m['mae']*100000:,.0f}")
                mc3.metric("R²",   f"{m['r2']:.4f}")

                plot_path = ROOT / "ml" / "plots" / "regression_metrics.png"
                fi_path   = ROOT / "ml" / "plots" / "regression_feature_importance.png"
                if plot_path.exists():
                    st.image(str(plot_path))
                if fi_path.exists():
                    st.image(str(fi_path))

    # ── Classification tab ───────────────────────────────────────────────────
    with ml_clf:
        st.subheader("Bank Marketing Subscription Prediction")
        st.markdown(
            "Predict whether a customer will subscribe to a term deposit using a "
            "**Logistic Regression** classifier deployed on Amazon SageMaker."
        )

        clf_endpoint = os.getenv("SAGEMAKER_CLASSIFICATION_ENDPOINT", "")
        if not clf_endpoint:
            st.warning("Set `SAGEMAKER_CLASSIFICATION_ENDPOINT` in .env to enable live predictions.")

        col3, col4 = st.columns(2)
        with col3:
            age      = st.slider("Age", 18, 95, 40)
            balance  = st.number_input("Account Balance ($)", value=1500, step=100)
            duration = st.slider("Last Contact Duration (seconds)", 0, 3000, 250)
            campaign = st.slider("Number of Contacts This Campaign", 1, 50, 2)
        with col4:
            pdays    = st.slider("Days Since Last Contact (-1 = never)", -1, 999, -1)
            previous = st.slider("Previous Campaign Contacts", 0, 275, 0)
            job      = st.selectbox("Job", ["management", "technician", "blue-collar", "admin.",
                                            "services", "retired", "self-employed", "unemployed",
                                            "entrepreneur", "housemaid", "student", "unknown"])
            marital  = st.selectbox("Marital Status", ["married", "single", "divorced"])

        col5, col6 = st.columns(2)
        with col5:
            education = st.selectbox("Education", ["secondary", "tertiary", "primary", "unknown"])
            housing   = st.selectbox("Has Housing Loan?", ["yes", "no"])
        with col6:
            loan      = st.selectbox("Has Personal Loan?", ["no", "yes"])
            contact   = st.selectbox("Contact Type", ["cellular", "telephone", "unknown"])
        month    = st.selectbox("Last Contact Month", ["jan","feb","mar","apr","may","jun",
                                                        "jul","aug","sep","oct","nov","dec"])
        poutcome = st.selectbox("Previous Campaign Outcome", ["unknown", "failure", "success", "other"])

        if st.button("🔮 Predict Subscription", use_container_width=True):
            payload = {
                "age": age, "balance": balance, "duration": duration,
                "campaign": campaign, "pdays": pdays, "previous": previous,
                "job": job, "marital": marital, "education": education,
                "default": "no", "housing": housing, "loan": loan,
                "contact": contact, "month": month, "poutcome": poutcome,
            }
            if clf_endpoint:
                with st.spinner("Calling SageMaker endpoint..."):
                    result = call_sagemaker(clf_endpoint, payload)
                if result:
                    label = result.get("label", "—")
                    prob  = result.get("probability", 0.0)
                    color = "#059669" if label == "yes" else "#dc2626"
                    st.markdown(f"""
                    <div class="prediction-box" style="border-color:{color}40;background:{color}0d">
                        <p style="color:#64748b;margin-bottom:8px">Subscription Prediction</p>
                        <div class="pred-value" style="color:{color}">{label.upper()}</div>
                        <p style="color:#64748b;margin-top:8px;font-size:0.85rem">
                            Confidence: {prob:.1%} · Logistic Regression · SageMaker sklearn 1.2-1
                        </p>
                    </div>""", unsafe_allow_html=True)
            else:
                st.info("SageMaker endpoint not configured. Showing local prediction.")
                try:
                    import joblib
                    model_path = ROOT / "ml" / "classification" / "model.joblib"
                    if model_path.exists():
                        model = joblib.load(model_path)
                        import pandas as pd
                        row = pd.DataFrame([payload])
                        pred_label = model.predict(row)[0]
                        pred_prob  = model.predict_proba(row)[0, 1]
                        label = "yes" if pred_label else "no"
                        color = "#059669" if pred_label else "#dc2626"
                        st.markdown(f"""
                        <div class="prediction-box" style="border-color:{color}40;background:{color}0d">
                            <p style="color:#64748b;margin-bottom:8px">Subscription Prediction (Local)</p>
                            <div class="pred-value" style="color:{color}">{label.upper()}</div>
                            <p style="color:#64748b;margin-top:8px;font-size:0.85rem">
                                Confidence: {pred_prob:.1%}
                            </p>
                        </div>""", unsafe_allow_html=True)
                    else:
                        st.warning("Run `python ml/classification/train.py` to generate a local model.")
                except Exception as e:
                    st.error(f"Local prediction error: {e}")

        # Show metrics if available
        clf_metrics_path = ROOT / "ml" / "classification" / "metrics.json"
        if clf_metrics_path.exists():
            with st.expander("📊 Model Performance Metrics"):
                cm_data = json.loads(clf_metrics_path.read_text())
                mc1, mc2, mc3, mc4 = st.columns(4)
                mc1.metric("Accuracy",  f"{cm_data['accuracy']:.3f}")
                mc2.metric("Precision", f"{cm_data['precision']:.3f}")
                mc3.metric("Recall",    f"{cm_data['recall']:.3f}")
                mc4.metric("F1 Score",  f"{cm_data['f1']:.3f}")

                plot_path = ROOT / "ml" / "plots" / "classification_metrics.png"
                cm_path   = ROOT / "ml" / "plots" / "classification_confusion_matrix.png"
                if plot_path.exists():
                    st.image(str(plot_path))
                if cm_path.exists():
                    st.image(str(cm_path))

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center;color:#475569;font-size:0.8rem'>"
    "Prologis Financial Assistant · "
    "Vertex AI (Gemini 2.5 Flash) · AWS SageMaker · AWS Bedrock · Supabase Postgres"
    "</p>",
    unsafe_allow_html=True,
)
