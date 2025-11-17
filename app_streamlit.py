# app_streamlit.py
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import traceback

import streamlit as st
import plotly.express as px
import streamlit_authenticator as stauth

from src.llm_explainer import llm_or_fallback_explain  # LLM explainer module

# -------------------------- Page config --------------------------
st.set_page_config(page_title="Predictive Transaction Intelligence", layout="wide")

# -------------------------- Auth (new API) -----------------------
credentials = {
    "usernames": {
        "analyst": {
            "email": "analyst@example.com",
            "first_name": "Analyst",
            "last_name": "One",
            "password": "pass123",
        },
        "manager": {
            "email": "manager@example.com",
            "first_name": "Manager",
            "last_name": "Two",
            "password": "pass456",
        },
    }
}

authenticator = stauth.Authenticate(
    credentials,
    "fraud_demo_cookie",  # cookie name
    "abcdef",             # cookie key/secret
    1.0                   # cookie expiry (days)
)

# Render login form on main area
authenticator.login(
    location="main",
    fields={"Form name": "Login", "Username": "Username", "Password": "Password", "Login": "Login"}
)

auth_status = st.session_state.get("authentication_status", None)
if auth_status is True:
    name = st.session_state.get("name", "")
    username = st.session_state.get("username", "")
    authenticator.logout(location="sidebar", button_name="Logout")
    st.sidebar.success(f"Welcome, {name}!")
elif auth_status is False:
    st.error("Username/Password incorrect.")
    st.stop()
else:
    st.info("Please enter your credentials.")
    st.stop()

# ---------------------- Artifact loading (robust) -------------------------
@st.cache_resource
def load_artifacts():
    """
    Try loading safer models first (LR -> RF -> GB).
    Return: model, encoders(dict), meta(dict), loaded_name (str)
    """
    base = Path(_file_).resolve().parent
    art = base / "artifacts"

    files = {
        "lr": art / "fraud_model_lr.joblib",
        "rf": art / "fraud_model_rf.joblib",
        "gb": art / "fraud_model.joblib",
    }

    # load encoders & metadata (required)
    try:
        encoders = joblib.load(art / "label_encoders.joblib")
    except Exception as e:
        st.error(f"❌ Could not load label_encoders.joblib: {e}")
        st.stop()
    try:
        meta = joblib.load(art / "metadata.joblib")
    except Exception as e:
        st.error(f"❌ Could not load metadata.joblib: {e}")
        st.stop()

    model = None
    loaded_name = None
    # prefer LR -> RF -> GB
    for key in ("lr", "rf", "gb"):
        p = files[key]
        if p.exists():
            try:
                model = joblib.load(p)
                loaded_name = p.name
                break
            except Exception as e:
                # warn and continue trying next artifact
                st.warning(f"Could not load {p.name}: {e}")
                continue

    if model is None:
        st.error(
            "❌ Could not load any model artifact. Place one of: "
            "fraud_model_lr.joblib, fraud_model_rf.joblib, fraud_model.joblib in ./artifacts"
        )
        st.stop()

    # Small safe debug info (sidebar)
    try:
        st.sidebar.markdown(f"*Loaded model artifact:* {loaded_name}")
        st.sidebar.markdown(f"*Model type:* {type(model).__name__}")
    except Exception:
        pass

    return model, encoders, meta, loaded_name

# load artifacts (must be before sliders/session-state usage)
model, encoders, meta, loaded_name = load_artifacts()
FEATURES = list(meta.get("feature_columns", []))
DEFAULT_THR = float(meta.get("threshold", 0.7)) if meta else 0.7
MODEL_NAME = loaded_name or meta.get("model_name", "Model")
HOME_CCY_META = meta.get("home_currency", None) if meta else None

# ---- Session-state init (needed before sidebar/widgets) ---------
st.session_state.setdefault("thr_pending", None)
# set slider defaults too (safe)
st.session_state.setdefault("thr_slider", float(max(0.05, min(0.95, DEFAULT_THR))))
st.session_state.setdefault("band_slider", 0.05)

# ------------------ Preprocessing to match training --------------
def clean_and_engineer(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()

    # Combine date+time if present (robust)
    if {"Transaction_Date", "Transaction_Time"}.issubset(df.columns):
        df["transaction_datetime"] = pd.to_datetime(
            df["Transaction_Date"].astype(str) + " " + df["Transaction_Time"].astype(str),
            errors="coerce"
        )

    # Normalize names
    df.columns = df.columns.str.strip().str.replace(" ", "_").str.lower()

    # Numerics (coerce)
    for c in [
        "transaction_amount",
        "distance_between_transactions_km",
        "time_since_last_transaction_min",
        "transaction_velocity",
    ]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Datetime features (month not used as a feature in training in your flow)
    if "transaction_datetime" in df.columns:
        dt = pd.to_datetime(df["transaction_datetime"], errors="coerce")
        df["hour"] = dt.dt.hour
        df["day"] = dt.dt.day
        df["day_of_week"] = dt.dt.dayofweek
        df["month"] = dt.dt.month
    else:
        df["hour"], df["day"], df["day_of_week"] = 0, 1, 0

    # Ratios (safeguards)
    ta = df.get("transaction_amount", 0)
    gap = df.get("time_since_last_transaction_min", 0)
    dist = df.get("distance_between_transactions_km", 0)
    df["amount_per_minute"] = ta / (gap + 1)
    df["distance_time_ratio"] = dist / (gap + 1)

    # Dynamic FX flag (use uploaded file's majority currency if metadata not present)
    if "transaction_currency" in df.columns:
        try:
            col = df["transaction_currency"].astype(str).str.upper()
            if HOME_CCY_META:
                home_ccy = str(HOME_CCY_META).upper()
            else:
                home_ccy = col.mode().iat[0] if not col.mode().empty else None
            if home_ccy:
                df["is_foreign_currency"] = np.where(col == home_ccy, 0, 1)
            else:
                df["is_foreign_currency"] = 0
        except Exception:
            df["is_foreign_currency"] = 0
    else:
        df["is_foreign_currency"] = 0

    return df

def safe_label_transform(series: pd.Series, le) -> pd.Series:
    """
    Transform a series using a saved LabelEncoder-like object but avoid ValueError
    by mapping unseen labels to -1.
    """
    try:
        # if sklearn LabelEncoder-like with classes_
        classes = getattr(le, "classes_", None)
        if classes is not None:
            mapping = {c: i for i, c in enumerate(classes)}
            out = series.astype(str).map(mapping)
            # unseen -> -1
            out = out.fillna(-1).astype(int)
            return out
        else:
            # fallback: try transform, but catch unknowns
            try:
                return le.transform(series.astype(str))
            except Exception:
                return series.astype(str).apply(lambda x: -1).astype(int)
    except Exception:
        return series.astype(str).apply(lambda x: -1).astype(int)

def apply_label_encoders(df: pd.DataFrame, encoders: dict) -> pd.DataFrame:
    out = df.copy()
    for c, le in encoders.items():
        if c in out.columns:
            out[c] = safe_label_transform(out[c], le)
    return out

def prepare_for_scoring(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = clean_and_engineer(df_raw)

    # never let target leak
    df = df.drop(columns=[c for c in ["isfraud", "is_fraud", "class", "target"] if c in df.columns], errors="ignore")

    # encoders (safe)
    df = apply_label_encoders(df, encoders)

    # clip extreme outliers (defensive guardrails)
    CLIP_PCTS = {
        "transaction_amount": 0.999,
        "amount_per_minute": 0.999,
        "distance_time_ratio": 0.999,
        "transaction_velocity": 0.999
    }
    for col, pct in CLIP_PCTS.items():
        if col in df.columns:
            try:
                q = df[col].quantile(pct)
                if pd.notnull(q) and q > 0:
                    df[col] = df[col].clip(lower=0, upper=q)
            except Exception:
                pass

    # drop non-features / IDs
    drop_cols = ["transaction_id", "user_id", "merchant_id", "device_id", "transaction_datetime"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    # align to training features and fill
    if FEATURES:
        df = df.reindex(columns=FEATURES, fill_value=np.nan)
    # fill numeric median for NaNs
    try:
        df = df.fillna(df.median(numeric_only=True))
    except Exception:
        df = df.fillna(0)
    return df

# ------------------------ Sidebar settings -----------------------
st.sidebar.header("Decision Settings")

# Initial threshold (if user used suggested thr earlier)
_pending = st.session_state.get("thr_pending", None)
if isinstance(_pending, (int, float)):
    init_thr = float(_pending)
else:
    init_thr = DEFAULT_THR if isinstance(DEFAULT_THR, (int, float)) and 0.05 <= DEFAULT_THR <= 0.95 else 0.7
init_thr = float(max(0.05, min(0.95, init_thr)))

thr = st.sidebar.slider(
    "Fraud threshold",
    0.05, 0.95,
    init_thr,
    0.01,
    key="thr_slider"
)

band = st.sidebar.slider(
    "Review band (±)",
    0.00, 0.20,
    0.05,
    0.01,
    key="band_slider"
)

# after widgets are created, clear pending so future changes stick to user input
if st.session_state.get("thr_pending") is not None:
    st.session_state["thr_pending"] = None

low, high = max(0.00, st.session_state["thr_slider"] - st.session_state["band_slider"]), min(1.00, st.session_state["thr_slider"] + st.session_state["band_slider"])
st.sidebar.caption(f"Risk bands → Low: < {low:.2f} | Review: {low:.2f}–{high:.2f} | High: ≥ {high:.2f}")
st.sidebar.info(f"Model: *{MODEL_NAME}*")

# --------------------------- UI layout ---------------------------
st.title("🛡 Predictive Transaction Intelligence — Fraud Detection")
st.markdown("Upload a CSV, get *fraud probabilities, **decisions, and **plain-English explanations*.")

tab1, tab2, tab3 = st.tabs(["📤 Upload & Score", "📊 Dashboard", "🧠 Explain"])

# ------------------------ Tab 1: Upload & Score ------------------
with tab1:
    st.header("Upload CSV")

    st.markdown("""
*📘 Dataset format requirement*

Upload a CSV with columns similar to the training schema.
If you're using a different dataset (e.g., from Kaggle), rename columns to match this format
—or let the app auto-map some common alternate names.

You can download a sample template below:
""")

    example_df = pd.DataFrame({
        "transaction_id": [1, 2],
        "transaction_amount": [1250.50, 4999.00],
        "distance_between_transactions_km": [2.5, 120.7],
        "time_since_last_transaction_min": [3, 240],
        "transaction_velocity": [5, 1],
        "transaction_currency": ["INR", "USD"],
        "card_type": ["Credit", "Debit"],
        "authentication_method": ["PIN", "OTP"],
        "transaction_category": ["Shopping", "Travel"],
        "transaction_datetime": ["2025-10-01 14:23:00", "2025-10-02 02:05:00"],
        "isfraud": [0, 1]
    })
    st.download_button(
        "⬇ Download sample CSV format",
        data=example_df.to_csv(index=False).encode("utf-8"),
        file_name="sample_transaction_data.csv",
        mime="text/csv",
    )

    # Column auto-normalization (maps common aliases → training names)
    COLUMN_ALIASES = {
        "amount": "transaction_amount",
        "amt": "transaction_amount",
        "transaction_amt": "transaction_amount",
        "class": "isfraud",
        "is_fraud": "isfraud",
        "fraud_flag": "isfraud",
        "time": "transaction_datetime",
        "timestamp": "transaction_datetime",
        "date_time": "transaction_datetime",
        "currency": "transaction_currency",
        "category": "transaction_category",
        "auth_method": "authentication_method",
        "velocity": "transaction_velocity",
        "between_transactions_km": "distance_between_transactions_km",
        "distance_km": "distance_between_transactions_km",
        "mins_since_last": "time_since_last_transaction_min",
    }

    CORE_RAW_COLUMNS = {
        "transaction_amount",
        "distance_between_transactions_km",
        "time_since_last_transaction_min",
        "transaction_velocity",
        "transaction_currency",
        "card_type",
        "authentication_method",
        "transaction_category",
    }

    def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
        df2 = df.copy()
        df2.columns = df2.columns.str.strip().str.replace(" ", "_").str.lower()
        df2 = df2.rename(columns={c: COLUMN_ALIASES.get(c, c) for c in df2.columns})

        # last-chance fixes (common alternate names)
        if "distance_between_transactions_km" not in df2.columns:
            for cand in ["between_transactions_km", "distance_km", "distance_between_tx_km", "between_tx_km"]:
                if cand in df2.columns:
                    df2["distance_between_transactions_km"] = pd.to_numeric(df2[cand], errors="coerce")
                    break
        if "time_since_last_transaction_min" not in df2.columns:
            for cand in ["mins_since_last", "minutes_since_last", "time_gap_min"]:
                if cand in df2.columns:
                    df2["time_since_last_transaction_min"] = pd.to_numeric(df2[cand], errors="coerce")
                    break
        return df2

    up = st.file_uploader("Choose a CSV with transactions", type=["csv"])

    if up:
        # Read + normalize columns
        raw = pd.read_csv(up)
        raw = normalize_columns(raw)

        # Diagnostics expander
        with st.expander("🔎 Diagnostics: columns & feature health", expanded=False):
            st.write("Columns after normalization:", list(raw.columns))
            try:
                feats_preview = prepare_for_scoring(raw.copy())
                st.write("Model feature columns (first 5 rows):")
                st.dataframe(feats_preview.head(), use_container_width=True)
                missing_feats = [c for c in FEATURES if c not in feats_preview.columns]
                if missing_feats:
                    st.warning("Missing model features (filled with NaN): " + ", ".join(missing_feats))
                nan_ratio = feats_preview.isna().mean().sort_values(ascending=False)
                st.write("NaN ratio (top 10):")
                st.dataframe(nan_ratio.head(10))
                for col in ["transaction_amount", "amount_per_minute", "distance_time_ratio", "transaction_velocity"]:
                    if col in feats_preview.columns:
                        desc = feats_preview[col].describe(percentiles=[0.5, 0.9, 0.99, 0.999]).to_frame().T
                        st.write(f"Distribution for *{col}*:")
                        st.dataframe(desc)
            except Exception as e:
                st.error("Diagnostics error: " + str(e))
                st.text(traceback.format_exc())

        # Warn if many core columns are missing
        missing = sorted([c for c in CORE_RAW_COLUMNS if c not in raw.columns])
        if missing:
            st.warning(
                "⚠ Some important columns are missing: "
                + ", ".join(missing)
                + ". The app will still attempt scoring, but results may be less accurate.\n"
                "Use the sample CSV to align column names if possible."
            )

        # Save original for other tabs
        st.session_state["uploaded_df"] = raw.copy()

        # Preview after normalization
        st.write("Preview (after auto-normalization):", raw.head())

        # Score
        feats = prepare_for_scoring(raw)
        try:
            probs = model.predict_proba(feats)[:, 1]
        except Exception as e:
            st.error(f"Model scoring error: {e}")
            st.stop()

        # Sanity readouts
        pmin, pmax, pmean = float(np.min(probs)), float(np.max(probs)), float(np.mean(probs))
        st.caption(f"Probabilities — min: {pmin:.3f} | max: {pmax:.3f} | mean: {pmean:.3f}")

        # Suggested threshold helper (90th percentile)
        suggest_thr = float(np.quantile(probs, 0.90))
        st.caption(f"💡 Suggested threshold (90th percentile of current file): {suggest_thr:.2f}")
        if st.button("Use suggested threshold"):
            st.session_state["thr_pending"] = float(max(0.05, min(0.95, suggest_thr)))
            st.experimental_rerun()

        def to_label(p):
            thr = st.session_state["thr_slider"]
            band = st.session_state["band_slider"]
            low, high = max(0.00, thr - band), min(1.00, thr + band)
            if p >= high:
                return "FRAUD"
            if p < low:
                return "LEGIT"
            return "REVIEW"

        decisions = np.array([to_label(p) for p in probs])
        share_fraud = (decisions == "FRAUD").mean()
        share_review = (decisions == "REVIEW").mean()
        share_legit = (decisions == "LEGIT").mean()
        st.caption(
            f"Decision mix — FRAUD: {share_fraud:.1%} | REVIEW: {share_review:.1%} | LEGIT: {share_legit:.1%}"
        )

        if share_fraud > 0.9:
            st.warning(
                "⚠ Most rows classified as FRAUD. Check column-name mapping "
                "(e.g., 'distance_between_transactions_km'), consider 'Use suggested threshold', "
                "or lower the threshold / widen the review band."
            )

        out = raw.copy()
        out["fraud_probability"] = probs
        out["decision"] = decisions

        st.success("Scoring complete.")
        st.dataframe(out.head(30), use_container_width=True)

        st.download_button(
            "⬇ Download scored CSV",
            data=out.to_csv(index=False).encode("utf-8"),
            file_name="scored_transactions.csv",
            mime="text/csv"
        )

    # Sidebar helper text
    st.sidebar.markdown("""
---
🧩 *Dataset Guidelines*
- Prefer the *sample template* to match column names.
- If your dataset uses different names (e.g., Amount instead of transaction_amount),
  the app will *auto-map* some common aliases, but please verify the preview.
- Missing key columns may reduce accuracy.
---
""")

# ------------------------ Tab 2: Dashboard -----------------------
with tab2:
    st.header("Dashboard")
    raw = st.session_state.get("uploaded_df")
    if raw is None:
        st.info("Upload a file in the *Upload & Score* tab.")
    else:
        feats = prepare_for_scoring(raw)
        probs = model.predict_proba(feats)[:, 1]
        thr = st.session_state["thr_slider"]
        band = st.session_state["band_slider"]
        low, high = max(0.00, thr - band), min(1.00, thr + band)
        decisions = pd.Series(
            np.where(probs >= high, "FRAUD", np.where(probs < low, "LEGIT", "REVIEW"))
        )

        # KPIs
        col1, col2, col3 = st.columns(3)
        col1.metric("Total", len(raw))
        col2.metric("Fraud (High)", int((decisions == "FRAUD").sum()))
        col3.metric("Review", int((decisions == "REVIEW").sum()))

        # Probability histogram
        fig = px.histogram(pd.DataFrame({"probability": probs}), x="probability", nbins=30,
                           title="Fraud Probability Distribution")
        st.plotly_chart(fig, use_container_width=True)

        # Example: Decisions by hour (if date/time present)
        temp = raw.copy()
        if "transaction_datetime" in temp.columns:
            dt = pd.to_datetime(temp["transaction_datetime"], errors="coerce")
            temp["hour"] = dt.dt.hour
            temp["decision"] = decisions.values
            fig2 = px.histogram(temp, x="hour", color="decision", barmode="group",
                                title="Decisions by Hour")
            st.plotly_chart(fig2, use_container_width=True)

# ------------------------ Tab 3: Explain -------------------------
with tab3:
    st.header("Explain a Row")
    raw = st.session_state.get("uploaded_df")
    if raw is None:
        st.info("Upload a file first in the *Upload & Score* tab.")
    else:
        idx = st.number_input("Row index", min_value=0, max_value=len(raw) - 1, value=0, step=1)
        row_display = raw.iloc[int(idx)].to_dict()

        feats_row = prepare_for_scoring(pd.DataFrame([row_display]))
        p = model.predict_proba(feats_row)[:, 1][0]

        thr = st.session_state["thr_slider"]
        band = st.session_state["band_slider"]
        low, high = max(0.00, thr - band), min(1.00, thr + band)
        label = "FRAUD" if p >= high else ("LEGIT" if p < low else "REVIEW")

        # Optional: context for LLM (p99 amount based on current upload)
        try:
            feats_all = prepare_for_scoring(raw)
            p99_amount = (
                float(np.nanpercentile(feats_all["transaction_amount"], 99))
                if "transaction_amount" in feats_all.columns
                else None
            )
        except Exception:
            p99_amount = None

        explain_row = {
            **row_display,
            **feats_row.iloc[0].to_dict(),
            "p99_amount": p99_amount,
        }

        st.subheader(f"Decision: {label}  |  Probability: {p:.3f}")
        st.caption("Explanation generated locally (FLAN-T5 if available), otherwise a rule-based fallback.")
        with st.spinner("Generating explanation..."):
            st.write(llm_or_fallback_explain(explain_row, label=label))

        with st.expander("Show raw row"):
            st.json(row_display)