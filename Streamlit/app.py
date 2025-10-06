# app.py
from __future__ import annotations
import os
import re
import base64
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

# Optional ML deps (guarded)
try:
    import tensorflow as tf
    from sklearn.preprocessing import StandardScaler
    SK_OK = True
except Exception as _e:
    tf = None  # type: ignore
    StandardScaler = None  # type: ignore
    SK_OK = False

# =============================================================================
# PATHS / CONSTANTS
# =============================================================================
APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "Mod8-16-8.h5"
TRAIN_CSV = APP_DIR / "train.csv"
LOGO_PATH = APP_DIR / "cube_logo.png"
SAMPLE_CSV = APP_DIR / "data" / "predicted.csv"   # optional, used as an example
COMPARISON_HTML = APP_DIR / "comparison.html"      # embedded view

REQUIRED_COLUMNS = [
    "Orbital Period (days)",
    "Transit Duration (hrs)",
    "Transit Depth (ppm)",
    "Planet Radius (Earth Radii)",
    "Planet Insolation Flux (Earth Flux)",
    "Planet Equilibrium Temperature (Kelvin)",
    "Star Effective Temperature (Kelvin)",
    "Star Radius (solar radii)",
]

# =============================================================================
# PAGE CONFIG & THEME
# =============================================================================
st.set_page_config(
    page_title="The Cube — Exoplanet Visualizer",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Starry backdrop
st.markdown("""
<style>
/* Background */
.stApp {
    background: url("https://i.ibb.co/fkLCh0J/star-bg.jpg");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}
/* Global text color */
.stMarkdown, .stText, h1, h2, h3, h4, h5, h6, p, div {
    color: white !important;
}
/* Core palette */
:root {
  --bg: #0e1117;          /* charcoal */
  --panel: #111827;       /* slate */
  --panel-2: #0f172a;     /* deep slate */
  --muted: #a3a3a3;
  --text: #e5e7eb;
  --border: rgba(255,255,255,0.08);
}
html, body, .stApp { background: var(--bg); color: var(--text); }
.block-container { padding-top: 1.0rem; }

/* Hide sidebar */
[data-testid="stSidebar"]{ display:none !important; }
[data-testid="collapsedControl"]{ display:none !important; }

/* Cards */
.dark-card{
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: 14px;
  padding: 16px 18px;
  box-shadow: 0 6px 16px rgba(0,0,0,.35);
}
.dark-card.tight { padding: 10px 12px; }

/* Hero */
.hero-banner {
    display:flex; align-items:center; justify-content:center;
    padding:28px 18px;
    background:linear-gradient(180deg,#000000 0%, #0b0b0b 100%);
    border-radius: 14px;
    margin: 6px auto 18px; max-width: 1200px;
    box-shadow: 0 8px 40px rgba(2,6,23,0.7);
    position: relative;
}
.hero-banner::before{
    content: ""; position: absolute; left:18px; top:18px; bottom:18px; width: 6px;
    border-radius: 4px; background: linear-gradient(180deg,#ff2d95,#7c3aed); opacity: 0.95;
}
.cube-title {
    margin: 0; color: #ffffff; font-weight: 800; font-size: 72px; letter-spacing: 8px;
    text-transform: uppercase; line-height: 0.95; text-shadow: 0 4px 30px rgba(0,0,0,0.7);
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial;
}
.cube-subtitle {
    margin: 6px 0 0; color: #cbd5e1; font-size: 13px; letter-spacing: 8px;
    text-transform: uppercase; opacity: 0.9;
}

/* Fixed cube logo */
#cube-logo {
  position: fixed; left: 14px; top: 12px; width: 56px; z-index: 9999;
  border-radius: 6px; box-shadow: 0 6px 24px rgba(0,0,0,0.6);
}

/* Hide Streamlit chrome */
header, footer, #MainMenu { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class='hero-banner'>
  <div style='text-align:left; max-width:920px;'>
    <h1 class='cube-title'>The Cube</h1>
    <div class='cube-subtitle'>Exoplanet Analyzer</div>
  </div>
</div>
""", unsafe_allow_html=True)

if LOGO_PATH.is_file():
    try:
        with open(LOGO_PATH, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        st.markdown(f"<img id='cube-logo' src='data:image/png;base64,{b64}' alt='cube'/>", unsafe_allow_html=True)
    except Exception:
        pass

# =============================================================================
# HELPERS
# =============================================================================
def explain_axis(name: str) -> str:
    AXIS_INFO = {
        "period": "Orbital period — time to orbit the star once (days).",
        "koi_period": "Orbital period — days per full orbit around the host star.",
        "trandur": "Transit duration — transit length (hours).",
        "koi_duration": "Transit duration — hours the planet takes to cross the star.",
        "duration": "Transit duration — transit length (hours).",
        "duration_hrs": "Transit duration — transit length (hours).",
        "rp_re": "Planet radius — Earth radii (R⊕).",
        "rade": "Planet radius — Earth radii (R⊕).",
        "koi_prad": "Planet radius — Earth radii (R⊕).",
        "teq": "Equilibrium temperature — blackbody estimate in Kelvin.",
        "teq_k": "Equilibrium temperature — Kelvin.",
        "koi_teq": "Equilibrium temperature — Kelvin.",
        "insol": "Incident flux — relative to Earth (S⊕).",
        "koi_insol": "Incident flux — relative to Earth (S⊕).",
        "teff": "Stellar effective temperature — surface temperature (K).",
        "koi_steff": "Stellar effective temperature — surface temperature (K).",
        "srad": "Stellar radius — solar radii (R☉).",
        "rads": "Stellar radius — solar radii (R☉).",
        "koi_srad": "Stellar radius — solar radii (R☉).",
        "snr": "Signal-to-noise ratio — detection strength.",
        "class": "Label — e.g., confirmed, candidate, or false positive.",
        "confidence_positive": "Model confidence for 'positive' (exoplanet) class (0–1).",
    }
    if not isinstance(name, str):
        return "Feature description unavailable."
    key = name.strip()
    return AXIS_INFO.get(key, AXIS_INFO.get(key.lower(), f"{name.replace('_', ' ').capitalize()} — feature description unavailable."))

def idx_for(columns: List[str], *candidates) -> int:
    cols_lower = [c.lower() for c in columns]
    for cand in candidates:
        if cand and cand.lower() in cols_lower:
            return cols_lower.index(cand.lower())
    return 0

def _normalize_header(s: str) -> str:
    s = s.lower()
    s = re.sub(r"\(.*?\)", "", s)
    s = re.sub(r"[^a-z0-9]+", " ", s).strip()
    return s

def guess_mapping(src_cols: List[str], required_cols: List[str]) -> Dict[str, Optional[str]]:
    norm_src = {_normalize_header(c): c for c in src_cols}
    mapping = {}
    for req in required_cols:
        key = _normalize_header(req)
        if key in norm_src:
            mapping[req] = norm_src[key]
            continue
        hit = next((norm_src[k] for k in norm_src if key in k or k in key), None)
        mapping[req] = hit
    return mapping

def _clean_numeric_block(df: pd.DataFrame) -> pd.DataFrame:
    def to_float(s: pd.Series) -> pd.Series:
        txt = s.astype(str)
        txt = txt.str.replace(",", "", regex=False)
        txt = txt.str.replace(r"[^\d\.\-eE+]", "", regex=True)
        return pd.to_numeric(txt, errors="coerce")
    X = df.apply(to_float)
    if X.isna().any().any():
        bad = X.columns[X.isna().any()].tolist()
        raise ValueError(f"Non-numeric values in: {bad}")
    return X

def make_confirmed_mask_from_series(series: pd.Series, confirmed_values: List[str]) -> pd.Series:
    """Flexible confirmed detector: supports numeric tokens and substrings."""
    tokens = [t.strip() for t in confirmed_values if str(t).strip()]
    # Numeric pass
    if pd.api.types.is_numeric_dtype(series):
        nums = set()
        for v in tokens:
            try:
                nums.add(float(v))
            except Exception:
                pass
        if nums:
            try:
                return series.astype(float).isin(nums)
            except Exception:
                pass
    # String/substring pass
    s = series.astype(str).str.strip().str.lower()
    if not tokens:
        return s.str.contains("confirm", na=False)
    mask = pd.Series(False, index=series.index)
    for tok in tokens:
        mask |= s.str.contains(re.escape(tok.lower()), na=False)
    return mask

# =============================================================================
# MODEL PIPELINE (guarded by availability)
# =============================================================================
@st.cache_resource
def load_model_and_scaler() -> Tuple[Optional[object], Optional[object]]:
    if not SK_OK:
        return None, None
    if not TRAIN_CSV.exists() or not MODEL_PATH.exists():
        return None, None
    X = np.loadtxt(str(TRAIN_CSV), delimiter=",", skiprows=1)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    X = X[:, 1:]
    scaler = StandardScaler().fit(X)
    model = tf.keras.models.load_model(str(MODEL_PATH))
    return model, scaler

def predict_probs_from_features(features_df: pd.DataFrame, model, scaler) -> np.ndarray:
    X = _clean_numeric_block(features_df)
    Xs = scaler.transform(X.values)
    probs = model.predict(Xs, verbose=0).ravel()
    return np.clip(probs, 0.0, 1.0)

def trim_by_percentiles(df: pd.DataFrame, cols: List[str], lo: int, hi: int) -> pd.DataFrame:
    if not cols:
        return df
    out = df
    for c in cols:
        if c not in out.columns or not pd.api.types.is_numeric_dtype(out[c]):
            continue
        qlo, qhi = out[c].quantile(lo/100.0), out[c].quantile(hi/100.0)
        out = out[(out[c] >= qlo) & (out[c] <= qhi)]
    return out

# =============================================================================
# TABS
# =============================================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["About Us", "Project Information", "Visualization", "Model Performance", "Compare Model"]
)

# -----------------------------------------------------------------------------
# TAB 1 — About Us
# -----------------------------------------------------------------------------
with tab1:
    st.markdown("""
        <div class="dark-card">
            <h2 style="margin:.25rem 0">About Us</h2>
            <p style="line-height: 1.2; font-size: 14px;">
                <strong>The Cube</strong><br>
                We are a team exploring exoplanet discovery and habitability.<br>
                The original repo link is here w/ more info —
                <a href="https://github.com/advikvenks/apollo67" target="_blank">GitHub Repo</a>.<br>
                We visualize NASA archive data and compare against our ML model.
            </p>
        </div>
    """, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# TAB 2 — Project Information
# -----------------------------------------------------------------------------
with tab2:
    st.markdown(f"""
        <div class='dark-card'>
            <h2 style='margin:.25rem 0'>Project Information</h2>
            <p><strong>Goal:</strong> Provide an interactive platform for visualizing and confirming exoplanet candidates.</p>
            <ul>
                <li>Supported archives: Kepler, K2, TOI (CSV)</li>
                <li>Features: 2D/3D scatter, curated histograms, compact model-vs-actual summary</li>
                <li>Please upload CSV files in the following format:<br>
                    <span style="font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', 'Courier New', monospace; background:#121622;border:1px solid #1d2230;border-radius:8px;padding:8px;display:block;">
                        {", ".join(REQUIRED_COLUMNS)}
                    </span>
                </li>
            </ul>
            <p class="small-note">Tip: If your headers differ, you can map them in the Visualization tab.</p>
        </div>
    """, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# TAB 3 — Visualization (Upload → Map → Run Model → Download → Explore)
# -----------------------------------------------------------------------------
with tab3:
    st.markdown("<div class='dark-card'><h2 style='margin:.25rem 0'>Visualization</h2>", unsafe_allow_html=True)

    c1, c2 = st.columns([2, 1])
    with c1:
        uploaded = st.file_uploader("Upload a CSV of candidates", type=["csv"])
    with c2:
        decision_thr = st.slider("Decision threshold", 0.0, 1.0, 0.5, 0.01)

    df = None
    scored = False
    user_has_model = SK_OK and TRAIN_CSV.exists() and MODEL_PATH.exists()

    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
        except Exception as e:
            st.error(f"Could not read CSV: {e}")
            st.markdown("</div>", unsafe_allow_html=True)
            st.stop()

        st.write("**Detected columns:**", list(df.columns))

        # Column mapping wizard (only needed if user wants to run the model)
        st.markdown("#### Map your columns to the required features (for scoring)")
        mapping_guess = guess_mapping(df.columns.tolist(), REQUIRED_COLUMNS)
        mapping: Dict[str, str] = {}

        mc1, mc2 = st.columns(2)
        with mc1:
            for req in REQUIRED_COLUMNS[:len(REQUIRED_COLUMNS)//2]:
                opts = ["-- Not present --"] + list(df.columns)
                idx = opts.index(mapping_guess[req]) if mapping_guess.get(req) in opts else 0
                mapping[req] = st.selectbox(f"{req}", opts, index=idx, key=f"viz_map_{req}")
        with mc2:
            for req in REQUIRED_COLUMNS[len(REQUIRED_COLUMNS)//2:]:
                opts = ["-- Not present --"] + list(df.columns)
                idx = opts.index(mapping_guess[req]) if mapping_guess.get(req) in opts else 0
                mapping[req] = st.selectbox(f"{req}", opts, index=idx, key=f"viz_map_{req}")

        can_score = all(v != "-- Not present --" for v in mapping.values())

        run_col1, run_col2 = st.columns([1, 3])
        with run_col1:
            run_model_btn = st.button("Run Model", type="primary", disabled=not (can_score and user_has_model))

        if not user_has_model:
            st.info("Model files not found or ML libraries not installed. You can still visualize any existing `confidence_positive` column if present.")

        # (Optional) local label detection to draw red rings
        st.markdown("---")
        st.markdown("**(Optional)** Provide actual/confirmed labels to show **red outlines** in the plots:")
        label_col_choice = st.selectbox("(Inline) Label column in this CSV", options=["(auto)"] + list(df.columns), index=0, key="viz_label_pick")
        custom_tokens_str = st.text_input(
            "Custom confirmed values (comma-separated, optional)",
            value="1, confirmed, true, yes",
            help="Values that should count as confirmed (e.g., 1/yes/true/confirmed)."
        )
        actuals_uploaded = st.file_uploader("Or upload a separate Actuals CSV (must include a label column)", type=["csv"], key="viz_acts_up")
        actual_df = None
        if actuals_uploaded:
            try:
                actual_df = pd.read_csv(actuals_uploaded)
            except Exception as e:
                st.error(f"Could not read Actuals CSV: {e}")

        # --- Run model only when user clicks the button ---
        if run_model_btn and user_has_model:
            try:
                feat_df = pd.DataFrame({req: df[mapping[req]] for req in REQUIRED_COLUMNS})
                model, scaler = load_model_and_scaler()
                if model is None or scaler is None:
                    raise RuntimeError("Model/scaler not available.")
                probs = predict_probs_from_features(feat_df, model=model, scaler=scaler)
                df = df.copy()
                df["confidence_positive"] = probs
                df["Predicted_Label"] = (df["confidence_positive"] >= decision_thr).astype(int)
                scored = True
                st.success(f"Scored {len(df)} rows. A 'confidence_positive' column was added.")
                st.download_button(
                    "Download scored CSV",
                    df.to_csv(index=False).encode("utf-8"),
                    file_name="scored_candidates.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            except Exception as e:
                st.error(f"Scoring failed: {e}")

        # If user did not run the model, but CSV already has confidence column, continue
        if not scored and "confidence_positive" in df.columns and pd.api.types.is_numeric_dtype(df["confidence_positive"]):
            st.info("Using existing `confidence_positive` column found in the uploaded CSV.")

        # ========== Visualization controls ==========
        if df is not None:
            numeric_cols = df.select_dtypes(include="number").columns.tolist()
            all_cols = df.columns.tolist()

            # Determine confirmed mask (base)
            base_confirmed = pd.Series([False]*len(df), index=df.index)
            tokens = custom_tokens_str.split(",") if custom_tokens_str else []
            # Prefer explicit pick
            if label_col_choice and label_col_choice != "(auto)" and label_col_choice in df.columns:
                try:
                    base_confirmed = make_confirmed_mask_from_series(df[label_col_choice], tokens)
                except Exception:
                    base_confirmed = pd.Series([False]*len(df), index=df.index)
            # Else try external actuals
            elif actual_df is not None and len(actual_df):
                actual_lower = {c.lower(): c for c in actual_df.columns}
                preferred = ["actual_label","koi_disposition","disposition","class","label","status"]
                act_col = next((actual_lower[n] for n in preferred if n in actual_lower), None)
                if act_col is None:
                    st.warning("Could not auto-detect label column in Actuals. Please pick one below.")
                    act_col = st.selectbox("Actuals label column", options=list(actual_df.columns), key="viz_act_pick")
                n = min(len(df), len(actual_df))
                df = df.iloc[:n].copy()
                series = actual_df.iloc[:n][act_col]
                try:
                    base_confirmed = make_confirmed_mask_from_series(series, tokens).reset_index(drop=True)
                    base_confirmed.index = df.index
                except Exception:
                    base_confirmed = pd.Series([False]*len(df), index=df.index)
            else:
                # Fallback: infer from common column names in df
                if df is None or not hasattr(df, "columns"):
                    base_confirmed = pd.Series([], dtype=bool)
                else:
                    lower_map = {c.lower(): c for c in df.columns}
                    guessed = None
                    for n in ["actual_label","koi_disposition","disposition","class","label","status"]:
                        if n in lower_map:
                            guessed = lower_map[n]
                            break
                    if not tokens:
                        tokens = ["1","confirmed","true","yes"]
                    if guessed is not None:
                        try:
                            base_confirmed = make_confirmed_mask_from_series(df[guessed], tokens)
                        except Exception:
                            base_confirmed = pd.Series([False]*len(df), index=df.index)
                    else:
                        base_confirmed = pd.Series([False]*len(df), index=df.index)

            # Axes and scaling/trimming UI
            x_default = idx_for(numeric_cols, "period", "koi_period")
            y_default = idx_for(numeric_cols, "trandur", "koi_duration", "duration", "duration_hrs")
            z_default = idx_for(numeric_cols, "rade", "rp_re", "koi_prad", "teq", "teq_k")

            st.markdown("<div class='dark-card'>", unsafe_allow_html=True)
            sc1, sc2, sc3 = st.columns(3)
            with sc1:
                x_axis = st.selectbox("X-axis", numeric_cols or all_cols, index=min(x_default, max(0, len(numeric_cols)-1)))
            with sc2:
                y_axis = st.selectbox("Y-axis", numeric_cols or all_cols, index=min(y_default, max(0, len(numeric_cols)-1)))
            with sc3:
                z_axis = st.selectbox("3D Z-axis", numeric_cols or all_cols, index=min(z_default, max(0, len(numeric_cols)-1)))

            s4, s5 = st.columns(2)
            with s4:
                lock_2d_equal = st.checkbox("Lock 2D aspect ratio (equal scale)", value=False)
            with s5:
                scaling_mode = st.selectbox(
                    "3D axis scaling",
                    options=["Equal cube (same scale)", "Data-driven (proportional)", "Normalize to 0–1 (for selected axes)"],
                    index=0
                )

            st.markdown("<div class='dark-card tight'>", unsafe_allow_html=True)
            t1, t2, t3 = st.columns(3)
            with t1:
                trim_low = st.slider("Percentile lower bound", 0, 10, 1)
            with t2:
                trim_high = st.slider("Percentile upper bound", 90, 100, 99)
            with t3:
                apply_trim = st.checkbox("Focus on dense region (trim outliers)", value=True)
            st.markdown("</div>", unsafe_allow_html=True)

            with st.expander("Axis explanations", expanded=False):
                st.markdown(
                    f"- **X-axis ({x_axis})** — {explain_axis(x_axis)}\n"
                    f"- **Y-axis ({y_axis})** — {explain_axis(y_axis)}\n"
                    f"- **Z-axis ({z_axis})** — {explain_axis(z_axis)}"
                )
            st.markdown("</div>", unsafe_allow_html=True)

            # Prepare plotting frame
            df_plot = df.copy()
            if apply_trim and pd.api.types.is_numeric_dtype(df_plot.get(x_axis, pd.Series(dtype=float))) \
               and pd.api.types.is_numeric_dtype(df_plot.get(y_axis, pd.Series(dtype=float))) \
               and pd.api.types.is_numeric_dtype(df_plot.get(z_axis, pd.Series(dtype=float))):
                df_plot = trim_by_percentiles(df_plot, [x_axis, y_axis, z_axis], trim_low, trim_high)

            mask_confirmed = base_confirmed.loc[df_plot.index] if isinstance(base_confirmed, pd.Series) else pd.Series([False]*len(df_plot), index=df_plot.index)

            # Ring controls
            st.markdown("<div class='dark-card tight' style='margin-bottom:8px'>", unsafe_allow_html=True)
            ring_size = st.slider("Confirmed ring size", 6, 20, 12)
            ring_width = st.slider("Confirmed ring line width", 1, 6, 3)
            st.markdown("</div>", unsafe_allow_html=True)

            if base_confirmed.any() and not mask_confirmed.any():
                st.warning("Confirmed points exist, but current trimming removed them. Loosen trim bounds.")
            elif not base_confirmed.any():
                st.info("No rows are marked confirmed. Select a label column or upload Actuals.")

            # Confidence buckets (colors)
            has_conf = "confidence_positive" in df_plot.columns and pd.api.types.is_numeric_dtype(df_plot["confidence_positive"])
            LOW_THR, HIGH_THR = 0.25, 0.75
            if has_conf:
                conf = df_plot["confidence_positive"]
                mask_not  = conf < LOW_THR
                mask_cand = (conf >= LOW_THR) & (conf < HIGH_THR)
                mask_plan = conf >= HIGH_THR
            else:
                mask_not  = pd.Series([True]*len(df_plot), index=df_plot.index)
                mask_cand = pd.Series([False]*len(df_plot), index=df_plot.index)
                mask_plan = pd.Series([False]*len(df_plot), index=df_plot.index)

            st.markdown(
                "<div class='dark-card tight' style='margin-bottom:12px'>"
                "<strong>Color rules:</strong> white &lt; 0.25 · Blues 0.25–0.75 · dark blue ≥ 0.75 · red ring = confirmed"
                "</div>",
                unsafe_allow_html=True,
            )

            # ---------- 2D SCATTER ----------
            traces2d = [
                go.Scatter(
                    x=df_plot.loc[mask_not, x_axis],
                    y=df_plot.loc[mask_not, y_axis],
                    mode="markers",
                    name=f"Not exoplanet (< {LOW_THR:.2f})",
                    marker=dict(color="white", size=3.0, line=dict(width=0.3, color="rgba(0,0,0,0.3)")),
                    hovertemplate=f"{x_axis}=%{{x}}<br>{y_axis}=%{{y}}<extra></extra>",
                )
            ]
            if mask_cand.any():
                traces2d.append(
                    go.Scatter(
                        x=df_plot.loc[mask_cand, x_axis],
                        y=df_plot.loc[mask_cand, y_axis],
                        mode="markers",
                        name=f"Candidate [{LOW_THR:.2f}, {HIGH_THR:.2f})",
                        marker=dict(
                            size=3.0,
                            color=df_plot.loc[mask_cand, "confidence_positive"],
                            colorscale="Blues",
                            cmin=LOW_THR, cmax=HIGH_THR, showscale=True,
                            colorbar=dict(title="Confidence+"),
                            line=dict(width=0.3, color="rgba(255,255,255,0.6)"),
                        ),
                        hovertemplate=f"{x_axis}=%{{x}}<br>{y_axis}=%{{y}}<br>conf+=%{{marker.color:.2f}}<extra></extra>",
                    )
                )
            if mask_plan.any():
                traces2d.append(
                    go.Scatter(
                        x=df_plot.loc[mask_plan, x_axis],
                        y=df_plot.loc[mask_plan, y_axis],
                        mode="markers",
                        name=f"Exoplanet (≥ {HIGH_THR:.2f})",
                        marker=dict(size=3.0, color="#1e3a8a", line=dict(width=0.3, color="rgba(255,255,255,0.6)")),
                        hovertemplate=f"{x_axis}=%{{x}}<br>{y_axis}=%{{y}}<extra>exoplanet</extra>",
                    )
                )
            # Red rings LAST for visibility
            if mask_confirmed.any():
                traces2d.append(
                    go.Scatter(
                        x=df_plot.loc[mask_confirmed, x_axis],
                        y=df_plot.loc[mask_confirmed, y_axis],
                        mode="markers",
                        name="Confirmed (ring)",
                        marker=dict(
                            symbol="circle-open",
                            size=ring_size,
                            color="red",
                            line=dict(width=ring_width, color="red"),
                        ),
                        hovertemplate=f"{x_axis}=%{{x}}<br>{y_axis}=%{{y}}<extra>confirmed</extra>",
                    )
                )

            fig2d = go.Figure(traces2d)
            fig2d.update_layout(
                template="plotly_dark",
                title=f"{x_axis} vs {y_axis}",
                margin=dict(l=40, r=20, t=40, b=40),
                height=520,
                plot_bgcolor="#0f172a",
                paper_bgcolor="#0e1117",
                font=dict(color="#e5e7eb"),
            )
            if lock_2d_equal:
                fig2d.update_layout(xaxis=dict(scaleanchor="y", scaleratio=1))
            st.plotly_chart(fig2d, use_container_width=True, config=dict(displaylogo=False))

            # ---------- 3D SCATTER ----------
            def minmax(series: pd.Series) -> pd.Series:
                vmin, vmax = series.min(), series.max()
                if pd.isna(vmin) or pd.isna(vmax) or vmax == vmin:
                    return pd.Series([0.5]*len(series), index=series.index)
                return (series - vmin) / (vmax - vmin)

            if scaling_mode == "Normalize to 0–1 (for selected axes)":
                X3 = minmax(pd.to_numeric(df_plot[x_axis], errors="coerce"))
                Y3 = minmax(pd.to_numeric(df_plot[y_axis], errors="coerce"))
                Z3 = minmax(pd.to_numeric(df_plot[z_axis], errors="coerce"))
                scene_aspect = dict(aspectmode="cube")
            else:
                X3 = df_plot[x_axis]; Y3 = df_plot[y_axis]; Z3 = df_plot[z_axis]
                scene_aspect = dict(aspectmode="cube" if scaling_mode.startswith("Equal") else "data")

            traces3d = [
                go.Scatter3d(
                    x=X3[mask_not], y=Y3[mask_not], z=Z3[mask_not],
                    mode="markers",
                    name=f"Not exoplanet (< {LOW_THR:.2f})",
                    marker=dict(color="white", size=2.0, line=dict(width=0.25)),
                    hovertemplate=f"{x_axis}=%{{x}}<br>{y_axis}=%{{y}}<br>{z_axis}=%{{z}}<extra></extra>",
                )
            ]
            if mask_cand.any():
                traces3d.append(
                    go.Scatter3d(
                        x=X3[mask_cand], y=Y3[mask_cand], z=Z3[mask_cand],
                        mode="markers",
                        name=f"Candidate [{LOW_THR:.2f}, {HIGH_THR:.2f})",
                        marker=dict(
                            size=2.2,
                            color=df_plot.loc[mask_cand, "confidence_positive"],
                            colorscale="Blues",
                            cmin=LOW_THR, cmax=HIGH_THR, showscale=True,
                            colorbar=dict(title="Confidence+"),
                            line=dict(width=0.25),
                        ),
                        hovertemplate=f"{x_axis}=%{{x}}<br>{y_axis}=%{{y}}<br>{z_axis}=%{{z}}<br>conf+=%{{marker.color:.2f}}<extra></extra>",
                    )
                )
            if mask_plan.any():
                traces3d.append(
                    go.Scatter3d(
                        x=X3[mask_plan], y=Y3[mask_plan], z=Z3[mask_plan],
                        mode="markers",
                        name=f"Exoplanet (≥ {HIGH_THR:.2f})",
                        marker=dict(size=2.4, color="#1e3a8a", line=dict(width=0.35)),
                        hovertemplate=f"{x_axis}=%{{x}}<br>{y_axis}=%{{y}}<br>{z_axis}=%{{z}}<extra>exoplanet</extra>",
                    )
                )
            if mask_confirmed.any():
                traces3d.append(
                    go.Scatter3d(
                        x=X3[mask_confirmed], y=Y3[mask_confirmed], z=Z3[mask_confirmed],
                        mode="markers",
                        name="Confirmed (ring)",
                        marker=dict(
                            size=max(2, ring_size - 4),
                            color="rgba(0,0,0,0)",
                            line=dict(width=ring_width + 1, color="red"),
                        ),
                        hovertemplate=f"{x_axis}=%{{x}}<br>{y_axis}=%{{y}}<br>{z_axis}=%{{z}}<extra>confirmed</extra>",
                    )
                )

            fig3d = go.Figure(traces3d)
            fig3d.update_layout(
                template="plotly_dark",
                title=f"3D: {x_axis}, {y_axis}, {z_axis}",
                margin=dict(l=0, r=0, t=40, b=0),
                scene=dict(
                    xaxis=dict(title=x_axis, showspikes=True, spikethickness=1, showbackground=True,
                               backgroundcolor="rgba(30,41,59,0.5)", gridcolor="rgba(200,200,200,0.12)"),
                    yaxis=dict(title=y_axis, showspikes=True, spikethickness=1, showbackground=True,
                               backgroundcolor="rgba(30,41,59,0.5)", gridcolor="rgba(200,200,200,0.12)"),
                    zaxis=dict(title=z_axis, showspikes=True, spikethickness=1, showbackground=True,
                               backgroundcolor="rgba(30,41,59,0.5)", gridcolor="rgba(200,200,200,0.12)"),
                    **scene_aspect,
                    dragmode="orbit",
                    camera=dict(eye=dict(x=1.8, y=1.8, z=1.8)),
                ),
                uirevision="keep",
                height=860,
                paper_bgcolor="#0e1117",
                font=dict(color="#e5e7eb"),
            )
            st.plotly_chart(fig3d, use_container_width=True, config=dict(displaylogo=False))

            # ---------- Distributions ----------
            st.markdown("<h3>Distributions</h3>", unsafe_allow_html=True)
            d1, d2, d3 = st.columns([1, 1, 1])

            with d1:
                fig_x = px.histogram(df_plot, x=x_axis, nbins=30, title=f"Distribution: {x_axis}", template="plotly_dark")
                fig_x.update_layout(height=280, paper_bgcolor="#0e1117", plot_bgcolor="#0f172a", font_color="#e5e7eb")
                fig_x.update_traces(marker_line_width=0)
                st.plotly_chart(fig_x, use_container_width=True, config=dict(displaylogo=False))

            with d2:
                fig_y = px.histogram(df_plot, x=y_axis, nbins=30, title=f"Distribution: {y_axis}", template="plotly_dark")
                fig_y.update_layout(height=280, paper_bgcolor="#0e1117", plot_bgcolor="#0f172a", font_color="#e5e7eb")
                fig_y.update_traces(marker_line_width=0)
                st.plotly_chart(fig_y, use_container_width=True, config=dict(displaylogo=False))

            with d3:
                actual_confirmed = mask_confirmed.sum()
                actual_other = len(df_plot) - actual_confirmed
                if "confidence_positive" in df_plot.columns and pd.api.types.is_numeric_dtype(df_plot["confidence_positive"]):
                    pred_exo = int((df_plot["confidence_positive"] >= 0.75).sum())
                    pred_other = len(df_plot) - pred_exo
                else:
                    pred_exo, pred_other = 0, len(df_plot)

                bar_df = pd.DataFrame({
                    "category": ["Actual Confirmed", "Actual Not", "Pred Exoplanet", "Pred Not"],
                    "count":    [actual_confirmed,   actual_other,  pred_exo,        pred_other]
                })
                fig_bar = px.bar(bar_df, x="category", y="count", title="Predicted vs Actual (counts)", template="plotly_dark")
                fig_bar.update_layout(height=280, paper_bgcolor="#0e1117", plot_bgcolor="#0f172a",
                                      font_color="#e5e7eb", xaxis_title="", yaxis_title="")
                st.plotly_chart(fig_bar, use_container_width=True, config=dict(displaylogo=False))

    else:
        st.info("Upload a CSV to begin.")
    st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# TAB 4 — Model Performance (images if present)
# -----------------------------------------------------------------------------
with tab4:
    st.markdown("<div class='dark-card'><h2 style='margin:.25rem 0'>Model Performance</h2>", unsafe_allow_html=True)
    cols = st.columns(2)
    paths = [("Accuracy.png", "Accuracy"), ("Model Loss.png", "Model Loss")]
    for (p, title), col in zip(paths, cols):
        with col:
            if (APP_DIR / p).exists():
                st.markdown(f"**{title}**")
                st.image(str(APP_DIR / p), use_container_width=True)
            else:
                st.warning(f"Could not find `{p}` in the app folder.")
    st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# TAB 5 — Compare Model (embed + inline comparison)
# -----------------------------------------------------------------------------
with tab5:
    st.markdown("<div class='dark-card'><h2 style='margin:.25rem 0'>Compare Model</h2>", unsafe_allow_html=True)
    st.caption("Explore the HTML comparison view below or use the inline comparison to overlay red confirmed rings and gradients.")
    # Embedded HTML
    try:
        if COMPARISON_HTML.exists():
            components.html(COMPARISON_HTML.read_text(encoding="utf-8"), height=1000, scrolling=True)
        else:
            st.info("`comparison.html` not found. Place it next to app.py.")
    except Exception as e:
        st.error(f"Could not load comparison.html: {e}")

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<h3>Inline Comparison</h3>", unsafe_allow_html=True)

    left, right = st.columns(2)
    with left:
        preds_file = st.file_uploader("Upload predictions or candidates CSV", type=["csv"], key="cmp_preds_up")
    with right:
        acts_file = st.file_uploader("Upload Actuals CSV (must contain a label column)", type=["csv"], key="cmp_acts_up")

    if preds_file is not None:
        try:
            dfp = pd.read_csv(preds_file)
        except Exception as e:
            st.error(f"Could not read predictions/candidates CSV: {e}")
            dfp = None

        if dfp is not None:
            # Add predictions if missing
            if "confidence_positive" not in dfp.columns and user_has_model:
                st.info("No `confidence_positive` column found. Map columns to score inline.")
                mapping_guess = guess_mapping(dfp.columns.tolist(), REQUIRED_COLUMNS)
                m1, m2 = st.columns(2)
                fmap: Dict[str, str] = {}
                with m1:
                    for req in REQUIRED_COLUMNS[:len(REQUIRED_COLUMNS)//2]:
                        opts = ["-- Not present --"] + list(dfp.columns)
                        idx = opts.index(mapping_guess[req]) if mapping_guess.get(req) in opts else 0
                        fmap[req] = st.selectbox(f"{req}", opts, index=idx, key=f"cmp_map_{req}")
                with m2:
                    for req in REQUIRED_COLUMNS[len(REQUIRED_COLUMNS)//2:]:
                        opts = ["-- Not present --"] + list(dfp.columns)
                        idx = opts.index(mapping_guess[req]) if mapping_guess.get(req) in opts else 0
                        fmap[req] = st.selectbox(f"{req}", opts, index=idx, key=f"cmp_map_{req}")

                can_inline_score = all(v != "-- Not present --" for v in fmap.values())
                if st.button("Score predictions (inline)", type="primary", disabled=not (can_inline_score and user_has_model)):
                    try:
                        feat_df2 = pd.DataFrame({req: dfp[fmap[req]] for req in REQUIRED_COLUMNS})
                        model, scaler = load_model_and_scaler()
                        if model is None or scaler is None:
                            raise RuntimeError("Model/scaler not available.")
                        dfp["confidence_positive"] = predict_probs_from_features(feat_df2, model=model, scaler=scaler)
                        st.success("Predictions computed.")
                    except Exception as e:
                        st.error(f"Could not compute predictions: {e}")
            elif "confidence_positive" not in dfp.columns and not user_has_model:
                st.info("No `confidence_positive` column and model not available. Upload a scored CSV or provide model files.")

            # Axes + trim
            numeric_cols = dfp.select_dtypes(include="number").columns.tolist()
            if not numeric_cols:
                st.error("No numeric columns to plot in predictions/candidates CSV.")
            else:
                cx = st.selectbox("X-axis", numeric_cols, index=idx_for(numeric_cols, "period","koi_period"), key="cmp_x")
                cy = st.selectbox("Y-axis", numeric_cols, index=idx_for(numeric_cols, "trandur","koi_duration","duration","duration_hrs"), key="cmp_y")
                cz = st.selectbox("3D Z-axis", numeric_cols, index=idx_for(numeric_cols, "rade","rp_re","koi_prad","teq","teq_k"), key="cmp_z")

                cl, cr = st.columns(2)
                with cl:
                    cmp_lo = st.slider("Percentile lower bound", 0, 10, 1, key="cmp_lo")
                with cr:
                    cmp_hi = st.slider("Percentile upper bound", 90, 100, 99, key="cmp_hi")

                lock_equal = st.checkbox("Lock 2D aspect ratio (equal)", value=False, key="cmp_lock")

                # Confirmed mask from Actuals CSV
                mask_conf_base = pd.Series([False]*len(dfp), index=dfp.index)
                custom_tokens_cmp = "1, confirmed, true, yes"
                if acts_file is not None:
                    try:
                        dfa = pd.read_csv(acts_file)
                        actual_lower = {c.lower(): c for c in dfa.columns}
                        preferred = ["actual_label","koi_disposition","disposition","class","label","status"]
                        act_col = next((actual_lower[n] for n in preferred if n in actual_lower), None)
                        if act_col is None:
                            act_col = st.selectbox("Label column in Actuals CSV", options=list(dfa.columns), key="cmp_lab_sel")
                        token_in = st.text_input(
                            "Custom confirmed values (comma-separated, optional) for Actuals",
                            value=custom_tokens_cmp,
                            key="cmp_tokens"
                        )
                        tokens_cmp = token_in.split(",") if token_in else custom_tokens_cmp.split(",")

                        n = min(len(dfp), len(dfa))
                        dfp = dfp.iloc[:n].copy()
                        dfa = dfa.iloc[:n].copy()

                        mask_conf_base = make_confirmed_mask_from_series(dfa[act_col], tokens_cmp).reset_index(drop=True)
                        mask_conf_base.index = dfp.index
                    except Exception as e:
                        st.error(f"Could not build confirmed mask from Actuals: {e}")

                # Trim frame
                def _trim(frame, cols, lo, hi):
                    out = frame
                    for c in cols:
                        if c in out and pd.api.types.is_numeric_dtype(out[c]):
                            qlo, qhi = out[c].quantile(lo/100.0), out[c].quantile(hi/100.0)
                            out = out[(out[c] >= qlo) & (out[c] <= qhi)]
                    return out

                dfp_plot = _trim(dfp, [cx, cy, cz], cmp_lo, cmp_hi)
                mask_conf = mask_conf_base.loc[dfp_plot.index]

                # Ring sliders
                st.markdown("<div class='dark-card tight' style='margin-bottom:8px'>", unsafe_allow_html=True)
                ring_size2 = st.slider("Confirmed ring size", 6, 20, 12, key="cmp_ring_size")
                ring_width2 = st.slider("Confirmed ring line width", 1, 6, 3, key="cmp_ring_width")
                st.markdown("</div>", unsafe_allow_html=True)

                if mask_conf_base.any() and not mask_conf.any():
                    st.warning("Confirmed points exist, but current trimming removed them.")

                # Confidence buckets
                has_conf2 = "confidence_positive" in dfp_plot.columns and pd.api.types.is_numeric_dtype(dfp_plot["confidence_positive"])
                LOW_THR, HIGH_THR = 0.25, 0.75
                if has_conf2:
                    conf2 = dfp_plot["confidence_positive"]
                    m_not  = conf2 < LOW_THR
                    m_cand = (conf2 >= LOW_THR) & (conf2 < HIGH_THR)
                    m_plan = conf2 >= HIGH_THR
                else:
                    m_not  = pd.Series([True]*len(dfp_plot), index=dfp_plot.index)
                    m_cand = pd.Series([False]*len(dfp_plot), index=dfp_plot.index)
                    m_plan = pd.Series([False]*len(dfp_plot), index=dfp_plot.index)

                # 2D
                t2 = [
                    go.Scatter(
                        x=dfp_plot.loc[m_not, cx], y=dfp_plot.loc[m_not, cy],
                        mode="markers", name=f"Not exoplanet (< {LOW_THR:.2f})",
                        marker=dict(color="white", size=3.0, line=dict(width=0.3, color="rgba(0,0,0,0.3)")),
                    )
                ]
                if m_cand.any():
                    t2.append(
                        go.Scatter(
                            x=dfp_plot.loc[m_cand, cx], y=dfp_plot.loc[m_cand, cy],
                            mode="markers", name=f"Candidate [{LOW_THR:.2f}, {HIGH_THR:.2f})",
                            marker=dict(
                                size=3.0,
                                color=dfp_plot.loc[m_cand, "confidence_positive"],
                                colorscale="Blues",
                                cmin=LOW_THR, cmax=HIGH_THR, showscale=True,
                                colorbar=dict(title="Confidence+"),
                                line=dict(width=0.3, color="rgba(255,255,255,0.6)"),
                            ),
                        )
                    )
                if m_plan.any():
                    t2.append(
                        go.Scatter(
                            x=dfp_plot.loc[m_plan, cx], y=dfp_plot.loc[m_plan, cy],
                            mode="markers", name=f"Exoplanet (≥ {HIGH_THR:.2f})",
                            marker=dict(size=3.0, color="#1e3a8a", line=dict(width=0.3, color="rgba(255,255,255,0.6)")),
                        )
                    )
                if mask_conf.any():
                    t2.append(
                        go.Scatter(
                            x=dfp_plot.loc[mask_conf, cx], y=dfp_plot.loc[mask_conf, cy],
                            mode="markers", name="Confirmed (ring)",
                            marker=dict(symbol="circle-open", size=ring_size2, color="red", line=dict(width=ring_width2, color="red")),
                        )
                    )

                fig_cmp_2d = go.Figure(t2)
                fig_cmp_2d.update_layout(
                    template="plotly_dark",
                    title=f"{cx} vs {cy}",
                    margin=dict(l=40, r=20, t=40, b=40),
                    height=520,
                    plot_bgcolor="#0f172a",
                    paper_bgcolor="#0e1117",
                    font=dict(color="#e5e7eb"),
                )
                if lock_equal:
                    fig_cmp_2d.update_layout(xaxis=dict(scaleanchor="y", scaleratio=1))
                st.plotly_chart(fig_cmp_2d, use_container_width=True, config=dict(displaylogo=False))

                # 3D
                X3, Y3, Z3 = dfp_plot[cx], dfp_plot[cy], dfp_plot[cz]
                t3 = [
                    go.Scatter3d(
                        x=X3[m_not], y=Y3[m_not], z=Z3[m_not],
                        mode="markers", name=f"Not exoplanet (< {LOW_THR:.2f})",
                        marker=dict(color="white", size=2.0, line=dict(width=0.25)),
                    )
                ]
                if m_cand.any():
                    t3.append(
                        go.Scatter3d(
                            x=X3[m_cand], y=Y3[m_cand], z=Z3[m_cand],
                            mode="markers", name=f"Candidate [{LOW_THR:.2f}, {HIGH_THR:.2f})",
                            marker=dict(
                                size=2.2,
                                color=dfp_plot.loc[m_cand, "confidence_positive"],
                                colorscale="Blues",
                                cmin=LOW_THR, cmax=HIGH_THR, showscale=True,
                                colorbar=dict(title="Confidence+"),
                                line=dict(width=0.25),
                            ),
                        )
                    )
                if m_plan.any():
                    t3.append(
                        go.Scatter3d(
                            x=X3[m_plan], y=Y3[m_plan], z=Z3[m_plan],
                            mode="markers", name=f"Exoplanet (≥ {HIGH_THR:.2f})",
                            marker=dict(size=2.4, color="#1e3a8a", line=dict(width=0.35)),
                        )
                    )
                if mask_conf.any():
                    t3.append(
                        go.Scatter3d(
                            x=X3[mask_conf], y=Y3[mask_conf], z=Z3[mask_conf],
                            mode="markers", name="Confirmed (ring)",
                            marker=dict(size=max(2, ring_size2-4), color="rgba(0,0,0,0)", line=dict(width=ring_width2+1, color="red")),
                        )
                    )

                fig_cmp_3d = go.Figure(t3)
                fig_cmp_3d.update_layout(
                    template="plotly_dark",
                    title=f"3D: {cx}, {cy}, {cz}",
                    margin=dict(l=0, r=0, t=40, b=0),
                    scene=dict(
                        xaxis=dict(title=cx),
                        yaxis=dict(title=cy),
                        zaxis=dict(title=cz),
                        aspectmode="cube",
                        camera=dict(eye=dict(x=1.8, y=1.8, z=1.8)),
                    ),
                    height=720,
                    paper_bgcolor="#0e1117",
                    font=dict(color="#e5e7eb"),
                )
                st.plotly_chart(fig_cmp_3d, use_container_width=True, config=dict(displaylogo=False))
    st.markdown("</div>", unsafe_allow_html=True)