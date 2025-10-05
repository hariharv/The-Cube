<<<<<<< HEAD
import base64
import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

# ---------- Repo-aware paths ----------
REPO_ROOT = Path(__file__).resolve().parent.parent
ART_DIR = REPO_ROOT / "artifacts"  # optional: model runs live here

def list_runs():
    if not ART_DIR.exists():
        return []
    return sorted(
        [p for p in ART_DIR.iterdir() if p.is_dir() and (p / "predictions.jsonl").exists()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

def load_predictions(run_dir: Path) -> pd.DataFrame:
    """Load predictions.jsonl and flatten feature dicts -> columns."""
    path = run_dir / "predictions.jsonl"
    if not path.exists():
        return pd.DataFrame()
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not records:
        return pd.DataFrame()
    df = pd.json_normalize(records, sep=".")
    df.columns = [c.replace("features.", "") for c in df.columns]
    for col in ["id", "score", "label_pred", "label_true", "candidate_name", "timestamp"]:
        if col not in df.columns:
            df[col] = None
    return df

def load_metrics(run_dir: Path) -> dict:
    path = run_dir / "metrics.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))

# ---------- Paths / assets ----------
APP_DIR = Path(__file__).resolve().parent
LOGO_PATH = APP_DIR / "cube_logo.png"
SAMPLE_CSV = APP_DIR / "data" / "predicted.csv"  # bundled dataset option

# ---------- Page config ----------
st.set_page_config(
    page_title="The Cube — Exoplanet Visualizer",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------- Global CSS (clean, organized dark theme) ----------
st.markdown(
    """
<style>
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

/* Hide sidebar and toggle */
[data-testid="stSidebar"]{ display:none !important; }
[data-testid="collapsedControl"]{ display:none !important; }

/* Cards */
.dark-card {
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: 14px;
  padding: 14px 16px;
}
.dark-card.tight { padding: 10px 12px; }

/* Centered hero */
.hero {
  text-align: center;
  padding: 8px 12px 18px;
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: 14px;
}
.hero h1 {
  margin: 0;
  font-size: 34px;
  line-height: 1.25;
  color: #f3f4f6;
}
.hero p {
  margin: 8px auto 0;
  max-width: 900px;
  color: #cbd5e1;
  font-size: 16px;
}

/* --- Large banner style for THE CUBE --- */
.hero-banner {
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 28px 18px;
    background: linear-gradient(180deg, #000000 0%, #0b0b0b 100%);
    border-radius: 14px;
    margin: 6px auto 18px;
    max-width: 1200px;
    box-shadow: 0 8px 40px rgba(2,6,23,0.7);
    position: relative;
}
.hero-banner::before{
    content: "";
    position: absolute;
    left: 18px;
    top: 18px;
    bottom: 18px;
    width: 6px;
    border-radius: 4px;
    background: linear-gradient(180deg,#ff2d95,#7c3aed);
    opacity: 0.95;
}
.cube-title {
    margin: 0;
    color: #ffffff;
    font-weight: 800;
    font-size: 72px;
    letter-spacing: 8px;
    text-transform: uppercase;
    line-height: 0.95;
    text-shadow: 0 4px 30px rgba(0,0,0,0.7);
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial;
}
.cube-subtitle {
    margin: 6px 0 0;
    color: #cbd5e1;
    font-size: 13px;
    letter-spacing: 8px;
    text-transform: uppercase;
    opacity: 0.9;
}

.hero-banner .hero-info {
    margin-top: 8px;
    color: #cbd5e1;
    max-width: 980px;
    font-size: 14px;
}

/* Fixed cube logo */
#cube-logo {
  position: fixed; left: 14px; top: 12px; width: 56px; z-index: 9999;
  border-radius: 6px; box-shadow: 0 6px 24px rgba(0,0,0,0.6);
}

/* Hide default chrome for a cleaner look */
header, footer, #MainMenu { visibility: hidden; }
</style>
""",
    unsafe_allow_html=True,
)

# ---------- Axis explanations ----------
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
def explain_axis(name: str) -> str:
    if not isinstance(name, str):
        return "Feature description unavailable."
    key = name.strip()
    return AXIS_INFO.get(key, AXIS_INFO.get(key.lower(), f"{name.replace('_', ' ').capitalize()} — feature description unavailable."))

def idx_for(columns, *candidates) -> int:
    cols_lower = [c.lower() for c in columns]
    for cand in candidates:
        if cand and cand.lower() in cols_lower:
            return cols_lower.index(cand.lower())
    return 0

# ---------- Hero (centered title + description) ----------
st.markdown(
        """
<div class='hero-banner'>
    <div style='text-align:left; max-width:920px;'>
        <h1 class='cube-title'>The Cube</h1>
        <div class='cube-subtitle'>Exoplanet Analyzer</div>
        
    
</div>
""",
        unsafe_allow_html=True,
)

# ---------- Fixed cube logo ----------
if LOGO_PATH.is_file():
    try:
        with open(LOGO_PATH, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
        st.markdown(f"<img id='cube-logo' src='data:image/png;base64,{b64}' alt='cube'/>", unsafe_allow_html=True)
    except Exception:
        pass

# ---------- Tabs ----------
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["About Us", "Project Information", "Visualization", "Model Performance", "Compare Model"]
)

with tab1:
    st.markdown(
        "<div class='dark-card'><h2 style='margin:.25rem 0'>About Us</h2>"
        "<p><strong>The Cube</strong> — a team exploring exoplanet discovery and habitability. "
        "We visualize NASA archive data and compare against our ML model.</p></div>",
        unsafe_allow_html=True,
    )

with tab2:
    st.markdown(
        "<div class='dark-card'><h2 style='margin:.25rem 0'>Project Information</h2>"
        "<p><strong>Goal:</strong> Provide an interactive platform for visualizing and confirming exoplanet candidates.</p>"
        "<ul><li>Supported archives: Kepler, K2, TOI (CSV)</li>"
        "<li>Features: 2D/3D scatter, curated histograms, compact model-vs-actual summary</li></ul></div>",
        unsafe_allow_html=True,
    )

# ----------------------- Visualization (Sample OR Upload) -----------------------
def trim_by_percentiles(df: pd.DataFrame, cols: list[str], lo: int, hi: int) -> pd.DataFrame:
    """Return filtered copy keeping only rows within [lo, hi] percentile for each numeric col in cols."""
    if not cols:
        return df
    out = df
    for c in cols:
        if c not in out.columns or not pd.api.types.is_numeric_dtype(out[c]):
            continue
        qlo, qhi = out[c].quantile(lo/100.0), out[c].quantile(hi/100.0)
        out = out[(out[c] >= qlo) & (out[c] <= qhi)]
    return out

with tab3:
    st.markdown("<h2>Visualization</h2>", unsafe_allow_html=True)

    source = st.radio(
        "Choose data source",
        options=["Sample data (recommended)", "Upload CSV"],
        index=0,
        horizontal=True,
        help="Use the built-in sample, or upload your own CSV with similar columns.",
    )

    df = None
    if source == "Sample data (recommended)":
        if not SAMPLE_CSV.is_file():
            st.error(f"Sample file not found: {SAMPLE_CSV}. Put a CSV at 'Streamlit/data/predicted.csv'.")
        else:
            df = pd.read_csv(SAMPLE_CSV, comment="#", skip_blank_lines=True)
    else:
        uploaded = st.file_uploader("Upload CSV file", type=["csv"])
        if uploaded:
            df = pd.read_csv(uploaded, comment="#", skip_blank_lines=True)
        else:
            st.info("Upload a CSV to continue.")

    if df is not None:
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        all_cols = df.columns.tolist()

        # Confirmed (ring) detection
        lower_map = {c.lower(): c for c in all_cols}
        preferred = ["actual_label", "koi_disposition", "disposition", "class", "label", "status"]
        label_col = next((lower_map[n] for n in preferred if n in lower_map), None)

        if label_col is not None and pd.api.types.is_numeric_dtype(df[label_col]):
            base_confirmed = (df[label_col] == 1)
        elif label_col is not None:
            base_confirmed = df[label_col].astype(str).str.lower().str.contains("confirm")
        else:
            base_confirmed = pd.Series([False] * len(df), index=df.index)

        # --- Axis controls + scaling & trimming options ---
        x_default = idx_for(numeric_cols, "period", "koi_period")
        y_default = idx_for(numeric_cols, "tranDur", "trandur", "koi_duration", "duration", "duration_hrs")
        z_default = idx_for(numeric_cols, "rade", "rp_re", "koi_prad", "teq", "teq_k")

        st.markdown("<div class='dark-card'>", unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            x_axis = st.selectbox("X-axis", numeric_cols, index=x_default, help="Select X feature")
        with c2:
            y_axis = st.selectbox("Y-axis", numeric_cols, index=y_default, help="Select Y feature")
        with c3:
            z_axis = st.selectbox("3D Z-axis", numeric_cols, index=z_default, help="Select Z feature")

        c4, c5 = st.columns([1,1])
        with c4:
            lock_2d_equal = st.checkbox(
                "Lock 2D aspect ratio (equal scale)",
                value=False,
                help="Keeps the same scale for X and Y in 2D to avoid distortion."
            )
        with c5:
            scaling_mode = st.selectbox(
                "3D axis scaling",
                options=["Equal cube (same scale)", "Data-driven (proportional)", "Normalize to 0–1 (for selected axes)"],
                index=0,
                help="Choose how the 3D axes are scaled."
            )

        # --- Trim outliers controls ---
        st.markdown("<div class='dark-card tight'>", unsafe_allow_html=True)
        trim_c1, trim_c2, trim_c3 = st.columns([1,1,1])
        with trim_c1:
            trim_low = st.slider("Percentile lower bound", min_value=0, max_value=10, value=1, step=1,
                                 help="Points below this percentile (per axis) are removed.")
        with trim_c2:
            trim_high = st.slider("Percentile upper bound", min_value=90, max_value=100, value=99, step=1,
                                  help="Points above this percentile (per axis) are removed.")
        with trim_c3:
            apply_trim = st.checkbox("Focus on dense region (trim outliers)", value=True)
        st.markdown("</div>", unsafe_allow_html=True)

        with st.expander("Axis explanations", expanded=False):
            st.markdown(
                f"- **X-axis ({x_axis})** — {explain_axis(x_axis)}\n"
                f"- **Y-axis ({y_axis})** — {explain_axis(y_axis)}\n"
                f"- **Z-axis ({z_axis})** — {explain_axis(z_axis)}"
            )
        st.markdown("</div>", unsafe_allow_html=True)

        # ---------- Prepare working (trimmed) DataFrame ----------
        df_plot = df.copy()
        if apply_trim:
            df_plot = trim_by_percentiles(df_plot, [x_axis, y_axis, z_axis], trim_low, trim_high)

        # Recompute masks on trimmed frame
        if label_col is not None and pd.api.types.is_numeric_dtype(df_plot[label_col]):
            mask_confirmed = (df_plot[label_col] == 1)
        elif label_col is not None:
            mask_confirmed = df_plot[label_col].astype(str).str.lower().str.contains("confirm")
        else:
            mask_confirmed = pd.Series([False] * len(df_plot), index=df_plot.index)

        has_conf = "confidence_positive" in df_plot.columns and pd.api.types.is_numeric_dtype(df_plot["confidence_positive"])
        LOW_THR, HIGH_THR = 0.25, 0.75
        if has_conf:
            conf = df_plot["confidence_positive"]
            mask_not  = conf < LOW_THR
            mask_cand = (conf >= LOW_THR) & (conf < HIGH_THR)
            mask_plan = conf >= HIGH_THR
        else:
            mask_not  = pd.Series([True] * len(df_plot), index=df_plot.index)
            mask_cand = pd.Series([False] * len(df_plot), index=df_plot.index)
            mask_plan = pd.Series([False] * len(df_plot), index=df_plot.index)

        st.markdown(
            "<div class='dark-card tight' style='margin-bottom:12px'>"
            "<strong>Color rules:</strong> white &lt; 0.25 · Blues 0.25–0.75 · dark blue ≥ 0.75 · red ring = confirmed"
            "</div>",
            unsafe_allow_html=True,
        )

        # -------------------- 2D scatter --------------------
        traces = [
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
            traces.append(
                go.Scatter(
                    x=df_plot.loc[mask_cand, x_axis],
                    y=df_plot.loc[mask_cand, y_axis],
                    mode="markers",
                    name=f"Candidate [{LOW_THR:.2f}, {HIGH_THR:.2f})",
                    marker=dict(
                        size=3.0,
                        color=df_plot.loc[mask_cand, "confidence_positive"],
                        colorscale="Blues",
                        cmin=LOW_THR,
                        cmax=HIGH_THR,
                        showscale=True,
                        colorbar=dict(title="Confidence+"),
                        line=dict(width=0.3, color="rgba(255,255,255,0.6)"),
                    ),
                    hovertemplate=f"{x_axis}=%{{x}}<br>{y_axis}=%{{y}}<br>conf+=%{{marker.color:.2f}}<extra></extra>",
                )
            )
        if mask_plan.any():
            traces.append(
                go.Scatter(
                    x=df_plot.loc[mask_plan, x_axis],
                    y=df_plot.loc[mask_plan, y_axis],
                    mode="markers",
                    name=f"Exoplanet (≥ {HIGH_THR:.2f})",
                    marker=dict(size=3.0, color="#1e3a8a", line=dict(width=0.3, color="rgba(255,255,255,0.6)")),
                    hovertemplate=f"{x_axis}=%{{x}}<br>{y_axis}=%{{y}}<extra>exoplanet</extra>",
                )
            )
        if mask_confirmed.any():
            traces.append(
                go.Scatter(
                    x=df_plot.loc[mask_confirmed, x_axis],
                    y=df_plot.loc[mask_confirmed, y_axis],
                    mode="markers",
                    name="Confirmed (ring)",
                    marker=dict(size=5.2, color="rgba(0,0,0,0)", line=dict(width=1.4, color="red")),
                    hovertemplate=f"{x_axis}=%{{x}}<br>{y_axis}=%{{y}}<extra>confirmed</extra>",
                )
            )

        fig2d = go.Figure(data=traces)
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
        config2d = dict(displaylogo=False, scrollZoom=True, doubleClick="reset",
                        modeBarButtonsToAdd=["toggleSpikelines", "hovercompare", "v1hovermode"])
        st.plotly_chart(fig2d, use_container_width=True, config=config2d)
        st.caption("Legend: white < 0.25 · Blues 0.25–0.75 · dark blue ≥ 0.75 · red ring = confirmed.")

        # -------------------- 3D scatter (scaling options) --------------------
        def minmax(series: pd.Series) -> pd.Series:
            vmin, vmax = series.min(), series.max()
            if pd.isna(vmin) or pd.isna(vmax) or vmax == vmin:
                return pd.Series([0.5] * len(series), index=series.index)
            return (series - vmin) / (vmax - vmin)

        if scaling_mode == "Normalize to 0–1 (for selected axes)":
            X3 = minmax(df_plot[x_axis]); Y3 = minmax(df_plot[y_axis]); Z3 = minmax(df_plot[z_axis])
            scene_aspect = dict(aspectmode="cube")
        else:
            X3 = df_plot[x_axis]; Y3 = df_plot[y_axis]; Z3 = df_plot[z_axis]
            scene_aspect = dict(aspectmode="cube" if scaling_mode.startswith("Equal") else "data")
            scene_aspect = dict(aspectmode=scene_aspect["aspectmode"]) if isinstance(scene_aspect, dict) else dict(aspectmode=scene_aspect)

        traces3 = [
            go.Scatter3d(
                x=X3[mask_not], y=Y3[mask_not], z=Z3[mask_not],
                mode="markers",
                name=f"Not exoplanet (< {LOW_THR:.2f})",
                marker=dict(color="white", size=2.0, line=dict(width=0.25)),
                hovertemplate=f"{x_axis}=%{{x}}<br>{y_axis}=%{{y}}<br>{z_axis}=%{{z}}<extra></extra>",
            )
        ]
        if mask_cand.any():
            traces3.append(
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
            traces3.append(
                go.Scatter3d(
                    x=X3[mask_plan], y=Y3[mask_plan], z=Z3[mask_plan],
                    mode="markers",
                    name=f"Exoplanet (≥ {HIGH_THR:.2f})",
                    marker=dict(size=2.4, color="#1e3a8a", line=dict(width=0.35)),
                    hovertemplate=f"{x_axis}=%{{x}}<br>{y_axis}=%{{y}}<br>{z_axis}=%{{z}}<extra>exoplanet</extra>",
                )
            )
        if mask_confirmed.any():
            traces3.append(
                go.Scatter3d(
                    x=X3[mask_confirmed], y=Y3[mask_confirmed], z=Z3[mask_confirmed],
                    mode="markers",
                    name="Confirmed (ring)",
                    marker=dict(size=3.2, color="rgba(0,0,0,0)", line=dict(width=1.6, color="red")),
                    hovertemplate=f"{x_axis}=%{{x}}<br>{y_axis}=%{{y}}<br>{z_axis}=%{{z}}<extra>confirmed</extra>",
                )
            )

        fig3d = go.Figure(data=traces3)
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
        config3d = dict(
            displaylogo=False,
            scrollZoom=True,
            doubleClick="reset",
            toImageButtonOptions=dict(format="png", filename="exoplanet-3d"),
            modeBarButtonsToAdd=[
                "resetCameraDefault3d","resetCameraLastSave3d",
                "zoom3d","pan3d","orbitRotation","tableRotation","toggleSpikelines",
            ],
        )
        st.plotly_chart(fig3d, use_container_width=True, config=config3d)
        st.caption(
            "3D scaling: "
            + ("Equal cube" if scaling_mode.startswith("Equal") else ("Data-driven" if "Data" in scaling_mode else "Normalized 0–1"))
            + ". Tip: drag to orbit, scroll to zoom, double-click to reset."
        )

        # ---------- Compact distributions ----------
        st.markdown("<h3>Distributions</h3>", unsafe_allow_html=True)
        cA, cB, cC = st.columns([1, 1, 1])

        with cA:
            fig_x = px.histogram(df_plot, x=x_axis, nbins=30, title=f"Distribution: {x_axis}", template="plotly_dark")
            fig_x.update_layout(height=280, paper_bgcolor="#0e1117", plot_bgcolor="#0f172a", font_color="#e5e7eb")
            fig_x.update_traces(marker_line_width=0)
            st.plotly_chart(fig_x, use_container_width=True, config=dict(displaylogo=False))

        with cB:
            fig_y = px.histogram(df_plot, x=y_axis, nbins=30, title=f"Distribution: {y_axis}", template="plotly_dark")
            fig_y.update_layout(height=280, paper_bgcolor="#0e1117", plot_bgcolor="#0f172a", font_color="#e5e7eb")
            fig_y.update_traces(marker_line_width=0)
            st.plotly_chart(fig_y, use_container_width=True, config=dict(displaylogo=False))

        with cC:
            actual_confirmed = base_confirmed.loc[df_plot.index].sum()
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

with tab4:
    st.markdown("<h2>Model Performance</h2>", unsafe_allow_html=True)

    # Where to look for images
    IMAGE_DIR = APP_DIR / "images"
    IMAGE_DIR.mkdir(exist_ok=True)  # safe if it already exists

    # Support both names/locations
    candidates = [
        ("Accuracy", [APP_DIR / "Accuracy.png", IMAGE_DIR / "Accuracy.png"]),
        ("Model Loss", [APP_DIR / "Model Loss.png", IMAGE_DIR / "Model Loss.png"]),
    ]

    cols = st.columns(2)
    any_found = False

    for col, (title, paths) in zip(cols, candidates):
        img_path = next((p for p in paths if p.is_file()), None)
        with col:
            st.markdown(f"<div class='dark-card'><h3 style='margin:.25rem 0'>{title}</h3>", unsafe_allow_html=True)
            if img_path:
                any_found = True
                st.image(str(img_path), use_container_width=True, caption=f"{title} over epochs")
            else:
                st.warning(
                    f"Could not find **{title}.png**. "
                    f"Place it at `{APP_DIR}` or `{IMAGE_DIR}`."
                )
            st.markdown("</div>", unsafe_allow_html=True)

    if not any_found:
        st.info(
            "Tip: export your training curves as **Accuracy.png** and **Model Loss.png**. "
            "You can also store them under `Streamlit/images/`."
        )
    
# ---------- Compare Model (embed HTML) ----------
with tab5:
    st.markdown("<h2>Compare Model vs Actual Data (HTML)</h2>", unsafe_allow_html=True)
    st.markdown(
        "<div class='dark-card tight' style='margin-bottom:12px'>"
        "Upload and explore in the embedded viewer. It uses the same color rules and also supports percentile trimming."
        "</div>",
        unsafe_allow_html=True,
    )
    try:
        with open(APP_DIR / "comparison.html", "r", encoding="utf-8") as f:
            html_content = f.read()
        components.html(html_content, height=1000, scrolling=True)
    except Exception as e:
        st.error(f"Could not load comparison.html: {e}")
=======
import streamlit as st
import pandas as pd
import plotly.express as px
import base64
from datetime import datetime
from pathlib import Path

# ---- Define paths early ----
APP_DIR = Path(__file__).resolve().parent
LOGO_PATH = APP_DIR / "cube_logo.png"  # adjust if inside /assets
background_path = Path(__file__).parent / "stars_bg.png"  # or .gif

# --- Inject starry background ---
if background_path.exists():
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{background_path.read_bytes().hex()}");
            background-size: cover;
            background-attachment: fixed;
            background-position: center;
            background-repeat: no-repeat;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )
else:
    st.markdown(
        """
        <style>
        .stApp {
            background: radial-gradient(circle at 20% 20%, #000010, #000020, #000000);
            background-attachment: fixed;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
# Page config
st.set_page_config(page_title="Exoplanet Data Visualizer", layout="wide")

# ---- App-wide dark theme + starfield-inspired background (CSS) ----
_CSS = r"""
<style>
/* Page base: solid black canvas */
html, body, .stApp {
    background: #000000;
    color: #e6e6e6;
}

/* Dense starfield using multiple tiny radial gradients and a repeating pattern */
.stApp:before {
    content: "";
    position: fixed;
    left: 0; top: 0; right: 0; bottom: 0;
    background-color: #000;
    z-index: -1;
        background-image:
            radial-gradient(circle at 3% 10%, rgba(255,255,255,0.95) 1px, transparent 2px),
            radial-gradient(circle at 12% 30%, rgba(255,255,255,0.9) 1px, transparent 2px),
            radial-gradient(circle at 20% 60%, rgba(255,255,255,0.85) 1px, transparent 2px),
            radial-gradient(circle at 30% 15%, rgba(255,255,255,0.9) 1px, transparent 2px),
            radial-gradient(circle at 40% 50%, rgba(255,255,255,0.8) 1px, transparent 2px),
            radial-gradient(circle at 50% 80%, rgba(255,255,255,0.75) 1px, transparent 2px),
            radial-gradient(circle at 60% 35%, rgba(255,255,255,0.7) 1px, transparent 2px),
            radial-gradient(circle at 68% 10%, rgba(255,255,255,0.6) 1px, transparent 2px),
            radial-gradient(circle at 78% 55%, rgba(255,255,255,0.65) 1px, transparent 2px),
            radial-gradient(circle at 85% 25%, rgba(255,255,255,0.5) 1px, transparent 2px),
            radial-gradient(circle at 92% 70%, rgba(255,255,255,0.45) 1px, transparent 2px);
    background-repeat: repeat;
    opacity: 0.95;
    filter: blur(0.2px);
}

/* Subtle twinkle animation (affects opacity of the star layer) */
@keyframes twinkle {
    0% {opacity: 0.9}
    50% {opacity: 1}
    100% {opacity: 0.9}
}
.stApp:before { animation: twinkle 10s ease-in-out infinite; }

/* Streamlit container tweaks */
.css-1d391kg, .main, .block-container {
    max-width: 100% !important;
    padding-left: 2rem !important;
    padding-right: 2rem !important;
}

/* Sidebar styling */
.stSidebar {
    background: linear-gradient(180deg, rgba(10,10,10,0.9), rgba(6,6,6,0.9));
    color: #ddd;
}

/* Hide the Streamlit header/menu for a cleaner look */
header, footer, #MainMenu {
    visibility: hidden;
}

/* Make titles pop */
h1, h2, h3, .streamlit-expanderHeader {
    color: #f5f5f5;
}

/* Button style */
.stButton>button {
    background: linear-gradient(90deg,#333 0%, #111 100%);
    color: #fff;
    border: 1px solid rgba(255,255,255,0.06);
}

/* Fixed cube logo in top-left (will be injected as an <img id="cube-logo">) */
#cube-logo {
    position: fixed;
    left: 14px;
    top: 12px;
    width: 56px; /* smaller/tiny cube */
    height: auto;
    z-index: 9999;
    border-radius: 6px;
    box-shadow: 0 6px 24px rgba(0,0,0,0.75);
}

</style>
"""
# ---- Paths & helpers (define ONCE and reuse) ----
APP_DIR = Path(__file__).resolve().parent
LOGO_PATH = APP_DIR / "cube_logo.png"   # change to (APP_DIR / "assets" / "cube_logo.png") if you move it

# Prefer loading from an external CSS file for easier tweaks
css_file = APP_DIR / "static" / "styles.css"
if css_file.is_file():
    # Wrap file contents in a <style> tag so it's interpreted as CSS, not rendered as text
    st.markdown(f"<style>{css_file.read_text()}</style>", unsafe_allow_html=True)
else:
    st.markdown(_CSS, unsafe_allow_html=True)

def show_logo(*, width=None, use_container_width=False):
    """Deprecated helper kept for compatibility (we render a fixed logo instead)."""
    # intentionally no-op; the app now renders a single fixed-position cube logo
    return


with st.sidebar:
    st.markdown("# The Cube")
    st.markdown("### NASA SpaceApps Challenge")
    st.markdown("---")

# Hero / header area inspired by a dark, minimal gallery
with st.container():
    left, right = st.columns([3,1])
    with left:
        st.markdown("# <span style='font-size:44px;color:#ffffff'>Exoplanet Data Visualizer</span>", unsafe_allow_html=True)
        st.markdown("<p style='color:#cfcfcf; font-size:18px'>Explore candidate exoplanets with dark, immersive visuals — curated plots, 3D views, and model performance overlays.</p>", unsafe_allow_html=True)
        st.markdown("<div style='margin-top:12px'><em style='color:#9a9a9a'>Tip:</em> upload a CSV in the Visualization tab to begin.</div>", unsafe_allow_html=True)
        # ...existing hero content...
    with right:
        # we'll keep the header clean; the cube logo is fixed to the top-left corner
        pass

# Inject the cube logo as an inline base64 image so it's always available and fixed on top-left
if LOGO_PATH.is_file():
    try:
        with open(LOGO_PATH, 'rb') as f:
            b64 = base64.b64encode(f.read()).decode('utf-8')
        img_html = f"<img id=\"cube-logo\" src=\"data:image/png;base64,{b64}\" alt=\"cube\" />"
        st.markdown(img_html, unsafe_allow_html=True)
    except Exception:
        # if embedding fails, fall back to a normal image render inside the sidebar area
        st.sidebar.image(str(LOGO_PATH), width=80)


tab1, tab2, tab3, tab4 = st.tabs(["About Us", "Project Information", "Visualization", "Model Performance"])

with tab1:
    st.title("About Us")
    st.markdown("""
    **The Cube**  
    We are a team passionate about exoplanet discovery and determining habitability.  
    This website proposes a novel AI/ML model for determining whether a celestial body is an exoplanet and provides cross-reference with NASA Exoplanet Archive data.
    """)

with tab2:
    st.title("Project Information")
    st.markdown("""
    **Project Goal:**  
    To provide an interactive platform for visualizing and confirming exoplanet candidates from NASA archives.

    **Supported Data:**  
    - Kepler, K2, TOI CSVs  
    - Columns: Orbital period, transit duration, planet radius, insolation flux, equilibrium temperature, stellar temperature, stellar radius, etc.

    **Features:**  
    - 2D/3D scatter plots  
    - Histograms  
    - Customizable axes  
    - Model performance tab (for future ML integration)
    """)

with tab3:
    st.title("Data Visualization")
    st.markdown("Upload a NASA Exoplanet Archive CSV and explore trends in actual exoplanets.")

    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"]) 

    if uploaded_file:
        df = pd.read_csv(uploaded_file, comment='#', skip_blank_lines=True)

        numeric_cols = df.select_dtypes(include='number').columns.tolist()
        all_cols = df.columns.tolist()

        st.write("### "+uploaded_file.name+" Data", df.head())

        # Identify disposition column for confirmed exoplanets
        disposition_col = None
        for col in all_cols:
            if col.lower() in ['koi_disposition', 'disposition']:
                disposition_col = col
                break

        st.subheader("2D Scatter Plot")
        x_axis = st.selectbox("X-axis", numeric_cols, index=0 if 'period' in numeric_cols else 0, help="Select the feature for the X-axis (e.g., orbital period, planet radius, etc.)")
        y_axis = st.selectbox("Y-axis", numeric_cols, index=1 if 'radE' in numeric_cols else 1, help="Select the feature for the Y-axis (e.g., planet radius, equilibrium temperature, etc.)")
        color_col_candidates = [c for c in all_cols if df[c].nunique() < 20 and df[c].dtype == 'object']
        color_col = st.selectbox("Color by", color_col_candidates, index=0 if color_col_candidates else None, help="Categorical feature to color the points (e.g., disposition)")

        st.markdown(f"""
        **Axis Explanation:**  
        - **{x_axis}:** {x_axis.replace('_', ' ').capitalize()}  
        - **{y_axis}:** {y_axis.replace('_', ' ').capitalize()}  
        - **Relationship:**  
            This plot shows how {y_axis.replace('_', ' ')} varies with {x_axis.replace('_', ' ')}. Patterns may reveal correlations, clusters, or outliers among exoplanets.
        """)

        # Highlight confirmed exoplanets in red if possible
        if disposition_col and color_col == disposition_col:
            color_map = {v: 'red' if str(v).lower() == 'confirmed' else '#1f77b4' for v in df[disposition_col].unique()}
            fig2d = px.scatter(
                df,
                x=x_axis,
                y=y_axis,
                color=disposition_col,
                color_discrete_map=color_map,
                labels={x_axis: x_axis, y_axis: y_axis, disposition_col: disposition_col},
                title=f"{x_axis} vs {y_axis}",
                template='plotly_dark'
            )
        else:
            fig2d = px.scatter(
                df,
                x=x_axis,
                y=y_axis,
                color=color_col if color_col_candidates else None,
                labels={x_axis: x_axis, y_axis: y_axis, color_col: color_col},
                title=f"{x_axis} vs {y_axis}",
                template='plotly_dark'
            )
        st.plotly_chart(fig2d, use_container_width=True)

        st.subheader("3D Scatter Plot")
        if len(numeric_cols) >= 3:
            x3d = st.selectbox("3D X-axis", numeric_cols, index=0, key="x3d", help="Select the feature for the X-axis in 3D plot")
            y3d = st.selectbox("3D Y-axis", numeric_cols, index=1, key="y3d", help="Select the feature for the Y-axis in 3D plot")
            z3d = st.selectbox("3D Z-axis", numeric_cols, index=2, key="z3d", help="Select the feature for the Z-axis in 3D plot")
            color3d_candidates = [c for c in all_cols if df[c].nunique() < 20 and df[c].dtype == 'object']
            color3d = st.selectbox("3D Color by", color3d_candidates, index=0 if color3d_candidates else None, key="color3d", help="Categorical feature to color the points in 3D plot")

            st.markdown(f"""
            **Axis Explanation:**  
            - **{x3d}:** {x3d.replace('_', ' ').capitalize()}  
            - **{y3d}:** {y3d.replace('_', ' ').capitalize()}  
            - **{z3d}:** {z3d.replace('_', ' ').capitalize()}  
            - **Relationship:**  
                This 3D plot visualizes the interaction between {x3d.replace('_', ' ')}, {y3d.replace('_', ' ')}, and {z3d.replace('_', ' ')}. It helps uncover multi-dimensional patterns in exoplanet properties.
            """)

            if disposition_col and color3d == disposition_col:
                color_map_3d = {v: 'red' if str(v).lower() == 'confirmed' else '#1f77b4' for v in df[disposition_col].unique()}
                fig3d = px.scatter_3d(
                    df,
                    x=x3d,
                    y=y3d,
                    z=z3d,
                    color=disposition_col,
                    color_discrete_map=color_map_3d,
                    labels={x3d: x3d, y3d: y3d, z3d: z3d, disposition_col: disposition_col},
                    title=f"3D Visualization: {x3d}, {y3d}, {z3d}",
                    template='plotly_dark'
                )
            else:
                fig3d = px.scatter_3d(
                    df,
                    x=x3d,
                    y=y3d,
                    z=z3d,
                    color=color3d if color3d_candidates else None,
                    labels={x3d: x3d, y3d: y3d, z3d: z3d, color3d: color3d},
                    title=f"3D Visualization: {x3d}, {y3d}, {z3d}",
                    template='plotly_dark'
                )
            st.plotly_chart(fig3d, use_container_width=True)

        st.subheader("Parameter Distributions")
        for col in numeric_cols:
            st.write(f"#### Histogram of {col}")
            st.markdown(f"Shows the distribution of {col.replace('_', ' ')} among all exoplanets in the dataset.")
            fig_hist = px.histogram(df, x=col, nbins=30, title=f"Distribution of {col}", template='plotly_dark')
            st.plotly_chart(fig_hist, use_container_width=True)

with tab4:
    st.title("Model Performance")
    st.markdown("""
    **Coming Soon:**  
    This section will display results & predictions of the machine learning model compared to actual exoplanet classifications.
    """)
>>>>>>> repoB/main
