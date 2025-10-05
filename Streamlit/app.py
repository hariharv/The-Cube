import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import base64
from pathlib import Path
import json
import streamlit.components.v1 as components

# Repo-aware paths (app is inside /Streamlit)
REPO_ROOT = Path(__file__).resolve().parent.parent
ART_DIR = REPO_ROOT / "artifacts"   # each run has predictions.jsonl + metrics.json

def list_runs():
    if not ART_DIR.exists():
        return []
    return sorted(
        [p for p in ART_DIR.iterdir() if p.is_dir() and (p / "predictions.jsonl").exists()],
        key=lambda p: p.stat().st_mtime,
        reverse=True
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

# ---- Define paths early ----
APP_DIR = Path(__file__).resolve().parent
LOGO_PATH = APP_DIR / "cube_logo.png"
background_path = Path(__file__).parent / "stars_bg.png"

# --- Inject starry background (fallback to radial gradient) ---
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

st.set_page_config(page_title="Exoplanet Data Visualizer", layout="wide")

# ---- Global CSS (dark, starfield look) ----
_CSS = r"""
<style>
html, body, .stApp { background: #000; color: #e6e6e6; }
.stApp:before {
    content: ""; position: fixed; left:0; top:0; right:0; bottom:0; z-index:-1;
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
    background-repeat: repeat; opacity:0.95; filter: blur(0.2px);
    animation: twinkle 10s ease-in-out infinite;
}
@keyframes twinkle { 0%{opacity:0.9} 50%{opacity:1} 100%{opacity:0.9} }
#cube-logo { position: fixed; left: 14px; top: 12px; width: 56px; z-index: 9999; border-radius: 6px;
             box-shadow: 0 6px 24px rgba(0,0,0,0.75); }
header, footer, #MainMenu { visibility: hidden; }
</style>
"""
css_file = APP_DIR / "static" / "styles.css"
if css_file.is_file():
    st.markdown(f"<style>{css_file.read_text()}</style>", unsafe_allow_html=True)
else:
    st.markdown(_CSS, unsafe_allow_html=True)

# ----------------------------- #
# Axis explanations
# ----------------------------- #
AXIS_INFO = {
    "koi_period": "Orbital period — days per full orbit around the host star.",
    "koi_duration": "Transit duration — hours the planet takes to cross the star.",
    "koi_depth": "Transit depth — fractional drop in brightness during transit.",
    "koi_prad": "Planet radius — in Earth radii (R⊕).",
    "koi_teq": "Equilibrium temperature — blackbody estimate in Kelvin.",
    "koi_insol": "Incident flux — stellar energy received vs. Earth (S⊕).",
    "koi_steff": "Stellar effective temperature — surface temperature (K).",
    "koi_srad": "Stellar radius — in solar radii (R☉).",
    "koi_slogg": "Stellar surface gravity — log(g) in cgs units.",
    "koi_smet": "Stellar metallicity — [Fe/H] dex.",

    "period": "Orbital period — time to orbit the star once (days).",
    "period_days": "Orbital period — time to orbit the star once (days).",
    "duration": "Transit duration — transit length (hours).",
    "duration_hrs": "Transit duration — transit length (hours).",
    "depth_ppm": "Transit depth — brightness drop in parts-per-million.",
    "rp_re": "Planet radius — Earth radii (R⊕).",
    "teq": "Equilibrium temperature — blackbody estimate in Kelvin.",
    "teq_k": "Equilibrium temperature — Kelvin.",
    "srad": "Stellar radius — solar radii (R☉).",
    "teff": "Stellar effective temperature — surface temperature (K).",
    "insol": "Incident flux — relative to Earth (S⊕).",
    "semi_major_axis": "Orbital semi-major axis — average star–planet distance (AU).",
    "eccentricity": "Orbital eccentricity — deviation from a circular orbit.",
    "impact_param": "Impact parameter — transit chord across the stellar disk.",
    "snr": "Signal-to-noise ratio — detection strength.",

    "class": "Predicted/true class label — e.g., confirmed, candidate, or false_positive.",
    "trandur": "Transit duration — transit length (hours).",
    "rade": "Planet radius — in Earth radii (R⊕).",
    "earthflux": "Incident stellar flux received by the planet relative to Earth (S⊕).",
    "eqtemp": "Equilibrium temperature — estimated blackbody temperature (K).",
    "efftemp": "Stellar effective temperature — surface temperature of the host star (K).",
    "rads": "Stellar radius — in solar radii (R☉).",
}

def explain_axis(name: str) -> str:
    if not isinstance(name, str):
        return "Feature description unavailable."
    key = name.strip()
    return AXIS_INFO.get(key, AXIS_INFO.get(key.lower(), f"{name.replace('_', ' ').capitalize()} — feature description unavailable."))

# Sidebar
with st.sidebar:
    st.markdown("# The Cube")
    st.markdown("### NASA SpaceApps Challenge")
    st.markdown("---")

# Hero
with st.container():
    left, right = st.columns([3,1])
    with left:
        st.markdown("# <span style='font-size:44px;color:#ffffff'>Exoplanet Data Visualizer</span>", unsafe_allow_html=True)
        st.markdown("<p style='color:#cfcfcf; font-size:18px'>Explore candidate exoplanets with dark, immersive visuals — curated plots, 3D views, and model performance overlays.</p>", unsafe_allow_html=True)
        st.markdown("<div style='margin-top:12px'><em style='color:#9a9a9a'>Tip:</em> upload a CSV in the Visualization tab to begin.</div>", unsafe_allow_html=True)
    with right:
        pass

# Fixed cube logo
if LOGO_PATH.is_file():
    try:
        with open(LOGO_PATH, 'rb') as f:
            b64 = base64.b64encode(f.read()).decode('utf-8')
        st.markdown(f"<img id='cube-logo' src='data:image/png;base64,{b64}' alt='cube'/>", unsafe_allow_html=True)
    except Exception:
        st.sidebar.image(str(LOGO_PATH), width=80)

# Tabs (added "Compare Model")
tab1, tab2, tab3, tab4, tab5 = st.tabs(["About Us", "Project Information", "Visualization", "Model Performance", "Compare Model"])

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

# ----------------------------- #
# Visualization (three-band rules + red ring)
# ----------------------------- #
with tab3:
    st.title("Data Visualization")
    st.markdown("Upload a NASA Exoplanet Archive CSV or your model output CSV (e.g. `predicted.csv`).")

    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"]) 

    if uploaded_file:
        df = pd.read_csv(uploaded_file, comment='#', skip_blank_lines=True)

        numeric_cols = df.select_dtypes(include='number').columns.tolist()
        all_cols = df.columns.tolist()
        object_cols = [c for c in all_cols if df[c].dtype == 'object']

        st.write("### "+uploaded_file.name+" Data", df.head())

        # Label/Disposition column detection
        lower_map = {c.lower(): c for c in all_cols}
        preferred_order = ["predicted_label", "actual_label", "koi_disposition", "disposition", "class", "label", "status"]
        default_label_col = next((lower_map[name] for name in preferred_order if name in lower_map), None)
        small_num_cols = [c for c in all_cols if pd.api.types.is_numeric_dtype(df[c]) and df[c].nunique() <= 10]
        label_candidates = list(dict.fromkeys(([default_label_col] if default_label_col else []) + object_cols + small_num_cols))
        label_candidates = [c for c in label_candidates if c is not None]
        label_col = st.selectbox("Label/Disposition column", options=label_candidates) if label_candidates else None

        # Confirmed mask (string contains 'confirm' or numeric equals chosen 'confirmed value')
        if label_col is not None and pd.api.types.is_numeric_dtype(df[label_col]):
            uniq = sorted(pd.Series(df[label_col].dropna().unique()).tolist())
            default_idx = uniq.index(1) if 1 in uniq else 0
            confirm_value = st.selectbox("Which numeric value means 'confirmed'?", options=uniq, index=default_idx, help="Used to draw the red ring around confirmed exoplanets.")
            mask_confirmed = (df[label_col] == confirm_value)
        elif label_col is not None:
            mask_confirmed = df[label_col].astype(str).str.lower().str.contains("confirm")
        else:
            mask_confirmed = pd.Series([False] * len(df))

        # Confidence column + fixed thresholds for three bands
        has_conf = ("confidence_positive" in df.columns) and pd.api.types.is_numeric_dtype(df["confidence_positive"])
        LOW_THR, HIGH_THR = 0.25, 0.75
        st.markdown(f"**Banding:** not exoplanet: < {LOW_THR}, candidate: [{LOW_THR}, {HIGH_THR}), exoplanet: ≥ {HIGH_THR}")

        # 2D Scatter
        st.subheader("2D Scatter Plot")
        if len(numeric_cols) == 0:
            st.warning("No numeric columns found.")
        else:
            x_axis = st.selectbox("X-axis", numeric_cols, index=0 if 'period' in numeric_cols else 0, help="Select numeric feature for X-axis")
            y_axis = st.selectbox("Y-axis", numeric_cols, index=1 if len(numeric_cols) > 1 else 0, help="Select numeric feature for Y-axis")

            st.markdown(f"""
            **Axis Explanation:**  
            - **X-axis ({x_axis})** — {explain_axis(x_axis)}  
            - **Y-axis ({y_axis})** — {explain_axis(y_axis)}  
            - **Color Rules:**  
              - Not exoplanet: **white** (confidence &lt; {LOW_THR})  
              - Candidate: **Blues gradient** (confidence in [{LOW_THR}, {HIGH_THR}))  
              - Exoplanet (model): **solid dark blue** (confidence ≥ {HIGH_THR})  
              - Confirmed: **red ring overlay**
            """)

            if has_conf:
                conf = df["confidence_positive"]
                mask_not   = conf < LOW_THR
                mask_cand  = (conf >= LOW_THR) & (conf < HIGH_THR)
                mask_plan  = conf >= HIGH_THR
            else:
                mask_not  = pd.Series([True] * len(df))
                mask_cand = pd.Series([False] * len(df))
                mask_plan = pd.Series([False] * len(df))

            # Smaller dots (2D)
            trace_not = go.Scatter(
                x=df.loc[mask_not, x_axis],
                y=df.loc[mask_not, y_axis],
                mode="markers",
                name=f"Not exoplanet (< {LOW_THR:.2f})",
                marker=dict(color="white", size=3.5, line=dict(width=0.3, color="rgba(0,0,0,0.3)")),
                hovertemplate=f"{x_axis}=%{{x}}<br>{y_axis}=%{{y}}<extra></extra>"
            )
            traces = [trace_not]

            if mask_cand.any():
                trace_cand = go.Scatter(
                    x=df.loc[mask_cand, x_axis],
                    y=df.loc[mask_cand, y_axis],
                    mode="markers",
                    name=f"Candidate [{LOW_THR:.2f}, {HIGH_THR:.2f})",
                    marker=dict(
                        size=3.5,
                        color=df.loc[mask_cand, "confidence_positive"],
                        colorscale="Blues",
                        cmin=LOW_THR, cmax=HIGH_THR,
                        showscale=True,
                        colorbar=dict(title="Confidence+"),
                        line=dict(width=0.3, color="rgba(255,255,255,0.6)")
                    ),
                    hovertemplate=f"{x_axis}=%{{x}}<br>{y_axis}=%{{y}}<br>conf+=%{{marker.color:.2f}}<extra></extra>"
                )
                traces.append(trace_cand)

            if mask_plan.any():
                trace_plan = go.Scatter(
                    x=df.loc[mask_plan, x_axis],
                    y=df.loc[mask_plan, y_axis],
                    mode="markers",
                    name=f"Exoplanet (≥ {HIGH_THR:.2f})",
                    marker=dict(
                        size=3.5,
                        color="#1e3a8a",
                        line=dict(width=0.3, color="rgba(255,255,255,0.6)")
                    ),
                    hovertemplate=f"{x_axis}=%{{x}}<br>{y_axis}=%{{y}}<extra>exoplanet</extra>"
                )
                traces.append(trace_plan)

            # Confirmed ring overlay (RED ring) — slightly larger but still subtle
            if mask_confirmed.any():
                trace_ring = go.Scatter(
                    x=df.loc[mask_confirmed, x_axis],
                    y=df.loc[mask_confirmed, y_axis],
                    mode="markers",
                    name="Confirmed (ring)",
                    marker=dict(
                        size=6,
                        color="rgba(0,0,0,0)",
                        line=dict(width=1.5, color="red")
                    ),
                    hovertemplate=f"{x_axis}=%{{x}}<br>{y_axis}=%{{y}}<extra>confirmed</extra>"
                )
                traces.append(trace_ring)

            fig2d = go.Figure(data=traces)
            fig2d.update_layout(
                template="plotly_dark",
                title=f"{x_axis} vs {y_axis}",
                margin=dict(l=40, r=20, t=40, b=40),
                uirevision="keep"
            )
            config2d = dict(
                displaylogo=False,
                scrollZoom=True,
                doubleClick="reset",
                modeBarButtonsToAdd=["toggleSpikelines", "hovercompare", "v1hovermode"]
            )
            st.plotly_chart(fig2d, use_container_width=True, config=config2d, theme=None)

        # 3D Scatter — improved exploration
        st.subheader("3D Scatter Plot")
        if len(numeric_cols) >= 3:
            x3d = st.selectbox("3D X-axis", numeric_cols, index=0, key="x3d")
            y3d = st.selectbox("3D Y-axis", numeric_cols, index=1 if len(numeric_cols) > 1 else 0, key="y3d")
            z3d = st.selectbox("3D Z-axis", numeric_cols, index=2 if len(numeric_cols) > 2 else 0, key="z3d")

            st.markdown(f"""
            **3D Axis Explanation:**  
            - **X-axis ({x3d})** — {explain_axis(x3d)}  
            - **Y-axis ({y3d})** — {explain_axis(y3d)}  
            - **Z-axis ({z3d})** — {explain_axis(z3d)}  
            - **Color Rules:**  
              - Not exoplanet: **white** (confidence &lt; {LOW_THR})  
              - Candidate: **Blues gradient** (confidence in [{LOW_THR}, {HIGH_THR}))  
              - Exoplanet (model): **solid dark blue** (confidence ≥ {HIGH_THR})  
              - Confirmed: **red ring overlay**
            """)

            if has_conf:
                conf = df["confidence_positive"]
                mask_not3 = conf < LOW_THR
                mask_cand3 = (conf >= LOW_THR) & (conf < HIGH_THR)
                mask_plan3 = conf >= HIGH_THR
            else:
                mask_not3  = pd.Series([True] * len(df))
                mask_cand3 = pd.Series([False] * len(df))
                mask_plan3 = pd.Series([False] * len(df))

            # Smaller 3D markers for dense clouds
            trace_not3 = go.Scatter3d(
                x=df.loc[mask_not3, x3d],
                y=df.loc[mask_not3, y3d],
                z=df.loc[mask_not3, z3d],
                mode="markers",
                name=f"Not exoplanet (< {LOW_THR:.2f})",
                marker=dict(color="white", size=2.2, line=dict(width=0.3)),
                hovertemplate=f"{x3d}=%{{x}}<br>{y3d}=%{{y}}<br>{z3d}=%{{z}}<extra></extra>"
            )
            traces3 = [trace_not3]

            if mask_cand3.any():
                trace_cand3 = go.Scatter3d(
                    x=df.loc[mask_cand3, x3d],
                    y=df.loc[mask_cand3, y3d],
                    z=df.loc[mask_cand3, z3d],
                    mode="markers",
                    name=f"Candidate [{LOW_THR:.2f}, {HIGH_THR:.2f})",
                    marker=dict(
                        size=2.4,
                        color=df.loc[mask_cand3, "confidence_positive"],
                        colorscale="Blues",
                        cmin=LOW_THR, cmax=HIGH_THR,
                        showscale=True,
                        colorbar=dict(title="Confidence+"),
                        line=dict(width=0.3)
                    ),
                    hovertemplate=f"{x3d}=%{{x}}<br>{y3d}=%{{y}}<br>{z3d}=%{{z}}<br>conf+=%{{marker.color:.2f}}<extra></extra>"
                )
                traces3.append(trace_cand3)

            if mask_plan3.any():
                trace_plan3 = go.Scatter3d(
                    x=df.loc[mask_plan3, x3d],
                    y=df.loc[mask_plan3, y3d],
                    z=df.loc[mask_plan3, z3d],
                    mode="markers",
                    name=f"Exoplanet (≥ {HIGH_THR:.2f})",
                    marker=dict(size=2.6, color="#1e3a8a", line=dict(width=0.4)),
                    hovertemplate=f"{x3d}=%{{x}}<br>{y3d}=%{{y}}<br>{z3d}=%{{z}}<extra>exoplanet</extra>"
                )
                traces3.append(trace_plan3)

            if mask_confirmed.any():
                trace_ring3d = go.Scatter3d(
                    x=df.loc[mask_confirmed, x3d],
                    y=df.loc[mask_confirmed, y3d],
                    z=df.loc[mask_confirmed, z3d],
                    mode="markers",
                    name="Confirmed (ring)",
                    marker=dict(size=3.6, color="rgba(0,0,0,0)", line=dict(width=2, color="red")),
                    hovertemplate=f"{x3d}=%{{x}}<br>{y3d}=%{{y}}<br>{z3d}=%{{z}}<extra>confirmed</extra>"
                )
                traces3.append(trace_ring3d)

            fig3d = go.Figure(data=traces3)
            fig3d.update_layout(
                template="plotly_dark",
                title=f"3D Visualization: {x3d}, {y3d}, {z3d}",
                margin=dict(l=0, r=0, t=40, b=0),
                scene=dict(
                    xaxis=dict(title=x3d, showspikes=True, spikethickness=1),
                    yaxis=dict(title=y3d, showspikes=True, spikethickness=1),
                    zaxis=dict(title=z3d, showspikes=True, spikethickness=1),
                    aspectmode="cube",
                    dragmode="orbit",
                    camera=dict(eye=dict(x=1.6, y=1.6, z=1.6))
                ),
                uirevision="keep"
            )
            # Better exploration: scroll zoom, orbit tools, etc.
            config3d = dict(
                displaylogo=False,
                scrollZoom=True,
                doubleClick="reset",
                toImageButtonOptions=dict(format="png", filename="exoplanet-3d"),
                modeBarButtonsToAdd=[
                    "resetCameraDefault3d",
                    "resetCameraLastSave3d",
                    "zoom3d",
                    "pan3d",
                    "orbitRotation",
                    "tableRotation",
                    "toggleSpikelines"
                ]
            )
            st.plotly_chart(fig3d, use_container_width=True, config=config3d, theme=None, height=760)

        st.subheader("Parameter Distributions")
        if len(numeric_cols):
            for col in numeric_cols:
                st.write(f"#### Histogram of {col}")
                st.markdown(f"Shows the distribution of {col.replace('_', ' ')} in the dataset.")
                fig_hist = px.histogram(df, x=col, nbins=30, title=f"Distribution of {col}", template='plotly_dark')
                fig_hist.update_traces(marker_line_width=0)  # cleaner bars
                st.plotly_chart(fig_hist, use_container_width=True)
        else:
            st.info("Upload a CSV with numeric columns to see distributions.")

with tab4:
    st.title("Model Performance")
    st.markdown("""
    **Coming Soon:**  
    This section will display results & predictions of the machine learning model compared to actual exoplanet classifications.
    """)

with tab5:
    st.title("Compare Model vs Actual Data (HTML)")
    st.markdown("""
    The embedded page shows side-by-side comparison with the same rules:
    **white** (&lt; 0.25), **Blues gradient** (0.25–0.75), **dark blue** (≥ 0.75), and **red ring** for confirmed.
    You can also pop either plot **full screen** for deeper inspection.
    """)
    try:
        with open(APP_DIR / "comparison.html", "r", encoding="utf-8") as f:
            html_content = f.read()
        components.html(html_content, height=1000, scrolling=True)
    except Exception as e:
        st.error(f"Could not load comparison.html: {e}")