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