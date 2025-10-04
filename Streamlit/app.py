import streamlit as st
import pandas as pd
import plotly.express as px

st.title("🌌 Exoplanet Data Visualizer")
st.markdown("Upload a NASA Exoplanet Archive CSV and explore key features.")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    needed_cols = ["koi_period", "koi_duration", "koi_prad", "koi_model_snr", "koi_disposition"]
    df = df[[c for c in needed_cols if c in df.columns]].dropna()

    st.write("### Sample Data", df.head())

    fig2d = px.scatter(
        df, 
        x="koi_period", 
        y="koi_prad", 
        color="koi_disposition",
        labels={"koi_period": "Orbital Period (days)", "koi_prad": "Planet Radius (Earth radii)"},
        title="Orbital Period vs Planet Radius"
    )
    st.plotly_chart(fig2d)

    if {"koi_period", "koi_duration", "koi_model_snr"}.issubset(df.columns):
        color_map = {
            "CONFIRMED": "green",
            "CANDIDATE": "yellow",
            "FALSE POSITIVE": "red"
        }

        fig3d = px.scatter_3d(
            df,
            x="koi_period",
            y="koi_duration",
            z="koi_model_snr",
            color="koi_disposition",
            color_discrete_map=color_map,
            labels={
                "koi_period": "Orbital Period (days)",
                "koi_duration": "Transit Duration (hours)",
                "koi_model_snr": "Signal-to-Noise Ratio"
            },
            title="3D Visualization: Period, Duration, and SNR"
        )
        st.plotly_chart(fig3d)
