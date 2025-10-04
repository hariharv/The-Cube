import streamlit as st
import pandas as pd
import plotly.express as px

# NASA SpaceApps UI: Add cube logo to sidebar and top of each tab
st.set_page_config(page_title="Exoplanet Data Visualizer", layout="wide")

with st.sidebar:
    st.image("cube_logo.png", use_column_width=True)
    st.markdown("### NASA SpaceApps Challenge")
    st.markdown("#### The Cube Project")
    st.markdown("---")

tab1, tab2, tab3, tab4 = st.tabs(["About Us", "Project Information", "Visualization", "Model Performance"])

with tab1:
    st.image("cube_logo.png", width=120)
    st.title("About Us")
    st.markdown("""
    **The Cube**  
    We are a team passionate about exoplanet discovery and determining habitability.  
    This website proposes a novel AI/ML model for determining whether a celestial body is an exoplanet and provides cross-reference with NASA Exoplanet Archive data.
    """)

with tab2:
    st.image("cube_logo.png", width=120)
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
    st.image("cube_logo.png", width=120)
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
                title=f"{x_axis} vs {y_axis}"
            )
        else:
            fig2d = px.scatter(
                df,
                x=x_axis,
                y=y_axis,
                color=color_col if color_col_candidates else None,
                labels={x_axis: x_axis, y_axis: y_axis, color_col: color_col},
                title=f"{x_axis} vs {y_axis}"
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
                    title=f"3D Visualization: {x3d}, {y3d}, {z3d}"
                )
            else:
                fig3d = px.scatter_3d(
                    df,
                    x=x3d,
                    y=y3d,
                    z=z3d,
                    color=color3d if color3d_candidates else None,
                    labels={x3d: x3d, y3d: y3d, z3d: z3d, color3d: color3d},
                    title=f"3D Visualization: {x3d}, {y3d}, {z3d}"
                )
            st.plotly_chart(fig3d, use_container_width=True)

        st.subheader("Parameter Distributions")
        for col in numeric_cols:
            st.write(f"#### Histogram of {col}")
            st.markdown(f"Shows the distribution of {col.replace('_', ' ')} among all exoplanets in the dataset.")
            fig_hist = px.histogram(df, x=col, nbins=30, title=f"Distribution of {col}")
            st.plotly_chart(fig_hist, use_container_width=True)

with tab4:
    st.image("cube_logo.png", width=120)
    st.title("Model Performance")
    st.markdown("""
    **Coming Soon:**  
    This section will display results & predictions of the machine learning model compared to actual exoplanet classifications.
    """)