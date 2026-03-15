import streamlit as st
import pandas as pd
import plotly.express as px
from ctgan import CTGAN
from sdmetrics.reports.single_table import QualityReport
import io

st.set_page_config(layout="wide")
st.title("CTGAN Synthetic Data Generator")


uploaded_file = st.file_uploader("Upload a CSV file", type="csv")

if uploaded_file is not None:
    try:
        original_df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Error reading file: {e}")
        st.stop()

    st.success("File uploaded successfully!")
    st.subheader("Original Dataset Preview")
    st.dataframe(original_df.head())

    discrete_columns = st.multiselect(
        "Select discrete columns",
        options=original_df.columns.tolist(),
        default=[c for c in original_df.columns if original_df[c].dtype == "object"]
    )

    epochs = st.number_input("Training epochs", min_value=10, max_value=1000, value=300)
    num_samples = st.number_input("Synthetic samples", min_value=1, value=len(original_df))

    if st.button("Generate Synthetic Data", use_container_width=True):
        with st.spinner("Training CTGAN..."):
            try:
                ctgan = CTGAN(epochs=epochs, verbose=True)
                ctgan.fit(original_df, discrete_columns)
                synthetic_df = ctgan.sample(num_samples)
                st.success("Synthetic data generated successfully!")
            except Exception as e:
                st.error(f"Error: {e}")
                st.stop()

            st.subheader("Generated Synthetic Data")
            st.dataframe(synthetic_df.head())

            csv_buffer = io.StringIO()
            synthetic_df.to_csv(csv_buffer, index=False)
            st.download_button(
                "Download Synthetic Data",
                data=csv_buffer.getvalue().encode("utf-8"),
                file_name="synthetic_data.csv",
                mime="text/csv"
            )

            st.subheader("Dashboard View (click to expand any graph)")

            numeric_cols = original_df.select_dtypes(include=["int64", "float64"]).columns
            categorical_cols = [c for c in discrete_columns if c in original_df.columns]

            plots = []

            # Numeric distributions 
            for col in numeric_cols:
                fig = px.histogram(
                    pd.DataFrame({
                        "Original": original_df[col],
                        "Synthetic": synthetic_df[col]
                    }).melt(var_name="Type", value_name=col),
                    x=col, color="Type", barmode="overlay", nbins=30,
                    title=f"Distribution: {col}"
                )
                plots.append(fig)

            # Categorical distributions
            for col in categorical_cols:
                df_compare = pd.concat([
                    original_df[col].value_counts(normalize=True).rename("Original"),
                    synthetic_df[col].value_counts(normalize=True).rename("Synthetic")
                ], axis=1).fillna(0).reset_index()

                df_compare = df_compare.rename(columns={"index": col})  # ✅ fix

                df_melted = df_compare.melt(
                    id_vars=col, var_name="Type", value_name="Proportion"
                )

                fig = px.bar(df_melted, x=col, y="Proportion", color="Type",
                             barmode="group", title=f"Category: {col}")
                plots.append(fig)

            # Correlation heatmaps
            if len(numeric_cols) > 1:
                fig1 = px.imshow(original_df[numeric_cols].corr(),
                                 text_auto=True, title="Original Correlation")
                fig2 = px.imshow(synthetic_df[numeric_cols].corr(),
                                 text_auto=True, title="Synthetic Correlation")
                plots.extend([fig1, fig2])

            # Grid layout for plots 
            n_cols = 3
            for i in range(0, len(plots), n_cols):
                cols = st.columns(n_cols)
                for j, fig in enumerate(plots[i:i+n_cols]):
                    with cols[j]:
                        fig.update_layout(height=250, margin=dict(l=10, r=10, t=30, b=10))
                        st.plotly_chart(fig, use_container_width=True)