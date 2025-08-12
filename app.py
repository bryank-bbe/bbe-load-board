
import streamlit as st

# Sidebar view selection with unique keys
view = st.sidebar.radio("View", ["Single Truck", "Multi-Truck"], index=1, key="view_selection_sidebar")

# Example placeholder content
if view == "Single Truck":
    st.title("Single Truck View")
    st.write("Details for a single truck will appear here.")
elif view == "Multi-Truck":
    st.title("Multi-Truck View")
    st.write("Details for multiple trucks will appear here.")
with st.sidebar.expander("Diagnostics"):
    import sys, pandas as pd, os
    st.write("Python version:", sys.version.split()[0])
    st.write("Streamlit version:", st.__version__)
    st.write("Installed files:", os.listdir('.'))
    if os.path.exists("zip_db.csv"):
        try:
            z = pd.read_csv("zip_db.csv")
            st.write("zip_db.csv rows:", len(z), "cols:", list(z.columns))
        except Exception as e:
            st.error(f"Error reading zip_db.csv: {e}")
