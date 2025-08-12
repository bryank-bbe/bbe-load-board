
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
