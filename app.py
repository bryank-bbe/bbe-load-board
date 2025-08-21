import streamlit as st
import pandas as pd
import pydeck as pdk
import numpy as np
import os
import hashlib
from datetime import datetime

# ---------------------------
# CONFIGURATION
# ---------------------------
st.set_page_config(page_title="BBE Load Match PRO", layout="wide")
st.title("🚛 BBE Load Match PRO (v7.1 – Cloud Ready)")

# Constants from environment or defaults
DEFAULT_ZIP = os.getenv("DEFAULT_ZIP", "36602")
MAP_STYLE = os.getenv("MAP_STYLE", "mapbox://styles/mapbox/light-v9")
DEFAULT_CPM = float(os.getenv("COST_PER_MILE", "0.35"))

# ---------------------------
# LOAD ZIP DATABASE
# ---------------------------
@st.cache_data
def load_zip_db():
    try:
        df = pd.read_csv("zip_db.csv", dtype={"ZIP": str})
        return df.set_index("ZIP")
    except Exception as e:
        st.error(f"❌ Could not load zip_db.csv: {e}")
        return pd.DataFrame(columns=["LAT", "LNG", "CITY", "STATE"])

zip_db = load_zip_db()

# ---------------------------
# HELPERS
# ---------------------------
def haversine(lat1, lon1, lat2, lon2):
    """Vectorized haversine formula in miles"""
    R = 3959.87433
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return R * 2 * np.arcsin(np.sqrt(a))

@st.cache_data
def load_sample_loads():
    try:
        return pd.read_csv("sample_loads.csv")
    except Exception as e:
        st.warning(f"⚠️ Could not load sample_loads.csv: {e}")
        return pd.DataFrame(columns=[
            "Pickup", "Drop", "Rate", "Miles", "Equipment",
            "DropZip", "Broker", "BrokerEmail", "BrokerPhone"
        ])

def broker_allowlist():
    try:
        df = pd.read_csv("brokers.csv")
        return df["Broker"].dropna().tolist()
    except Exception:
        return []

# ---------------------------
# SIDEBAR SETTINGS
# ---------------------------
st.sidebar.header("Settings")
load_source = st.sidebar.radio("Loads Source", ["Sample CSV", "Truckstop API (future)", "Email Parsing (future)"])
radius_step = st.sidebar.slider("Deadhead Radius Step (miles)", 50, 300, 100, 50)
max_radius = st.sidebar.slider("Maximum Deadhead Radius (miles)", 100, 500, 300, 50)
default_zip = st.sidebar.text_input("Default Truck ZIP", DEFAULT_ZIP)
cost_per_mile = st.sidebar.number_input("Cost per Mile ($)", value=DEFAULT_CPM, step=0.05)

# ---------------------------
# TRUCK ENTRY
# ---------------------------
st.subheader("Truck Entry Form")
with st.form("truck_entry_form"):
    truck_id = st.text_input("Truck Number / Identifier")
    delivery_zip = st.text_input("Delivery ZIP", value=default_zip)
    delivery_date = st.date_input("Delivery Date", datetime.today())
    submit_button = st.form_submit_button("Add Truck")

# ---------------------------
# LOAD SEARCH
# ---------------------------
st.subheader("Reload Match Display")

if submit_button:
    if delivery_zip not in zip_db.index:
        st.error(f"❌ ZIP {delivery_zip} not found in database.")
    else:
        loads_df = load_sample_loads()
        if loads_df.empty:
            st.warning("⚠️ No loads available.")
        else:
            origin = zip_db.loc[delivery_zip]
            lat1, lon1 = origin["LAT"], origin["LNG"]

            # Expand search radius
            results = pd.DataFrame()
            radius = radius_step
            while radius <= max_radius and results.empty:
                loads_df["Deadhead"] = haversine(lat1, lon1,
                                                 zip_db.loc[loads_df["DropZip"].astype(str)]["LAT"].values,
                                                 zip_db.loc[loads_df["DropZip"].astype(str)]["LNG"].values)
                filtered = loads_df[loads_df["Deadhead"] <= radius].copy()
                results = filtered
                radius += radius_step

            if results.empty:
                st.warning("⚠️ No loads found within max radius.")
            else:
                results["Profit"] = results["Rate"].str.replace("[$,]", "", regex=True).astype(float) - \
                                    cost_per_mile * (results["Miles"] + results["Deadhead"])
                results["Booked"] = False

                # Broker allowlist
                brokers = broker_allowlist()
                if brokers:
                    results = results[results["Broker"].isin(brokers)]

                # Display
                st.dataframe(results)

                # CSV download
                st.download_button(
                    "⬇️ Download Results CSV",
                    results.to_csv(index=False).encode("utf-8"),
                    "load_matches.csv",
                    "text/csv",
                )

                # ---------------------------
                # MAP DISPLAY
                # ---------------------------
                st.subheader("Load Match Map")
                if not results.empty:
                    map_data = results.merge(zip_db, left_on="DropZip", right_index=True)
                    layer = pdk.Layer(
                        "ScatterplotLayer",
                        data=map_data,
                        get_position="[LNG, LAT]",
                        get_radius=5000,
                        get_color=[0, 128, 255],
                        pickable=True,
                    )
                    view_state = pdk.ViewState(latitude=lat1, longitude=lon1, zoom=6)
                    st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view_state, map_style=MAP_STYLE))
