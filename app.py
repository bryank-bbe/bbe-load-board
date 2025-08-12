
import streamlit as st
import pandas as pd
import math

# ---------------------------
# PAGE CONFIG
# ---------------------------
st.set_page_config(page_title="BBE Load Match", layout="wide")

st.title("🚛 BBE Load Match (v6)")

# ---------------------------
# SIDEBAR SETTINGS
# ---------------------------
st.sidebar.header("Settings")

load_source = st.sidebar.radio("Loads Source", ["Use sample", "Upload CSV"], index=0, key="load_source")
if load_source == "Upload CSV":
    uploaded_loads = st.sidebar.file_uploader("Upload Loads CSV", type="csv", key="loads_csv")
else:
    uploaded_loads = None

view_mode = st.sidebar.radio("View", ["Single Truck", "Multi-Truck"], index=0, key="view_mode")

# ---------------------------
# LOAD DATA
# ---------------------------
@st.cache_data
def load_csv(file):
    return pd.read_csv(file)

if uploaded_loads is not None:
    loads_df = load_csv(uploaded_loads)
else:
    loads_df = pd.read_csv("sample_loads.csv")

# ---------------------------
# FUNCTIONS
# ---------------------------
def haversine(lat1, lon1, lat2, lon2):
    R = 3959
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))

zip_df = pd.read_csv("zip_db.csv")

def find_zip_info(zip_code):
    row = zip_df[zip_df['ZIP'] == int(zip_code)]
    if row.empty:
        return None
    return float(row.iloc[0]['LAT']), float(row.iloc[0]['LNG']), row.iloc[0]['CITY'], row.iloc[0]['STATE']

def match_loads(truck_zip, equipment, max_radius, auto_expand=False):
    truck_info = find_zip_info(truck_zip)
    if truck_info is None:
        return pd.DataFrame()
    t_lat, t_lon, _, _ = truck_info

    radius = max_radius
    matches = pd.DataFrame()
    while matches.empty and auto_expand and radius <= 300:
        matches = loads_df[loads_df["Equipment"] == equipment].copy()
        matches["Deadhead"] = matches.apply(
            lambda row: haversine(t_lat, t_lon, row["DestLat"], row["DestLon"]), axis=1
        )
        matches = matches[matches["Deadhead"] <= radius]
        if matches.empty:
            radius += 50
        else:
            break
    return matches

# ---------------------------
# SINGLE TRUCK VIEW
# ---------------------------
if view_mode == "Single Truck":
    st.subheader("Single Truck Search")
    truck_zip = st.text_input("Truck Delivery ZIP", "36602", key="truck_zip")
    equipment = st.selectbox("Equipment", sorted(loads_df["Equipment"].unique()), key="equip")
    start_radius = st.number_input("Start radius (miles)", 50, 300, 100, key="radius")
    auto_expand = st.checkbox("Auto-expand if no matches", value=True, key="expand")

    if st.button("Find Loads", key="find_loads"):
        results = match_loads(truck_zip, equipment, start_radius, auto_expand)
        if results.empty:
            st.warning("No loads found.")
        else:
            st.dataframe(results)

# ---------------------------
# MULTI-TRUCK VIEW
# ---------------------------
else:
    st.subheader("Multi-Truck Search")
    trucks_file = st.file_uploader("Upload Trucks CSV", type="csv", key="trucks_csv")
    top_n = st.number_input("Top matches per truck", 1, 20, 3, key="topn")

    if trucks_file is not None:
        trucks_df = pd.read_csv(trucks_file)
        all_matches = []
        for _, truck in trucks_df.iterrows():
            truck_zip = str(truck["ZIP"])
            equip = truck["Equipment"]
            results = match_loads(truck_zip, equip, 100, auto_expand=True)
            if not results.empty:
                results = results.head(top_n)
                results.insert(0, "TruckZIP", truck_zip)
                all_matches.append(results)
        if all_matches:
            final_df = pd.concat(all_matches, ignore_index=True)
            st.dataframe(final_df)
        else:
            st.warning("No matches found.")
