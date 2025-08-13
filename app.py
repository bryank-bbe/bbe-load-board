import streamlit as st
import pandas as pd
import pydeck as pdk
import math
import re
import os
import logging
from datetime import datetime
import hashlib

# ---------------------------
# CONFIGURATION
# ---------------------------
st.set_page_config(page_title="BBE Load Match", layout="wide")
st.title("🚛 BBE Load Match (v7 – Enhanced UI & Features)")

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants from environment or defaults
DEFAULT_ZIP = os.getenv("DEFAULT_ZIP", "36602")
MAP_STYLE = os.getenv("MAP_STYLE", "mapbox://styles/mapbox/light-v9")

# ---------------------------
# SIDEBAR
# ---------------------------
st.sidebar.header("Settings")
load_source = st.sidebar.radio("Loads Source", ["Use sample", "Upload CSV"], index=0, key="load_source")
uploaded_loads = st.sidebar.file_uploader("Upload Loads CSV", type="csv", key="loads_csv") if load_source == "Upload CSV" else None
view_mode = st.sidebar.radio("View", ["Single Truck", "Multi-Truck"], index=0, key="view_mode")

# ---------------------------
# UTILITIES
# ---------------------------
@st.cache_data(ttl=300)  # Refresh every 5 minutes
def read_csv(file_or_path, **kwargs):
    return pd.read_csv(file_or_path, **kwargs)

def haversine(lat1, lon1, lat2, lon2):
    R = 3959.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    return 2 * R * math.asin(min(1, math.sqrt(a)))

def first_col(df: pd.DataFrame, *candidates):
    lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand in lower:
            return lower[cand]
    return None

# ---------------------------
# ZIP DB (lazy loading)
# ---------------------------
_zip = None
def get_zip_db():
    global _zip
    if _zip is None:
        try:
            _zip_raw = read_csv("zip_db.csv")
            _zip = _zip_raw.rename(columns={c: c.strip().upper() for c in _zip_raw.columns})
            if "LONG" in _zip.columns and "LNG" not in _zip.columns: _zip["LNG"] = _zip["LONG"]
            if "LON" in _zip.columns and "LNG" not in _zip.columns: _zip["LNG"] = _zip["LON"]
            if "ZIP" not in _zip.columns or "LAT" not in _zip.columns or "LNG" not in _zip.columns:
                st.warning("zip_db.csv is missing required columns (ZIP, LAT, LNG).")
            if "ZIP" in _zip.columns:
                _zip["ZIP"] = pd.to_numeric(_zip["ZIP"], errors="coerce").astype("Int64")
            if "LAT" in _zip.columns:
                _zip["LAT"] = pd.to_numeric(_zip["LAT"], errors="coerce")
            if "LNG" in _zip.columns:
                _zip["LNG"] = pd.to_numeric(_zip["LNG"], errors="coerce")
        except Exception as e:
            st.error(f"Failed to read zip_db.csv: {e}")
            _zip = pd.DataFrame(columns=["ZIP", "LAT", "LNG", "CITY", "STATE"])
    return _zip

def find_zip_info(zip_code: str):
    try:
        z = int(str(zip_code).strip())
    except Exception:
        return None
    zip_db = get_zip_db()
    if zip_db.empty or "ZIP" not in zip_db.columns:
        return None
    row = zip_db.loc[zip_db["ZIP"] == z]
    if row.empty:
        return None
    lat = float(row.iloc[0]["LAT"]) if pd.notna(row.iloc[0]["LAT"]) else None
    lng = float(row.iloc[0]["LNG"]) if pd.notna(row.iloc[0]["LNG"]) else None
    city = row.iloc[0]["CITY"] if "CITY" in zip_db.columns else ""
    state = row.iloc[0]["STATE"] if "STATE" in zip_db.columns else ""
    if lat is None or lng is None:
        return None
    return lat, lng, city, state

# ---------------------------
# LOADS (robust normalization)
# ---------------------------
@st.cache_data
def normalize_loads(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty or not any(col in df.columns for col in ["Pickup", "Drop", "Rate", "Miles"]):
        st.error("Loads CSV must contain at least Pickup, Drop, Rate, and Miles columns.")
        return pd.DataFrame(columns=["Equipment", "DestLat", "DestLon", "Rate", "Miles", "Pickup", "Drop", "Broker", "BrokerEmail", "BrokerPhone", "DropZip"])
    d = df.copy()
    d.columns = [str(c).strip() for c in d.columns]

    # Equipment
    equip_col = first_col(d, "equipment", "equip", "trailer", "trailertype", "type")
    d["Equipment"] = d[equip_col].astype(str).str.strip() if equip_col else "Flat"

    # Rate (USD)
    rate_col = first_col(d, "rate", "price", "pay", "freight", "offer")
    d["Rate"] = pd.to_numeric(d[rate_col].astype(str).str.replace(r"[^0-9.\-]", "", regex=True), errors="coerce") if rate_col else pd.NA

    # Miles (loaded)
    miles_col = first_col(d, "miles", "distance", "loaded_miles", "mi")
    d["Miles"] = pd.to_numeric(d[miles_col], errors="coerce") if miles_col else pd.NA

    # Pickup / Drop text
    pick_city = first_col(d, "pickup_city", "pick_city", "origin_city", "from_city", "pickup")
    pick_state = first_col(d, "pickup_state", "pick_state", "origin_state", "from_state", "pickup_st", "pickupstate")
    drop_city = first_col(d, "drop_city", "delivery_city", "dest_city", "to_city", "drop")
    drop_state = first_col(d, "drop_state", "delivery_state", "dest_state", "to_state", "drop_st", "dropstate")
    d["Pickup"] = d[[c for c in [pick_city] if c]].astype(str).agg(", ".join, axis=1) if pick_city else ""
    if pick_state:
        d["Pickup"] = (d["Pickup"] + ", " + d[pick_state].astype(str)).str.strip(", ")
    d["Drop"] = d[[c for c in [drop_city] if c]].astype(str).agg(", ".join, axis=1) if drop_city else ""
    if drop_state:
        d["Drop"] = (d["Drop"] + ", " + d[drop_state].astype(str)).str.strip(", ")

    # Broker info
    broker_col = first_col(d, "broker", "company", "contact", "provider")
    email_col = first_col(d, "broker_email", "email", "e-mail", "mail")
    phone_col = first_col(d, "broker_phone", "phone", "tel", "telephone", "cell")
    d["Broker"] = d[broker_col].astype(str) if broker_col else ""
    d["BrokerEmail"] = d[email_col].astype(str) if email_col else ""
    d["BrokerPhone"] = d[phone_col].astype(str) if phone_col else ""

    # Destination Lat/Lon
    lat_col = first_col(d, "destlat", "drop_lat", "droplat", "delivery_lat", "dlat", "lat2")
    lon_col = first_col(d, "destlon", "drop_lon", "droplon", "delivery_lon", "dlon", "lng2", "lon2")
    if lat_col and lon_col:
        d["DestLat"] = pd.to_numeric(d[lat_col], errors="coerce")
        d["DestLon"] = pd.to_numeric(d[lon_col], errors="coerce")
    else:
        zip_col = first_col(d, "drop_zip", "dest_zip", "delivery_zip", "to_zip", "zip_to", "dropzip", "zip")
        d["DropZip"] = d[zip_col].astype(str) if zip_col else ""
        def to_ll(z):
            info = find_zip_info(z) if pd.notna(z) and str(z).strip() else None
            return (info[0], info[1]) if info else (None, None)
        lat, lon = [], []
        for z in (d["DropZip"] if "DropZip" in d.columns else []):
            a, b = to_ll(z)
            lat.append(a); lon.append(b)
        if lat:
            d["DestLat"] = lat
            d["DestLon"] = lon
        else:
            d["DestLat"] = pd.NA
            d["DestLon"] = pd.NA

    return d

try:
    raw_loads = read_csv(uploaded_loads) if uploaded_loads is not None else read_csv("sample_loads.csv")
    loads_df = normalize_loads(raw_loads)
except Exception as e:
    st.error(f"Failed to read loads CSV: {e}")
    loads_df = pd.DataFrame()
equip_options = sorted({str(x) for x in loads_df.get("Equipment", pd.Series(["Flat"])).dropna().unique()}) or ["Flat", "Step"]

# ---------------------------
# MATCHING LOGIC
# ---------------------------
def compute_deadhead_rows(df: pd.DataFrame, t_lat: float, t_lon: float) -> pd.DataFrame:
    subset = df.dropna(subset=["DestLat", "DestLon"]).copy()
    if subset.empty:
        return subset
    subset["Deadhead"] = subset.apply(
        lambda r: haversine(t_lat, t_lon, float(r["DestLat"]), float(r["DestLon"])),
        axis=1
    )
    if "Rate" in subset.columns and "Miles" in subset.columns:
        subset["RPM"] = (pd.to_numeric(subset["Rate"], errors="coerce") /
                         pd.to_numeric(subset["Miles"], errors="coerce")).round(2)
        subset["Profit"] = (subset["Rate"] * subset["Miles"]) - (subset["Deadhead"] * 1.5)
    return subset

def match_loads(loads_df: pd.DataFrame, truck_zip: str, equipment: str, max_deadhead: int, auto_expand: bool):
    truck_zip = re.sub(r'[^0-9]', '', str(truck_zip))[:5]
    info = find_zip_info(truck_zip)
    if not info:
        st.warning("Invalid ZIP, using default location (Mobile, AL).")
        info = (30.6954, -88.0399, "Mobile", "AL")
    t_lat, t_lon, t_city, t_state = info

    subset = loads_df.copy()
    if "Equipment" in subset.columns:
        subset = subset[subset["Equipment"].astype(str).str.lower() == str(equipment).lower()]
    subset = compute_deadhead_rows(subset, t_lat, t_lon)
    if subset.empty:
        return pd.DataFrame(), max_deadhead, (t_lat, t_lon)

    hits = subset[subset["Deadhead"] <= max_deadhead].sort_values("Deadhead")
    if not hits.empty:
        trusted_brokers = read_csv("brokers.csv")["Broker"].tolist() if os.path.exists("brokers.csv") else []
        hits = hits[hits["Broker"].isin(trusted_brokers)] if trusted_brokers else hits
    return hits, max_deadhead, (t_lat, t_lon)

# ---------------------------
# RENDERING
# ---------------------------
def render_map(truck_latlon, rows: pd.DataFrame):
    if not truck_latlon:
        return
    tlat, tlon = truck_latlon
    layers = []
    if not rows.empty:
        drop = rows.dropna(subset=["DestLat", "DestLon"]).head(20).copy()
        if not drop.empty:
            drop["lat"] = drop["DestLat"].astype(float)
            drop["lon"] = drop["DestLon"].astype(float)
            layers.append(
                pdk.Layer(
                    "ScatterplotLayer",
                    data=drop,
                    get_position='[lon, lat]',
                    get_fill_color=[0, 122, 255, 160],
                    get_radius=6000,
                    pickable=True,
                    tooltip={"html": "<b>Drop Location</b><br>Load: {Pickup} → {Drop}<br>Rate: ${Rate}<br>Miles: {Miles}"}
                )
            )
    truck_df = pd.DataFrame([{"lat": tlat, "lon": tlon}])
    layers.append(
        pdk.Layer(
            "ScatterplotLayer",
            data=truck_df,
            get_position='[lon, lat]',
            get_fill_color=[255, 0, 0, 200],
            get_radius=7000,
            pickable=True,
            tooltip={"html": "<b>Truck Location</b>"}
        )
    )
    deck = pdk.Deck(
        map_style=MAP_STYLE,
        initial_view_state=pdk.ViewState(latitude=tlat, longitude=tlon, zoom=6),
        layers=layers
    )
    st.pydeck_chart(deck, use_container_width=True)

def load_card(row, i):
    key = f"booked_{i}"
    if key not in st.session_state:
        st.session_state[key] = False
    booked = st.session_state[key]

    rate = row.get("Rate", "")
    miles = row.get("Miles", "")
    rpm = row.get("RPM", "")
    profit = row.get("Profit", 0)
    pickup = row.get("Pickup", "")
    drop = row.get("Drop", "")
    broker = row.get("Broker", "")
    b_email = str(row.get("BrokerEmail", "") or "")
    b_phone = str(row.get("BrokerPhone", "") or "")

    c1, c2, c3, c4 = st.columns([3, 2, 2, 2])

    with c1:
        st.markdown(f"**{pickup or 'Origin ?'} → {drop or 'Destination ?'}**")
        st.caption(f"Equip: {row.get('Equipment', '?')} • Deadhead: {round(row.get('Deadhead', 0), 1)} mi")

    with c2:
        st.metric("Rate", f"${int(rate):,}" if pd.notna(rate) else "—")
        st.metric("Profit", f"${int(profit):,}" if pd.notna(profit) else "—")

    with c3:
        st.metric("Miles", f"{int(miles):,}" if pd.notna(miles) else "—")
        st.metric("RPM", f"{rpm:.2f}" if pd.notna(rpm) else "—")

    with c4:
        subject = f"BBE Interest in Load - {pickup} to {drop}"
        body = f"Hello {broker or ''},%0D%0A%0D%0AWe are interested in this load for a nearby truck.%0D%0AThanks,%0D%0ABBE"
        st.markdown(
            f"[📧 Email](mailto:{b_email if booked else '***@***'}?subject={subject}&body={body}) &nbsp;&nbsp; "
            f"[📞 Call](tel:{b_phone})",
            unsafe_allow_html=True
        )
        clicked = st.toggle("Booked", value=booked, key=key)
        st.session_state[key] = clicked

    st.divider()

# ---------------------------
# UI: SINGLE / MULTI
# ---------------------------
if view_mode == "Single Truck":
    st.subheader("Single Truck Search")
    col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
    with col1:
        truck_zip = st.text_input("Truck Delivery ZIP", DEFAULT_ZIP, key="truck_zip")
    with col2:
        equipment = st.selectbox("Equipment", equip_options, index=0, key="equip")
    with col3:
        max_deadhead = st.slider("Max Deadhead (mi)", 50, 300, 200, step=25, key="max_deadhead")
    with col4:
        auto_expand = st.checkbox("Auto-expand if no matches", value=True, key="expand")
        pickup_date = st.date_input("Pickup Date", value=datetime.now(), key="pickup_date")

    if st.button("Find Loads", key="find_loads"):
        results, used_radius, truck_latlon = match_loads(loads_df, truck_zip, equipment, max_deadhead, auto_expand)
        if results.empty:
            st.warning(f"No loads found within {used_radius} miles.")
        else:
            st.success(f"Found {len(results)} load(s) within {used_radius} miles.")
            with st.expander("Map (truck and drops)", expanded=False):
                render_map(truck_latlon, results)
            for i, (_, r) in enumerate(results.head(25).iterrows()):
                load_card(r, i)
            if st.button("Download Matches", key="download"):
                csv = results.to_csv(index=False)
                st.download_button("Download CSV", csv, "matches.csv", "text/csv")
            if st.button("Refresh Loads", key="refresh"):
                st.experimental_rerun()

else:
    st.subheader("Multi-Truck Search")
    trucks_file = st.file_uploader("Upload Trucks CSV", type="csv", key="trucks_csv")
    top_n = st.number_input("Top matches per truck", 1, 20, 3, key="topn")

    if trucks_file is not None:
        try:
            tdf_raw = read_csv(trucks_file)
            tdf = tdf_raw.rename(columns={c: c.strip() for c in tdf_raw.columns})
            zip_col = first_col(tdf, "zip", "truck_zip", "delivery_zip")
            equip_col = first_col(tdf, "equipment", "equip", "type", "trailer")
            if not zip_col:
                st.error("Truck CSV must have a ZIP column (e.g., ZIP / truck_zip / delivery_zip).")
            else:
                all_rows = []
                for _, tr in tdf.iterrows():
                    tz = str(tr.get(zip_col, "")).strip()
                    eq = str(tr.get(equip_col, equip_options[0]))
                    hits, used_radius, _ = match_loads(loads_df, tz, eq, 100, True)
                    if not hits.empty:
                        hits = hits.head(int(top_n)).copy()
                        hits.insert(0, "TruckZIP", tz)
                        hits.insert(1, "EquipmentReq", eq)
                        all_rows.append(hits)
                if all_rows:
                    final_df = pd.concat(all_rows, ignore_index=True)
                    st.dataframe(final_df)
                    if st.button("Download All Matches", key="download_multi"):
                        csv = final_df.to_csv(index=False)
                        st.download_button("Download CSV", csv, "multi_matches.csv", "text/csv")
                else:
                    st.warning("No matches found for uploaded trucks.")
        except Exception as e:
            st.error(f"Failed to process trucks CSV: {e}")

# ---------------------------
# DIAGNOSTICS (optional)
# ---------------------------
with st.sidebar.expander("Diagnostics"):
    st.write("loads_df columns:", list(loads_df.columns))
    st.write("equip options:", equip_options[:10])
    if "Rate" in loads_df.columns:
        st.write("Sample Rates:", loads_df["Rate"].head().tolist())

