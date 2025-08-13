import streamlit as st
import pandas as pd
import math

st.set_page_config(page_title="BBE Load Match", layout="wide")
st.title("🚛 BBE Load Match (v6 – hardened)")

# =============== Sidebar ===============
st.sidebar.header("Settings")
load_source = st.sidebar.radio("Loads Source", ["Use sample", "Upload CSV"], index=0, key="load_source")
uploaded_loads = st.sidebar.file_uploader("Upload Loads CSV", type="csv", key="loads_csv") if load_source == "Upload CSV" else None
view_mode = st.sidebar.radio("View", ["Single Truck", "Multi-Truck"], index=0, key="view_mode")

# =============== Helpers ===============
def haversine(lat1, lon1, lat2, lon2):
    R = 3959.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    return 2 * R * math.asin(min(1, math.sqrt(a)))

@st.cache_data
def read_csv(file_or_path, **kwargs):
    return pd.read_csv(file_or_path, **kwargs)

def first_col(df: pd.DataFrame, *candidates):
    lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand in lower:
            return lower[cand]
    return None

# =============== Load ZIP DB (normalize headers) ===============
try:
    _zip_raw = read_csv("zip_db.csv")
    # normalize header names to upper for robust lookup
    _zip = _zip_raw.rename(columns={c: c.strip().upper() for c in _zip_raw.columns})
    # expected: ZIP, LAT, LNG (allow LONG, LON), CITY, STATE
    if "LONG" in _zip.columns and "LNG" not in _zip.columns: _zip["LNG"] = _zip["LONG"]
    if "LON" in _zip.columns and "LNG" not in _zip.columns: _zip["LNG"] = _zip["LON"]
    if "ZIP" not in _zip.columns or "LAT" not in _zip.columns or "LNG" not in _zip.columns:
        st.warning("zip_db.csv is missing required columns. Expected headers: ZIP, LAT, LNG, CITY, STATE.")
    # coerce types safely
    if "ZIP" in _zip.columns:
        _zip["ZIP"] = pd.to_numeric(_zip["ZIP"], errors="coerce").astype("Int64")
    if "LAT" in _zip.columns:
        _zip["LAT"] = pd.to_numeric(_zip["LAT"], errors="coerce")
    if "LNG" in _zip.columns:
        _zip["LNG"] = pd.to_numeric(_zip["LNG"], errors="coerce")
except Exception as e:
    st.error(f"Failed to read zip_db.csv: {e}")
    _zip = pd.DataFrame(columns=["ZIP","LAT","LNG","CITY","STATE"])

def find_zip_info(zip_code: str):
    try:
        z = int(str(zip_code).strip())
    except Exception:
        return None
    if _zip.empty or "ZIP" not in _zip.columns:
        return None
    row = _zip.loc[_zip["ZIP"] == z]
    if row.empty:
        return None
    lat = float(row.iloc[0]["LAT"]) if pd.notna(row.iloc[0]["LAT"]) else None
    lng = float(row.iloc[0]["LNG"]) if pd.notna(row.iloc[0]["LNG"]) else None
    city = row.iloc[0]["CITY"] if "CITY" in _zip.columns else ""
    state = row.iloc[0]["STATE"] if "STATE" in _zip.columns else ""
    if lat is None or lng is None:
        return None
    return lat, lng, city, state

# =============== Load loads file (sample or uploaded) ===============
try:
    raw_loads = read_csv(uploaded_loads) if uploaded_loads is not None else read_csv("sample_loads.csv")
except Exception as e:
    st.error(f"Failed to read loads CSV: {e}")
    raw_loads = pd.DataFrame()

def normalize_loads(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["Equipment","DestLat","DestLon"])
    d = df.copy()
    # Trim headers
    d.columns = [str(c).strip() for c in d.columns]

    # Equipment under many names
    equip_col = first_col(d, "equipment","equip","trailer","trailertype","type")
    if equip_col is None:
        d["Equipment"] = "Flat"
        st.info("No Equipment column found; defaulting to 'Flat' for all loads.")
    else:
        d["Equipment"] = d[equip_col].astype(str).str.strip()

    # Destination lat/lon or ZIP
    lat_col = first_col(d, "destlat","drop_lat","droplat","delivery_lat","dlat","lat2")
    lon_col = first_col(d, "destlon","drop_lon","droplon","delivery_lon","dlon","lng2","lon2")
    if lat_col and lon_col:
        d["DestLat"] = pd.to_numeric(d[lat_col], errors="coerce")
        d["DestLon"] = pd.to_numeric(d[lon_col], errors="coerce")
    else:
        zip_col = first_col(d, "drop_zip","dest_zip","delivery_zip","to_zip","zip_to","dropzip","zip")
        if zip_col is not None:
            def to_ll(z):
                info = find_zip_info(z) if pd.notna(z) else None
                return (info[0], info[1]) if info else (None, None)
            lat, lon = [], []
            for z in d[zip_col]:
                a,b = to_ll(z)
                lat.append(a); lon.append(b)
            d["DestLat"] = lat
            d["DestLon"] = lon
        else:
            d["DestLat"] = pd.NA
            d["DestLon"] = pd.NA
            st.warning("No destination lat/lon or destination ZIP found; cannot compute deadhead for these rows.")
    return d

loads_df = normalize_loads(raw_loads)

# Equipment options (safe)
equip_options = sorted({str(x) for x in loads_df.get("Equipment", pd.Series(["Flat"])).dropna().unique()}) or ["Flat","Step"]

def match_loads(loads_df: pd.DataFrame, truck_zip: str, equipment: str, start_radius: int, auto_expand: bool):
    info = find_zip_info(truck_zip)
    if not info:
        return pd.DataFrame(), start_radius
    t_lat, t_lon, _, _ = info

    subset = loads_df.copy()
    if "Equipment" in subset.columns:
        subset = subset[subset["Equipment"].astype(str).str.lower() == str(equipment).lower()]
    subset = subset.dropna(subset=["DestLat","DestLon"])
    if subset.empty:
        return pd.DataFrame(), start_radius

    radius = int(start_radius)
    while True:
        tmp = subset.copy()
        tmp["Deadhead"] = tmp.apply(lambda r: haversine(float(r["DestLat"]), float(r["DestLon"]), t_lat, t_lon)
                                    if pd.notna(r["DestLat"]) and pd.notna(r["DestLon"]) else 99999, axis=1)
        hits = tmp[tmp["Deadhead"] <= radius].sort_values("Deadhead")
        if not hits.empty or not auto_expand or radius >= 300:
            return hits, radius
        radius += 50

# =============== UI ===============
if view_mode == "Single Truck":
    st.subheader("Single Truck Search")
    col1, col2, col3, col4 = st.columns([1,1,1,1])
    with col1:
        truck_zip = st.text_input("Truck Delivery ZIP", "36602", key="truck_zip")
    with col2:
        equipment = st.selectbox("Equipment", equip_options, index=0, key="equip")
    with col3:
        start_radius = st.number_input("Start radius (mi)", 50, 300, 100, step=25, key="radius")
    with col4:
        auto_expand = st.checkbox("Auto-expand if no matches", value=True, key="expand")

    if st.button("Find Loads", key="find_loads"):
        results, used_radius = match_loads(loads_df, truck_zip, equipment, start_radius, auto_expand)
        if results.empty:
            st.warning(f"No loads found within {used_radius} miles.")
        else:
            st.success(f"Found {len(results)} load(s) within {used_radius} miles.")
            st.dataframe(results.reset_index(drop=True))

else:
    st.subheader("Multi-Truck Search")
    trucks_file = st.file_uploader("Upload Trucks CSV", type="csv", key="trucks_csv")
    top_n = st.number_input("Top matches per truck", 1, 20, 3, key="topn")

    if trucks_file is not None:
        try:
            tdf_raw = read_csv(trucks_file)
            # normalize truck CSV headers too
            tdf = tdf_raw.rename(columns={c: c.strip() for c in tdf_raw.columns})
            zip_col = first_col(tdf, "zip","truck_zip","delivery_zip")
            equip_col = first_col(tdf, "equipment","equip","type","trailer")
            if not zip_col:
                st.error("Truck CSV must have a ZIP column (e.g., ZIP / truck_zip / delivery_zip).")
            else:
                all_rows = []
                for _, tr in tdf.iterrows():
                    tz = str(tr.get(zip_col, "")).strip()
                    eq = str(tr.get(equip_col, equip_options[0]))
                    hits, used_radius = match_loads(loads_df, tz, eq, 100, True)
                    if not hits.empty:
                        hits = hits.head(int(top_n)).copy()
                        hits.insert(0, "TruckZIP", tz)
                        hits.insert(1, "EquipmentReq", eq)
                        all_rows.append(hits)
                if all_rows:
                    final_df = pd.concat(all_rows, ignore_index=True)
                    st.dataframe(final_df)
                else:
                    st.warning("No matches found for uploaded trucks.")
        except Exception as e:
            st.error(f"Failed to process trucks CSV: {e}")

# =============== Diagnostics (optional) ===============
with st.sidebar.expander("Diagnostics"):
    st.write("zip_db.csv columns:", list(_zip.columns))
    st.write("loads_df columns:", list(loads_df.columns))
    st.write("Equip options:", equip_options[:10])
