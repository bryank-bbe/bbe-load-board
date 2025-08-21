

# app.py — BBE Load Match PRO (v7.2 – Polished UI, fixed normalize_loads + toggle key)

import sys, traceback
try:
    import streamlit as st
    import pandas as pd
    import numpy as np
    import pydeck as pdk
    import os, re, hashlib, logging
    from datetime import datetime
except Exception as e:
    print("STARTUP IMPORT ERROR:", e)
    traceback.print_exc()
    sys.exit(1)

# -----------------------------------------------------------------------------
# PAGE CONFIG / THEME TWEAKS
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="BBE Load Match PRO",
    page_icon="🚛",
    layout="wide"
)

st.markdown("""
<style>
.block-container {padding-top: 1.2rem;}
.badge {display:inline-block; padding:3px 10px; border-radius:999px; font-size:11px; background:#eef2ff; color:#1e40af; font-weight:600;}
.badge-green {background:#ecfdf5; color:#065f46;}
.badge-amber {background:#fffbeb; color:#92400e;}
.load-card {border:1px solid #e5e7eb; border-radius:16px; padding:14px 16px; margin-bottom:10px; background:#fff; box-shadow:0 1px 0 rgba(0,0,0,0.03);}
.load-title {font-weight:700; font-size:16px; margin:0;}
.load-sub {color:#6b7280; font-size:12px;}
.metric {font-weight:700; font-size:18px;}
.kv {color:#6b7280; font-size:12px;}
hr {border:none; height:1px; background:#f3f4f6; margin:8px 0;}
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# HEADER / LOGGING
# -----------------------------------------------------------------------------
st.markdown("### 🚛 BBE Load Match **PRO**")
st.caption("Find reloads fast. Prioritize profit. Keep deadhead tight.")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# CONSTANTS / ENV
# -----------------------------------------------------------------------------
DEFAULT_ZIP = os.getenv("DEFAULT_ZIP", "36602")
MAP_STYLE = os.getenv("MAP_STYLE", "mapbox://styles/mapbox/light-v9")

# -----------------------------------------------------------------------------
# SIDEBAR — SETTINGS
# -----------------------------------------------------------------------------
with st.sidebar:
    st.subheader("Settings")
    source = st.radio("Loads Source", ["Sample CSV", "Upload CSV"], index=0, horizontal=True)
    uploaded_loads = st.file_uploader("Upload Loads CSV", type=["csv"]) if source == "Upload CSV" else None

    st.divider()
    st.subheader("Search Radius")
    step = st.slider("Auto-expand step (mi)", 25, 150, 50, step=25)
    ceiling = st.slider("Max radius (mi)", 100, 500, 300, step=25)

    st.divider()
    st.subheader("Costs & Ranking")
    cost_per_mile = st.number_input("Cost per mile (est.)", min_value=0.0, value=1.50, step=0.05)
    prefer_se = st.toggle("Prefer Southeast lanes (AL/MS/LA/GA/FL panhandle)", value=True)

    st.divider()
    st.subheader("View")
    view_mode = st.radio("Mode", ["Single Truck", "Multi-Truck"])

# -----------------------------------------------------------------------------
# HELPERS
# -----------------------------------------------------------------------------
@st.cache_data(ttl=300)
def read_csv(file_or_path, **kwargs):
    return pd.read_csv(file_or_path, **kwargs)

def first_col(df: pd.DataFrame, *candidates):
    lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand in lower:
            return lower[cand]
    return None

@st.cache_data(ttl=3600)
def get_zip_db():
    try:
        z = read_csv("zip_db.csv")
        z = z.rename(columns={c: c.strip().upper() for c in z.columns})
        if "LONG" in z.columns and "LNG" not in z.columns: z["LNG"] = z["LONG"]
        if "LON" in z.columns and "LNG" not in z.columns: z["LNG"] = z["LON"]
        need = {"ZIP","LAT","LNG"}
        if not need.issubset(z.columns):
            st.warning("zip_db.csv requires ZIP, LAT, LNG.")
        z["ZIP"] = pd.to_numeric(z["ZIP"], errors="coerce").astype("Int64")
        z["LAT"] = pd.to_numeric(z["LAT"], errors="coerce")
        z["LNG"] = pd.to_numeric(z["LNG"], errors="coerce")
        return z
    except Exception as e:
        st.error(f"Failed to read zip_db.csv: {e}")
        return pd.DataFrame(columns=["ZIP","LAT","LNG","CITY","STATE"])

def find_zip_info(zip_code: str):
    try:
        z = int(str(zip_code).strip())
    except Exception:
        return None
    db = get_zip_db()
    if db.empty: return None
    row = db.loc[db["ZIP"] == z]
    if row.empty: return None
    lat = float(row.iloc[0]["LAT"]) if pd.notna(row.iloc[0]["LAT"]) else None
    lng = float(row.iloc[0]["LNG"]) if pd.notna(row.iloc[0]["LNG"]) else None
    city = row.iloc[0]["CITY"] if "CITY" in db.columns else ""
    state = row.iloc[0]["STATE"] if "STATE" in db.columns else ""
    if lat is None or lng is None: return None
    return (lat, lng, city, state)

# -----------------------------------------------------------------------------
# LOADS (robust normalization) — FIXED DropZip iteration
# -----------------------------------------------------------------------------
@st.cache_data
def normalize_loads(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    d = df.copy()
    d.columns = [str(c).strip() for c in d.columns]

    # Required-ish columns
    must = {"Pickup","Drop","Rate","Miles"}
    if not must.intersection(set(d.columns)):
        st.error("Loads CSV must contain at least Pickup, Drop, Rate, and Miles columns.")
        return pd.DataFrame()

    # Equipment
    equip_col = first_col(d, "equipment","equip","trailer","trailertype","type")
    d["Equipment"] = d[equip_col].astype(str).str.strip() if equip_col else "Flat"

    # Rate (total $)
    rate_col = first_col(d, "rate","price","pay","freight","offer")
    if rate_col:
        parsed = (
            d[rate_col]
            .astype(str).str.lower()
            .str.replace(",", "", regex=False)
            .str.replace("$", "", regex=False)
            .str.replace("k", "000", regex=False)
            .str.extract(r"(-?\d+\.?\d*)")[0]
        )
        d["Rate"] = pd.to_numeric(parsed, errors="coerce")
    else:
        d["Rate"] = pd.NA

    # Miles (loaded)
    miles_col = first_col(d, "miles","distance","loaded_miles","mi")
    d["Miles"] = pd.to_numeric(d[miles_col], errors="coerce") if miles_col else pd.NA

    # Pickup / Drop labels
    pick_city = first_col(d, "pickup_city","pick_city","origin_city","from_city","pickup")
    pick_state = first_col(d, "pickup_state","pick_state","origin_state","from_state","pickup_st","pickupstate")
    drop_city = first_col(d, "drop_city","delivery_city","dest_city","to_city","drop")
    drop_state = first_col(d, "drop_state","delivery_state","dest_state","to_state","drop_st","dropstate")
    d["Pickup"] = d[[c for c in [pick_city] if c]].astype(str).agg(", ".join, axis=1) if pick_city else d.get("Pickup","")
    d["Drop"]   = d[[c for c in [drop_city] if c]].astype(str).agg(", ".join, axis=1) if drop_city else d.get("Drop","")
    if pick_state: d["Pickup"] = (d["Pickup"] + ", " + d[pick_state].astype(str)).str.strip(", ")
    if drop_state: d["Drop"]   = (d["Drop"] + ", " + d[drop_state].astype(str)).str.strip(", ")
    d["Pickup"] = d["Pickup"].replace(["nan","None"], "", regex=False)
    d["Drop"]   = d["Drop"].replace(["nan","None"], "", regex=False)

    # Broker info
    broker_col = first_col(d, "broker","company","contact","provider")
    email_col  = first_col(d, "broker_email","email","e-mail","mail")
    phone_col  = first_col(d, "broker_phone","phone","tel","telephone","cell")
    d["Broker"]      = d[broker_col].astype(str) if broker_col else ""
    d["BrokerEmail"] = d[email_col].astype(str) if email_col else ""
    d["BrokerPhone"] = d[phone_col].astype(str) if phone_col else ""

    # Destination Lat/Lon
    lat_col = first_col(d, "destlat","drop_lat","droplat","delivery_lat","dlat","lat2")
    lon_col = first_col(d, "destlon","drop_lon","droplon","delivery_lon","dlon","lng2","lon2")
    if lat_col and lon_col:
        d["DestLat"] = pd.to_numeric(d[lat_col], errors="coerce")
        d["DestLon"] = pd.to_numeric(d[lon_col], errors="coerce")
    else:
        zip_col = first_col(d, "drop_zip","dest_zip","delivery_zip","to_zip","zip_to","dropzip","zip")
        d["DropZip"] = d[zip_col].astype(str) if zip_col else ""
        # SAFE iteration over Series (no boolean ambiguity)
        ser = d["DropZip"] if "DropZip" in d.columns else pd.Series([], dtype="object")
        ser = ser.astype(str).fillna("")
        lat, lon = [], []
        for z in ser:
            info = find_zip_info(z) if z.strip() else None
            a, b = (info[0], info[1]) if info else (None, None)
            lat.append(a); lon.append(b)
        d["DestLat"] = lat if len(lat) == len(d) else pd.NA
        d["DestLon"] = lon if len(lon) == len(d) else pd.NA

    return d

def haversine_vec(lat1, lon1, lat2, lon2):
    R = 3959.0
    lat1 = np.radians(lat1); lon1 = np.radians(lon1)
    lat2 = np.radians(lat2); lon2 = np.radians(lon2)
    dlat = lat2 - lat1; dlon = lon2 - lon1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2.0)**2
    return 2 * R * np.arcsin(np.minimum(1, np.sqrt(a)))

def compute_deadhead_rows(df: pd.DataFrame, t_lat: float, t_lon: float) -> pd.DataFrame:
    subset = df.dropna(subset=["DestLat","DestLon"]).copy()
    if subset.empty:
        return subset
    subset["Deadhead"] = haversine_vec(
        t_lat, t_lon,
        subset["DestLat"].astype(float).values,
        subset["DestLon"].astype(float).values
    )
    if "Rate" in subset.columns and "Miles" in subset.columns:
        rate_num = pd.to_numeric(subset["Rate"], errors="coerce")
        miles_num = pd.to_numeric(subset["Miles"], errors="coerce").replace(0, np.nan)
        subset["RPM"] = (rate_num / miles_num).round(2)
    return subset

def add_profit(df: pd.DataFrame, cpm: float) -> pd.DataFrame:
    if df.empty:
        return df
    rate_num = pd.to_numeric(df["Rate"], errors="coerce")
    miles_num = pd.to_numeric(df["Miles"], errors="coerce")
    deadhead_num = pd.to_numeric(df["Deadhead"], errors="coerce")
    total_miles = (miles_num.fillna(0) + deadhead_num.fillna(0))
    est_cost = cpm * total_miles
    df["Profit"] = (rate_num.fillna(0) - est_cost).round(0)
    return df

def se_lane_boost(city_state: str) -> int:
    if not city_state: return 0
    val = city_state.lower()
    states = [" al", " ms", " la", " ga", " fl"]
    return 1 if any(val.endswith(s) or f", {s.strip()}" in val for s in states) else 0

def rank_results(df: pd.DataFrame, prefer_se: bool) -> pd.DataFrame:
    if df.empty: return df
    out = df.copy()
    out["SEBoost"] = out["Drop"].astype(str).apply(se_lane_boost) if prefer_se else 0
    return out.sort_values(by=["SEBoost","Profit","Deadhead"], ascending=[False, False, True])

def load_source_df(uploaded_file):
    if uploaded_file is not None:
        return read_csv(uploaded_file)
    try:
        return read_csv("sample_loads.csv")
    except Exception:
        return pd.DataFrame()

# -----------------------------------------------------------------------------
# CARDS / MAP RENDER
# -----------------------------------------------------------------------------
def load_card(row):
    # Stable unique key per load
    load_id = str(row.get("Load ID", f"{row.get('Pickup','?')}-{row.get('Drop','?')}-{row.get('Rate','?')}"))
    key = f"booked_{hashlib.md5(load_id.encode()).hexdigest()[:8]}"
    booked = st.session_state.get(key, False)

    rate = row.get("Rate", "")
    miles = row.get("Miles", "")
    rpm = row.get("RPM", "")
    profit = row.get("Profit", 0)
    pickup = row.get("Pickup", "")
    drop = row.get("Drop", "")
    broker = str(row.get("Broker", "") or "")
    b_email = str(row.get("BrokerEmail", "") or "")
    b_phone = str(row.get("BrokerPhone", "") or "")
    deadhead = float(row.get("Deadhead", 0) or 0)

    with st.container():
        st.markdown('<div class="load-card">', unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns([3, 2, 2, 2])

        with c1:
            st.markdown(f'<div class="load-title">{pickup or "Origin ?"} → {drop or "Destination ?"}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="load-sub">Equip: {row.get("Equipment","?")} &nbsp;&nbsp; <span class="badge">Deadhead: {deadhead:.1f} mi</span></div>', unsafe_allow_html=True)

        with c2:
            st.markdown("**Rate**")
            st.markdown(f'<div class="metric">${int(rate):,}</div>' if pd.notna(rate) and rate != "" else "—", unsafe_allow_html=True)
            st.markdown('<div class="kv">Total</div>', unsafe_allow_html=True)

        with c3:
            st.markdown("**Profit**")
            st.markdown(f'<div class="metric">${int(profit):,}</div>' if pd.notna(profit) and profit != "" else "—", unsafe_allow_html=True)
            st.markdown(f'<div class="kv">CPM={cost_per_mile:.2f}</div>', unsafe_allow_html=True)

        with c4:
            st.markdown("**Miles / RPM**")
            rpm_txt = f"{float(rpm):.2f}" if pd.notna(rpm) and rpm != "" else "—"
            st.markdown(f'<div class="metric">{int(miles) if pd.notna(miles) and miles!="" else "—"} mi</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="kv">RPM: {rpm_txt}</div>', unsafe_allow_html=True)

        st.markdown("<hr/>", unsafe_allow_html=True)
        cta1, cta2, cta3 = st.columns([2,2,1])
        with cta1:
            subject = f"BBE Interest in Load - {pickup} to {drop}"
            body = f"Hello {broker or ''},%0D%0A%0D%0AWe are interested in this load for a nearby truck.%0D%0AThanks,%0D%0ABBE"
            st.markdown(f"[📧 Email Broker](mailto:{b_email}?subject={subject}&body={body})", unsafe_allow_html=True)
        with cta2:
            st.markdown(f"[📞 Call Broker](tel:{b_phone})", unsafe_allow_html=True)
        with cta3:
            # ✅ Do not write to session_state with same key as widget
            booked = st.toggle("Booked", value=booked, key=key)
        st.markdown('</div>', unsafe_allow_html=True)

def render_map(truck_latlon, rows: pd.DataFrame):
    if not truck_latlon:
        return
    tlat, tlon = truck_latlon
    layers = []

    if not rows.empty and {"DestLat","DestLon"}.issubset(rows.columns):
        drop = rows.dropna(subset=["DestLat","DestLon"]).head(400).copy()
        if not drop.empty:
            drop["lat"] = drop["DestLat"].astype(float)
            drop["lon"] = drop["DestLon"].astype(float)
            layers.append(
                pdk.Layer(
                    "ScatterplotLayer",
                    data=drop,
                    get_position='[lon, lat]',
                    get_radius=5500,
                    get_fill_color=[0, 122, 255, 160],
                    pickable=True,
                )
            )

    truck_df = pd.DataFrame([{"lat": tlat, "lon": tlon}])
    layers.append(
        pdk.Layer(
            "ScatterplotLayer",
            data=truck_df,
            get_position='[lon, lat]',
            get_radius=7000,
            get_fill_color=[255, 0, 0, 200],
            pickable=True,
        )
    )

    deck = pdk.Deck(
        map_style=MAP_STYLE,
        initial_view_state=pdk.ViewState(latitude=tlat, longitude=tlon, zoom=6),
        layers=layers,
        tooltip={"html": "<b>{Pickup}</b> → <b>{Drop}</b><br>Rate: ${Rate}<br>Miles: {Miles}<br>Deadhead: {Deadhead}"}
    )
    st.pydeck_chart(deck, use_container_width=True)

# -----------------------------------------------------------------------------
# DATA LOAD
# -----------------------------------------------------------------------------
def load_source_df(uploaded_file):
    if uploaded_file is not None:
        return read_csv(uploaded_file)
    try:
        return read_csv("sample_loads.csv")
    except Exception:
        return pd.DataFrame()

if source == "Upload CSV" and uploaded_loads is not None:
    raw_loads = load_source_df(uploaded_loads)
else:
    raw_loads = load_source_df(None)

loads_df = normalize_loads(raw_loads)
equip_options = sorted({str(x) for x in loads_df.get("Equipment", pd.Series(["Flat"])).dropna().unique()}) or ["Flat","Step"]

# -----------------------------------------------------------------------------
# UI FLOW
# -----------------------------------------------------------------------------
if view_mode == "Single Truck":
    st.subheader("Single Truck Search")
    c1, c2, c3, c4 = st.columns([1.2, 1, 1, 1.2])
    with c1:
        truck_zip = st.text_input("Truck Delivery ZIP", DEFAULT_ZIP, key="truck_zip")
    with c2:
        equipment = st.selectbox("Equipment", equip_options, index=0, key="equip")
    with c3:
        start_radius = st.slider("Start Radius (mi)", 25, 300, 150, step=25, key="start_radius")
    with c4:
        auto_expand = st.checkbox("Auto-expand if no matches", value=True, key="expand")

    go = st.button("🔎 Find Loads", type="primary")
    st.caption("Tip: tweak **Cost per mile** and **Prefer Southeast lanes** in the sidebar.")

    if go:
        info = find_zip_info(truck_zip)
        if not info:
            st.warning("Invalid ZIP. Using Mobile, AL.")
            info = (30.6954, -88.0399, "Mobile", "AL")
        t_lat, t_lon, t_city, t_state = info

        subset = loads_df.copy()
        if "Equipment" in subset.columns:
            subset = subset[subset["Equipment"].astype(str).str.lower() == equipment.lower()]

        subset = compute_deadhead_rows(subset, t_lat, t_lon)
        subset = add_profit(subset, cost_per_mile)

        if subset.empty:
            st.warning("No loads data available.")
        else:
            used = start_radius
            hits = pd.DataFrame()
            while True:
                hits = subset[subset["Deadhead"] <= used]
                if not hits.empty or not auto_expand or used >= ceiling:
                    break
                used += step

            if hits.empty:
                st.warning(f"No loads within {used} miles.")
            else:
                ranked = rank_results(hits, prefer_se)

                t_results, t_map, t_diag = st.tabs(["📋 Results", "🗺️ Map", "🛠 Diagnostics"])

                with t_results:
                    colA, colB, colC, colD = st.columns(4)
                    colA.metric("Matches", len(ranked))
                    colB.metric("Radius Used", f"{used} mi")
                    colC.metric("Median Profit", f"${int(ranked['Profit'].median()):,}" if not ranked['Profit'].empty else "—")
                    colD.metric("Median RPM", f"{ranked['RPM'].median():.2f}" if pd.notna(ranked['RPM']).any() else "—")

                    st.download_button(
                        "⬇️ Download Matches (CSV)",
                        data=ranked.to_csv(index=False).encode("utf-8"),
                        file_name="matches.csv",
                        mime="text/csv",
                        use_container_width=True
                    )

                    st.markdown("—")
                    for _, r in ranked.head(60).iterrows():
                        load_card(r)

                with t_map:
                    render_map((t_lat, t_lon), ranked)

                with t_diag:
                    st.write("Used radius:", used)
                    st.write("Columns:", list(ranked.columns))
                    st.dataframe(ranked.head(100), use_container_width=True)

else:
    st.subheader("Multi-Truck Search")
    trucks_file = st.file_uploader("Upload Trucks CSV (needs ZIP, Equipment)", type="csv")
    top_n = st.number_input("Top matches per truck", 1, 20, 3)

    if trucks_file is not None:
        try:
            tdf = read_csv(trucks_file)
            tdf = tdf.rename(columns={c: c.strip() for c in tdf.columns})
            zip_col = first_col(tdf, "zip","truck_zip","delivery_zip","ZIP")
            equip_col = first_col(tdf, "equipment","equip","type","trailer")

            if not zip_col:
                st.error("Truck CSV must have a ZIP column (ZIP / truck_zip / delivery_zip).")
            else:
                all_rows = []
                for _, tr in tdf.iterrows():
                    tz = str(tr.get(zip_col, "")).strip()
                    eq = str(tr.get(equip_col, "Flat"))
                    info = find_zip_info(tz)
                    if not info:
                        continue
                    t_lat, t_lon, *_ = info
                    sub = loads_df.copy()
                    if "Equipment" in sub.columns:
                        sub = sub[sub["Equipment"].astype(str).str.lower() == eq.lower()]
                    sub = compute_deadhead_rows(sub, t_lat, t_lon)
                    sub = add_profit(sub, cost_per_mile)
                    if sub.empty:
                        continue

                    used = 100
                    hits = pd.DataFrame()
                    while True:
                        hits = sub[sub["Deadhead"] <= used]
                        if not hits.empty or used >= ceiling:
                            break
                        used += step

                    if hits.empty:
                        continue
                    ranked = rank_results(hits, prefer_se).head(int(top_n)).copy()
                    ranked.insert(0, "TruckZIP", tz)
                    ranked.insert(1, "EquipmentReq", eq)
                    all_rows.append(ranked)

                if all_rows:
                    final_df = pd.concat(all_rows, ignore_index=True)
                    st.dataframe(final_df, use_container_width=True)
                    st.download_button(
                        "⬇️ Download All Matches (CSV)",
                        data=final_df.to_csv(index=False).encode("utf-8"),
                        file_name="multi_matches.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                else:
                    st.warning("No matches found for uploaded trucks.")
        except Exception as e:
            st.error(f"Failed to process trucks CSV: {e}")

