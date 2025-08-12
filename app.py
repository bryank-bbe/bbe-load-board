
import streamlit as st
import pandas as pd
from math import radians, cos, sin, asin, sqrt
from datetime import datetime
import pydeck as pdk
import urllib.parse as urlenc
import os

st.set_page_config(page_title="BBE Load Match", page_icon="🚚", layout="wide")

SOUTHEAST = {"AL","MS","LA","GA","FL","SC","NC","TN"}

def haversine(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1; dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
    return 2 * 3956 * asin((a**0.5))

@st.cache_data
def load_zip_db_from_file(path):
    try:
        df = pd.read_csv(path, dtype={"zip":"str","lat":"float","lng":"float"})
        df = df.rename(columns={c:c.lower() for c in df.columns})
        if "lon" in df.columns and "lng" not in df.columns: df["lng"] = df["lon"]
        return df[["zip","lat","lng"]].dropna()
    except Exception:
        return None

@st.cache_data
def load_zip_db_upload(file):
    try:
        df = pd.read_csv(file, dtype={"zip":"str","lat":"float","lng":"float"})
        df = df.rename(columns={c:c.lower() for c in df.columns})
        if "lon" in df.columns and "lng" not in df.columns: df["lng"] = df["lon"]
        return df[["zip","lat","lng"]].dropna()
    except Exception:
        return None

def default_zip_df():
    return pd.DataFrame([
        {"zip":"36602","lat":30.6944,"lng":-88.0431},
        {"zip":"39201","lat":32.2988,"lng":-90.1848},
        {"zip":"35203","lat":33.5186,"lng":-86.8104},
        {"zip":"32501","lat":30.4213,"lng":-87.2169},
        {"zip":"30301","lat":33.7490,"lng":-84.3880},
        {"zip":"31401","lat":32.0809,"lng":-81.0912},
        {"zip":"70560","lat":29.9941,"lng":-91.8187},
        {"zip":"36104","lat":32.3777,"lng":-86.3006},
        {"zip":"36507","lat":30.8810,"lng":-87.7770},
        {"zip":"39401","lat":31.2850,"lng":-89.2903},
        {"zip":"35242","lat":33.4090,"lng":-86.6730},
    ])

def latlng_for(zip_df, z):
    row = zip_df.loc[zip_df["zip"]==str(z).zfill(5)]
    if row.empty: return None
    r = row.iloc[0]; return float(r["lat"]), float(r["lng"])

def deadhead(zip_df, a_zip, b_zip):
    a = latlng_for(zip_df, a_zip); b = latlng_for(zip_df, b_zip)
    if not a or not b: return None
    return round(haversine(a[1], a[0], b[1], b[0]))

@st.cache_data
def load_sample_loads():
    return pd.DataFrame([
        {"Load ID":"TS-884211","Source":"Truckstop","Pickup Zip":"36602","Pickup City":"Mobile, AL","Delivery City":"Atlanta, GA","Delivery Zip":"30301","Miles":345,"Rate":1300,"Equip":"Flat","Tarp":False,"When":"2025-08-05 13:00","Broker":"ABC Logistics","Broker Email":"broker1@example.com","Broker Phone":"555-123-4567","Needs Review":False},
        {"Load ID":"EM-40219","Source":"Email","Pickup Zip":"36532","Pickup City":"Fairhope, AL","Delivery City":"Savannah, GA","Delivery Zip":"31401","Miles":420,"Rate":1600,"Equip":"Flat","Tarp":True,"When":"2025-08-05 15:30","Broker":"Sherwood","Broker Email":"dispatch@sherwoodlumber.com","Broker Phone":"413-289-7013","Needs Review":True},
        {"Load ID":"INT-90012","Source":"McLeod","Pickup Zip":"39201","Pickup City":"Jackson, MS","Delivery City":"New Iberia, LA","Delivery Zip":"70560","Miles":312,"Rate":1250,"Equip":"Flat","Tarp":False,"When":"2025-08-05 10:30","Broker":"Southern Freight","Broker Email":"loads@southernfreight.com","Broker Phone":"555-777-8888","Needs Review":False},
        {"Load ID":"TS-884355","Source":"Truckstop","Pickup Zip":"35203","Pickup City":"Birmingham, AL","Delivery City":"Savannah, GA","Delivery Zip":"31401","Miles":360,"Rate":1500,"Equip":"Flat","Tarp":False,"When":"2025-08-05 12:00","Broker":"XYZ Freight","Broker Email":"broker2@example.com","Broker Phone":"555-987-6543","Needs Review":True},
        {"Load ID":"TS-900999","Source":"Truckstop","Pickup Zip":"32501","Pickup City":"Pensacola, FL","Delivery City":"Montgomery, AL","Delivery Zip":"36104","Miles":169,"Rate":900,"Equip":"Step","Tarp":False,"When":"2025-08-05 11:30","Broker":"Delta Carriers","Broker Email":"broker3@example.com","Broker Phone":"555-654-3210","Needs Review":False},
    ])

def mailto_link(broker_email, subject, body):
    if not broker_email: return None
    return f"mailto:{broker_email}?subject={urlenc.quote(subject)}&body={urlenc.quote(body)}"

# Sidebar templates
st.sidebar.header("Templates")
trucks_template = pd.DataFrame([{"Truck ID":"", "Zip":"", "Equip":"Flat", "Home State":"AL", "Home Zip":"", "Days Out":"0"}])
st.sidebar.download_button("Download Trucks Template", trucks_template.to_csv(index=False).encode(), "trucks_template.csv", "text/csv")
loads_template = pd.DataFrame([{"Load ID":"", "Source":"Truckstop/Email/McLeod", "Pickup Zip":"", "Pickup City":"",
    "Delivery City":"", "Delivery Zip":"", "Miles":"", "Rate":"", "Equip":"Flat/Step", "Tarp":"True/False", "When":"YYYY-MM-DD HH:MM",
    "Broker":"", "Broker Email":"", "Broker Phone":"", "Needs Review":"False"}])
st.sidebar.download_button("Download Loads Template", loads_template.to_csv(index=False).encode(), "loads_template.csv", "text/csv")

# Data
st.sidebar.header("Data")
zip_df = load_zip_db_from_file("zip_db.csv")
if zip_df is not None:
    st.sidebar.success("ZIP DB loaded from zip_db.csv")
else:
    st.sidebar.info("No saved ZIP DB found. Upload one or use default.")
    zip_up = st.sidebar.file_uploader("US ZIP database (zip,lat,lng)", type=["csv"])
    zip_df = load_zip_db_upload(zip_up) if zip_up else default_zip_df()
    if zip_up and st.sidebar.button("Save as default (zip_db.csv)"):
        zip_up.seek(0); open("zip_db.csv","wb").write(zip_up.read()); st.sidebar.success("Saved.")

loads_source = st.sidebar.radio("Loads source", ["Use sample","Upload CSV"], index=0)
if loads_source == "Upload CSV":
    up = st.sidebar.file_uploader("Upload loads CSV", type=["csv"])
    loads_df = pd.read_csv(up, dtype=str) if up else load_sample_loads()
else:
    loads_df = load_sample_loads()

df = loads_df.copy()
for col in ["Miles","Rate"]:
    if col in df.columns: df[col] = pd.to_numeric(df[col], errors="coerce")
for c in ["Pickup Zip","Delivery Zip"]:
    if c in df.columns: df[c] = df[c].astype(str).str.zfill(5)

# Trucks
st.sidebar.header("Trucks")
trucks_up = st.sidebar.file_uploader("Upload trucks CSV", type=["csv"], help="Cols: Truck ID, Zip, Equip, Home State, Home Zip, Days Out")

# Controls
st.sidebar.header("Search Controls")
view = st.sidebar.radio("View", ["Single Truck","Multi-Truck"], index=1)
start_radius = st.sidebar.slider("Start radius (mi)", 50, 300, 100, 25)
auto_expand = st.sidebar.toggle("Auto-expand if few matches", True)
min_results = st.sidebar.slider("Target results per truck", 1, 10, 3)
max_radius = st.sidebar.slider("Max radius (mi)", 100, 600, 400, 50)
min_rpm = st.sidebar.number_input("Min RPM", value=2.5, step=0.1)
top_n = st.sidebar.slider("Top matches per truck (Multi)", 1, 10, 3)

st.sidebar.header("Filters")
se_only = st.sidebar.toggle("Southeast backhaul only", False)
home_only = st.sidebar.toggle("Home-time only", False)

st.sidebar.header("Home-Time Bias")
home_bias_weight = st.sidebar.slider("Home ZIP bias weight", 0.0, 1.0, 0.4, 0.05)
days_out_boost_after = st.sidebar.slider("Boost if days out ≥", 0, 14, 5, 1)

st.sidebar.header("Booking")
if "booked" not in st.session_state: st.session_state.booked = {}
dispatcher = st.sidebar.selectbox("Booked by", ["Bryan","Michelle","Greg","Diane","Dispatcher"], index=0)
hide_booked = st.sidebar.toggle("Hide booked", False)

def compute_matches_for_truck(zip_df, loads_df, delivery_zip, equip, home_state,
                              start_radius, auto_expand, min_results, max_radius, min_rpm,
                              se_only=False, home_only=False,
                              home_zip=None, days_out=0, home_bias_weight=0.4, days_out_boost_after=5):
    dfx = loads_df.copy()
    dfx["Deadhead"] = dfx["Pickup Zip"].apply(lambda z: deadhead(zip_df, delivery_zip, z))
    if "Equip" in dfx.columns: dfx = dfx[dfx["Equip"].fillna(equip)==equip]
    dfx = dfx[(dfx["Rate"]/dfx["Miles"].clip(lower=1)).fillna(0) >= float(min_rpm)]
    if se_only: dfx = dfx[dfx["Delivery City"].fillna("").str[-2:].str.upper().isin(SOUTHEAST)]
    if home_only: dfx = dfx[dfx["Delivery City"].fillna("").str[-2:].str.upper()==str(home_state).upper()]
    r = int(start_radius)
    filt = lambda dfi, R: dfi[(dfi["Deadhead"].notna()) & (dfi["Deadhead"] <= R)]
    matches = filt(dfx, r)
    while auto_expand and len(matches) < int(min_results) and r < int(max_radius):
        r += 50; matches = filt(dfx, r)
    def home_zip_bonus(row):
        if not home_zip: return 0.0
        dz = str(row.get("Delivery Zip","")).zfill(5)
        miles = deadhead(zip_df, home_zip, dz)
        if miles is None: return 0.0
        scale = 1.0 - min(max(miles, 0), 600)/600.0
        return round(scale * home_bias_weight, 3)
    def home_state_bonus(row):
        st2 = str(row.get("Delivery City","")).strip()[-2:]
        return 0.2 if st2.upper()==str(home_state).upper() else 0.0
    matches["RPM"] = (matches["Rate"]/matches["Miles"].clip(lower=1)).round(2)
    base = matches["RPM"] + matches.apply(home_state_bonus, axis=1) + matches.apply(home_zip_bonus, axis=1)
    if int(days_out) >= int(days_out_boost_after): base = base + 0.2
    matches["Score"] = base.round(2)
    matches["Booked"] = matches["Load ID"].map(lambda i: st.session_state.booked.get(i))
    if hide_booked: matches = matches[matches["Booked"].isna()]
    return matches.sort_values(["Score","RPM"], ascending=[False,False]), r

def mailto_link(broker_email, subject, body):
    if not broker_email: return None
    return f"mailto:{broker_email}?subject={urlenc.quote(subject)}&body={urlenc.quote(body)}"

def render_email_button(row, truck_id, delivery_zip, deadhead_miles):
    subj = f"BBE – Load Inquiry: {row.get('Pickup City','')} → {row.get('Delivery City','')} ({row.get('When','')})"
    body = f"""Hello {row.get('Broker','')},

I'm interested in this load:
Load ID: {row.get('Load ID','')}
Pickup: {row.get('Pickup City','')} ({row.get('Pickup Zip','')}) {row.get('When','')}
Delivery: {row.get('Delivery City','')} ({row.get('Delivery Zip','')})
Equipment: {row.get('Equip','')}{' • Tarp' if row.get('Tarp') else ''}
Rate: ${int(row.get('Rate',0)):,}
Miles: {row.get('Miles','')}
Deadhead from truck {truck_id} (ZIP {delivery_zip}): {deadhead_miles} mi

Please confirm availability.
Thanks,
Billy Barnes Enterprises
"""
    link = mailto_link(row.get("Broker Email",""), subj, body)
    if link: st.markdown(f"[📧 Email this load]({link})")

def render_email_summary_button(truck_id, to_email, top_rows):
    if not to_email: return
    lines = []
    for _, r in top_rows.iterrows():
        dead = int(r['Deadhead']) if pd.notna(r['Deadhead']) else None
        lines.append(f"- {r['Pickup City']} → {r['Delivery City']} | ${int(r['Rate']):,} | RPM {r['RPM']} | Deadhead {dead if dead is not None else '–'} mi | Load {r['Load ID']}")
    body = f"Hello,\n\nTop suggestions for {truck_id}:\n\n" + "\n".join(lines) + "\n\nThanks,\nBBE Dispatcher\n"
    link = mailto_link(to_email, f"BBE – Top loads for {truck_id}", body)
    st.markdown(f"[📧 Email summary for {truck_id}]({link})")

st.title("BBE Load Match")
st.caption("Multi-Truck • Progressive radius • Home-time bias • Map polylines + labels • Email actions")

if st.sidebar.radio("Mode", ["Operate","Deploy Guide"], index=0) == "Deploy Guide":
    st.header("Deploy on Streamlit Cloud")
    st.markdown("- Push files to GitHub → New app in Streamlit Cloud → select `app.py` → Deploy.\n- Upload `zip_db.csv` in-app and click **Save as default**.")

view = st.sidebar.radio("View", ["Single Truck","Multi-Truck"], index=1)

# Prepare loads DF
df = df = (pd.read_csv("sample_loads.csv") if not os.path.exists("sample_loads.csv") else pd.read_csv("sample_loads.csv")) if False else df

if view == "Single Truck":
    truck_id = st.text_input("Truck ID", "BBE-1023")
    delivery_zip = st.text_input("Truck Delivery ZIP", "36602")
    equip = st.selectbox("Equipment", ["Flat","Step"], index=0)
    home_state = st.text_input("Driver Home State (2-letter)", "AL")
    home_zip = st.text_input("Driver Home ZIP (optional)", "35242")
    days_out = st.number_input("Driver days out", min_value=0, value=3, step=1)
    to_email = st.text_input("Send summary to (optional)", "bryank@billybarnes.net")

    matches, final_r = compute_matches_for_truck(zip_df, df, delivery_zip, equip, home_state,
                                                 start_radius, auto_expand, min_results, max_radius, min_rpm,
                                                 se_only=se_only, home_only=home_only,
                                                 home_zip=home_zip, days_out=days_out,
                                                 home_bias_weight=home_bias_weight, days_out_boost_after=days_out_boost_after)
    st.subheader(f"Results for {truck_id} near {delivery_zip} • radius {final_r} mi • {len(matches)} matches")
    if matches.empty:
        st.warning("No matches. Try increasing radius or lowering RPM threshold.")
    else:
        top_rows = matches.head(5)
        for _, row in matches.iterrows():
            dead = int(row['Deadhead']) if pd.notna(row['Deadhead']) else None
            with st.expander(f"{row['Pickup City']} → {row['Delivery City']}  •  ${row['Rate']:,} • RPM {row['RPM']} • Deadhead {dead if dead is not None else '–'} mi"):
                st.write(f"**Load ID:** {row['Load ID']} | **Source:** {row.get('Source','N/A')} | **Equip:** {row.get('Equip','')} {'• Tarp' if row.get('Tarp') else ''}")
                st.write(f"**Pickup:** {row.get('Pickup City','')} ({row.get('Pickup Zip','')})")
                st.write(f"**Delivery:** {row.get('Delivery City','')} ({row.get('Delivery Zip','')})")
                st.write(f"**Miles:** {row.get('Miles','')} | **Rate:** ${int(row.get('Rate',0)):,} | **Score:** {row['Score']}")
                st.write(f"**Broker:** {row.get('Broker','')} | **Phone:** {row.get('Broker Phone','')} | **Email:** {row.get('Broker Email','')}")
                c1, c2, c3, c4 = st.columns([1,1,2,2])
                with c1:
                    if row["Load ID"] in st.session_state.booked:
                        st.button("Unbook", key=f"unbook_{row['Load ID']}", on_click=lambda i=row["Load ID"]: st.session_state.booked.pop(i, None))
                    else:
                        st.button("Book", key=f"book_{row['Load ID']}", on_click=lambda i=row["Load ID"]: st.session_state.booked.update({i: {'by': dispatcher, 'at': datetime.now().isoformat(timespec='seconds')}}))
                with c2: st.markdown(f"[📞 Call](tel:{row.get('Broker Phone','')})")
                with c3: st.markdown(f"[📧 Email](mailto:{row.get('Broker Email','')})")
                with c4: render_email_button(row, truck_id, delivery_zip, dead if dead is not None else 0)
        render_email_summary_button(truck_id, to_email, top_rows)

else:
    st.subheader("Multi-Truck")
    trucks_up = trucks_up or st.sidebar.file_uploader("Upload trucks CSV", type=["csv"])
    if not trucks_up:
        st.info("Upload Trucks CSV (Truck ID, Zip, Equip, Home State, Home Zip, Days Out).")
    else:
        trucks_df = pd.read_csv(trucks_up, dtype=str).rename(columns=lambda c: c.strip())
        cols = {c.lower(): c for c in trucks_df.columns}
        id_col = cols.get("truck id") or cols.get("truck_id") or list(trucks_df.columns)[0]
        zip_col = cols.get("zip") or cols.get("current zip") or cols.get("delivery zip") or list(trucks_df.columns)[1]
        eq_col = cols.get("equip") or cols.get("equipment")
        home_state_col = cols.get("home state") or cols.get("home_state")
        home_zip_col = cols.get("home zip") or cols.get("home_zip")
        days_out_col = cols.get("days out") or cols.get("days_out")

        summary_rows, map_lines, map_labels = [], [], []
        for _, r in trucks_df.iterrows():
            tid = str(r.get(id_col, ""))
            tzip = str(r.get(zip_col, "")).zfill(5)
            teq = str(r.get(eq_col, "Flat")) if eq_col else "Flat"
            thome_state = str(r.get(home_state_col, "AL")) if home_state_col else "AL"
            thome_zip = str(r.get(home_zip_col, "")) if home_zip_col else ""
            tdays_out = int(str(r.get(days_out_col, "0")).strip() or 0) if days_out_col else 0

            matches, final_r = compute_matches_for_truck(zip_df, df, tzip, teq, thome_state,
                                                         start_radius, auto_expand, min_results, max_radius, min_rpm,
                                                         se_only=se_only, home_only=home_only,
                                                         home_zip=thome_zip, days_out=tdays_out,
                                                         home_bias_weight=home_bias_weight, days_out_boost_after=days_out_boost_after)
            top = matches.head(int(top_n))
            with st.expander(f"{tid} • ZIP {tzip} • Equip {teq} • Home {thome_state} ({thome_zip or '—'}) • Days out {tdays_out} • radius {final_r} mi • matches: {len(matches)}"):
                if top.empty:
                    st.warning("No matches for this truck.")
                else:
                    for _, row in top.iterrows():
                        dead = int(row['Deadhead']) if pd.notna(row['Deadhead']) else None
                        st.write(f"**{row['Pickup City']} → {row['Delivery City']}** | ${int(row['Rate']):,} | RPM {row['RPM']} | Deadhead {dead if dead is not None else '–'} mi | Load {row['Load ID']}")
                        c1, c2, c3, c4 = st.columns([1,1,2,2])
                        with c1:
                            if row["Load ID"] in st.session_state.booked:
                                st.button("Unbook", key=f"unbook_{tid}_{row['Load ID']}", on_click=lambda i=row["Load ID"]: st.session_state.booked.pop(i, None))
                            else:
                                st.button("Book", key=f"book_{tid}_{row['Load ID']}", on_click=lambda i=row["Load ID"]: st.session_state.booked.update({i: {'by': thome_state, 'at': datetime.now().isoformat(timespec='seconds')}}))
                        with c2: st.markdown(f"[📞 Call](tel:{row.get('Broker Phone','')})")
                        with c3: st.markdown(f"[📧 Email](mailto:{row.get('Broker Email','')})")
                        with c4: render_email_button(row, tid, tzip, dead if dead is not None else 0)

                    # Map lines + distance labels for the best pickup
                    if not top.empty:
                        a = latlng_for(zip_df, tzip)
                        b = latlng_for(zip_df, top.iloc[0]["Pickup Zip"])
                        dead = int(top.iloc[0]["Deadhead"]) if pd.notna(top.iloc[0]["Deadhead"]) else None
                        if a and b:
                            map_lines.append({"from_lng":a[1],"from_lat":a[0],"to_lng":b[1],"to_lat":b[0]})
                            map_labels.append({"text": f"{dead if dead is not None else '–'} mi", "longitude": (a[1]+b[1])/2, "latitude": (a[0]+b[0])/2})

        if map_lines:
            st.subheader("Map: Trucks → best pickups (with distance labels)")
            lines_df = pd.DataFrame(map_lines)
            layers = [pdk.Layer("LineLayer", lines_df, get_source_position="[from_lng, from_lat]", get_target_position="[to_lng, to_lat]", width_min_pixels=2)]
            if map_labels:
                labels_df = pd.DataFrame(map_labels)
                layers.append(pdk.Layer("TextLayer", labels_df, get_position='[longitude, latitude]', get_text="text", get_size=16, get_color=[0,0,0], get_alignment_baseline='"bottom"'))
            init = pdk.ViewState(latitude=float(lines_df["from_lat"].iloc[0]), longitude=float(lines_df["from_lng"].iloc[0]), zoom=5)
            st.pydeck_chart(pdk.Deck(layers=layers, initial_view_state=init, map_style="mapbox://styles/mapbox/light-v9"))

        if 'summary_rows' in locals() and summary_rows:
            st.subheader("Summary")
            sum_df = pd.DataFrame(summary_rows)
            st.dataframe(sum_df, use_container_width=True, hide_index=True)
            st.download_button("Download summary CSV", sum_df.to_csv(index=False).encode(), "multi_truck_summary.csv", "text/csv")

st.subheader("Booked Loads (all)")
booked_rows = []
if "Delivery Zip" not in df.columns: df["Delivery Zip"] = ""
for _, row in df.iterrows():
    if row.get("Load ID") in st.session_state.booked:
        b = st.session_state.booked[row["Load ID"]]
        booked_rows.append({"Booked By": b["by"], "Booked At": b["at"], **{k: row.get(k, "") for k in ["Load ID","Source","Pickup City","Pickup Zip","Delivery City","Delivery Zip","Miles","Rate","Equip","Broker","Broker Phone","Broker Email"]}})
if booked_rows:
    exp_df = pd.DataFrame(booked_rows)
    st.download_button("Download booked loads (CSV)", exp_df.to_csv(index=False).encode(), "booked_loads.csv", "text/csv")
else:
    st.caption("No booked loads yet.")
