# BBE Load Match (v6)
Deploy steps:
1) Create a GitHub repo and upload: app.py, requirements.txt, sample_loads.csv, sample_trucks.csv, zip_db.csv
2) In Streamlit Cloud: New app → point to repo → main file `app.py` → Deploy
3) In the app: upload your full `zip_db.csv` once and click **Save as default**

CSV formats:
- Trucks: Truck ID,Zip,Equip,Home State,Home Zip,Days Out
- Loads: Load ID,Source,Pickup Zip,Pickup City,Delivery City,Delivery Zip,Miles,Rate,Equip,Tarp,When,Broker,Broker Email,Broker Phone,Needs Review
