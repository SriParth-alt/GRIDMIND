import requests
import pandas as pd

def fetch_nasa_solar(lat=13.0, lon=80.0):
    url = "https://power.larc.nasa.gov/api/temporal/hourly/point"

    params = {
        "parameters": "ALLSKY_SFC_SW_DWN",
        "community": "RE",
        "longitude": lon,
        "latitude": lat,
        "start": "20220101",
        "end": "20220110",
        "format": "JSON"
    }

    response = requests.get(url, params=params)
    data = response.json()

    solar_data = data["properties"]["parameter"]["ALLSKY_SFC_SW_DWN"]

    df = pd.DataFrame(list(solar_data.items()), columns=["timestamp", "irradiance"])

    df["timestamp"] = pd.to_datetime(df["timestamp"], format="%Y%m%d%H")

    # Convert irradiance → solar power (simple model)
    panel_efficiency = 0.2
    panel_area = 50  # m²

    df["solar_kw"] = df["irradiance"] * panel_efficiency * panel_area / 1000

    return df[["timestamp", "solar_kw"]]