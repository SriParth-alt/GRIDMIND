import os
import requests
import pandas as pd


class SolarModel:

    def __init__(
        self,
        location_name,
        latitude,
        longitude,
        year,
        panel_area=500,
        panel_efficiency=0.20,
        derating_factor=0.85
    ):
        self.location_name = location_name
        self.latitude = latitude
        self.longitude = longitude
        self.year = year
        self.panel_area = panel_area
        self.panel_efficiency = panel_efficiency
        self.derating_factor = derating_factor

        self.data_path = os.path.join("data", "renewable")
        os.makedirs(self.data_path, exist_ok=True)

        self.file_path = os.path.join(
            self.data_path,
            f"solar_{self.location_name}_{self.year}.csv"
        )

    def fetch_nasa_power_data(self):

        print(f"Fetching solar data for {self.location_name} - {self.year}")

        url = (
            "https://power.larc.nasa.gov/api/temporal/hourly/point?"
            f"parameters=ALLSKY_SFC_SW_DWN&"
            f"community=RE&"
            f"longitude={self.longitude}&"
            f"latitude={self.latitude}&"
            f"start={self.year}0101&"
            f"end={self.year}1231&"
            f"format=JSON"
        )

        response = requests.get(url, timeout=60)
        response.raise_for_status()

        data = response.json()
        ghi = data["properties"]["parameter"]["ALLSKY_SFC_SW_DWN"]

        df = pd.DataFrame({
            "timestamp": pd.to_datetime(list(ghi.keys()), format="%Y%m%d%H"),
            "ghi": list(ghi.values())
        })

        df.to_csv(self.file_path, index=False)
        return df

    def load_data(self):

        if not os.path.exists(self.file_path):
            return self.fetch_nasa_power_data()

        return pd.read_csv(self.file_path, parse_dates=["timestamp"])

    def compute_solar_power(self, df):

        df["solar_kw"] = (
            self.panel_area
            * self.panel_efficiency
            * self.derating_factor
            * df["ghi"]
        )

        df["solar_kw"] = df["solar_kw"].clip(lower=0)
        return df

    def get_solar_generation(self):

        df = self.load_data()
        df = self.compute_solar_power(df)
        return df[["timestamp", "solar_kw"]]
# import os
# import requests
# import pandas as pd
# import numpy as np

# class SolarModel:
#     def __init__(
#         self,
#         latitude=13.0827,
#         longitude=80.2707,
#         panel_area=500,
#         panel_efficiency=0.20,
#         derating_factor=0.85
#     ):
#         self.latitude = latitude
#         self.longitude = longitude
#         self.panel_area = panel_area
#         self.panel_efficiency = panel_efficiency
#         self.derating_factor = derating_factor
#         self.data_path = os.path.join("data", "renewable")
#         os.makedirs(self.data_path, exist_ok=True)
#         self.file_path = os.path.join(self.data_path, "solar_chennai.csv")

#     def fetch_nasa_power_data(self, start_year=2016, end_year=2016):
#         print(f"Fetching NASA POWER data for Lat: {self.latitude}, Lon: {self.longitude}...")

#         # Construct parameters for the API call
#         params = {
#             "parameters": "ALLSKY_SFC_SW_DWN,T2M,CLOUD_AMT",
#             "community": "RE",
#             "longitude": self.longitude,
#             "latitude": self.latitude,
#             "start": f"{start_year}0101",
#             "end": f"{end_year}1231",
#             "format": "JSON"
#         }
        
#         url = "https://power.larc.nasa.gov/api/temporal/hourly/point"

#         try:
#             response = requests.get(url, params=params, timeout=60)
#             response.raise_for_status()
#             data = response.json()
            
#             # Validation: Check if the API returned an error message in the JSON
#             if "messages" in data and len(data["messages"]) > 0:
#                 print(f"NASA Warning: {data['messages']}")
                
#         except Exception as e:
#             raise RuntimeError(f"API Request failed: {e}")

#         # Extracting data safely
#         try:
#             param_data = data["properties"]["parameter"]
#             # Create a list of all timestamps from one of the parameters
#             timestamps = list(param_data["ALLSKY_SFC_SW_DWN"].keys())
            
#             # Build the DataFrame
#             df = pd.DataFrame({
#                 "timestamp": pd.to_datetime(timestamps, format="%Y%m%d%H"),
#                 "ghi": [param_data["ALLSKY_SFC_SW_DWN"][t] for t in timestamps],
#                 "temperature": [param_data["T2M"][t] for t in timestamps],
#                 "cloud_cover": [param_data["CLOUD_AMT"][t] for t in timestamps]
#             })
            
#             # NASA uses -999 for missing values
#             df = df.replace(-999, np.nan).fillna(method='ffill')
            
#             df.to_csv(self.file_path, index=False)
#             print(f"Successfully saved {len(df)} rows to {self.file_path}")
#             return df

#         except KeyError as e:
#             print(f"Error parsing JSON structure: Missing key {e}")
#             return pd.DataFrame()

#     def load_data(self):
#         if not os.path.exists(self.file_path):
#             return self.fetch_nasa_power_data()
#         return pd.read_csv(self.file_path, parse_dates=["timestamp"])

#     def compute_solar_power(self, df):
#         # Calculation: P = Area * Efficiency * Irradiance * Derating
#         # NASA's ALLSKY_SFC_SW_DWN is usually in kW-hr/m^2/day for daily, 
#         # but in Wh/m^2 for hourly. Ensure it's treated as W/m^2.
        
#         df["solar_kw"] = (
#             self.panel_area 
#             * self.panel_efficiency 
#             * self.derating_factor 
#             * df["ghi"]  # Assuming ghi is in W/m^2
#         ) / 1000.0

#         df["solar_kw"] = df["solar_kw"].clip(lower=0)
#         return df

#     def get_solar_generation(self):
#         df = self.load_data()
#         if df.empty:
#             return None
#         df = self.compute_solar_power(df)
#         return df[["timestamp", "solar_kw"]]

# # Test execution
# if __name__ == "__main__":
#     model = SolarModel()
#     results = model.get_solar_generation()
#     if results is not None:
#         print(results.head())