import pandas as pd
import numpy as np

def load_ashrae_data(path):
    df = pd.read_csv(path)

    # Convert timestamp
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # 🔥 OPTIONAL: filter one building (VERY IMPORTANT)
    df = df[df["building_id"] == 0]

    # Extract time features
    df["hour"] = df["timestamp"].dt.hour
    df["day"] = df["timestamp"].dt.day

    # Demand
    df["predicted_load"] = df["meter_reading"]

    # 🔥 Handle missing values
    df = df.dropna()

    # 🔥 LIMIT DATA (important for training speed)
    df = df.iloc[:2000]

    # 🔥 SIMPLE SOLAR (temporary)
    df["solar_kw"] = np.maximum(0, 50 * np.sin((df["hour"] - 6) * np.pi / 12))

    # 🔥 PRICING
    def get_price(hour):
        if 18 <= hour <= 22:
            return 10
        elif 10 <= hour <= 17:
            return 6
        else:
            return 3

    df["price_per_kwh"] = df["hour"].apply(get_price)

    return df[["predicted_load", "solar_kw", "price_per_kwh", "hour", "day"]]