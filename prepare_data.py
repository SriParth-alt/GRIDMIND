import pandas as pd
from ashrae_loader import load_ashrae_data
from nasa_solar import fetch_nasa_solar
from pricing import get_price


def prepare_final_data(ashrae_path):
    demand_df = load_ashrae_data(ashrae_path)
    solar_df = fetch_nasa_solar()

    # Merge on timestamp
    df = pd.merge(demand_df, solar_df, on="timestamp", how="inner")

    # Add pricing
    df["price_per_kwh"] = df["hour"].apply(get_price)

    return df