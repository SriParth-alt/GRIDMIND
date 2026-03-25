import os
import pandas as pd


class PricingModel:

    def __init__(self, base_year=2016, escalation_rate=0.03):
        self.base_year = base_year
        self.escalation_rate = escalation_rate

        self.data_path = os.path.join("data", "pricing")
        os.makedirs(self.data_path, exist_ok=True)

    def get_base_price(self, hour):
        """
        Time-of-Use pricing (₹ per kWh)
        """

        if 0 <= hour < 6:
            return 5     # Off-peak
        elif 6 <= hour < 18:
            return 8     # Normal
        elif 18 <= hour < 22:
            return 12    # Peak
        else:
            return 6     # Late night

    def generate_yearly_pricing(self, year):

        file_path = os.path.join(
            self.data_path,
            f"pricing_{year}.csv"
        )

        if os.path.exists(file_path):
            return pd.read_csv(file_path, parse_dates=["timestamp"])

        print(f"Generating pricing data for {year}...")

        date_range = pd.date_range(
            start=f"{year}-01-01",
            end=f"{year}-12-31 23:00:00",
            freq="h"
        )

        escalation_multiplier = (1 + self.escalation_rate) ** (year - self.base_year)

        prices = []

        for timestamp in date_range:
            base_price = self.get_base_price(timestamp.hour)
            escalated_price = base_price * escalation_multiplier
            prices.append(escalated_price)

        df = pd.DataFrame({
            "timestamp": date_range,
            "price_per_kwh": prices
        })

        df.to_csv(file_path, index=False)
        return df