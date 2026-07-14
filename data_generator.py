"""
data_generator.py
Generates a realistic 24h x N-day synthetic microgrid dataset with:
  - Diurnal solar curve (bell-shaped, zero at night)
  - Demand with morning/evening peaks + Gaussian noise
  - Time-of-use electricity price with random spikes
  - Day-to-day variability (cloud cover, demand scaling)
  - day_of_week column (0=Mon … 6=Sun) for seasonal/weekly patterns
"""

import numpy as np
import pandas as pd


def generate_microgrid_data(n_days: int = 365, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    total_rows = n_days * 24
    hours_arr  = np.tile(np.arange(24), n_days)
    days_arr   = np.repeat(np.arange(n_days), 24)

    # Per-day variability
    cloud_factors = rng.uniform(0.1, 1.0, size=n_days)
    demand_scales = rng.uniform(0.8, 1.3, size=n_days)
    spike_hours   = rng.choice([-1, 17, 18, 19, 20],
                               size=n_days,
                               p=[0.5, 0.125, 0.125, 0.125, 0.125])

    # Weekend demand is ~15% lower
    day_of_week_arr = days_arr % 7   # 0=Mon … 6=Sun
    is_weekend      = (day_of_week_arr >= 5).astype(float)
    weekend_scale   = 1.0 - 0.15 * is_weekend[days_arr] if False else (
        np.where(day_of_week_arr >= 5, 0.85, 1.0)
    )

    # Solar: bell curve peaked at noon, zero outside 6-18
    solar_base = np.where(
        (hours_arr >= 6) & (hours_arr <= 18),
        150.0 * np.exp(-0.5 * ((hours_arr - 12) / 3.0) ** 2),
        0.0
    )
    solar_kw = np.maximum(
        0.0,
        solar_base * cloud_factors[days_arr] + rng.normal(0, 5, total_rows)
    )

    # Demand: morning peak (8h) + evening peak (19h) + base
    morning_peak = 40.0 * np.exp(-0.5 * ((hours_arr - 8)  / 1.5) ** 2)
    evening_peak = 60.0 * np.exp(-0.5 * ((hours_arr - 19) / 2.0) ** 2)
    demand = np.maximum(
        20.0,
        (60.0 + morning_peak + evening_peak)
        * demand_scales[days_arr]
        * weekend_scale
        + rng.normal(0, 8, total_rows)
    )

    # Price: time-of-use with random spike
    price = np.where((hours_arr >= 7)  & (hours_arr <= 9),  rng.uniform(7,  9,  total_rows),
            np.where((hours_arr >= 17) & (hours_arr <= 21), rng.uniform(9,  12, total_rows),
            np.where((hours_arr >= 23) | (hours_arr <= 5),  rng.uniform(2,  4,  total_rows),
                                                             rng.uniform(4,  7,  total_rows))))
    spike_mask = (hours_arr == spike_hours[days_arr])
    price = np.where(spike_mask, price * rng.uniform(1.5, 2.5, total_rows), price)

    df = pd.DataFrame({
        "day":            days_arr.astype("int32"),
        "day_of_week":    day_of_week_arr.astype("int32"),
        "hour":           hours_arr.astype("int32"),
        "predicted_load": np.round(demand,   2),
        "solar_kw":       np.round(solar_kw, 2),
        "price_per_kwh":  np.round(price,    3),
    })

    print(f"[data_generator] Generated {len(df)} rows — {n_days} days x 24 hours")
    return df


if __name__ == "__main__":
    df = generate_microgrid_data(n_days=60)
    df.to_csv("microgrid_data.csv", index=False)
    print("Saved to microgrid_data.csv")
