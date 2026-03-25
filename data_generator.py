"""
data_generator.py
Generates a realistic 24h × N-day synthetic microgrid dataset with:
  - Diurnal solar curve (bell-shaped, zero at night)
  - Demand with morning/evening peaks + Gaussian noise
  - Time-of-use electricity price with random spikes
  - Day-to-day variability (cloud cover, demand scaling)
"""

import numpy as np
import pandas as pd


def generate_microgrid_data(n_days: int = 60, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    records = []

    for day in range(n_days):
        # Per-day variability
        cloud_factor   = rng.uniform(0.1, 1.0)          # 0=overcast, 1=clear sky
        demand_scale   = rng.uniform(0.8, 1.3)          # high/low demand day
        price_spike_hr = rng.choice([-1, 17, 18, 19, 20], p=[0.5, 0.125, 0.125, 0.125, 0.125])

        for hour in range(24):
            # ── Solar (bell curve peaked at noon, zero outside 6-18) ──
            if 6 <= hour <= 18:
                solar_base = 150 * np.exp(-0.5 * ((hour - 12) / 3) ** 2)
            else:
                solar_base = 0.0
            solar_noise = rng.normal(0, 5)
            solar_kw    = max(0.0, solar_base * cloud_factor + solar_noise)

            # ── Demand (morning peak 7-9, evening peak 18-21) ──
            morning_peak = 40 * np.exp(-0.5 * ((hour - 8) / 1.5) ** 2)
            evening_peak = 60 * np.exp(-0.5 * ((hour - 19) / 2.0) ** 2)
            base_load    = 60.0
            demand_noise = rng.normal(0, 8)
            demand       = max(20.0, (base_load + morning_peak + evening_peak) * demand_scale + demand_noise)

            # ── Electricity price (time-of-use + spike) ──
            if 7 <= hour <= 9:
                price = rng.uniform(7, 9)
            elif 17 <= hour <= 21:
                price = rng.uniform(9, 12)
            elif 23 <= hour or hour <= 5:
                price = rng.uniform(2, 4)
            else:
                price = rng.uniform(4, 7)

            if hour == price_spike_hr:
                price *= rng.uniform(1.5, 2.5)          # random price spike

            records.append({
                "day":            day,
                "hour":           hour,
                "predicted_load": round(demand, 2),
                "solar_kw":       round(solar_kw, 2),
                "price_per_kwh":  round(price, 3),
            })

    df = pd.DataFrame(records)
    print(f"[data_generator] Generated {len(df)} rows — {n_days} days × 24 hours")
    print(df.describe().to_string())
    return df


if __name__ == "__main__":
    df = generate_microgrid_data(n_days=60)
    df.to_csv("microgrid_data.csv", index=False)
    print("\nSaved to microgrid_data.csv")
