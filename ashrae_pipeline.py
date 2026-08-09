"""
ashrae_pipeline.py

Loads real ASHRAE building energy data and produces a DataFrame in the same
format as data_generator.py, so the RL environment works with both.

Differences from the old ashrae_loader.py:
  - Uses multiple well-behaved buildings (no zero-reading gaps)
  - Physics-based solar model (clear-sky + cloud attenuation)
  - Realistic TOU pricing with per-day variability
  - Adds day_of_week column
  - Fills demand gaps by interpolation rather than dropping rows
"""

from pathlib import Path

import numpy as np
import pandas as pd


# Simulation-ready data (~1 MB) committed to the repo so clones and deployments
# never need the 700 MB raw dataset. Regenerate: python ashrae_pipeline.py --rebuild
PRECOMPUTED_PATH = Path(__file__).resolve().parent / "data" / "simulation_data.csv"

# Buildings at site 0 with full-year electricity data and <5% zeros
GOOD_BUILDINGS = [105, 107, 108]

# Approximate latitude for ASHRAE site 0 (used in solar model)
SITE_LATITUDE_DEG = 30.0


# ─────────────────────────────────────────────────────────────────────────── #
#  Solar model                                                                 #
# ─────────────────────────────────────────────────────────────────────────── #

def _clear_sky_solar(timestamps: pd.Series, lat_deg: float = SITE_LATITUDE_DEG,
                     peak_kw: float = 150.0) -> np.ndarray:
    """
    Simplified clear-sky irradiance model.
    Returns kW output for a solar array with `peak_kw` rated capacity.
    """
    lat   = np.radians(lat_deg)
    doy   = timestamps.dt.dayofyear.values
    hour  = timestamps.dt.hour.values

    # Solar declination (radians)
    decl = np.radians(23.45 * np.sin(2 * np.pi * (doy - 81) / 365))

    # Hour angle (radians) — solar noon = 0
    ha = np.radians(15.0 * (hour - 12))

    # Solar elevation angle
    sin_elev = (np.sin(lat) * np.sin(decl)
                + np.cos(lat) * np.cos(decl) * np.cos(ha))
    elev = np.arcsin(np.clip(sin_elev, -1, 1))

    return np.maximum(0.0, peak_kw * np.sin(elev))


def _add_solar(df: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    """Attenuate clear-sky solar by cloud coverage from weather data."""
    clear_sky = _clear_sky_solar(df["timestamp"])

    # Merge cloud_coverage (okta 0-8; NaN when unknown)
    w = (weather[weather["site_id"] == 0][["timestamp", "cloud_coverage"]]
         .copy())
    w["timestamp"] = pd.to_datetime(w["timestamp"])
    # Forward-fill then back-fill NaN cloud values
    w["cloud_coverage"] = w["cloud_coverage"].ffill().bfill().fillna(3.0)
    w["cloud_frac"]     = (w["cloud_coverage"] / 8.0).clip(0, 1)

    df = df.merge(w[["timestamp", "cloud_frac"]], on="timestamp", how="left")
    df["cloud_frac"] = df["cloud_frac"].fillna(0.375)   # median okta 3/8

    df["solar_kw"] = np.round(
        clear_sky * (1.0 - 0.75 * df["cloud_frac"])     # clouds cut ~75% at max
        + np.random.default_rng(0).normal(0, 3, len(df)).clip(-10, 10),
        2
    )
    df["solar_kw"] = df["solar_kw"].clip(lower=0.0)
    df.drop(columns=["cloud_frac"], inplace=True)
    return df


# ─────────────────────────────────────────────────────────────────────────── #
#  Pricing model                                                               #
# ─────────────────────────────────────────────────────────────────────────── #

def _add_pricing(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    rng   = np.random.default_rng(seed)
    hour  = df["timestamp"].dt.hour.values
    n     = len(df)

    price = np.where(
        (hour >= 7)  & (hour <= 9),  rng.uniform(7,  9,  n),
        np.where(
        (hour >= 17) & (hour <= 21), rng.uniform(9,  12, n),
        np.where(
        (hour >= 23) | (hour <= 5),  rng.uniform(2,  4,  n),
                                     rng.uniform(4,  7,  n)
        )))

    # Random daily spike in one evening hour
    days   = df["timestamp"].dt.date.unique()
    spikes = {d: rng.choice([-1, 18, 19, 20], p=[0.5, 1/6, 1/6, 1/6]) for d in days}
    spike_mask = df["timestamp"].dt.date.map(spikes).values == hour
    price = np.where(spike_mask, price * rng.uniform(1.5, 2.5, n), price)

    df["price_per_kwh"] = np.round(price, 3)
    return df


# ─────────────────────────────────────────────────────────────────────────── #
#  Main loader                                                                 #
# ─────────────────────────────────────────────────────────────────────────── #

def _build_from_raw(
    dataset_dir: str,
    building_ids: list,
    n_days: int,
    seed: int,
    real_inputs: bool,
) -> pd.DataFrame:
    """
    Build simulation data from the raw ASHRAE dataset (~700 MB).
    Only needed when regenerating the cache — see build_cache().
    """
    real = None
    if real_inputs:
        from external_data import load_real_inputs
        real = load_real_inputs()

    train   = pd.read_csv(f"{dataset_dir}/train.csv")
    weather = pd.read_csv(f"{dataset_dir}/weather_train.csv")
    weather["timestamp"] = pd.to_datetime(weather["timestamp"])

    # Keep electricity meter (meter == 0) for selected buildings
    elec = train[(train["meter"] == 0) & (train["building_id"].isin(building_ids))].copy()
    elec["timestamp"] = pd.to_datetime(elec["timestamp"])
    elec = elec.sort_values(["building_id", "timestamp"]).reset_index(drop=True)

    frames = []
    global_day = 0

    for bid in building_ids:
        bdf = elec[elec["building_id"] == bid].copy()

        # Interpolate zero / missing readings (sensor dropouts)
        bdf["meter_reading"] = bdf["meter_reading"].replace(0.0, np.nan)
        bdf["meter_reading"] = bdf["meter_reading"].interpolate("linear").bfill().ffill()

        # Ensure complete hourly index
        full_idx = pd.date_range(bdf["timestamp"].min(),
                                 bdf["timestamp"].max(), freq="h")
        bdf = (bdf.set_index("timestamp")
                  .reindex(full_idx)
                  .rename_axis("timestamp")
                  .reset_index())
        bdf["meter_reading"] = bdf["meter_reading"].interpolate("linear")

        bdf["predicted_load"] = np.round(bdf["meter_reading"], 2)
        bdf["hour"]           = bdf["timestamp"].dt.hour
        bdf["day_of_week"]    = bdf["timestamp"].dt.dayofweek   # 0=Mon

        if real is not None:
            # Merge measured solar / prices / carbon by calendar hour
            bdf["month"]        = bdf["timestamp"].dt.month
            bdf["day_of_month"] = bdf["timestamp"].dt.day
            bdf = bdf.merge(real, on=["month", "day_of_month", "hour"], how="left")
            for col in ("solar_kw", "price_per_kwh", "carbon_kg_kwh"):
                bdf[col] = bdf[col].ffill().bfill()
            bdf.drop(columns=["month", "day_of_month"], inplace=True)
        else:
            bdf = _add_solar(bdf, weather)
            bdf = _add_pricing(bdf, seed=seed + bid)

        # Assign sequential day numbers
        dates      = bdf["timestamp"].dt.date
        unique_days = sorted(dates.unique())
        if n_days is not None:
            unique_days = unique_days[:n_days]
        day_map    = {d: global_day + i for i, d in enumerate(unique_days)}
        global_day += len(unique_days)

        bdf = bdf[dates.isin(unique_days)].copy()
        bdf["day"] = bdf["timestamp"].dt.date.map(day_map).astype("int32")

        bdf["building_id"] = bid
        cols = ["building_id", "day", "day_of_week", "hour",
                "predicted_load", "solar_kw", "price_per_kwh"]
        if "carbon_kg_kwh" in bdf.columns:
            cols.append("carbon_kg_kwh")
        frames.append(bdf[cols])

    result = pd.concat(frames, ignore_index=True)
    result["day_of_week"] = result["day_of_week"].astype("int32")
    result["hour"]        = result["hour"].astype("int32")

    total_days = result["day"].nunique()
    print(f"[ashrae_pipeline] Built {len(result)} rows from raw ASHRAE — "
          f"{total_days} days from buildings {building_ids}")
    return result


# ─────────────────────────────────────────────────────────────────────────── #
#  Precomputed cache — what deployments and fresh clones actually use
# ─────────────────────────────────────────────────────────────────────────── #

def _slice_precomputed(df: pd.DataFrame, building_ids: list, n_days: int) -> pd.DataFrame:
    """Apply the same building / day filtering the raw builder does."""
    df = df[df["building_id"].isin(building_ids)].copy()

    frames, global_day = [], 0
    for bid in building_ids:
        bdf  = df[df["building_id"] == bid]
        days = sorted(bdf["day"].unique())
        if n_days is not None:
            days = days[:n_days]
        remap = {d: global_day + i for i, d in enumerate(days)}
        global_day += len(days)

        bdf = bdf[bdf["day"].isin(days)].copy()
        bdf["day"] = bdf["day"].map(remap).astype("int32")
        frames.append(bdf)

    out = pd.concat(frames, ignore_index=True)
    # Match the raw builder's dtypes exactly (CSV round-trip widens ints)
    for col in ("day_of_week", "hour"):
        out[col] = out[col].astype("int32")
    return out


def build_cache(dataset_dir: str = "dataset") -> pd.DataFrame:
    """Rebuild the precomputed CSV from the raw ASHRAE dataset."""
    df = _build_from_raw(dataset_dir=dataset_dir, building_ids=GOOD_BUILDINGS,
                         n_days=None, seed=42, real_inputs=True)
    PRECOMPUTED_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PRECOMPUTED_PATH, index=False)
    size_mb = PRECOMPUTED_PATH.stat().st_size / 1e6
    print(f"[ashrae_pipeline] Cache written -> {PRECOMPUTED_PATH} ({size_mb:.1f} MB)")
    return df


def load_ashrae_data(
    dataset_dir: str = "dataset",
    building_ids: list = None,
    n_days: int = None,
    seed: int = 42,
    real_inputs: bool = True,
) -> pd.DataFrame:
    """
    Load simulation-ready microgrid data.

    Uses the small precomputed cache (data/simulation_data.csv, ~1 MB) when it
    exists, so clones and deployments run without the 700 MB raw ASHRAE dump.
    Falls back to building from raw when the cache is absent or when synthetic
    solar/pricing is requested (real_inputs=False).

    Columns: building_id, day, day_of_week, hour, predicted_load, solar_kw,
             price_per_kwh, carbon_kg_kwh

    Parameters
    ----------
    dataset_dir  : folder holding train.csv / weather_train.csv (raw build only)
    building_ids : buildings to include (default: GOOD_BUILDINGS)
    n_days       : cap days per building (None = all available)
    seed         : random seed for synthetic pricing (real_inputs=False only)
    real_inputs  : use measured solar/price/carbon (see external_data.py)
    """
    if building_ids is None:
        building_ids = GOOD_BUILDINGS

    if real_inputs and PRECOMPUTED_PATH.exists():
        df  = _slice_precomputed(pd.read_csv(PRECOMPUTED_PATH), building_ids, n_days)
        print(f"[ashrae_pipeline] Loaded {len(df)} rows from cache — "
              f"{df['day'].nunique()} days from buildings {building_ids}")
        return df

    return _build_from_raw(dataset_dir, building_ids, n_days, seed, real_inputs)


if __name__ == "__main__":
    import sys

    if "--rebuild" in sys.argv:
        df = build_cache()
    else:
        df = load_ashrae_data()

    print(df.head())
    print("\nDemand stats:")
    print(df["predicted_load"].describe())
    print("\nSolar stats:")
    print(df["solar_kw"].describe())
