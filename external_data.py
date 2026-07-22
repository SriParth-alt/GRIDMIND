"""
external_data.py

Downloads and caches REAL 2016 hourly inputs to replace the synthetic parts
of the pipeline:

  1. Solar     — NASA POWER hourly irradiance (free API, no key) converted to
                 array output for a rated PV system.
  2. Prices    — NYISO day-ahead hourly LBMP for the N.Y.C. zone (public CSV
                 archive, no registration).
  3. Carbon    — real hourly grid CO2 intensity computed from NYISO's actual
                 5-minute fuel mix and standard per-fuel emission factors.

Everything is cached in data/external/ so the download runs once.

Usage:
    from external_data import load_real_inputs
    df = load_real_inputs()   # columns: month, day_of_month, hour,
                              #          solar_kw, price_per_kwh, carbon_kg_kwh
"""

import io
import json
import zipfile
from pathlib import Path
from urllib.request import urlopen, Request

import numpy as np
import pandas as pd

CACHE_DIR = Path(__file__).resolve().parent / "data" / "external"

YEAR      = 2016            # matches the ASHRAE dataset year
LATITUDE  = 40.71           # New York City
LONGITUDE = -74.01
PV_RATED_KW          = 150.0   # rated capacity of the simulated array
PV_PERFORMANCE_RATIO = 0.80    # inverter/soiling/temperature losses

NYISO_ZONE = "N.Y.C."

# kg CO2 per kWh generated, by NYISO fuel category (EPA/eGRID-typical values)
EMISSION_FACTORS = {
    "Natural Gas":        0.42,
    "Dual Fuel":          0.44,
    "Other Fossil Fuels": 0.90,
    "Other Renewables":   0.10,   # biomass / landfill gas
    "Nuclear":            0.0,
    "Hydro":              0.0,
    "Wind":               0.0,
}


def _get(url: str, timeout: int = 120) -> bytes:
    req = Request(url, headers={"User-Agent": "sustain-ai/1.0"})
    with urlopen(req, timeout=timeout) as r:
        return r.read()


# ─────────────────────────────────────────────────────────────────────────── #
#  1. NASA POWER solar
# ─────────────────────────────────────────────────────────────────────────── #

def fetch_solar(force: bool = False) -> pd.DataFrame:
    """Hourly PV output (kW) for the full year, local standard time."""
    cache = CACHE_DIR / f"solar_nyc_{YEAR}.csv"
    if cache.exists() and not force:
        return pd.read_csv(cache)

    print(f"[external_data] Downloading NASA POWER solar for {YEAR}…")
    url = (
        "https://power.larc.nasa.gov/api/temporal/hourly/point"
        f"?parameters=ALLSKY_SFC_SW_DWN&start={YEAR}0101&end={YEAR}1231"
        f"&latitude={LATITUDE}&longitude={LONGITUDE}"
        "&community=RE&time-standard=lst&format=JSON"
    )
    payload = json.loads(_get(url, timeout=300))
    vals = payload["properties"]["parameter"]["ALLSKY_SFC_SW_DWN"]

    rows = []
    for key, ghi in vals.items():            # key = "YYYYMMDDHH"
        ghi = max(0.0, float(ghi))           # -999 fill values → 0
        kw  = PV_RATED_KW * (ghi / 1000.0) * PV_PERFORMANCE_RATIO
        rows.append({
            "month":        int(key[4:6]),
            "day_of_month": int(key[6:8]),
            "hour":         int(key[8:10]),
            "solar_kw":     round(kw, 2),
        })

    df = pd.DataFrame(rows)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache, index=False)
    print(f"[external_data] Solar cached: {len(df)} hours -> {cache.name}")
    return df


# ─────────────────────────────────────────────────────────────────────────── #
#  2. NYISO day-ahead prices
# ─────────────────────────────────────────────────────────────────────────── #

def fetch_prices(force: bool = False) -> pd.DataFrame:
    """Hourly day-ahead LBMP for the N.Y.C. zone, converted to cents/kWh."""
    cache = CACHE_DIR / f"prices_nyiso_{YEAR}.csv"
    if cache.exists() and not force:
        return pd.read_csv(cache)

    print(f"[external_data] Downloading NYISO day-ahead prices for {YEAR}…")
    frames = []
    for month in range(1, 13):
        url = (f"http://mis.nyiso.com/public/csv/damlbmp/"
               f"{YEAR}{month:02d}01damlbmp_zone_csv.zip")
        z = zipfile.ZipFile(io.BytesIO(_get(url)))
        for name in sorted(z.namelist()):
            day = pd.read_csv(z.open(name))
            day = day[day["Name"] == NYISO_ZONE]
            frames.append(day[["Time Stamp", "LBMP ($/MWHr)"]])
        print(f"  month {month:02d} done")

    df = pd.concat(frames, ignore_index=True)
    ts = pd.to_datetime(df["Time Stamp"], format="%m/%d/%Y %H:%M")
    df = pd.DataFrame({
        "month":         ts.dt.month,
        "day_of_month":  ts.dt.day,
        "hour":          ts.dt.hour,
        # $/MWh -> cents/kWh, floored at 0 (rare negative prices)
        "price_per_kwh": (df["LBMP ($/MWHr)"].clip(lower=0) / 10.0).round(3),
    })
    # DST duplicates → average; missing hours filled later on merge
    df = df.groupby(["month", "day_of_month", "hour"], as_index=False).mean()

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache, index=False)
    print(f"[external_data] Prices cached: {len(df)} hours -> {cache.name}")
    return df


# ─────────────────────────────────────────────────────────────────────────── #
#  3. NYISO fuel mix → hourly carbon intensity
# ─────────────────────────────────────────────────────────────────────────── #

def fetch_carbon(force: bool = False) -> pd.DataFrame:
    """Real hourly grid CO2 intensity (kg/kWh) from the NYISO fuel mix."""
    cache = CACHE_DIR / f"carbon_nyiso_{YEAR}.csv"
    if cache.exists() and not force:
        return pd.read_csv(cache)

    print(f"[external_data] Downloading NYISO fuel mix for {YEAR}…")
    frames = []
    for month in range(1, 13):
        url = (f"http://mis.nyiso.com/public/csv/rtfuelmix/"
               f"{YEAR}{month:02d}01rtfuelmix_csv.zip")
        z = zipfile.ZipFile(io.BytesIO(_get(url)))
        for name in sorted(z.namelist()):
            frames.append(pd.read_csv(z.open(name)))
        print(f"  month {month:02d} done")

    df = pd.concat(frames, ignore_index=True)
    ts = pd.to_datetime(df["Time Stamp"], format="mixed")
    df["month"]        = ts.dt.month
    df["day_of_month"] = ts.dt.day
    df["hour"]         = ts.dt.hour
    df["ef"]           = df["Fuel Category"].map(EMISSION_FACTORS).fillna(0.5)
    df["co2"]          = df["Gen MWh"] * df["ef"]

    hourly = (df.groupby(["month", "day_of_month", "hour"])
                .agg(gen=("Gen MWh", "sum"), co2=("co2", "sum"))
                .reset_index())
    hourly["carbon_kg_kwh"] = (hourly["co2"] / hourly["gen"]).round(4)
    hourly = hourly[["month", "day_of_month", "hour", "carbon_kg_kwh"]]

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    hourly.to_csv(cache, index=False)
    print(f"[external_data] Carbon cached: {len(hourly)} hours -> {cache.name}")
    return hourly


# ─────────────────────────────────────────────────────────────────────────── #
#  Combined loader
# ─────────────────────────────────────────────────────────────────────────── #

def load_real_inputs(force: bool = False) -> pd.DataFrame:
    """
    One row per hour of the year with all three real inputs:
      month, day_of_month, hour, solar_kw, price_per_kwh, carbon_kg_kwh
    """
    solar  = fetch_solar(force)
    prices = fetch_prices(force)
    carbon = fetch_carbon(force)

    keys = ["month", "day_of_month", "hour"]
    df = solar.merge(prices, on=keys, how="left").merge(carbon, on=keys, how="left")
    df["price_per_kwh"] = df["price_per_kwh"].ffill().bfill()
    df["carbon_kg_kwh"] = df["carbon_kg_kwh"].ffill().bfill()
    return df


if __name__ == "__main__":
    df = load_real_inputs()
    print(df.head(30).to_string())
    print("\nHours:", len(df))
    print("\nSolar  kW    :", df["solar_kw"].describe().round(2).to_dict())
    print("Price  c/kWh :", df["price_per_kwh"].describe().round(2).to_dict())
    print("Carbon kg/kWh:", df["carbon_kg_kwh"].describe().round(3).to_dict())
