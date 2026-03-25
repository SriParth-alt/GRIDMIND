from models.solar_model import SolarModel
import pandas as pd
import matplotlib.pyplot as plt


def assign_season(month):
    if month in [1, 2]:
        return "Winter"
    elif month in [3, 4, 5, 6]:
        return "Summer"
    elif month in [7, 8, 9]:
        return "Monsoon"
    else:
        return "PostMonsoon"


def main():

    locations = {
        "north_chennai": (13.1600, 80.3000),
        "central_chennai": (13.0827, 80.2707),
        "south_chennai": (12.9500, 80.2000)
    }

    years = [2016, 2017, 2018, 2019, 2020]

    annual_results = []
    seasonal_results = []

    for name, coords in locations.items():
        for year in years:

            model = SolarModel(
                location_name=name,
                latitude=coords[0],
                longitude=coords[1],
                year=year
            )

            df = model.get_solar_generation()
            df["year"] = year
            df["month"] = df["timestamp"].dt.month
            df["season"] = df["month"].apply(assign_season)

            # ---- Annual Metrics ----
            mean_val = df["solar_kw"].mean()
            max_val = df["solar_kw"].max()
            std_val = df["solar_kw"].std()
            cov_val = std_val / mean_val if mean_val != 0 else 0

            annual_results.append({
                "Region": name,
                "Year": year,
                "Mean_kW": mean_val,
                "Max_kW": max_val,
                "Std_kW": std_val,
                "CoV": cov_val
            })

            # ---- Seasonal Metrics ----
            seasonal_group = df.groupby("season")["solar_kw"].mean()

            for season, value in seasonal_group.items():
                seasonal_results.append({
                    "Region": name,
                    "Year": year,
                    "Season": season,
                    "Mean_kW": value
                })

    annual_df = pd.DataFrame(annual_results)
    seasonal_df = pd.DataFrame(seasonal_results)

    annual_df.to_csv("annual_summary.csv", index=False)
    seasonal_df.to_csv("seasonal_summary.csv", index=False)

    print("\nAnnual Summary Saved to annual_summary.csv")
    print("\nSeasonal Summary Saved to seasonal_summary.csv")

    # ---- Plot Annual Trend ----
    plt.figure()

    for region in locations.keys():
        subset = annual_df[annual_df["Region"] == region]
        plt.plot(subset["Year"], subset["Mean_kW"], label=region)

    plt.xlabel("Year")
    plt.ylabel("Annual Mean Solar Generation (kW)")
    plt.title("Annual Solar Generation Trend (2016–2020)")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
