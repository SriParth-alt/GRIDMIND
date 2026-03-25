import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import models.pricing_model
import matplotlib.pyplot as plt


def main():

    pricing = models.pricing_model.PricingModel()

    df_2016 = pricing.generate_yearly_pricing(2016)
    df_2020 = pricing.generate_yearly_pricing(2020)

    print("\n2016 Pricing Sample:")
    print(df_2016.head())

    print("\n2020 Pricing Sample:")
    print(df_2020.head())

    print("\nMean Price 2016:", df_2016["price_per_kwh"].mean())
    print("Mean Price 2020:", df_2020["price_per_kwh"].mean())

    # Plot comparison
    plt.figure()
    plt.plot(df_2016["timestamp"][:48], df_2016["price_per_kwh"][:48], label="2016")
    plt.plot(df_2020["timestamp"][:48], df_2020["price_per_kwh"][:48], label="2020")
    plt.xlabel("Hour")
    plt.ylabel("Price (₹/kWh)")
    plt.title("TOU Pricing Comparison (First 48 Hours)")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()