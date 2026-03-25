from models.battery_model import BatteryModel


def main():
    battery = BatteryModel()

    print("\nInitial SOC:", battery.get_soc())

    print("\nCharging 120 kWh...")
    battery.charge(120)
    print("SOC after charge:", battery.get_soc())

    print("\nDischarging 80 kWh...")
    battery.discharge(80)
    print("SOC after discharge:", battery.get_soc())


if __name__ == "__main__":
    main()