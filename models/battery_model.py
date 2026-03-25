class BatteryModel:
    def __init__(
        self,
        capacity_kwh=500,          # Total battery capacity
        max_charge_kw=100,         # Max charging rate per hour
        max_discharge_kw=100,      # Max discharging rate per hour
        efficiency=0.9,            # Round-trip efficiency
        soc_init=0.5               # Initial state of charge (50%)
    ):
        self.capacity = capacity_kwh
        self.max_charge = max_charge_kw
        self.max_discharge = max_discharge_kw
        self.efficiency = efficiency

        # State of Charge (in kWh)
        self.soc = soc_init * capacity_kwh

    def charge(self, energy_kwh):
        """
        Charge the battery with given energy (kWh)
        """
        charge_energy = min(energy_kwh, self.max_charge)

        # Apply efficiency loss
        charge_energy *= self.efficiency

        # Prevent overcharging
        available_space = self.capacity - self.soc
        actual_charge = min(charge_energy, available_space)

        self.soc += actual_charge

        return actual_charge

    def discharge(self, energy_kwh):
        """
        Discharge the battery to supply energy (kWh)
        """
        discharge_energy = min(energy_kwh, self.max_discharge)

        # Prevent over-discharge
        actual_discharge = min(discharge_energy, self.soc)

        # Apply efficiency loss
        actual_discharge *= self.efficiency

        self.soc -= actual_discharge

        return actual_discharge

    def get_soc(self):
        """
        Return SOC as percentage
        """
        return (self.soc / self.capacity) * 100

    def get_available_energy(self):
        """
        Energy that can be discharged
        """
        return self.soc

    def reset(self, soc_init=0.5):
        """
        Reset battery state
        """
        self.soc = soc_init * self.capacity