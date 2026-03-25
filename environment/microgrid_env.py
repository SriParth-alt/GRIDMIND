"""
microgrid_env.py
Microgrid environment with:
  - Proper battery physics (efficiency, rate limits, SOC bounds)
  - Shaped reward with peak-hour penalty
  - Reward scaled to ~[-1, 1] range for stable Q-network learning
  - Episode-level day shuffling to prevent cycle memorisation
"""

import numpy as np
import pandas as pd


class MicrogridEnv:

    BATTERY_CAPACITY   = 100.0
    MAX_CHARGE_RATE    = 30.0
    MAX_DISCHARGE_RATE = 30.0
    CHARGE_EFF         = 0.95
    DISCHARGE_EFF      = 0.95
    SOC_MIN            = 10.0
    SOC_MAX            = 95.0

    def __init__(self, data: pd.DataFrame, shuffle_episodes: bool = True):
        self.full_data        = data.reset_index(drop=True)
        self.shuffle_episodes = shuffle_episodes
        self.data             = self.full_data.copy()
        self.current_step     = 0
        self.battery_soc      = 50.0

    # ------------------------------------------------------------------ #
    #  Reset                                                               #
    # ------------------------------------------------------------------ #
    def reset(self):
        if self.shuffle_episodes and "day" in self.full_data.columns:
            days = self.full_data["day"].unique().copy()
            np.random.shuffle(days)
            self.data = (
                pd.concat([self.full_data[self.full_data["day"] == d] for d in days])
                .reset_index(drop=True)
            )
        self.current_step = 0
        self.battery_soc  = 50.0
        return self._get_state()

    # ------------------------------------------------------------------ #
    #  State                                                               #
    # ------------------------------------------------------------------ #
    def _get_state(self):
        row = self.data.iloc[self.current_step]
        if self.current_step + 1 < len(self.data):
            next_price = float(self.data.iloc[self.current_step + 1]["price_per_kwh"])
        else:
            next_price = float(row["price_per_kwh"])

        return {
            "demand":      float(row["predicted_load"]),
            "solar":       float(row["solar_kw"]),
            "price":       float(row["price_per_kwh"]),
            "next_price":  next_price,
            "battery_soc": self.battery_soc,
            "hour":        float(row["hour"]),
        }

    # ------------------------------------------------------------------ #
    #  Battery helpers                                                     #
    # ------------------------------------------------------------------ #
    def _charge(self, kw: float) -> float:
        headroom = (self.SOC_MAX - self.battery_soc) / self.CHARGE_EFF
        actual   = min(kw, self.MAX_CHARGE_RATE, max(0.0, headroom))
        self.battery_soc += actual * self.CHARGE_EFF
        return actual

    def _discharge(self, kw: float) -> float:
        available = (self.battery_soc - self.SOC_MIN) * self.DISCHARGE_EFF
        actual    = min(kw, self.MAX_DISCHARGE_RATE, max(0.0, available))
        self.battery_soc -= actual / self.DISCHARGE_EFF
        return actual

    # ------------------------------------------------------------------ #
    #  Step                                                                #
    # ------------------------------------------------------------------ #
    def step(self, action: int):
        row    = self.data.iloc[self.current_step]
        demand = float(row["predicted_load"])
        solar  = float(row["solar_kw"])
        price  = float(row["price_per_kwh"])
        hour   = int(row["hour"])

        excess_solar       = max(0.0, solar - demand)
        net_demand         = max(0.0, demand - solar)
        battery_charged    = 0.0
        battery_discharged = 0.0

        # ── Actions ────────────────────────────────────────────────── #
        if action == 0:
            pass                                        # idle — grid only

        elif action == 1:                               # charge from excess solar
            if excess_solar > 0:
                battery_charged = self._charge(excess_solar)

        elif action == 2:                               # discharge to cover demand
            if net_demand > 0:
                battery_discharged = self._discharge(net_demand)

        elif action == 3:                               # smart: price-aware charge/discharge
            if self.current_step + 1 < len(self.data):
                next_price = float(self.data.iloc[self.current_step + 1]["price_per_kwh"])
            else:
                next_price = price
            if excess_solar > 5 and next_price > price:
                battery_charged = self._charge(excess_solar)
            elif net_demand > 0 and price > 7:
                battery_discharged = self._discharge(net_demand)

        # ── Energy balance ─────────────────────────────────────────── #
        grid_used       = max(0.0, net_demand - battery_discharged)
        curtailed_solar = max(0.0, excess_solar - battery_charged)

        # ── Shaped reward ──────────────────────────────────────────── #
        grid_cost      = grid_used * price
        missed_storage = curtailed_solar * price * 0.6
        dispatch_value = battery_discharged * price

        avoidable_grid = 0.0
        if action == 0 and net_demand > 0:
            dischargeable  = (self.battery_soc - self.SOC_MIN) * self.DISCHARGE_EFF
            avoidable      = min(net_demand, dischargeable)
            avoidable_grid = avoidable * price * 0.8

        soc_penalty = 0.0
        if self.battery_soc < self.SOC_MIN + 5:
            soc_penalty = 1.5
        elif self.battery_soc > self.SOC_MAX - 5:
            soc_penalty = 0.8

        peak_penalty = 0.0
        if 18 <= hour <= 22 and grid_used > 0:
            peak_penalty = grid_used * price * 0.8

        reward = (
            - grid_cost
            - missed_storage
            + dispatch_value
            - avoidable_grid
            - soc_penalty
            - peak_penalty
        )

        # Scale reward: max cost per step ~30kW * $25 = 750 → divide by 1000
        # Keeps reward in ~[-1, 1] range for healthy Q-network gradients
        reward = reward / 1000.0

        self.current_step += 1
        done = self.current_step >= len(self.data) - 1
        return self._get_state(), reward, done, {
            "grid_used":       round(grid_used, 3),
            "battery_charged": round(battery_charged, 3),
            "battery_used":    round(battery_discharged, 3),
            "curtailed_solar": round(curtailed_solar, 3),
            "cost":            round(grid_cost, 3),
            "battery_soc":     round(self.battery_soc, 2),
        }