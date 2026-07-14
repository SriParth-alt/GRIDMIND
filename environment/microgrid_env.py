"""
microgrid_env.py

Microgrid RL environment (13D state, 4 discrete actions).

Battery physics: energy tracked in kWh against BATTERY_CAPACITY.
SOC exposed to the agent as a percentage.

Reward: SAVINGS-BASED — the reward is the cost the agent avoided relative
to a "do nothing, buy everything from grid" policy, minus battery wear and
an emission-weighted term. This isolates the value created by each action
instead of drowning it under the unavoidable demand cost.
"""

import numpy as np
import pandas as pd

from config import (
    BATTERY_CAPACITY, MAX_CHARGE_RATE, MAX_DISCHARGE_RATE,
    CHARGE_EFF, DISCHARGE_EFF, SOC_MIN, SOC_MAX,
    EMISSION_FACTOR, EMISSION_WEIGHT,
    PEAK_WEIGHT, WEAR_COST_PER_KWH, REWARD_SCALE,
    PEAK_HOURS_MORNING, PEAK_HOURS_EVENING,
    SMART_PRICE_HIGH_THRESHOLD,
)


class MicrogridEnv:

    def __init__(self, data: pd.DataFrame, shuffle_episodes: bool = True):
        self.full_data        = data.reset_index(drop=True)
        self.shuffle_episodes = shuffle_episodes
        self.data             = self.full_data.copy()
        self.current_step     = 0

        # Battery state in kWh (not percent!)
        self.soc_min_kwh    = SOC_MIN / 100.0 * BATTERY_CAPACITY
        self.soc_max_kwh    = SOC_MAX / 100.0 * BATTERY_CAPACITY
        self.battery_energy = 0.5 * BATTERY_CAPACITY

        self.prev_demand    = 0.0
        self.total_emission = 0.0

    # ------------------------------------------------------------------ #
    @property
    def battery_soc(self) -> float:
        """State of charge as a percentage of capacity."""
        return self.battery_energy / BATTERY_CAPACITY * 100.0

    # ------------------------------------------------------------------ #
    #  Reset
    # ------------------------------------------------------------------ #
    def reset(self):
        if self.shuffle_episodes and "day" in self.full_data.columns:
            # One random day per episode — 24 steps, fast training
            day = np.random.choice(self.full_data["day"].unique())
            self.data = (
                self.full_data[self.full_data["day"] == day]
                .reset_index(drop=True)
            )
        else:
            self.data = self.full_data.copy()

        self.current_step   = 0
        self.battery_energy = 0.5 * BATTERY_CAPACITY
        self.total_emission = 0.0
        self.prev_demand    = float(self.data.iloc[0]["predicted_load"])

        return self._get_state()

    # ------------------------------------------------------------------ #
    #  State  (13D)
    # ------------------------------------------------------------------ #
    def _get_state(self):
        row    = self.data.iloc[self.current_step]
        demand = float(row["predicted_load"])

        if self.current_step + 1 < len(self.data):
            nxt         = self.data.iloc[self.current_step + 1]
            next_price  = float(nxt["price_per_kwh"])
            next_demand = float(nxt["predicted_load"])
            next_solar  = float(nxt["solar_kw"])
        else:
            next_price  = float(row["price_per_kwh"])
            next_demand = demand
            next_solar  = float(row["solar_kw"])

        # 24-hour planning horizon
        horizon_end   = min(self.current_step + 24, len(self.data))
        horizon       = self.data.iloc[self.current_step:horizon_end]["predicted_load"].values
        peak_idx      = int(horizon.argmax())
        next24_peak   = float(horizon.max())
        next24_avg    = float(horizon.mean())
        next24_peak_hour = float(self.data.iloc[self.current_step + peak_idx]["hour"])

        dow = int(row["day_of_week"]) if "day_of_week" in row.index else int(row["day"]) % 7

        return {
            "demand":            demand,
            "solar":             float(row["solar_kw"]),
            "price":             float(row["price_per_kwh"]),
            "next_price":        next_price,
            "next_demand":       next_demand,
            "next_solar":        next_solar,
            "battery_soc":       self.battery_soc,
            "hour":              float(row["hour"]),
            "day_of_week":       float(dow),
            "demand_delta":      demand - self.prev_demand,
            "next24_peak_demand": next24_peak,
            "next24_peak_hour":   next24_peak_hour,
            "next24_avg_demand":  next24_avg,
        }

    # ------------------------------------------------------------------ #
    #  Battery helpers — all quantities in kWh
    # ------------------------------------------------------------------ #
    def _charge(self, kwh: float) -> float:
        """Charge battery; returns kWh drawn from the source."""
        headroom = (self.soc_max_kwh - self.battery_energy) / CHARGE_EFF
        actual   = min(kwh, MAX_CHARGE_RATE, max(0.0, headroom))
        self.battery_energy += actual * CHARGE_EFF
        return actual

    def _discharge(self, kwh: float) -> float:
        """Discharge battery; returns kWh delivered to the load."""
        available = (self.battery_energy - self.soc_min_kwh) * DISCHARGE_EFF
        actual    = min(kwh, MAX_DISCHARGE_RATE, max(0.0, available))
        self.battery_energy -= actual / DISCHARGE_EFF
        return actual

    # ------------------------------------------------------------------ #
    #  Step
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
        grid_charge        = 0.0   # kWh bought from grid to charge battery

        # ── Actions ────────────────────────────────────────────────── #
        # 0: idle       — grid covers all net demand, battery untouched
        # 1: charge     — solar first, then grid (agent decides when via price in state)
        # 2: discharge  — battery covers as much net demand as possible
        # 3: arbitrage  — heuristic: charge if cheap-now/expensive-next, discharge if expensive

        if action == 0:
            pass

        elif action == 1:
            if excess_solar > 0:
                battery_charged = self._charge(excess_solar)
            headroom = (self.soc_max_kwh - self.battery_energy) / CHARGE_EFF
            if headroom > 1.0:
                grid_charge      = self._charge(min(MAX_CHARGE_RATE - battery_charged, headroom))
                battery_charged += grid_charge

        elif action == 2:
            if net_demand > 0:
                battery_discharged = self._discharge(net_demand)

        elif action == 3:
            if self.current_step + 1 < len(self.data):
                next_price = float(self.data.iloc[self.current_step + 1]["price_per_kwh"])
            else:
                next_price = price

            if price <= SMART_PRICE_HIGH_THRESHOLD and next_price > price:
                headroom         = (self.soc_max_kwh - self.battery_energy) / CHARGE_EFF
                grid_charge      = self._charge(min(MAX_CHARGE_RATE, headroom))
                battery_charged += grid_charge
            elif net_demand > 0 and price > SMART_PRICE_HIGH_THRESHOLD:
                battery_discharged = self._discharge(net_demand)

        # ── Energy balance ─────────────────────────────────────────── #
        grid_used       = max(0.0, net_demand - battery_discharged) + grid_charge
        curtailed_solar = max(0.0, excess_solar - max(0.0, battery_charged - grid_charge))

        emission             = grid_used * EMISSION_FACTOR
        self.total_emission += emission

        # ── Reward: savings vs "do nothing" ───────────────────────── #
        # Peak-hour grid usage is priced up so the agent learns to avoid it.
        in_peak         = (PEAK_HOURS_MORNING[0] <= hour <= PEAK_HOURS_MORNING[1]) or \
                          (PEAK_HOURS_EVENING[0] <= hour <= PEAK_HOURS_EVENING[1])
        effective_price = price * (1.0 + PEAK_WEIGHT) if in_peak else price

        do_nothing_cost = net_demand * effective_price
        actual_cost     = grid_used * effective_price
        savings         = do_nothing_cost - actual_cost   # >0 when battery helped

        wear           = WEAR_COST_PER_KWH * (battery_charged + battery_discharged)
        emission_delta = (grid_used - net_demand) * EMISSION_FACTOR  # >0 if we bought extra

        reward = (savings - wear - EMISSION_WEIGHT * emission_delta) / REWARD_SCALE

        # ── Advance ────────────────────────────────────────────────── #
        self.prev_demand   = demand
        self.current_step += 1
        done = self.current_step >= len(self.data) - 1

        return self._get_state(), reward, done, {
            "grid_used":       round(grid_used, 3),
            "battery_charged": round(battery_charged, 3),
            "battery_used":    round(battery_discharged, 3),
            "curtailed_solar": round(curtailed_solar, 3),
            "cost":            round(grid_used * price, 3),
            "battery_soc":     round(self.battery_soc, 2),
            "emission":        round(emission, 3),
        }
