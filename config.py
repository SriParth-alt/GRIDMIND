"""
config.py
Single source of truth for all hyperparameters and constants.
Edit here — never scatter magic numbers across files.
"""

# ── Environment ───────────────────────────────────────────────────────────── #
BATTERY_CAPACITY    = 500.0   # kWh — scaled to real ASHRAE demand (mean 184, max 591)
MAX_CHARGE_RATE     = 120.0   # kW
MAX_DISCHARGE_RATE  = 120.0   # kW
CHARGE_EFF          = 0.95
DISCHARGE_EFF       = 0.95
SOC_MIN             = 10.0    # % — lower hard limit
SOC_MAX             = 95.0    # % — upper hard limit
SOC_WARN_LOW        = 15.0    # % — soft warning (starts penalty)
SOC_WARN_HIGH       = 90.0    # % — soft warning (starts penalty)

EMISSION_FACTOR     = 0.82    # kg CO2 per kWh from grid

# ── Reward shaping (savings-based) ────────────────────────────────────────── #
# reward = (cost avoided vs do-nothing) - battery wear - emission penalty
EMISSION_WEIGHT     = 0.5     # relative weight of CO2 vs cost
PEAK_WEIGHT         = 0.8     # peak-hour grid price multiplier (1 + this)
WEAR_COST_PER_KWH   = 0.15    # ¢/kWh cycled — light degradation cost; real
                              # wholesale spreads are thin, so heavy wear
                              # makes all arbitrage unprofitable
REWARD_SCALE        = 500.0   # max savings per step ~ MAX_DISCHARGE * price

# Legacy weights kept for reference (no longer used by the env)
MISSED_SOLAR_WEIGHT = 0.6
AVOIDABLE_WEIGHT    = 0.8
SOC_PENALTY_LOW     = 1.5
SOC_PENALTY_HIGH    = 0.8

# Peak hours (both morning and evening)
PEAK_HOURS_MORNING  = (7, 9)   # 07:00–09:00
PEAK_HOURS_EVENING  = (18, 22) # 18:00–22:00

# Smart-strategy thresholds (action 3)
SMART_SOLAR_EXCESS_THRESHOLD = 5.0   # kW — charge only if excess > this
SMART_PRICE_HIGH_THRESHOLD   = 7.0   # $/kWh — discharge only if price > this

# ── State normalisation bounds ────────────────────────────────────────────── #
# Bounds cover real ASHRAE demand (max ~591 kWh) and synthetic data.
STATE_BOUNDS = {
    "demand":            600.0,  # kW  — raised to cover real building demand
    "solar":             160.0,  # kW
    "price":              15.0,  # ¢/kWh — covers real NYISO spikes (max ~14.2)
    "next_price":         15.0,  # ¢/kWh
    "next_demand":       600.0,  # kW  — 1-step-ahead demand forecast
    "next_solar":        160.0,  # kW  — 1-step-ahead solar forecast
    "battery_soc":       100.0,  # %
    "hour":               24.0,  # 0–23
    "day_of_week":         6.0,  # 0=Mon … 6=Sun
    "demand_delta":      100.0,  # kW — max hourly swing in real data
    # 24-hour planning horizon features (ground-truth during training)
    "next24_peak_demand": 600.0, # kW  — highest demand in next 24 steps
    "next24_peak_hour":    24.0, # 0–23 — hour that peak occurs
    "next24_avg_demand":  600.0, # kW  — mean demand over next 24 steps
}

STATE_DIM  = len(STATE_BOUNDS)  # 13
ACTION_DIM = 4

# ── DQN hyperparameters ───────────────────────────────────────────────────── #
LEARNING_RATE   = 3e-4
WEIGHT_DECAY    = 1e-5
GAMMA           = 0.97   # discount factor (hourly tasks; 0.97 ≈ 12-hr half-life)
TAU             = 0.005  # Polyak soft-update rate
BATCH_SIZE      = 64
MIN_REPLAY      = 256
REPLAY_MAXLEN   = 50_000
PER_ALPHA       = 0.6    # PER priority exponent
BETA_START      = 0.4    # importance-sampling correction start
BETA_INCREMENT  = 0.001

EPSILON_START   = 1.0
EPSILON_MIN     = 0.01
EPISODES        = 5000   # each episode = 1 day (24 steps), so more episodes needed
# decay calibrated so epsilon reaches EPSILON_MIN after EPISODES episodes
EPSILON_DECAY   = EPSILON_MIN ** (1.0 / EPISODES)  # decays to 0.01 over EPISODES

# ── Training ──────────────────────────────────────────────────────────────── #
N_DAYS_TRAIN    = 365
DATA_SEED       = 42
CHECKPOINT_FREQ = 100   # save checkpoint every N episodes
LOG_FREQ        = 5     # print metrics every N episodes
REWARD_WINDOW   = 50    # rolling window size for avg reward
PLATEAU_THRESH  = 0.5   # warn if improvement < this over window

MODEL_PATH      = "microgrid_dqn.pth"
