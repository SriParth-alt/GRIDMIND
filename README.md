# GRIDMING ⚡ — Deep-RL Microgrid Energy Optimization

A reinforcement-learning agent that manages the energy flow of a building microgrid —
deciding every hour whether to draw from the grid, charge or discharge a battery, or
run price arbitrage. **Every input is measured data**: real building demand (ASHRAE),
real solar irradiance (NASA POWER), real day-ahead market prices (NYISO), and real
hourly grid carbon intensity computed from NYISO's actual 2016 fuel mix.

**Results (90-day evaluation, all-real 2016 inputs):**

| Policy | Energy cost | Savings vs do-nothing |
|---|---|---|
| Do nothing (grid only) | 1.70M | — |
| Rule-based heuristic | 1.68M | 0.99% |
| **RL agent (this project)** | **1.62M** | **4.56%** |

The interesting result: on **real** market prices — where arbitrage spreads are thin —
the hand-written heuristic captures barely 1% of value, while the learned agent finds
**4.6× more**. The harder the market, the bigger RL's relative advantage.

## How it works

```
real demand + solar + prices + carbon ──► MicrogridEnv (13D state) ──► Dueling DQN
   ASHRAE  NASA POWER  NYISO   NYISO           savings-based           charge / discharge /
                                               reward                 arbitrage decisions
```

- **Environment** ([environment/microgrid_env.py](environment/microgrid_env.py)) — hourly
  microgrid simulation: 500 kWh battery with charge/discharge efficiency and rate
  limits, hourly CO₂ tracking, and price thresholds derived from the data's own
  percentiles (no magic numbers — works on any price regime).
- **Real inputs** ([external_data.py](external_data.py)) — downloads and caches all
  measured 2016 data: NASA POWER hourly irradiance → PV output, NYISO day-ahead
  LBMP (N.Y.C. zone) → prices, NYISO 5-min fuel mix + per-fuel emission factors
  → hourly grid carbon intensity. Free public sources, no API keys.
- **State (13D)** — current demand/solar/price, 1-hour-ahead lookahead, battery SOC,
  time features, and a 24-hour planning horizon (peak demand, peak hour, average).
- **Reward** — *savings-based*: the cost avoided relative to a "buy everything from
  the grid" policy, minus battery wear and an emissions penalty. This isolates the
  value of each decision instead of drowning it in unavoidable base cost.
- **Agent** ([agents/dqn_agent.py](agents/dqn_agent.py)) — Dueling DQN with Double-DQN
  targets, Prioritised Experience Replay, and Polyak soft target updates.
- **All hyperparameters** live in [config.py](config.py).

## Quick start

```bash
pip install -r requirements.txt
```

**1. Get the data** — download the [ASHRAE Great Energy Predictor III dataset](https://www.kaggle.com/c/ashrae-energy-prediction/data)
and place `train.csv`, `weather_train.csv`, `building_metadata.csv` in `dataset/`.

**2. Train the agent** (~5–10 min on CPU):

```bash
python train_agent.py
```

**3. Launch the live dashboard:**

```bash
uvicorn webapp.server:app --port 8000
```

Open **http://localhost:8000** and hit *Start simulation* — it streams the RL agent
and the baseline hour-by-hour over WebSocket, with live cost/emissions comparison.

**Or run the terminal pipeline:**

```bash
python main.py     # comparison, plots, and a 2-day demo
```

## Project layout

```
├── agents/dqn_agent.py          # Dueling DQN + PER agent
├── environment/microgrid_env.py # Microgrid simulation (battery physics, reward)
├── ashrae_pipeline.py           # Data loader: ASHRAE demand + real inputs
├── external_data.py             # NASA / NYISO downloaders (solar, prices, carbon)
├── data_generator.py            # Synthetic data generator (for experiments)
├── config.py                    # Every hyperparameter in one place
├── train_agent.py               # Training loop
├── compare_baseline.py          # RL vs rule-based heuristic evaluation
├── main.py                      # Terminal pipeline entry point
└── webapp/                      # FastAPI + WebSocket live dashboard
    ├── server.py
    └── static/index.html
```

## Data sources

| Input | Source | Detail |
|---|---|---|
| Building demand | [ASHRAE GEPIII](https://www.kaggle.com/c/ashrae-energy-prediction/data) | 3 buildings, hourly meter readings, 2016 |
| Solar | [NASA POWER](https://power.larc.nasa.gov/) | Measured hourly irradiance, NYC, → 150 kW array |
| Prices | [NYISO](http://mis.nyiso.com/public/) | Day-ahead hourly LBMP, N.Y.C. zone, 2016 |
| Carbon intensity | NYISO fuel mix | Real hourly kg CO₂/kWh from actual generation |

## Roadmap

- [x] Real day-ahead market prices (NYISO) instead of synthetic TOU
- [x] NASA measured solar irradiance
- [x] Hourly grid carbon intensity (charge when the grid is cleanest)
- [ ] Continuous action space (variable charge/discharge rates)
- [ ] Held-out building validation
- [ ] Demand-charge modeling (peak-kW billing, where batteries earn most)
