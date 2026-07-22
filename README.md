# SUSTAIN_AI ⚡ — Deep-RL Microgrid Energy Optimization

A reinforcement-learning agent that manages the energy flow of a building microgrid —
deciding every hour whether to draw from the grid, charge or discharge a battery, or
run price arbitrage — trained and evaluated on **real building energy data** from the
ASHRAE Great Energy Predictor dataset.

**Results (90-day evaluation on real building data):**

| Policy | Total cost | vs do-nothing |
|---|---|---|
| Do nothing (grid only) | $3.01M | — |
| Rule-based heuristic | $2.79M | −7.4% |
| **RL agent (this project)** | **$2.67M** | **−11.5%** |

The RL agent beats the hand-written heuristic by **~4.9%** on cost — learned purely
from experience, with no dispatch rules programmed in.

## How it works

```
ASHRAE building data ──► MicrogridEnv (13D state, battery physics) ──► Dueling DQN
        │                        │                                        │
   real demand,            savings-based reward                 charge / discharge /
   solar, prices          (cost avoided vs no-op)               arbitrage decisions
```

- **Environment** ([environment/microgrid_env.py](environment/microgrid_env.py)) — hourly
  microgrid simulation: 500 kWh battery with charge/discharge efficiency and rate
  limits, real demand from three ASHRAE buildings, physics-based solar (clear-sky
  model attenuated by recorded cloud cover), time-of-use pricing, CO₂ tracking.
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
├── ashrae_pipeline.py           # Real-data loader (demand, solar, pricing)
├── data_generator.py            # Synthetic data generator (for experiments)
├── config.py                    # Every hyperparameter in one place
├── train_agent.py               # Training loop
├── compare_baseline.py          # RL vs rule-based heuristic evaluation
├── main.py                      # Terminal pipeline entry point
└── webapp/                      # FastAPI + WebSocket live dashboard
    ├── server.py
    └── static/index.html
```

## Roadmap

- [ ] Real day-ahead market prices (CAISO/ERCOT) instead of synthetic TOU
- [ ] NREL/NASA measured solar irradiance
- [ ] Hourly grid carbon intensity (charge when the grid is cleanest)
- [ ] Continuous action space (variable charge/discharge rates)
- [ ] Held-out building validation
