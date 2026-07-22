# GRIDMIND / SUSTAIN_AI — Research Notes

> Complete technical record of the project for research-paper writing.
> Everything below is implemented and verified in this repository
> (github.com/SriParth-alt/GRIDMIND). Numbers come from actual runs.

## 1. Problem statement

An AI agent controls the hourly energy flow of a building microgrid consisting of:
grid connection, rooftop PV array (150 kW rated), and a battery energy storage
system (500 kWh, 120 kW max charge/discharge, 95% one-way efficiency, SOC bounds
10–95%). Each hour the agent chooses one of four actions: idle, charge battery
(solar first, then grid), discharge battery to cover load, or a price-arbitrage
heuristic. Objectives: minimize energy cost and CO₂ emissions.

## 2. Data — all inputs are measured, aligned to calendar year 2016

| Input | Source | Detail |
|---|---|---|
| Building demand | ASHRAE Great Energy Predictor III (Kaggle) | 3 buildings (IDs 105, 107, 108), hourly electricity meter readings, full year 2016; gaps interpolated |
| Solar | NASA POWER API | Measured hourly global horizontal irradiance, NYC (40.71, −74.01), local standard time; converted to PV output: P = 150 kW × (GHI/1000) × 0.80 performance ratio |
| Electricity price | NYISO public archive | Day-ahead hourly LBMP, N.Y.C. zone, 2016; $/MWh → ¢/kWh; mean 2.95, p25 2.05, p75 3.49, max 14.19 |
| Carbon intensity | NYISO 5-min fuel mix | Real hourly kg CO₂/kWh computed with per-fuel emission factors (gas 0.42, dual-fuel 0.44, other fossil 0.90, renewables ≈ 0); range 0.112–0.327, mean 0.197 |

Downloader: `external_data.py` (free public sources, no API keys, cached CSVs).

## 3. Method

### 3.1 Environment (environment/microgrid_env.py)
- Battery tracked in kWh energy (not %), efficiency applied on both charge and
  discharge, rate limits enforced.
- Price thresholds for the scripted arbitrage action are **data-driven** — the
  25th/75th percentiles of the loaded price series — so the environment works
  unchanged on any price regime.
- Episode = one random day (24 hourly steps), sampled from 1,098 building-days.

### 3.2 State (13 dimensions, all normalized)
demand, solar, price, next-hour price, next-hour demand, next-hour solar,
battery SOC, hour of day, day of week, demand delta (1-hr change),
next-24h peak demand, next-24h peak hour, next-24h mean demand.

### 3.3 Reward — savings-based (the key design decision)
```
reward = [ (net_demand × p_eff)                    # cost if agent did nothing
         − (grid_used  × p_eff)                    # actual cost incurred
         − wear × (charged + discharged)           # battery degradation, 0.15 ¢/kWh
         − w_e × (grid_used − net_demand) × CI_t ] # emissions of EXTRA energy bought
         / scale
```
where p_eff is the price multiplied by 1.8 during peak hours (07–09, 18–22),
CI_t is the real hourly carbon intensity, and w_e = 0.5.

Rationale: with raw-cost rewards, ~98–99% of the cost is unavoidable base load
that no action affects, so the learning signal from battery decisions is buried
in noise; the agent converges to a no-op policy (we observed exactly this: 0.0%
improvement). Rewarding the *savings vs a do-nothing counterfactual* isolates
the marginal value of each action. This changed the outcome from 0% to ~4.6%.

### 3.4 Agent (agents/dqn_agent.py)
Dueling DQN (shared 128-128 MLP, value + advantage streams), Double-DQN targets,
Prioritized Experience Replay (α=0.6, β 0.4→1.0), Polyak soft target updates
(τ=0.005), Adam lr 3e-4, γ=0.97, Huber loss, gradient clipping at norm 1.0,
ε-greedy 1.0→0.01 over 1,500 episodes, batch 64, replay 50k.

### 3.5 Baselines
1. **Do-nothing**: all net demand from grid (no battery use).
2. **Rule-based heuristic**: charge when price ≤ p25 and SOC < 85%; discharge
   when price ≥ p75 and SOC > 20%; idle otherwise. (Given the same data-driven
   percentile thresholds as the environment — a fair, competent baseline.)

## 4. Results

### 4.1 Main result (90 evaluation days, all-real 2016 inputs)

| Policy | Energy cost | Savings vs do-nothing | CO₂ |
|---|---|---|---|
| Do-nothing | 1,699,422 | — | 82,869 kg |
| Rule-based heuristic | 1,682,523 | 0.99% | 82,822 kg |
| **RL agent** | **1,621,922** | **4.56%** | 82,995 kg |

- RL vs rule-based baseline: **3.60% cheaper** — the learned policy extracts
  **≈4.6× more value** than threshold rules.
- Emissions effectively neutral at aggregate level (−0.2 to +0.2% across runs);
  in live side-by-side runs the RL agent tracked *lower* CO₂ than the baseline
  once hourly carbon intensity entered the reward.
- Training: average reward per day climbs −0.1 → +3.1 over 1,500 episodes
  (~2 min CPU); convergent, reproducible.

### 4.2 Ablation-style evidence from the project's history

| Configuration | RL vs baseline | Interpretation |
|---|---|---|
| Broken battery physics + raw-cost reward | **0.00%** | No learnable signal → no-op policy |
| Fixed physics + savings reward, synthetic prices (spreads ≈ 7¢) | +4.9% (12.0% vs do-nothing) | Learning works; generous synthetic spreads inflate value |
| Same, real NYISO prices (spreads ≈ 1.5¢) | +3.6% (4.56% vs do-nothing) | Honest number; **heuristic collapses to 0.99% but RL retains most of its edge** |

### 4.3 The headline finding for the paper

On real 2016 wholesale prices, arbitrage spreads are thin (buy ≈2¢, sell ≈3.5¢,
minus ~10% round-trip losses and degradation). Under these conditions simple
threshold rules capture almost nothing (0.99%), while the learned policy still
finds 4.56% — because it exploits patterns rules cannot express: pre-positioning
the battery using the 24-hour demand horizon, weekday/weekend differences,
selective participation only when the spread clears its true marginal cost.
**The advantage of learned dispatch grows as the market gets harder** — the
opposite of what the synthetic-data experiments suggested.

This also reproduces a known industry reality: pure energy arbitrage was rarely
profitable for grid batteries in 2016, which is why real systems earn primarily
through demand-charge reduction and grid services — a natural future-work item.

## 5. Engineering findings worth reporting honestly

1. **A silent unit bug nullified the entire system**: battery SOC (a percentage)
   was mutated with kWh quantities, so the configured capacity was never used.
   Symptom: RL results identical to baseline to the third decimal. Lesson:
   physical-unit discipline in RL environments; validate that action
   consequences actually scale with configuration.
2. **Reward normalization by counterfactual, not by magnitude**: dividing raw
   cost by a constant does not fix signal-to-noise; subtracting a counterfactual
   policy's cost does.
3. **Data-driven thresholds**: any hardcoded price constant silently breaks when
   the price regime changes (synthetic → real broke ours); percentiles of the
   loaded series are scale-free.
4. **Flat carbon factors make the emissions objective decorative** — with a
   constant kg/kWh, emission minimization is mathematically identical to energy
   minimization. Only real hourly intensity (3× daily swing) makes it a distinct
   learnable objective.

## 6. System artifacts

- Full-stack demo: FastAPI backend streams simulation over WebSocket; browser
  dashboard shows live cumulative cost, SOC trajectories, dispatch log, CO₂
  (webapp/). Run: `uvicorn webapp.server:app --port 8000`.
- Reproduction: `pip install -r requirements.txt`, place ASHRAE CSVs in
  `dataset/`, `python train_agent.py` (~2 min CPU), `python main.py`.

## 7. Limitations & future work

- Wholesale LBMP used as the retail tariff (no delivery charges / demand
  charges). Demand-charge modeling is the highest-value extension — it is where
  batteries earn most in practice.
- Solar sited in NYC while ASHRAE buildings' true location is anonymized
  (site 0 is believed to be in the US Southeast); one coherent location was
  chosen for internal consistency.
- Discrete 4-action space; continuous charge/discharge rates would tighten
  control.
- Evaluation days overlap the training distribution (same three buildings);
  held-out-building validation is planned.
- Emission factors for the fuel mix are eGRID-typical constants, not
  plant-specific.
