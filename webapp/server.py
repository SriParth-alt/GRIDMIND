"""
webapp/server.py
FastAPI backend that streams a live microgrid simulation over WebSocket.

Run from the project root:
    uvicorn webapp.server:app --reload --port 8000

Then open http://localhost:8000 in a browser.
"""

import asyncio
import sys
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# Make the project root importable (webapp/ is a subfolder)
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import STATE_BOUNDS, STATE_DIM, ACTION_DIM, MODEL_PATH
from ashrae_pipeline import load_ashrae_data
from environment.microgrid_env import MicrogridEnv
from agents.dqn_agent import DQNAgent
from compare_baseline import _baseline_action

ACTION_NAMES = {0: "Idle", 1: "Charge", 2: "Discharge", 3: "Arbitrage"}


def state_to_vector(state: dict) -> np.ndarray:
    return np.array([
        state["demand"]             / STATE_BOUNDS["demand"],
        state["solar"]              / STATE_BOUNDS["solar"],
        state["price"]              / STATE_BOUNDS["price"],
        state["next_price"]         / STATE_BOUNDS["next_price"],
        state["next_demand"]        / STATE_BOUNDS["next_demand"],
        state["next_solar"]         / STATE_BOUNDS["next_solar"],
        state["battery_soc"]        / STATE_BOUNDS["battery_soc"],
        state["hour"]               / STATE_BOUNDS["hour"],
        state["day_of_week"]        / STATE_BOUNDS["day_of_week"],
        np.clip(state["demand_delta"] / STATE_BOUNDS["demand_delta"], -1.0, 1.0),
        state["next24_peak_demand"] / STATE_BOUNDS["next24_peak_demand"],
        state["next24_peak_hour"]   / STATE_BOUNDS["next24_peak_hour"],
        state["next24_avg_demand"]  / STATE_BOUNDS["next24_avg_demand"],
    ], dtype=np.float32)


app = FastAPI(title="SUSTAIN_AI Live Microgrid")

_agent = None


def get_agent() -> DQNAgent:
    global _agent
    if _agent is None:
        model_file = ROOT / MODEL_PATH
        if not model_file.exists():
            raise FileNotFoundError(
                f"Trained model not found at {model_file}. "
                "Run 'python train_agent.py' first to train the agent."
            )
        agent = DQNAgent(STATE_DIM, ACTION_DIM)
        agent.model.load_state_dict(
            torch.load(model_file, map_location="cpu", weights_only=True)
        )
        agent.model.eval()
        agent.epsilon = 0.0
        _agent = agent
    return _agent


@app.get("/")
async def index():
    return FileResponse(str(Path(__file__).parent / "static" / "index.html"))


@app.websocket("/ws/simulate")
async def simulate(ws: WebSocket):
    """
    Streams one microgrid hour per tick.
    Client sends: {"n_days": int, "speed_ms": int}
    Server sends per-tick JSON with both RL-agent and baseline state,
    so the frontend can show them side by side.
    """
    await ws.accept()
    try:
        params    = await ws.receive_json()
        n_days    = max(1, min(int(params.get("n_days", 5)), 60))
        speed_ms  = max(50, min(int(params.get("speed_ms", 400)), 3000))

        agent = get_agent()
        data  = load_ashrae_data(n_days=n_days, seed=42)

        env_rl  = MicrogridEnv(data, shuffle_episodes=False)
        env_bl  = MicrogridEnv(data, shuffle_episodes=False)

        state_rl = env_rl.reset()
        state_bl = env_bl.reset()

        rl_total_cost = rl_total_emit = 0.0
        bl_total_cost = bl_total_emit = 0.0

        total_steps = len(data) - 1
        await ws.send_json({"type": "start", "total_steps": total_steps})

        done = False
        step = 0
        while not done:
            vec        = state_to_vector(state_rl)
            rl_action  = agent.choose_action(vec)
            bl_action  = _baseline_action(state_bl, env_bl.price_low, env_bl.price_high)

            next_rl, _, done, info_rl = env_rl.step(rl_action)
            next_bl, _, _,    info_bl = env_bl.step(bl_action)

            rl_total_cost += info_rl["cost"]
            rl_total_emit += info_rl["emission"]
            bl_total_cost += info_bl["cost"]
            bl_total_emit += info_bl["emission"]

            await ws.send_json({
                "type": "tick",
                "step": step,
                "hour": int(state_rl["hour"]),
                "demand": round(state_rl["demand"], 1),
                "solar": round(state_rl["solar"], 1),
                "price": round(state_rl["price"], 2),
                "rl": {
                    "action": rl_action,
                    "action_name": ACTION_NAMES[rl_action],
                    "grid_used": info_rl["grid_used"],
                    "battery_soc": info_rl["battery_soc"],
                    "cost": info_rl["cost"],
                    "emission": info_rl["emission"],
                    "total_cost": round(rl_total_cost, 2),
                    "total_emission": round(rl_total_emit, 2),
                },
                "baseline": {
                    "action": bl_action,
                    "action_name": ACTION_NAMES[bl_action],
                    "grid_used": info_bl["grid_used"],
                    "battery_soc": info_bl["battery_soc"],
                    "cost": info_bl["cost"],
                    "emission": info_bl["emission"],
                    "total_cost": round(bl_total_cost, 2),
                    "total_emission": round(bl_total_emit, 2),
                },
            })

            state_rl = next_rl
            state_bl = next_bl
            step += 1

            await asyncio.sleep(speed_ms / 1000.0)

        savings_pct  = (bl_total_cost - rl_total_cost) / bl_total_cost * 100 if bl_total_cost else 0.0
        emission_pct = (bl_total_emit - rl_total_emit) / bl_total_emit * 100 if bl_total_emit else 0.0

        await ws.send_json({
            "type": "done",
            "rl_total_cost": round(rl_total_cost, 2),
            "bl_total_cost": round(bl_total_cost, 2),
            "rl_total_emission": round(rl_total_emit, 2),
            "bl_total_emission": round(bl_total_emit, 2),
            "cost_savings_pct": round(savings_pct, 2),
            "emission_savings_pct": round(emission_pct, 2),
        })

    except WebSocketDisconnect:
        pass
    except Exception as e:  # noqa: BLE001
        try:
            await ws.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass
    finally:
        try:
            await ws.close()
        except Exception:
            pass


app.mount("/static", StaticFiles(directory=str(Path(__file__).parent / "static")), name="static")
