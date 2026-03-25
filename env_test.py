import pandas as pd
from environment.microgrid_env import MicrogridEnv


def main():
    # Fake small dataset for testing
    data = pd.DataFrame({
        "predicted_load": [100, 120, 90, 150],
        "solar_kw": [80, 130, 20, 0],
        "price_per_kwh": [5, 6, 8, 10],
        "hour": [10, 12, 18, 21]
    })

    env = MicrogridEnv(data)

    state = env.reset()
    print("Initial State:", state)

    for _ in range(len(data)):
        action = 2  # try discharge
        next_state, reward, done, info = env.step(action)

        print("\nNext State:", next_state)
        print("Reward:", reward)
        print("Info:", info)

        if done:
            break


if __name__ == "__main__":
    main()