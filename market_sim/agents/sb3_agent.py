"""
sb3_agent.py
------------
Deep-RL upgrade path for the bot. Uses stable-baselines3's PPO algorithm
on top of the GymTradingEnv from env.py. This needs packages that aren't
available in the sandbox this project was written in, so it's provided
ready-to-run on your own machine:

    pip install gymnasium stable-baselines3 torch

Usage:
    python -m market_sim.agents.sb3_agent
"""

from __future__ import annotations

import pandas as pd

from ..env import GymTradingEnv, _HAS_GYM


def train_ppo(df: pd.DataFrame, total_timesteps: int = 100_000, model_path: str = "outputs/ppo_trader"):
    if not _HAS_GYM:
        raise ImportError(
            "gymnasium not installed. Run: pip install gymnasium stable-baselines3 torch"
        )
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env

    def _make_env():
        return GymTradingEnv(df, window_size=30, allow_short=True)

    env = make_vec_env(_make_env, n_envs=4)

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=256,
        gamma=0.99,
        ent_coef=0.01,
    )
    model.learn(total_timesteps=total_timesteps)
    model.save(model_path)
    print(f"saved model to {model_path}.zip")
    return model


def evaluate(model, df: pd.DataFrame):
    """Greedy rollout of a trained SB3 model on a df, returns the underlying PaperBroker."""
    env = GymTradingEnv(df, window_size=30, allow_short=True)
    obs, _ = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(int(action))
        done = terminated or truncated
    return env._env.broker


if __name__ == "__main__":
    from ..market import SyntheticMarket

    df = SyntheticMarket(n_candles=3000, seed=42).generate()
    model = train_ppo(df, total_timesteps=50_000)
    broker = evaluate(model, df)
    print("final equity:", broker.equity_curve[-1])
