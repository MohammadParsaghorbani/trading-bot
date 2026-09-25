"""
env.py
------
A Gymnasium-style trading environment. Works standalone (pure numpy/pandas,
no gymnasium required) but also plugs straight into stable-baselines3 if
you `pip install gymnasium stable-baselines3 torch` on your own machine,
since it exposes the same reset()/step() contract.

State  = a rolling window of normalized indicator features
Action = discrete: 0 = go/stay flat, 1 = go/stay long, 2 = go/stay short
Reward = change in portfolio equity that step, minus a small penalty for
         flipping position too often (discourages overtrading/commission bleed)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .broker import PaperBroker
from .indicators import add_all_indicators

try:
    import gymnasium as gym
    from gymnasium import spaces
    _HAS_GYM = True
except ImportError:
    _HAS_GYM = False


FEATURE_COLUMNS = [
    "return_1", "return_5", "volatility_10",
    "rsi_14", "macd", "macd_signal", "macd_hist",
]


class TradingEnv:
    """
    Parameters
    ----------
    df : OHLCV dataframe (raw, indicators added internally)
    window_size : how many past candles the agent sees per observation
    starting_cash, commission_rate, slippage_rate, allow_short : passed to PaperBroker
    turnover_penalty : reward penalty applied when the agent changes position
    """

    ACTIONS = {0: "flat", 1: "long", 2: "short"}

    def __init__(
        self,
        df: pd.DataFrame,
        window_size: int = 30,
        starting_cash: float = 10_000.0,
        commission_rate: float = 0.001,
        slippage_rate: float = 0.0005,
        allow_short: bool = True,
        turnover_penalty: float = 0.0002,
    ):
        self.raw_df = df.reset_index(drop=True)
        self.df = add_all_indicators(self.raw_df)
        self.window_size = window_size
        self.turnover_penalty = turnover_penalty
        self.broker_kwargs = dict(
            starting_cash=starting_cash,
            commission_rate=commission_rate,
            slippage_rate=slippage_rate,
            allow_short=allow_short,
        )
        self.broker = PaperBroker(**self.broker_kwargs)

        # precompute normalized feature matrix (z-score per column over the episode)
        feats = self.df[FEATURE_COLUMNS].values.astype(np.float32)
        mean = feats.mean(axis=0, keepdims=True)
        std = feats.std(axis=0, keepdims=True) + 1e-8
        self._feats = (feats - mean) / std

        self.n_features = len(FEATURE_COLUMNS)
        self.obs_dim = self.window_size * self.n_features + 1  # +1 for current position sign

        if _HAS_GYM:
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32
            )
            self.action_space = spaces.Discrete(3)

        self._step_idx = self.window_size
        self._last_action = 0

    def _get_obs(self) -> np.ndarray:
        window = self._feats[self._step_idx - self.window_size:self._step_idx].flatten()
        pos_sign = np.sign(self.broker.position_qty)
        return np.concatenate([window, [pos_sign]]).astype(np.float32)

    def reset(self, seed: int | None = None):
        self.broker = PaperBroker(**self.broker_kwargs)
        self._step_idx = self.window_size
        self._last_action = 0
        obs = self._get_obs()
        info = {}
        return obs, info

    def step(self, action: int):
        price = float(self.df.loc[self._step_idx, "close"])
        equity_before = self.broker.equity(price)

        target_frac = {0: 0.0, 1: 1.0, 2: -1.0}[int(action)]
        target_qty = target_frac * self.broker.max_affordable_qty(price)
        self.broker.set_target_position(
            target_qty, price,
            time=self.df.loc[self._step_idx, "open_time"],
            step=self._step_idx,
        )

        self._step_idx += 1
        terminated = self._step_idx >= len(self.df) - 1
        truncated = False

        next_price = float(self.df.loc[self._step_idx, "close"])
        equity_after = self.broker.equity(next_price)

        reward = (equity_after - equity_before) / max(equity_before, 1e-8)
        if int(action) != self._last_action:
            reward -= self.turnover_penalty
        self._last_action = int(action)

        obs = self._get_obs()
        info = {"equity": equity_after, "position": self.broker.position_qty}
        return obs, reward, terminated, truncated, info

    def render(self):
        print(
            f"step={self._step_idx} equity={self.broker.equity_curve[-1] if self.broker.equity_curve else self.broker.starting_cash:.2f} "
            f"position={self.broker.position_qty:.4f}"
        )


if _HAS_GYM:
    class GymTradingEnv(gym.Env):
        """Thin gymnasium.Env wrapper around TradingEnv, for stable-baselines3."""

        metadata = {"render_modes": []}

        def __init__(self, df: pd.DataFrame, **kwargs):
            super().__init__()
            self._env = TradingEnv(df, **kwargs)
            self.observation_space = self._env.observation_space
            self.action_space = self._env.action_space

        def reset(self, *, seed=None, options=None):
            return self._env.reset(seed=seed)

        def step(self, action):
            return self._env.step(action)

        def render(self):
            self._env.render()
