"""
q_agent.py
----------
A tabular Q-learning agent. Runs with nothing but numpy + pandas, so it
works immediately even before you `pip install torch stable-baselines3`.
It's the "first working bot" -- swap in agents/sb3_agent.py later for a
deep-RL upgrade using the exact same TradingEnv.

State = a small discretized fingerprint of market condition + current
         position, e.g. (rsi_bucket, macd_sign, trend_sign, position_sign)
Action = 0 flat, 1 long, 2 short   (same convention as env.py)

train() trains on one fixed price history (fast, good for a quick demo, but
the agent can overfit to that one specific path). train_multi_market() is
the more honest way to train a policy that's meant to generalize: each
episode gets a *freshly generated* market, so the table learns a general
"in this kind of situation, do this" rule instead of memorizing one
particular sequence of ups and downs.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from ..broker import PaperBroker
from ..indicators import add_all_indicators


def _precompute(df: pd.DataFrame):
    """Turn a raw OHLCV df into the numpy arrays the training loop needs.
    Doing this once per episode (instead of pandas .loc/.iloc per step)
    is what makes 100s of episodes actually feasible."""
    data = add_all_indicators(df)
    close = data["close"].to_numpy(dtype=np.float64)
    open_time = data["open_time"].to_numpy()

    rsi = data["rsi_14"].to_numpy(dtype=np.float64)
    rsi_bucket = np.where(rsi < 30, 0, np.where(rsi > 70, 2, 1)).astype(np.int8)

    macd_sign = np.sign(data["macd_hist"].to_numpy(dtype=np.float64)).astype(np.int8)

    # trend sign with a dead-zone: only call it a real trend once the
    # SMA10/SMA50 gap is a meaningful fraction of price. Without this,
    # every tiny noise-driven crossover near the flip point flips the
    # agent's position too, and commission/slippage on that chatter
    # eats any real trend-following edge before it can pay off.
    close_ = data["close"].to_numpy(dtype=np.float64)
    gap = (data["sma_10"] - data["sma_50"]).to_numpy(dtype=np.float64) / close_
    band = 0.004  # 0.4% of price
    trend_sign = np.where(gap > band, 1, np.where(gap < -band, -1, 0)).astype(np.int8)

    return close, open_time, rsi_bucket, macd_sign, trend_sign


class QLearningAgent:
    def __init__(
        self,
        alpha: float = 0.1,
        gamma: float = 0.95,
        epsilon: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995,
        seed: int | None = None,
    ):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.rng = np.random.default_rng(seed)
        self.q_table: dict[tuple, np.ndarray] = {}
        self.n_actions = 3

    def _q(self, state: tuple) -> np.ndarray:
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.n_actions)
        return self.q_table[state]

    @staticmethod
    def _bucket_rsi(rsi: float) -> int:
        if rsi < 30:
            return 0
        if rsi > 70:
            return 2
        return 1

    @staticmethod
    def state_from_row(row: pd.Series, position_qty: float) -> tuple:
        rsi_b = QLearningAgent._bucket_rsi(row["rsi_14"])
        macd_sign = int(np.sign(row["macd_hist"]))
        gap = (row["sma_10"] - row["sma_50"]) / row["close"]
        trend_sign = 1 if gap > 0.004 else (-1 if gap < -0.004 else 0)
        pos_sign = int(np.sign(position_qty))
        return (rsi_b, macd_sign, trend_sign, pos_sign)

    def act(self, state: tuple, greedy: bool = False) -> int:
        if not greedy and self.rng.random() < self.epsilon:
            return int(self.rng.integers(0, self.n_actions))
        q = self._q(state)
        best = np.flatnonzero(q == q.max())
        return int(self.rng.choice(best))

    def update(self, state, action, reward, next_state, done):
        q = self._q(state)
        next_q = self._q(next_state)
        target = reward if done else reward + self.gamma * np.max(next_q)
        q[action] += self.alpha * (target - q[action])

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, path: str):
        """Persist the learned Q-table to a JSON file (tuple keys -> strings)."""
        import json
        serializable = {"|".join(map(str, k)): v.tolist() for k, v in self.q_table.items()}
        with open(path, "w") as f:
            json.dump(serializable, f)

    @classmethod
    def load(cls, path: str) -> "QLearningAgent":
        """Load a Q-table saved with save(). Returned agent is greedy-ready
        (epsilon set to 0) -- retrain further or set .epsilon if you want it
        to keep exploring."""
        import json
        agent = cls(epsilon=0.0, epsilon_min=0.0)
        with open(path) as f:
            raw = json.load(f)
        agent.q_table = {
            tuple(int(x) for x in k.split("|")): np.array(v) for k, v in raw.items()
        }
        return agent

    def _run_precomputed(self, arrays, broker_kwargs, greedy: bool, learn: bool, turnover_penalty: float = 0.0003):
        close, open_time, rsi_bucket, macd_sign, trend_sign = arrays
        broker = PaperBroker(**broker_kwargs)
        n = len(close)

        pos_sign = 0
        state = (int(rsi_bucket[1]), int(macd_sign[1]), int(trend_sign[1]), pos_sign)
        last_action = 0

        for i in range(1, n - 1):
            price = close[i]
            action = self.act(state, greedy=greedy)
            target_frac = 0.0 if action == 0 else (1.0 if action == 1 else -1.0)
            target_qty = target_frac * broker.max_affordable_qty(price)
            equity_before = broker.equity(price)
            broker.set_target_position(target_qty, price, time=open_time[i], step=i)

            next_price = close[i + 1]
            equity_after = broker.equity(next_price)

            pos_sign = int(np.sign(broker.position_qty))
            next_state = (int(rsi_bucket[i + 1]), int(macd_sign[i + 1]), int(trend_sign[i + 1]), pos_sign)

            if learn:
                reward = (equity_after - equity_before) / max(equity_before, 1e-8)
                if action != last_action:
                    reward -= turnover_penalty
                done = i == n - 2
                self.update(state, action, reward, next_state, done)

            last_action = action
            state = next_state

        return broker

    def train(
        self,
        df: pd.DataFrame,
        n_episodes: int = 50,
        starting_cash: float = 10_000.0,
        commission_rate: float = 0.001,
        slippage_rate: float = 0.0005,
        allow_short: bool = True,
        turnover_penalty: float = 0.0003,
        verbose: bool = True,
    ) -> list[float]:
        """Train repeatedly on ONE fixed price history. Fast, but the agent
        can end up memorizing that specific path rather than a general rule
        -- prefer train_multi_market for anything you plan to trust
        out-of-sample."""
        arrays = _precompute(df)
        broker_kwargs = dict(
            starting_cash=starting_cash, commission_rate=commission_rate,
            slippage_rate=slippage_rate, allow_short=allow_short,
        )
        episode_returns = []
        for ep in range(n_episodes):
            broker = self._run_precomputed(arrays, broker_kwargs, greedy=False, learn=True, turnover_penalty=turnover_penalty)
            self.decay_epsilon()
            final_equity = broker.equity_curve[-1] if broker.equity_curve else starting_cash
            ep_return = (final_equity / starting_cash - 1) * 100
            episode_returns.append(ep_return)
            if verbose and (ep + 1) % max(1, n_episodes // 10) == 0:
                print(f"episode {ep + 1}/{n_episodes}  return={ep_return:+.2f}%  epsilon={self.epsilon:.3f}  states={len(self.q_table)}")
        return episode_returns

    def train_multi_market(
        self,
        market_factory: Callable[[], pd.DataFrame],
        n_episodes: int = 300,
        starting_cash: float = 10_000.0,
        commission_rate: float = 0.001,
        slippage_rate: float = 0.0005,
        allow_short: bool = True,
        turnover_penalty: float = 0.0003,
        verbose: bool = True,
    ) -> list[float]:
        """Train on a FRESH randomly generated market every episode.
        `market_factory` is a zero-arg callable that returns a new OHLCV
        DataFrame each time it's called, e.g.:

            lambda: SyntheticMarket(n_candles=600, seed=None, ...).generate()

        This is slower per-episode-count but produces a policy that has
        actually seen many different trend timings/durations, so the
        (rsi_bucket, macd_sign, trend_sign, position_sign) -> action table
        reflects a general rule rather than one memorized price path.
        """
        broker_kwargs = dict(
            starting_cash=starting_cash, commission_rate=commission_rate,
            slippage_rate=slippage_rate, allow_short=allow_short,
        )
        episode_returns = []
        for ep in range(n_episodes):
            arrays = _precompute(market_factory())
            broker = self._run_precomputed(arrays, broker_kwargs, greedy=False, learn=True, turnover_penalty=turnover_penalty)
            self.decay_epsilon()
            final_equity = broker.equity_curve[-1] if broker.equity_curve else starting_cash
            ep_return = (final_equity / starting_cash - 1) * 100
            episode_returns.append(ep_return)
            if verbose and (ep + 1) % max(1, n_episodes // 10) == 0:
                avg_recent = np.mean(episode_returns[-max(1, n_episodes // 10):])
                print(f"episode {ep + 1}/{n_episodes}  avg_recent_return={avg_recent:+.2f}%  "
                      f"epsilon={self.epsilon:.3f}  states={len(self.q_table)}")
        return episode_returns

    def run_episode(
        self,
        df: pd.DataFrame,
        starting_cash: float = 10_000.0,
        commission_rate: float = 0.001,
        slippage_rate: float = 0.0005,
        allow_short: bool = True,
    ) -> PaperBroker:
        """Greedy (no exploration) run for backtesting/reporting."""
        arrays = _precompute(df)
        broker_kwargs = dict(
            starting_cash=starting_cash, commission_rate=commission_rate,
            slippage_rate=slippage_rate, allow_short=allow_short,
        )
        return self._run_precomputed(arrays, broker_kwargs, greedy=True, learn=False)

    def decide_live(self, window_df: pd.DataFrame, position_qty: float, price: float) -> float:
        """Same interface as MLAgent.decide_live -- lets live_trader.py treat
        either agent type uniformly. Returns a target position fraction in
        {-1, 0, 1} (this agent doesn't size by confidence, unlike MLAgent)."""
        data = add_all_indicators(window_df)
        last = data.iloc[-1]
        state = self.state_from_row(last, position_qty)
        action = self.act(state, greedy=True)
        return {0: 0.0, 1: 1.0, 2: -1.0}[action]
