"""
ml_agent.py
-----------
A supervised-learning trading agent -- and the one actually recommended for
real use, over agents/q_agent.py.

Why: extensive testing (see README's "Honest result" section) showed tabular
Q-learning's own estimation variance was swamping a real-but-small trading
signal, causing it to collapse to "never trade" even when trading was
correctly the better choice. A direct supervised classifier -- "given these
indicators right now, will price be meaningfully higher/lower/flat in N
candles?" -- is a much better-posed, more sample-efficient learning problem,
and it makes DETERMINISTIC decisions (no epsilon-greedy randomness) once
trained: same inputs always produce the same trade.

Strategy
--------
1. Train a gradient-boosted classifier to predict the 3-way direction label
   (up / down / flat) of the return over the next `horizon` candles.
2. At each decision point, take the model's predicted probabilities.
   - Go long if P(up) clears a confidence threshold and beats P(down)
   - Go short if P(down) clears the threshold and beats P(up)
   - Otherwise stay flat
   Position SIZE scales with confidence (a barely-over-threshold signal
   opens a small position; a highly confident one opens a large one) --
   this is what makes it size-aware rather than a binary flip-flop.
3. Decisions are made every `horizon` candles, not every single candle --
   matching the re-decision cadence to the prediction horizon is what stops
   it from "whipsawing" on short-term noise inside a single prediction
   window (this alone was the single biggest lever found during tuning).

This works on ANY OHLCV data with the market_sim contract -- synthetic or
real Binance history. See train_ml.py for both usages.
"""

from __future__ import annotations

import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

from ..indicators import add_all_indicators
from ..broker import PaperBroker

FEATURES = [
    "rsi_14", "macd", "macd_hist", "return_1", "return_5",
    "volatility_10", "atr_14", "sma_10", "sma_50",
]


def _make_features_labels(df: pd.DataFrame, horizon: int, band: float):
    data = add_all_indicators(df)
    close = data["close"].to_numpy()
    fwd_ret = np.zeros(len(close))
    fwd_ret[:-horizon] = close[horizon:] / close[:-horizon] - 1
    label = np.where(fwd_ret > band, 1, np.where(fwd_ret < -band, -1, 0))
    X = data[FEATURES].to_numpy()
    return X[:-horizon], label[:-horizon]


class MLAgent:
    def __init__(
        self,
        horizon: int = 15,
        band: float = 0.004,
        long_threshold: float = 0.40,
        short_threshold: float = 0.40,
        n_estimators: int = 200,
        max_depth: int = 3,
        learning_rate: float = 0.05,
        random_state: int = 0,
    ):
        self.horizon = horizon
        self.band = band
        self.long_threshold = long_threshold
        self.short_threshold = short_threshold
        self.model = GradientBoostingClassifier(
            n_estimators=n_estimators, max_depth=max_depth,
            learning_rate=learning_rate, subsample=0.8, random_state=random_state,
        )
        self._fitted = False

    def fit(self, dfs: list[pd.DataFrame]):
        """Train on one or more OHLCV histories. Pass MANY different
        synthetic markets for a general policy, or a single long real
        history (e.g. years of BTCUSDT) for a market-specific one."""
        Xs, ys = [], []
        for df in dfs:
            X, y = _make_features_labels(df, self.horizon, self.band)
            Xs.append(X)
            ys.append(y)
        X_train = np.vstack(Xs)
        y_train = np.concatenate(ys)
        self.model.fit(X_train, y_train)
        self._fitted = True
        return self

    def predict_proba_updown(self, data: pd.DataFrame):
        """data must already have indicators added (add_all_indicators).
        Returns (p_up, p_down) arrays aligned to data's rows."""
        proba = self.model.predict_proba(data[FEATURES].to_numpy())
        classes = list(self.model.classes_)
        p_up = proba[:, classes.index(1)] if 1 in classes else np.zeros(len(data))
        p_down = proba[:, classes.index(-1)] if -1 in classes else np.zeros(len(data))
        return p_up, p_down

    def target_fraction(self, p_up: float, p_down: float) -> float:
        """Confidence-scaled position size in [-1, 1]. A signal right at the
        threshold opens a small position; a highly confident one opens a
        large one -- this is what makes the agent size-aware instead of a
        binary long/short/flat flip."""
        if p_up > self.long_threshold and p_up > p_down:
            return float(min(1.0, (p_up - 0.30) / 0.4))
        if p_down > self.short_threshold and p_down > p_up:
            return -float(min(1.0, (p_down - 0.30) / 0.4))
        return 0.0

    def backtest(
        self,
        df: pd.DataFrame,
        starting_cash: float = 10_000.0,
        commission_rate: float = 0.001,
        slippage_rate: float = 0.0005,
        allow_short: bool = True,
    ) -> PaperBroker:
        """Greedy, deterministic backtest -- re-decides every `horizon`
        candles (see module docstring for why)."""
        data = add_all_indicators(df)
        p_up, p_down = self.predict_proba_updown(data)
        close = data["close"].to_numpy()
        open_time = data["open_time"].to_numpy()
        broker = PaperBroker(
            starting_cash=starting_cash, commission_rate=commission_rate,
            slippage_rate=slippage_rate, allow_short=allow_short,
        )
        i = 0
        n = len(close) - 1
        while i < n:
            price = close[i]
            frac = self.target_fraction(p_up[i], p_down[i])
            target_qty = frac * broker.max_affordable_qty(price)
            broker.set_target_position(target_qty, price, time=open_time[i], step=i)
            i += self.horizon
        return broker

    def decide_live(self, window_df: pd.DataFrame, position_qty: float, price: float) -> float:
        """For live_trader.py: given the current rolling window (with the
        newest candle last), return the target position fraction in
        [-1, 1] to hand to broker.max_affordable_qty()."""
        data = add_all_indicators(window_df)
        p_up, p_down = self.predict_proba_updown(data.iloc[[-1]])
        return self.target_fraction(float(p_up[0]), float(p_down[0]))

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str) -> "MLAgent":
        with open(path, "rb") as f:
            return pickle.load(f)
