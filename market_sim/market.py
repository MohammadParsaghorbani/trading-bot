"""
market.py
---------
Produces OHLCV candle data for the simulator, either:
  1) SyntheticMarket  -> randomly generated (like a fake TradingView chart)
  2) BinanceMarket    -> real historical data pulled from Binance's public REST API

Both return the same shape: a pandas.DataFrame with columns
    ['open_time', 'open', 'high', 'low', 'close', 'volume']
so the rest of the system (broker, env, indicators) never needs to know
which source the candles came from.
"""

from __future__ import annotations

import time
import numpy as np
import pandas as pd


class SyntheticMarket:
    """
    Generates a fake but realistic-looking OHLCV series using a
    Geometric Brownian Motion price path with an added volatility
    regime switch (calm / volatile periods) so it doesn't look like
    pure noise, plus randomly sized wicks for open/high/low.

    Parameters
    ----------
    n_candles : number of candles to generate
    start_price : starting close price
    mu : annualized base drift (e.g. 0.0 = no trend, 0.2 = 20%/yr uptrend)
    sigma : annualized volatility (e.g. 0.5 = 50%/yr)
    candle_seconds : how many seconds each candle represents (60=1m, 3600=1h, 86400=1d)
    seed : RNG seed for reproducibility
    regime_switch_prob : probability per candle of flipping calm/volatile regime
    trend_switch_prob : probability per candle of switching bull/bear/choppy trend regime.
        Kept low on purpose (trend regimes should last many candles, like real
        markets) so there is an actual learnable pattern -- not just noise --
        for momentum/trend indicators (and the RL agent) to pick up on.
    trend_strength : extra annualized drift added during a trending regime
    """

    def __init__(
        self,
        n_candles: int = 2000,
        start_price: float = 100.0,
        mu: float = 0.05,
        sigma: float = 0.6,
        candle_seconds: int = 3600,
        seed: int | None = None,
        regime_switch_prob: float = 0.01,
        trend_switch_prob: float = 0.003,
        trend_strength: float = 0.9,
    ):
        self.n_candles = n_candles
        self.start_price = start_price
        self.mu = mu
        self.sigma = sigma
        self.candle_seconds = candle_seconds
        self.rng = np.random.default_rng(seed)
        self.regime_switch_prob = regime_switch_prob
        self.trend_switch_prob = trend_switch_prob
        self.trend_strength = trend_strength

    def generate(self) -> pd.DataFrame:
        dt = self.candle_seconds / (365 * 24 * 3600)  # fraction of a year per candle
        n = self.n_candles

        # volatility regime: alternate between calm (0.6x) and volatile (1.8x) multipliers
        regime = np.ones(n)
        current = 1.0
        for i in range(n):
            if self.rng.random() < self.regime_switch_prob:
                current = 1.8 if current == 0.6 else 0.6
            regime[i] = current

        # trend regime: -1 (bear), 0 (choppy/range), +1 (bull) -- persists for long
        # stretches (mean duration ~ 1/trend_switch_prob candles) so a trend-
        # following signal (e.g. SMA10 vs SMA50) has something real to catch,
        # instead of every market being an unpredictable pure random walk.
        trend = np.zeros(n)
        current_trend = 0
        trend_choices = np.array([-1, 0, 1])
        for i in range(n):
            if self.rng.random() < self.trend_switch_prob:
                current_trend = self.rng.choice(trend_choices)
            trend[i] = current_trend

        shocks = self.rng.normal(0.0, 1.0, size=n)
        vol = self.sigma * regime
        mu_t = self.mu + trend * self.trend_strength
        log_returns = (mu_t - 0.5 * vol**2) * dt + vol * np.sqrt(dt) * shocks
        close = self.start_price * np.exp(np.cumsum(log_returns))
        close = np.insert(close, 0, self.start_price)[:n]  # keep length n

        open_ = np.empty(n)
        open_[0] = self.start_price
        open_[1:] = close[:-1]

        # intrabar wiggle for high/low, proportional to that candle's volatility
        wiggle = np.abs(self.rng.normal(0.0, vol * np.sqrt(dt), size=n)) * close
        high = np.maximum(open_, close) + wiggle * self.rng.uniform(0.1, 1.0, size=n)
        low = np.minimum(open_, close) - wiggle * self.rng.uniform(0.1, 1.0, size=n)
        low = np.clip(low, 0.01, None)

        # volume loosely correlated with |return| and volatility regime
        base_vol = self.rng.lognormal(mean=8.0, sigma=0.5, size=n)
        volume = base_vol * (1 + 3 * np.abs(log_returns) / (vol * np.sqrt(dt) + 1e-9))

        start_ts = int(time.time()) - n * self.candle_seconds
        open_time = start_ts + np.arange(n) * self.candle_seconds

        df = pd.DataFrame(
            {
                "open_time": pd.to_datetime(open_time, unit="s"),
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
            }
        )
        return df

    def stream(self):
        """
        Generator version of generate(): yields one new candle (as a dict)
        at a time, forever, continuing the same regime/trend state. This is
        what a *live* paper-trading loop consumes -- instead of a fixed
        pre-generated history, it's an endless feed, just like a real
        exchange's websocket would be.
        """
        dt = self.candle_seconds / (365 * 24 * 3600)
        vol_regime = 1.0
        trend_regime = 0
        trend_choices = np.array([-1, 0, 1])
        last_close = self.start_price
        t = int(time.time())

        while True:
            if self.rng.random() < self.regime_switch_prob:
                vol_regime = 1.8 if vol_regime == 0.6 else 0.6
            if self.rng.random() < self.trend_switch_prob:
                trend_regime = self.rng.choice(trend_choices)

            vol = self.sigma * vol_regime
            mu_t = self.mu + trend_regime * self.trend_strength
            shock = self.rng.normal(0.0, 1.0)
            log_ret = (mu_t - 0.5 * vol**2) * dt + vol * np.sqrt(dt) * shock

            open_ = last_close
            close_ = open_ * np.exp(log_ret)
            wiggle = abs(self.rng.normal(0.0, vol * np.sqrt(dt))) * close_
            high_ = max(open_, close_) + wiggle * self.rng.uniform(0.1, 1.0)
            low_ = max(0.01, min(open_, close_) - wiggle * self.rng.uniform(0.1, 1.0))
            volume_ = self.rng.lognormal(mean=8.0, sigma=0.5) * (
                1 + 3 * abs(log_ret) / (vol * np.sqrt(dt) + 1e-9)
            )

            row = {
                "open_time": pd.Timestamp(t, unit="s"),
                "open": open_, "high": high_, "low": low_, "close": close_, "volume": volume_,
            }
            last_close = close_
            t += self.candle_seconds
            yield row


class BinanceMarket:
    """
    Pulls real historical OHLCV candles from Binance's public REST API
    (no API key needed for public klines data).

    NOTE: requires outbound internet access to api.binance.com, which the
    sandbox this code was authored in does not have. It is written and
    ready to run on your own machine (`pip install requests`, then run).

    Parameters
    ----------
    symbol : e.g. "BTCUSDT"
    interval : e.g. "1m","5m","15m","1h","4h","1d"
    limit : number of candles to fetch (Binance caps a single call at 1000)
    """

    BASE_URL = "https://api.binance.com/api/v3/klines"

    def __init__(self, symbol: str = "BTCUSDT", interval: str = "1h", limit: int = 1000):
        self.symbol = symbol
        self.interval = interval
        self.limit = min(limit, 1000)

    def fetch(self) -> pd.DataFrame:
        import requests

        params = {"symbol": self.symbol, "interval": self.interval, "limit": self.limit}
        resp = requests.get(self.BASE_URL, params=params, timeout=15)
        resp.raise_for_status()
        raw = resp.json()

        cols = [
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "n_trades",
            "taker_buy_base", "taker_buy_quote", "ignore",
        ]
        df = pd.DataFrame(raw, columns=cols)
        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
        for c in ["open", "high", "low", "close", "volume"]:
            df[c] = df[c].astype(float)
        return df[["open_time", "open", "high", "low", "close", "volume"]]

    def fetch_history(self, total_candles: int) -> pd.DataFrame:
        """Page backwards through Binance's 1000-candle limit to get a longer history."""
        import requests

        all_frames = []
        end_time = None
        remaining = total_candles
        while remaining > 0:
            batch = min(remaining, 1000)
            params = {"symbol": self.symbol, "interval": self.interval, "limit": batch}
            if end_time is not None:
                params["endTime"] = end_time
            resp = requests.get(self.BASE_URL, params=params, timeout=15)
            resp.raise_for_status()
            raw = resp.json()
            if not raw:
                break
            cols = [
                "open_time", "open", "high", "low", "close", "volume",
                "close_time", "quote_volume", "n_trades",
                "taker_buy_base", "taker_buy_quote", "ignore",
            ]
            df = pd.DataFrame(raw, columns=cols)
            all_frames.append(df)
            end_time = int(raw[0][0]) - 1  # page further back in time
            remaining -= len(raw)
            if len(raw) < batch:
                break

        full = pd.concat(all_frames[::-1], ignore_index=True)
        full["open_time"] = pd.to_datetime(full["open_time"], unit="ms")
        for c in ["open", "high", "low", "close", "volume"]:
            full[c] = full[c].astype(float)
        return full[["open_time", "open", "high", "low", "close", "volume"]].drop_duplicates("open_time")

    def poll_latest(self) -> pd.Series:
        """Fetch just the most recently CLOSED candle -- what a live trading
        loop calls every `interval` on a timer to get the newest bar."""
        df = self.fetch()
        return df.iloc[-2]  # -1 is often the still-forming current candle
