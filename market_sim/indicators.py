"""
indicators.py
-------------
Pure pandas/numpy technical indicators (no ta-lib needed) used both for
display and as feature inputs to the trading agent.
"""

import numpy as np
import pandas as pd


def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window).mean()


def ema(series: pd.Series, window: int) -> pd.Series:
    return series.ewm(span=window, adjust=False).mean()


def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return (100 - (100 / (1 + rs))).fillna(50)


def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    macd_line = ema(series, fast) - ema(series, slow)
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    return tr.ewm(alpha=1 / window, adjust=False).mean()


def bollinger_bands(series: pd.Series, window: int = 20, n_std: float = 2.0):
    mid = sma(series, window)
    std = series.rolling(window).std()
    upper = mid + n_std * std
    lower = mid - n_std * std
    return upper, mid, lower


def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Returns a copy of df with a standard indicator feature set appended."""
    out = df.copy()
    out["sma_10"] = sma(out["close"], 10)
    out["sma_50"] = sma(out["close"], 50)
    out["ema_12"] = ema(out["close"], 12)
    out["ema_26"] = ema(out["close"], 26)
    out["rsi_14"] = rsi(out["close"], 14)
    macd_line, signal_line, hist = macd(out["close"])
    out["macd"] = macd_line
    out["macd_signal"] = signal_line
    out["macd_hist"] = hist
    out["atr_14"] = atr(out, 14)
    bb_up, bb_mid, bb_low = bollinger_bands(out["close"])
    out["bb_upper"] = bb_up
    out["bb_lower"] = bb_low
    out["return_1"] = out["close"].pct_change(1)
    out["return_5"] = out["close"].pct_change(5)
    out["volatility_10"] = out["return_1"].rolling(10).std()
    return out.bfill().fillna(0)
