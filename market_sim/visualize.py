"""
visualize.py
------------
Two outputs:
  1) plot_backtest() -> a matplotlib PNG with candlesticks + trade markers + equity curve
  2) export_dashboard_json() -> a JSON blob the web dashboard (dashboard.html) reads
"""

from __future__ import annotations

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from .broker import PaperBroker


def plot_backtest(df: pd.DataFrame, broker: PaperBroker, title: str = "Backtest", out_path: str = "outputs/backtest.png"):
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(14, 8), sharex=False, gridspec_kw={"height_ratios": [3, 1]}
    )

    # --- candlesticks ---
    df = df.reset_index(drop=True)
    width = 0.6
    for i, row in df.iterrows():
        color = "#26a69a" if row["close"] >= row["open"] else "#ef5350"
        ax1.plot([i, i], [row["low"], row["high"]], color=color, linewidth=0.7)
        ax1.add_patch(
            plt.Rectangle(
                (i - width / 2, min(row["open"], row["close"])),
                width,
                max(abs(row["close"] - row["open"]), 1e-9),
                color=color,
            )
        )

    for t in broker.trades:
        marker = "^" if t.side == "buy" else "v"
        color = "lime" if t.side == "buy" else "red"
        ax1.scatter(t.step, t.price, marker=marker, color=color, s=60, zorder=5, edgecolors="black", linewidths=0.5)

    ax1.set_title(title)
    ax1.set_ylabel("price")
    ax1.set_xlim(0, len(df))

    # --- equity curve ---
    eq = broker.equity_curve
    ax2.plot(range(len(eq)), eq, color="#42a5f5", linewidth=1.2)
    ax2.axhline(broker.starting_cash, color="gray", linestyle="--", linewidth=0.8)
    ax2.set_ylabel("equity")
    ax2.set_xlabel("step")

    plt.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def export_dashboard_json(df: pd.DataFrame, broker: PaperBroker, out_path: str = "outputs/backtest_data.json"):
    df = df.reset_index(drop=True)
    candles = [
        {
            "t": str(row["open_time"]),
            "o": round(float(row["open"]), 6),
            "h": round(float(row["high"]), 6),
            "l": round(float(row["low"]), 6),
            "c": round(float(row["close"]), 6),
            "v": round(float(row["volume"]), 2),
        }
        for _, row in df.iterrows()
    ]
    trades = [
        {"step": t.step, "time": str(t.time), "side": t.side, "price": round(t.price, 6), "qty": round(t.qty, 6)}
        for t in broker.trades
    ]
    equity = [round(float(e), 4) for e in broker.equity_curve]

    stats = compute_stats(broker)

    payload = {"candles": candles, "trades": trades, "equity": equity, "stats": stats}
    with open(out_path, "w") as f:
        json.dump(payload, f)
    return out_path


def compute_stats(broker: PaperBroker) -> dict:
    eq = np.array(broker.equity_curve) if broker.equity_curve else np.array([broker.starting_cash])
    total_return = (eq[-1] / broker.starting_cash - 1) * 100
    returns = np.diff(eq) / eq[:-1] if len(eq) > 1 else np.array([0.0])
    sharpe = float(np.mean(returns) / (np.std(returns) + 1e-9) * np.sqrt(252)) if len(returns) > 1 else 0.0
    running_max = np.maximum.accumulate(eq)
    drawdown = (eq - running_max) / running_max
    max_drawdown = float(drawdown.min() * 100)
    n_trades = len(broker.trades)
    total_commission = float(sum(t.commission for t in broker.trades))

    return {
        "starting_cash": broker.starting_cash,
        "final_equity": float(eq[-1]),
        "total_return_pct": round(float(total_return), 2),
        "sharpe_approx": round(sharpe, 3),
        "max_drawdown_pct": round(max_drawdown, 2),
        "n_trades": n_trades,
        "total_commission": round(total_commission, 4),
    }
