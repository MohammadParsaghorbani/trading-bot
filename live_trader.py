"""
live_trader.py
--------------
The actual "paper trading" runtime: a loop that watches a live feed --
either the endless SyntheticMarket.stream() (for testing/demo, works with
zero internet) or real Binance candles via BinanceMarket.poll_latest() (for
real crypto data, needs internet + `pip install requests`) -- and executes
trades through PaperBroker in real time, exactly like it would against a
real exchange.

Defaults to the ML agent (agents/ml_agent.py, trained by train_ml.py) --
see the README for why that's the recommended one over the tabular
Q-learning agent. Both share a decide_live() interface so either can be
dropped in.

It keeps a rolling window of the last `window` candles, recomputes
indicators + the trained agent's decision each new candle, and executes it.
Every candle it writes a fresh outputs/live_data.json in the exact shape
dashboard.html expects, so you can just keep reloading the dashboard to
watch it trade live.

Usage:
    python train_ml.py                                     # train once (synthetic, or --symbol BTCUSDT)
    python live_trader.py                                  # synthetic feed, demo speed
    python live_trader.py --source binance --symbol BTCUSDT --interval 1h
    python live_trader.py --agent-type q --agent-path outputs/trained_agent.json   # use the Q-agent instead
"""

from __future__ import annotations

import argparse
import json
import os
import time

import pandas as pd

from market_sim.broker import PaperBroker
from market_sim.market import SyntheticMarket, BinanceMarket
from market_sim.agents.q_agent import QLearningAgent
from market_sim.agents.ml_agent import MLAgent
from market_sim.visualize import compute_stats


def build_agent(agent_type: str, agent_path: str):
    if agent_type == "ml":
        if os.path.exists(agent_path):
            print(f"loading trained ML agent from {agent_path}")
            return MLAgent.load(agent_path)
        print(f"no saved ML agent found at {agent_path} -- run train_ml.py first. "
              "Falling back to an untrained (always-flat) placeholder.")
        agent = MLAgent()
        return agent
    else:
        if os.path.exists(agent_path):
            print(f"loading trained Q-agent from {agent_path}")
            return QLearningAgent.load(agent_path)
        print(f"no saved Q-agent found at {agent_path} -- run train_and_save.py first. "
              "Falling back to an untrained (always-flat) placeholder.")
        return QLearningAgent(epsilon=0.0, epsilon_min=0.0)


def synthetic_feed(n_candles: int | None = None):
    market = SyntheticMarket(
        start_price=100.0, mu=0.05, sigma=0.5,
        trend_switch_prob=0.003, trend_strength=3.0, seed=None,
    )
    count = 0
    for row in market.stream():
        yield row
        count += 1
        if n_candles is not None and count >= n_candles:
            return


def binance_feed(symbol: str, interval: str, poll_seconds: int):
    source = BinanceMarket(symbol=symbol, interval=interval)
    last_time = None
    while True:
        row = source.poll_latest()
        if last_time is None or row["open_time"] > last_time:
            last_time = row["open_time"]
            yield row.to_dict()
        time.sleep(poll_seconds)


def run(
    feed,
    agent,
    decision_interval: int = 1,
    window: int = 80,
    starting_cash: float = 10_000.0,
    commission_rate: float = 0.001,
    slippage_rate: float = 0.0005,
    allow_short: bool = True,
    out_path: str = "outputs/live_data.json",
    sleep_seconds: float = 0.0,
):
    """decision_interval: re-decide the position every N candles rather than
    every single one. For MLAgent, pass its own .horizon here (train_ml.py
    prints this) -- deciding at the same cadence the model was trained to
    predict at is what stops it from reacting to noise between predictions.
    For the Q-agent, 1 is fine (it decides every candle)."""
    broker = PaperBroker(
        starting_cash=starting_cash, commission_rate=commission_rate,
        slippage_rate=slippage_rate, allow_short=allow_short,
    )
    rows = []
    step = 0
    candles_since_decision = decision_interval  # force a decision on the first candle

    for row in feed:
        rows.append(row)
        if len(rows) > window + 5:  # keep a bounded rolling window in memory
            rows.pop(0)
        if len(rows) < 20:
            continue  # not enough history yet for indicators

        df = pd.DataFrame(rows)
        price = float(df["close"].iloc[-1])

        if candles_since_decision >= decision_interval:
            frac = agent.decide_live(df, broker.position_qty, price)
            candles_since_decision = 0
        else:
            frac = None  # hold whatever position we already have
        candles_since_decision += 1

        if frac is not None:
            target_qty = frac * broker.max_affordable_qty(price)
            broker.set_target_position(target_qty, price, time=df["open_time"].iloc[-1], step=step)
        step += 1

        equity = broker.equity(price)
        side = "LONG" if broker.position_qty > 0 else ("SHORT" if broker.position_qty < 0 else "FLAT")
        tag = " (new decision)" if frac is not None else ""
        print(f"[{df['open_time'].iloc[-1]}] price={price:.2f}  {side:<5}{tag}  "
              f"position={broker.position_qty:+.4f}  equity=${equity:,.2f}")

        # write a live snapshot every candle so the dashboard can be reloaded at any time
        stats = compute_stats(broker)
        payload = {
            "candles": [
                {"t": str(r["open_time"]), "o": round(float(r["open"]), 6),
                 "h": round(float(r["high"]), 6), "l": round(float(r["low"]), 6),
                 "c": round(float(r["close"]), 6), "v": round(float(r["volume"]), 2)}
                for r in rows
            ],
            "trades": [
                {"step": t.step, "time": str(t.time), "side": t.side,
                 "price": round(t.price, 6), "qty": round(t.qty, 6)}
                for t in broker.trades[-200:]
            ],
            "equity": [round(float(e), 4) for e in broker.equity_curve[-len(rows):]],
            "stats": stats,
        }
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(payload, f)

        if sleep_seconds:
            time.sleep(sleep_seconds)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["synthetic", "binance"], default="synthetic")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--interval", default="1h")
    parser.add_argument("--poll-seconds", type=int, default=60,
                         help="how often to poll Binance for a new candle (only used with --source binance)")
    parser.add_argument("--n-candles", type=int, default=300,
                         help="stop after this many candles (synthetic only; omit/None to run forever)")
    parser.add_argument("--sleep-seconds", type=float, default=0.05,
                         help="pause between candles (synthetic demo pacing; 0 = as fast as possible)")
    parser.add_argument("--agent-type", choices=["ml", "q"], default="ml")
    parser.add_argument("--agent-path", default=None,
                         help="defaults to outputs/trained_ml_agent.pkl or outputs/trained_agent.json depending on --agent-type")
    parser.add_argument("--decision-interval", type=int, default=None,
                         help="defaults to the ML agent's own .horizon, or 1 for the Q-agent")
    parser.add_argument("--out", default="outputs/live_data.json")
    args = parser.parse_args()

    agent_path = args.agent_path or (
        "outputs/trained_ml_agent.pkl" if args.agent_type == "ml" else "outputs/trained_agent.json"
    )
    agent = build_agent(args.agent_type, agent_path)
    decision_interval = args.decision_interval or (getattr(agent, "horizon", 1) if args.agent_type == "ml" else 1)

    if args.source == "synthetic":
        feed = synthetic_feed(n_candles=args.n_candles)
    else:
        feed = binance_feed(args.symbol, args.interval, args.poll_seconds)

    run(feed, agent, decision_interval=decision_interval, out_path=args.out, sleep_seconds=args.sleep_seconds)
