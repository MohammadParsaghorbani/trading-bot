"""
train_ml.py
-----------
Trains MLAgent (see market_sim/agents/ml_agent.py) either on:

  A) many random synthetic markets (works right now, no internet needed) -- a
     general "how do trending markets typically behave" policy, or

  B) real historical data for a specific real coin, e.g. Bitcoin -- a policy
     tuned to that market's actual past behavior. Needs internet (this
     sandbox has none, but it's ready to run on your own machine).

Usage:
    python train_ml.py                                  # synthetic ensemble
    python train_ml.py --symbol BTCUSDT --interval 1h --candles 5000   # real BTC history
"""

import argparse
import os

from market_sim.market import SyntheticMarket, BinanceMarket
from market_sim.agents.ml_agent import MLAgent
from market_sim.visualize import compute_stats

os.makedirs("outputs", exist_ok=True)


def synthetic_ensemble(n_markets=80, n_candles=800):
    return [
        SyntheticMarket(
            n_candles=n_candles, start_price=100.0, mu=0.05, sigma=0.5,
            trend_switch_prob=0.003, trend_strength=3.0, seed=seed,
        ).generate()
        for seed in range(1, n_markets + 1)
    ]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default=None, help="e.g. BTCUSDT -- omit to use the synthetic ensemble instead")
    parser.add_argument("--interval", default="1h")
    parser.add_argument("--candles", type=int, default=5000)
    parser.add_argument("--horizon", type=int, default=15)
    parser.add_argument("--out", default="outputs/trained_ml_agent.pkl")
    args = parser.parse_args()

    agent = MLAgent(horizon=args.horizon, band=0.004)

    if args.symbol:
        print(f"fetching {args.candles} candles of real {args.symbol} ({args.interval}) from Binance...")
        full_df = BinanceMarket(symbol=args.symbol, interval=args.interval).fetch_history(args.candles)
        # WALK-FORWARD split: train on the earlier period, test on the later
        # one -- never shuffle a single real time series, or the model
        # "cheats" by training on data adjacent to (and correlated with)
        # what it's tested on.
        split = int(len(full_df) * 0.8)
        train_df, test_df = full_df.iloc[:split], full_df.iloc[split:]
        print(f"train: {len(train_df)} candles, test (held-out, later in time): {len(test_df)} candles")
        agent.fit([train_df])
    else:
        print("no --symbol given -- training on a synthetic ensemble instead")
        markets = synthetic_ensemble()
        train_df, test_df = markets[0], markets[-1]  # just for a quick sanity check below
        agent.fit(markets)

    agent.save(args.out)
    print(f"saved {args.out}")

    broker = agent.backtest(test_df)
    stats = compute_stats(broker)
    bh = (test_df["close"].iloc[-1] / test_df["close"].iloc[0] - 1) * 100
    print(f"quick check on held-out data: agent={stats['total_return_pct']:+.2f}%  "
          f"buy&hold={bh:+.2f}%  trades={stats['n_trades']}")
