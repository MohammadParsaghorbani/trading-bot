"""
run_demo.py
-----------
End-to-end smoke test of the whole pipeline, using only numpy/pandas/
matplotlib/sklearn (no internet, no torch needed):

    1. Generate a synthetic market (fake TradingView-style candles)
    2. Train the tabular Q-learning agent on it (paper trading)
    3. Run a greedy backtest
    4. Save a matplotlib chart + a JSON file for the web dashboard

Run: python run_demo.py
"""

import os
from market_sim.market import SyntheticMarket
from market_sim.agents.q_agent import QLearningAgent
from market_sim.visualize import plot_backtest, export_dashboard_json, compute_stats

os.makedirs("outputs", exist_ok=True)

print("1. Generating synthetic market...")
market = SyntheticMarket(
    n_candles=3000, start_price=100.0, mu=0.05, sigma=0.5,
    trend_switch_prob=0.003, trend_strength=0.9, seed=7,
)
df = market.generate()
train_df = df.iloc[: int(len(df) * 0.8)].reset_index(drop=True)
test_df = df.iloc[int(len(df) * 0.8):].reset_index(drop=True)

print("2. Training Q-learning agent...")
agent = QLearningAgent(alpha=0.15, gamma=0.95, epsilon=1.0, epsilon_decay=0.985, seed=1)
returns = agent.train(train_df, n_episodes=120, verbose=True)

print("3. Backtesting (greedy, out-of-sample)...")
broker = agent.run_episode(test_df)
stats = compute_stats(broker)
print("Backtest stats:", stats)

print("4. Saving chart + dashboard data...")
plot_backtest(test_df, broker, title="Q-Agent Out-of-Sample Backtest", out_path="outputs/backtest.png")
export_dashboard_json(test_df, broker, out_path="outputs/backtest_data.json")

print("Done. See outputs/backtest.png and outputs/backtest_data.json")
