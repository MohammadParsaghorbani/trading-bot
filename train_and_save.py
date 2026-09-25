"""
train_and_save.py
------------------
Trains the tabular Q-learning agent across many random synthetic markets
(so it learns a general policy, not one memorized price history) and saves
the result to outputs/trained_agent.json for live_trader.py to load.

Run: python train_and_save.py
"""

import os
from market_sim.market import SyntheticMarket
from market_sim.agents.q_agent import QLearningAgent

os.makedirs("outputs", exist_ok=True)


def make_market(seed=None):
    return SyntheticMarket(
        n_candles=800, start_price=100.0, mu=0.05, sigma=0.5,
        trend_switch_prob=0.003, trend_strength=3.0, seed=seed,
    ).generate()


if __name__ == "__main__":
    agent = QLearningAgent(alpha=0.15, gamma=0.97, epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.999, seed=1)
    agent.train_multi_market(lambda: make_market(seed=None), n_episodes=3000, turnover_penalty=0.0006)
    agent.save("outputs/trained_agent.json")
    print("saved outputs/trained_agent.json  (", len(agent.q_table), "states learned )")
