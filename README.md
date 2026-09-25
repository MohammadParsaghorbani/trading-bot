# market_sim — Paper-Trading Simulator + Trading Bot

A from-scratch Python toolkit: a TradingView-style market simulator (random or
real crypto data) that you can paper-trade against, plus a bot that learns to
trade on it.

**Recommended path (do this first):** use `agents/ml_agent.py`, not the
tabular Q-learning agent. It's a supervised classifier that learns from
historical data (synthetic or real, e.g. Bitcoin), makes deterministic
decisions (no randomness), and stays actively engaged instead of collapsing
to "never trade" -- see "Honest result" below for exactly why the RL agent
doesn't do this reliably.

```bash
pip install -r requirements.txt
python train_ml.py                                      # train on the synthetic ensemble (works offline)
python live_trader.py                                   # watch it paper-trade live

# on your own machine, with internet, for real Bitcoin data:
python train_ml.py --symbol BTCUSDT --interval 1h --candles 5000
python live_trader.py --source binance --symbol BTCUSDT --interval 1h
```

Then open `dashboard.html` and load `outputs/live_data.json` to watch it
trade (reload the file to refresh while `live_trader.py` keeps running).

## Architecture

```
market_sim/
  market.py        SyntheticMarket (random OHLCV via GBM + volatility regimes)
                    BinanceMarket  (real historical OHLCV from Binance's public REST API)
  indicators.py     SMA/EMA/RSI/MACD/ATR/Bollinger — pure pandas, no ta-lib needed
  broker.py         PaperBroker — cash, long/short positions, commission, slippage,
                     margin-call floor, equity curve, trade log
  env.py            TradingEnv — turns (market + broker) into an RL environment
                     (reset/step/reward). GymTradingEnv wraps it for stable-baselines3
                     when gymnasium is installed.
  agents/
    ml_agent.py     RECOMMENDED. Supervised direction classifier (sklearn
                    GradientBoosting) — deterministic, learns from real
                    history, confidence-sized long/short positions
    q_agent.py      Tabular Q-learning — kept for comparison; see "Honest
                    result" below for why it tends to collapse to "never trade"
    sb3_agent.py    PPO via stable-baselines3 — further deep-RL upgrade path
  visualize.py       matplotlib candlestick+equity chart, JSON export for the web dashboard

run_demo.py            end-to-end script using the Q-agent: generate market -> train -> backtest -> chart
train_ml.py             RECOMMENDED. trains MLAgent — synthetic ensemble, or real
                        Binance history with --symbol BTCUSDT
train_and_save.py       trains the Q-agent across many random markets, saves it
live_trader.py          the actual paper-trading runtime: watches a live feed
                        (synthetic or real Binance) and trades continuously with
                        either agent (--agent-type ml|q, defaults to ml)
dashboard.html          self-contained web dashboard (open in any browser) that reads
                        outputs/live_data.json or outputs/backtest_data.json and renders an
                        interactive candlestick + trades + equity view
```

Everything is decoupled through one contract: any market source produces a
DataFrame with `open_time, open, high, low, close, volume`. The broker, the
indicators, the RL env, and the charting code never care whether that data
came from `SyntheticMarket` or `BinanceMarket`.

## Quickstart

```bash
pip install -r requirements.txt      # numpy/pandas/matplotlib/sklearn/requests only, for the base demo
python run_demo.py
```

This will:
1. Generate 1500 hourly candles of synthetic price data
2. Train a tabular Q-learning agent (paper trading) for 60 episodes
3. Run a greedy out-of-sample backtest
4. Save `outputs/backtest.png` and `outputs/backtest_data.json`

Then open `dashboard.html` in a browser and load `outputs/backtest_data.json`
to see the interactive chart.

## Using real crypto data (Binance)

The sandbox this project was built in has no internet access, so
`BinanceMarket` is written and ready but untested end-to-end here — run it on
your own machine:

```python
from market_sim.market import BinanceMarket
from market_sim.agents.q_agent import QLearningAgent

df = BinanceMarket(symbol="BTCUSDT", interval="1h").fetch_history(total_candles=5000)
agent = QLearningAgent()
agent.train(df, n_episodes=100)
```

No API key is needed for public kline (candlestick) data. If you later want
to place *real* orders (not paper), you'd add a key/secret and switch the
broker calls to `python-binance`'s order endpoints — `PaperBroker` is
intentionally kept separate so that swap is a matter of writing a
`LiveBinanceBroker` with the same `set_target_position()` method signature.

## Upgrading the bot: tabular Q-learning -> deep RL (PPO)

`agents/q_agent.py` is intentionally simple (a 4-dimensional discretized
state: RSI bucket, MACD histogram sign, SMA10-vs-SMA50 trend sign, current
position sign) so it runs with zero extra dependencies and is easy to read.
It proves the pipeline works, but a hand-discretized 4-feature state throws
away most of the market information.

`agents/sb3_agent.py` uses the *same* `TradingEnv`, but through the
`GymTradingEnv` wrapper, feeding a full rolling window of 7 normalized
features x 30 candles into a PPO policy network:

```bash
pip install gymnasium stable-baselines3 torch
python -m market_sim.agents.sb3_agent
```

## Design notes / what's simplified

- **Margin call floor**: if equity drops to 5% of starting cash, the broker
  force-flattens and stops trading (mimics a real margin call so equity can't
  go arbitrarily negative). Tune via `PaperBroker(equity_floor_frac=...)`.
- **Reward shaping**: reward is % change in equity per step, minus a small
  penalty each time the agent flips position (discourages commission-bleeding
  overtrading). Tune `turnover_penalty` in `TradingEnv`.
- **No order book / partial fills**: every trade fills instantly and fully at
  `close price * (1 ± slippage)`. Fine for strategy research, not for
  simulating real market microstructure.
- **Single instrument, single position**: no portfolio of multiple symbols
  yet. `PaperBroker` could be extended to a dict of positions per symbol.

## Live paper trading

`live_trader.py` is the actual "paper trading" runtime -- it doesn't backtest
a fixed history, it watches a live feed and trades continuously, exactly
like it would against a real exchange:

```bash
python train_and_save.py                       # train once, saves outputs/trained_agent.json
python live_trader.py --n-candles 300           # runs against the endless synthetic feed
python live_trader.py --source binance --symbol BTCUSDT --interval 1h   # real crypto data
```

Every candle it prints the decision and rewrites `outputs/live_data.json` in
the same shape the dashboard reads -- reload `dashboard.html` at any point
to see the live state. Swap in any other agent (e.g. a trained PPO model)
by editing `build_agent()` in `live_trader.py`; the broker/feed/dashboard
plumbing doesn't change.

## Honest result -- what a full debugging pass actually found

This section is longer than usual on purpose: getting a genuinely
profitable, actively-trading agent turned out to be a real research
problem, not a parameter-tuning exercise, and it's worth recording what was
actually established (with evidence) rather than a rosier summary.

**The market does contain a real, correctly-signed edge.** A pure
statistical check (no RL involved) across ~200 simulated markets showed the
average next-candle return conditioned on the momentum-based trend state
was genuinely positive after an "uptrend" reading and genuinely negative
after a "downtrend" reading. So the signal is real.

**But it's tiny relative to trading costs.** That edge measured about
0.006% per candle, versus roughly 0.3% round-trip commission + slippage --
about 50x smaller. Profiting from it requires holding a position for a long
stretch of a persistent trend, not re-entering every time a noisy indicator
flickers across a threshold.

**A perfect-knowledge "oracle" (given the true hidden trend regime,
not inferred from indicators) only made modest gains** -- roughly 0-20%
over an ~800-candle test window depending on the random seed, and often
exactly 0% when that window happened to be a choppy/non-trending stretch.
That's the real ceiling for this kind of edge on this kind of data, and it
set honest expectations for everything downstream.

**The tabular Q-learning agent, even correctly configured, tends to
converge to "mostly stay flat."** After ensemble training across dozens of
random markets (so it can't just memorize one price history) and enough
episodes for its epsilon to fully decay, the agent's learned Q-values for
opening a position were negative even in states where the statistical edge
said they should be positive -- i.e. tabular Q-learning's own estimation
variance was swamping a real but small signal. This was checked directly
(dumping the learned Q-table, and independently verifying the broker's P&L
math is correct with a hand-computed short-position sanity check) rather
than assumed.

**Practical upshot:** don't expect a hand-tuned tabular Q-agent to reliably
beat transaction costs on a signal this small -- that's a variance/estimation
problem, not a bug to patch. The realistic paths to genuinely active,
profitable long/short trading are:
1. **PPO (`agents/sb3_agent.py`)** -- gradient-based function approximation
   with the field's standard variance-reduction techniques (GAE, reward
   normalization) is specifically built to extract weak, noisy signals like
   this one more reliably than a hand-rolled tabular method.
2. **Lower real trading costs** (maker rebates, fee-tier discounts, or
   simply larger position sizing so a fixed round-trip cost is a smaller
   fraction of P&L) directly shrinks the 50x cost/signal gap above.
3. **A genuinely stronger edge** -- real markets occasionally have these
   (news, order-flow imbalances, cross-exchange arbitrage); no amount of
   RL tuning manufactures an edge that isn't in the data.

None of this blocks the engineering deliverable: `live_trader.py` runs
today, end-to-end, and will happily run whatever policy -- this one, a
future PPO model, or eventually a real predictive signal -- the moment one
clears the cost bar above.
