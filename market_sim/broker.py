"""
broker.py
---------
A simple paper-trading broker. Tracks cash, a single position (long or
short, with size), applies commission + slippage, and records an equity
curve + trade log. This is the thing your bot "talks to" instead of a
real exchange.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Trade:
    step: int
    time: object
    side: str          # "buy" or "sell"
    price: float
    qty: float
    commission: float
    reason: str = ""


class PaperBroker:
    """
    Position convention: `position_qty` can be positive (long), negative
    (short), or zero (flat). `avg_entry_price` tracks the cost basis of
    the current open position.

    commission_rate: fraction of notional charged per trade (e.g. 0.001 = 0.1%)
    slippage_rate: fraction of price added against you on execution
    """

    def __init__(
        self,
        starting_cash: float = 10_000.0,
        commission_rate: float = 0.001,
        slippage_rate: float = 0.0005,
        allow_short: bool = True,
        max_leverage: float = 1.0,
        equity_floor_frac: float = 0.05,
    ):
        self.starting_cash = starting_cash
        self.cash = starting_cash
        self.commission_rate = commission_rate
        self.slippage_rate = slippage_rate
        self.allow_short = allow_short
        self.max_leverage = max_leverage
        self.equity_floor_frac = equity_floor_frac
        self.bankrupt = False

        self.position_qty = 0.0
        self.avg_entry_price = 0.0

        self.equity_curve: list[float] = []
        self.trades: list[Trade] = []

    def _exec_price(self, price: float, is_buy: bool) -> float:
        # slippage always works against you
        return price * (1 + self.slippage_rate) if is_buy else price * (1 - self.slippage_rate)

    def equity(self, mark_price: float) -> float:
        return self.cash + self.position_qty * mark_price

    def max_affordable_qty(self, price: float) -> float:
        equity = max(self.equity(price), 0.0)
        return (equity * self.max_leverage) / price

    def set_target_position(self, target_qty: float, price: float, time=None, step: int = 0, reason: str = ""):
        """
        Move the current position to `target_qty` (can be negative for
        short if allow_short). This is the single entry point agents use:
        e.g. target_qty=0 -> flatten, target_qty=+X -> go/extend long,
        target_qty=-X -> go/extend short.
        """
        if not self.allow_short and target_qty < 0:
            target_qty = 0.0

        # margin call: once equity falls to the floor, force-flatten and stop trading
        current_equity = self.equity(price)
        if self.bankrupt or current_equity <= self.starting_cash * self.equity_floor_frac:
            self.bankrupt = True
            target_qty = 0.0

        max_qty = self.max_affordable_qty(price)
        target_qty = max(-max_qty, min(max_qty, target_qty))

        delta = target_qty - self.position_qty
        if abs(delta) < 1e-9:
            self.equity_curve.append(self.equity(price))
            return

        is_buy = delta > 0
        exec_price = self._exec_price(price, is_buy)
        notional = abs(delta) * exec_price
        commission = notional * self.commission_rate

        # Pure cash-flow accounting: buying spends cash (delta>0 -> cash down),
        # selling/shorting raises cash (delta<0 -> cash up). Because equity is
        # always cash + position_qty*price, this one line is correct for
        # opening, closing, flipping, and partial long/short adjustments alike
        # -- no separate realized-PnL bookkeeping needed for equity to be right.
        self.cash -= delta * exec_price
        self.cash -= commission

        # cost basis, kept only for reporting (not used in equity calc)
        new_qty = self.position_qty + delta
        if (delta > 0 and target_qty > 0) or (delta < 0 and target_qty < 0):
            opened_qty = target_qty - self.position_qty if abs(new_qty) > abs(self.position_qty) else 0
            if opened_qty != 0:
                total_cost = self.avg_entry_price * self.position_qty + exec_price * opened_qty
                self.avg_entry_price = total_cost / (self.position_qty + opened_qty) if (self.position_qty + opened_qty) != 0 else exec_price

        self.position_qty = target_qty
        if self.position_qty == 0:
            self.avg_entry_price = 0.0

        self.trades.append(
            Trade(step=step, time=time, side="buy" if is_buy else "sell",
                  price=exec_price, qty=abs(delta), commission=commission, reason=reason)
        )
        self.equity_curve.append(self.equity(price))

    def reset(self):
        self.cash = self.starting_cash
        self.position_qty = 0.0
        self.avg_entry_price = 0.0
        self.equity_curve = []
        self.trades = []
