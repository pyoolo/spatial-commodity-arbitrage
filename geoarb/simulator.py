"""
simulator.py
============

A back-testable physical-arbitrage trading simulator.

Strategy (per lane, each day):
  * If the arb margin exceeds an entry threshold `entry_bps` (in USD/tonne)
    AND we have free capital AND lane capacity, BOOK a cargo:
        - buy `cargo_size` tonnes FOB at origin,
        - (optionally) hedge by SELLING futures to lock the flat price, so the
          PnL is driven by the *basis/spatial spread*, not outright flat price,
        - the cargo is "in transit" for `transit_days`.
  * On arrival, settle: realised PnL = (dest_price_arrival - delivered_cost_entry)
    per tonne, plus the futures hedge leg if hedging is on. Capital is freed.

This captures the two real risks a spatial arb trader actually runs:
  1. Freight/price move while the cargo is at sea (the window can close mid-voyage).
  2. Capital is locked during transit (financing + opportunity cost).

Hedging with futures removes the *flat-price* component of (2)'s risk, leaving
the basis/spatial component — which is the thing the trader is actually long.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
import pandas as pd


@dataclass
class SimConfig:
    entry_threshold: float = 4.0     # USD/tonne min margin to book a cargo
    cargo_size: float = 60_000.0     # tonnes (Panamax-ish)
    capital: float = 60_000_000.0    # USD working capital
    max_open_per_lane: int = 2       # concurrent cargoes per lane
    hedge_with_futures: bool = True  # sell futures against the long cargo
    financing_rate: float = 0.06
    speed_kn: float = 13.0


@dataclass
class _OpenCargo:
    lane: str
    entry_date: pd.Timestamp
    arrival_idx: int
    tonnes: float
    delivered_cost: float       # locked at entry, USD/tonne
    entry_futures: float        # futures level at entry (for hedge)
    capital_used: float


class ArbitrageSimulator:
    """Event-driven simulator over the arb-margin panel."""

    def __init__(self, config: SimConfig | None = None):
        self.cfg = config or SimConfig()

    def run(self, arb_df: pd.DataFrame,
            futures: pd.DataFrame) -> dict[str, pd.DataFrame]:
        cfg = self.cfg
        fut = futures.set_index("date")["futures"]
        dates = sorted(arb_df["date"].unique())
        date_to_idx = {d: i for i, d in enumerate(dates)}

        # pivot for fast access: dict[date] -> sub-df
        by_date = {d: g for d, g in arb_df.groupby("date")}

        free_capital = cfg.capital
        open_cargoes: list[_OpenCargo] = []
        open_count: dict[str, int] = {}
        trade_log = []
        equity_curve = []

        realised_pnl = 0.0

        for di, d in enumerate(dates):
            # 1) settle arrivals scheduled for today
            still_open = []
            for c in open_cargoes:
                if c.arrival_idx <= di:
                    sub = by_date[dates[c.arrival_idx]]
                    row = sub[sub["lane"] == c.lane]
                    if row.empty:
                        arrival_dest_price = c.delivered_cost  # fallback: flat
                    else:
                        arrival_dest_price = float(row["dest_price"].iloc[0])

                    # physical leg PnL per tonne
                    phys = arrival_dest_price - c.delivered_cost

                    # futures hedge leg: we SOLD futures at entry, buy back now.
                    hedge = 0.0
                    if cfg.hedge_with_futures:
                        f_now = float(fut.iloc[c.arrival_idx])
                        hedge = c.entry_futures - f_now   # short futures gain

                    pnl_per_t = phys + hedge
                    pnl = pnl_per_t * c.tonnes
                    realised_pnl += pnl
                    free_capital += c.capital_used + pnl
                    open_count[c.lane] = open_count.get(c.lane, 1) - 1

                    trade_log.append(dict(
                        lane=c.lane, entry_date=c.entry_date,
                        arrival_date=dates[c.arrival_idx],
                        tonnes=c.tonnes,
                        delivered_cost=c.delivered_cost,
                        arrival_dest_price=arrival_dest_price,
                        phys_pnl_per_t=phys, hedge_pnl_per_t=hedge,
                        pnl_per_t=pnl_per_t, pnl=pnl,
                    ))
                else:
                    still_open.append(c)
            open_cargoes = still_open

            # 2) look for new entries today (greedy by margin)
            sub = by_date[d].sort_values("arb_margin", ascending=False)
            f_today = float(fut.iloc[di])
            for _, row in sub.iterrows():
                if row["arb_margin"] < cfg.entry_threshold:
                    break
                lane = row["lane"]
                if open_count.get(lane, 0) >= cfg.max_open_per_lane:
                    continue
                cap_needed = row["delivered_cost"] * cfg.cargo_size
                if cap_needed > free_capital:
                    continue
                arrival_idx = min(di + int(round(row["transit_days"])), len(dates) - 1)
                open_cargoes.append(_OpenCargo(
                    lane=lane, entry_date=d, arrival_idx=arrival_idx,
                    tonnes=cfg.cargo_size,
                    delivered_cost=float(row["delivered_cost"]),
                    entry_futures=f_today,
                    capital_used=cap_needed,
                ))
                open_count[lane] = open_count.get(lane, 0) + 1
                free_capital -= cap_needed

            # 3) mark equity (realised + free capital; open cargoes held at cost)
            locked = sum(c.capital_used for c in open_cargoes)
            equity = free_capital + locked
            equity_curve.append(dict(date=d, equity=equity,
                                     realised_pnl=realised_pnl,
                                     open_cargoes=len(open_cargoes),
                                     free_capital=free_capital))

        return {
            "trades": pd.DataFrame(trade_log),
            "equity": pd.DataFrame(equity_curve),
        }
