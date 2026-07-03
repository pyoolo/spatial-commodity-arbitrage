"""
simulator.py
============

A back-testable physical-arbitrage trading simulator.

Strategy (per lane, each day):
  * If the arb margin exceeds an entry threshold ``entry_threshold`` (USD/tonne)
    AND we have free capital AND lane capacity, BOOK a cargo:
        - buy ``cargo_size`` tonnes FOB at origin,
        - (optionally) hedge by SELLING futures to lock the flat price, so the
          PnL is driven by the *basis / spatial spread*, not outright flat price,
        - the cargo is "in transit" for ``transit_days``.
  * On arrival, settle: realised PnL = (dest_price_arrival - delivered_cost_entry)
    per tonne, plus the futures hedge leg if hedging is on. Capital is freed.

This captures the two real risks a spatial-arb trader actually runs:
  1. Freight / price move while the cargo is at sea (the window can close
     mid-voyage).
  2. Capital is locked during transit (financing + opportunity cost).

Hedging with futures removes the *flat-price* component of that risk, leaving
the basis / spatial component -- which is the thing the trader is actually long.

Accounting notes
----------------
* **Capital locked at booking** is the cash actually outlaid up front: the FOB
  purchase plus the in-transit financing charge. Freight, destination handling
  and tariff are settlement-side costs and are *not* pre-funded here; they are
  still fully reflected in ``delivered_cost`` and therefore in realised PnL.
  (The previous version locked the entire delivered cost, which overstated
  working-capital usage and throttled position count.)
* **Equity is marked to market** each day: open cargoes are revalued at the
  current lane margin rather than frozen at cost. This makes the equity path --
  and hence volatility and drawdown -- reflect in-transit risk instead of
  jumping discretely only on settlement.
"""

from __future__ import annotations

from dataclasses import dataclass
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
    entry_idx: int
    arrival_idx: int
    tonnes: float
    delivered_cost: float       # locked at entry, USD/tonne
    entry_futures: float        # futures level at entry (for the hedge leg)
    capital_used: float         # cash outlaid up front (FOB + financing)


class ArbitrageSimulator:
    """Event-driven simulator over the arb-margin panel."""

    def __init__(self, config: SimConfig | None = None):
        self.cfg = config or SimConfig()

    def _upfront_capital(self, row) -> float:
        """Cash outlaid at booking: FOB purchase + in-transit financing.

        ``delivered_cost`` bakes in freight/handling/tariff which settle later;
        here we fund only the parts paid to take the position.
        """
        cfg = self.cfg
        fob = float(row["origin_price"]) / float(row["fx"])
        transit_yrs = float(row["transit_days"]) / 360.0
        financing = fob * cfg.financing_rate * transit_yrs
        return (fob + financing) * cfg.cargo_size

    def run(self, arb_df: pd.DataFrame,
            futures: pd.DataFrame) -> dict[str, pd.DataFrame]:
        cfg = self.cfg
        fut = futures.set_index("date")["futures"]
        dates = sorted(arb_df["date"].unique())

        # dict[date] -> sub-df, and a per-date lane lookup for fast MTM
        by_date = {d: g for d, g in arb_df.groupby("date")}
        margin_lookup = {
            d: dict(zip(g["lane"], g["arb_margin"])) for d, g in by_date.items()
        }

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
                if c.arrival_idx > di:
                    still_open.append(c)
                    continue

                sub = by_date[dates[c.arrival_idx]]
                row = sub[sub["lane"] == c.lane]
                arrival_dest_price = (
                    c.delivered_cost if row.empty
                    else float(row["dest_price"].iloc[0])
                )

                # physical leg PnL per tonne
                phys = arrival_dest_price - c.delivered_cost

                # futures hedge leg: we SOLD futures at entry, buy back now.
                hedge = 0.0
                if cfg.hedge_with_futures:
                    f_now = float(fut.iloc[c.arrival_idx])
                    hedge = c.entry_futures - f_now   # short-futures gain

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
                cap_needed = self._upfront_capital(row)
                if cap_needed > free_capital:
                    continue
                # never arrive the same day we book (guards short/zero lanes)
                dt = max(1, int(round(row["transit_days"])))
                arrival_idx = min(di + dt, len(dates) - 1)
                open_cargoes.append(_OpenCargo(
                    lane=lane, entry_date=d, entry_idx=di,
                    arrival_idx=arrival_idx,
                    tonnes=cfg.cargo_size,
                    delivered_cost=float(row["delivered_cost"]),
                    entry_futures=f_today,
                    capital_used=cap_needed,
                ))
                open_count[lane] = open_count.get(lane, 0) + 1
                free_capital -= cap_needed

            # 3) mark equity to market: cash + upfront capital returned at
            #    settlement + unrealised MTM on open cargoes (current lane
            #    margin vs the margin locked at entry, scaled by tonnage).
            locked = 0.0
            unrealised = 0.0
            today_margins = margin_lookup[d]
            for c in open_cargoes:
                locked += c.capital_used
                cur_margin = today_margins.get(c.lane)
                if cur_margin is not None:
                    # entry margin was dest_price(entry) - delivered_cost;
                    # approximate MTM as change in lane margin since entry.
                    entry_margin = margin_lookup[dates[c.entry_idx]].get(
                        c.lane, cur_margin)
                    unrealised += (cur_margin - entry_margin) * c.tonnes
            equity = free_capital + locked + unrealised
            equity_curve.append(dict(
                date=d, equity=equity, realised_pnl=realised_pnl,
                unrealised_pnl=unrealised, open_cargoes=len(open_cargoes),
                free_capital=free_capital,
            ))

        return {
            "trades": pd.DataFrame(trade_log),
            "equity": pd.DataFrame(equity_curve),
        }
