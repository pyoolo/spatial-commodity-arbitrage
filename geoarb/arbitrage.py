"""
arbitrage.py
============

The delivered-cost / arbitrage-window engine.

Core economic identity (per tonne, USD):

    DeliveredCost^{i->j}_t = P^i_t / e^i_t      (origin cash, FX-adjusted)
                           + Handling^i + Handling^j
                           + Freight^{ij}_t
                           + Tariff^{ij}
                           + FinancingCost^{ij}_t

    ArbMargin^{i->j}_t = P^j_t - DeliveredCost^{i->j}_t

The *spatial arbitrage condition* (no-arbitrage / Law of One Price band) says
that in equilibrium  ArbMargin <= 0  for every lane: you cannot make a
risk-free profit by shipping. A *positive* margin is an open "arb window" — a
trade signal — that physical traders act on, and whose execution pushes prices
back toward the band (P^j down, P^i up).

Financing cost models the cost of capital tied up while cargo is in transit:

    FinancingCost = (P^i/e^i + Handling^i) * r * (transit_days/360)

with transit_days = distance_nm / (speed_kn * 24).
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd

from .geography import Location, Route, build_default_network


def transit_days(distance_nm: float, speed_kn: float = 13.0) -> float:
    """Voyage time in days for a bulk carrier at `speed_kn` knots."""
    return distance_nm / (speed_kn * 24.0)


def delivered_cost(origin_price, handling_o, handling_d, freight, tariff,
                   fx=1.0, financing_rate=0.06, dist_nm=10000.0,
                   speed_kn=13.0):
    """Per-tonne delivered cost (USD). Scalars or numpy arrays both work."""
    fob = origin_price / fx + handling_o
    fin = fob * financing_rate * (transit_days(dist_nm, speed_kn) / 360.0)
    return fob + freight + tariff + handling_d + fin


def arbitrage_signal(dest_price, delivered):
    """Per-tonne arbitrage margin = destination price - delivered cost."""
    return dest_price - delivered


@dataclass
class ArbitrageEngine:
    """Compute delivered-cost and arb-margin time series for every lane."""

    locations: list[Location] = None
    routes: list[Route] = None
    financing_rate: float = 0.06
    speed_kn: float = 13.0

    def __post_init__(self):
        if self.locations is None or self.routes is None:
            self.locations, self.routes = build_default_network()
        self._by_code = {l.code: l for l in self.locations}
        self._country = {l.code: l.country for l in self.locations}

    # ------------------------------------------------------------------
    def compute(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Return a long DataFrame with one row per (date, lane)."""
        prices = data["prices"].set_index("date")
        freight = data["freight"].set_index("date")
        fx = data["fx"].set_index("date")
        dates = prices.index

        rows = []
        for r in self.routes:
            o = self._by_code[r.origin]
            d = self._by_code[r.destination]
            lane = f"{r.origin}->{r.destination}"
            p_o = prices[r.origin].to_numpy()
            p_d = prices[r.destination].to_numpy()
            frt = freight[lane].to_numpy()
            e_o = fx[o.country].to_numpy() if o.country in fx.columns else np.ones(len(dates))

            dc = delivered_cost(
                origin_price=p_o, handling_o=o.handling_cost,
                handling_d=d.handling_cost, freight=frt, tariff=r.tariff,
                fx=e_o, financing_rate=self.financing_rate,
                dist_nm=r.distance_nm, speed_kn=self.speed_kn,
            )
            margin = arbitrage_signal(p_d, dc)

            lane_df = pd.DataFrame({
                "date": dates,
                "lane": lane,
                "origin": r.origin,
                "destination": r.destination,
                "origin_price": p_o,
                "dest_price": p_d,
                "freight": frt,
                "fx": e_o,
                "delivered_cost": dc,
                "arb_margin": margin,
                "distance_nm": r.distance_nm,
                "transit_days": transit_days(r.distance_nm, self.speed_kn),
            })
            rows.append(lane_df)

        out = pd.concat(rows, ignore_index=True)
        out["window_open"] = out["arb_margin"] > 0
        return out

    # ------------------------------------------------------------------
    @staticmethod
    def window_stats(arb_df: pd.DataFrame) -> pd.DataFrame:
        """Summarise, per lane, how often the window is open and how rich it is."""
        g = arb_df.groupby("lane")
        stats = pd.DataFrame({
            "pct_open": g["window_open"].mean() * 100.0,
            "mean_margin": g["arb_margin"].mean(),
            "max_margin": g["arb_margin"].max(),
            "mean_margin_when_open": g.apply(
                lambda x: x.loc[x["window_open"], "arb_margin"].mean(),
                include_groups=False),
            "mean_freight": g["freight"].mean(),
            "distance_nm": g["distance_nm"].first(),
        })
        return stats.sort_values("pct_open", ascending=False)
