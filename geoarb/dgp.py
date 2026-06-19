"""
dgp.py
======

Synthetic Data-Generating Process (DGP) for a spatial commodity market.

The design goal is *economic realism with full mathematical control*: every
series is generated from an explicit stochastic model so the paper can derive
the arbitrage statistics in closed form and the notebooks can verify them.

Components
----------
1. A global "world" price factor  F_t  following a random walk with drift
   (the common stochastic trend that ties all locations together — this is
   what makes regional prices *cointegrated*).

2. Location cash prices

       P^i_t = F_t + a_i + u^i_t ,            u^i_t = rho_i u^i_{t-1} + eps^i_t

   i.e. a common trend  F_t  plus a location intercept  a_i  plus a
   stationary (mean-reverting, AR(1)) idiosyncratic spread  u^i_t.
   Because every P^i_t shares the SAME F_t, any pair (P^i, P^j) is
   cointegrated with cointegrating vector (1, -1); the spread P^i - P^j is
   stationary. This is the formal version of the Law of One Price.

3. Freight rates per lane  f^{ij}_t  follow a (log) Ornstein-Uhlenbeck /
   AR(1) mean-reverting process around the lane baseline — freight is
   volatile but mean-reverting (it does not trend to infinity).

4. FX (USD per local currency at each origin) follows a random walk in logs;
   it shifts the USD-denominated cost of origin product.

5. A futures price  H_t  on the benchmark exchange, modelled as the world
   factor plus a small carry term; local *basis* is then  b^i_t = P^i_t - H_t.

All randomness goes through a single numpy Generator so runs are reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
import pandas as pd

from .geography import Location, Route, build_default_network


@dataclass
class DGPConfig:
    """Parameters of the synthetic data-generating process."""

    n_days: int = 750                 # ~3 trading years
    seed: int = 42

    # world common trend F_t : random walk with drift
    world_drift: float = 0.02         # USD/tonne/day expected drift
    world_vol: float = 4.5            # USD/tonne daily shock sd

    # idiosyncratic stationary spread u^i_t : AR(1)
    spread_ar: float = 0.92           # persistence (|rho|<1 => stationary)
    spread_vol: float = 3.0           # innovation sd (USD/tonne)

    # freight (log-OU / AR1 around baseline)
    freight_ar: float = 0.95
    freight_vol: float = 0.05         # in log space
    freight_seasonal_amp: float = 0.08  # seasonal amplitude (log)

    # FX log random walk (origin currency strength vs USD)
    fx_vol: float = 0.006

    # futures carry
    carry_per_day: float = 0.015      # USD/tonne/day contango-ish

    # demand/supply shock occasionally widening destination premia
    shock_prob: float = 0.01          # prob/day of a regional demand shock
    shock_size: float = 25.0          # USD/tonne added to a random dest spread


class SyntheticDGP:
    """Generate synthetic, internally-consistent market data on the network."""

    def __init__(self, config: DGPConfig | None = None,
                 locations: list[Location] | None = None,
                 routes: list[Route] | None = None):
        self.cfg = config or DGPConfig()
        if locations is None or routes is None:
            locations, routes = build_default_network()
        self.locations = locations
        self.routes = routes
        self.rng = np.random.default_rng(self.cfg.seed)
        self._by_code = {l.code: l for l in locations}

    # ------------------------------------------------------------------
    def _ar1(self, n: int, rho: float, vol: float, x0: float = 0.0) -> np.ndarray:
        """Simulate a stationary AR(1): x_t = rho x_{t-1} + eps_t."""
        eps = self.rng.normal(0.0, vol, size=n)
        x = np.empty(n)
        x[0] = x0 + eps[0]
        for t in range(1, n):
            x[t] = rho * x[t - 1] + eps[t]
        return x

    def _random_walk(self, n: int, drift: float, vol: float, x0: float = 0.0):
        steps = self.rng.normal(drift, vol, size=n)
        return x0 + np.cumsum(steps)

    # ------------------------------------------------------------------
    def generate(self) -> dict[str, pd.DataFrame]:
        """Return a dict of tidy DataFrames: prices, freight, fx, futures, basis."""
        cfg = self.cfg
        n = cfg.n_days
        dates = pd.bdate_range("2022-01-03", periods=n)
        t = np.arange(n)

        # 1) world common trend (random walk w/ drift), starts near 0 offset
        F = self._random_walk(n, cfg.world_drift, cfg.world_vol, x0=0.0)

        # 5) futures = world factor anchored to a benchmark level + carry
        bench_anchor = 430.0
        H = bench_anchor + F + cfg.carry_per_day * t
        futures = pd.DataFrame({"date": dates, "futures": H})

        # 2) location cash prices
        price_cols = {}
        basis_cols = {}
        dest_codes = [l.code for l in self.locations if l.role == "destination"]
        for loc in self.locations:
            a_i = loc.base_price - bench_anchor      # location intercept
            u = self._ar1(n, cfg.spread_ar, cfg.spread_vol)
            P = bench_anchor + F + a_i + u
            price_cols[loc.code] = P
            basis_cols[loc.code] = P - H             # local basis vs futures

        # occasional regional demand shocks on destinations (adds persistence)
        for code in dest_codes:
            shock = np.zeros(n)
            active = 0.0
            for ti in range(n):
                if self.rng.random() < cfg.shock_prob:
                    active += cfg.shock_size
                active *= 0.90  # decay
                shock[ti] = active
            price_cols[code] = price_cols[code] + shock
            basis_cols[code] = basis_cols[code] + shock

        prices = pd.DataFrame({"date": dates, **price_cols})
        basis = pd.DataFrame({"date": dates, **basis_cols})

        # 3) freight per lane: baseline * exp(seasonal + OU)
        freight_data = {"date": dates}
        season = cfg.freight_seasonal_amp * np.sin(2 * np.pi * t / 252.0)
        for r in self.routes:
            ou = self._ar1(n, cfg.freight_ar, cfg.freight_vol)
            mult = np.exp(season + ou)
            freight_data[f"{r.origin}->{r.destination}"] = r.base_freight * mult
        freight = pd.DataFrame(freight_data)

        # 4) FX per origin country (log random walk). USD per unit local ccy,
        #    normalised to 1.0 at t0. A *weaker* origin currency lowers the
        #    USD price of origin product (export competitiveness).
        origin_countries = sorted({l.country for l in self.locations
                                   if l.role == "origin"})
        fx_data = {"date": dates}
        for c in origin_countries:
            logfx = self._random_walk(n, 0.0, cfg.fx_vol, x0=0.0)
            fx_data[c] = np.exp(logfx)
        fx = pd.DataFrame(fx_data)

        return {
            "prices": prices,
            "basis": basis,
            "freight": freight,
            "fx": fx,
            "futures": futures,
        }

    # ------------------------------------------------------------------
    def to_long(self, data: dict[str, pd.DataFrame] | None = None) -> pd.DataFrame:
        """Melt prices into long/tidy form for plotting libraries."""
        data = data or self.generate()
        long = data["prices"].melt(id_vars="date", var_name="location",
                                   value_name="price")
        roles = {l.code: l.role for l in self.locations}
        long["role"] = long["location"].map(roles)
        return long
