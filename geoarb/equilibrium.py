"""
equilibrium.py
==============

Spatial Price Equilibrium (SPE) — the Takayama-Judge transportation problem.

Given, at a point in time:
  * origin supplies  s_i  (tonnes available),
  * destination demands  d_j  (tonnes required),
  * delivered unit costs  c_{ij}  (origin price + freight + tariff + ... ),
  * destination willingness-to-pay (prices)  p_j,

the competitive spatial equilibrium maximises total surplus, equivalently it
solves the *minimum-cost transportation problem* of moving product from
origins to destinations:

    maximise   sum_{ij} (p_j - c_{ij}) x_{ij}        (trader gross margin)
    subject to sum_j x_{ij} <= s_i        (cannot ship more than supply)
               sum_i x_{ij} <= d_j        (cannot deliver more than demand)
               x_{ij} >= 0

The dual variables on the supply/demand constraints are the equilibrium
*location prices*; in equilibrium, for any lane actually used, the margin
(p_j - c_{ij}) equals the price gap implied by the duals, and for unused lanes
the margin is <= 0. That is exactly the spatial-arbitrage no-arb band.

We solve it with scipy.optimize.linprog (HiGHS). The flows x_{ij} are the
equilibrium trade pattern; comparing them to the *naive* "ship wherever margin
is positive" rule shows why capacity/competition closes most windows.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy.optimize import linprog


@dataclass
class SpatialEquilibrium:
    """Solve the spatial transportation LP for a single time slice."""

    origins: list[str]
    destinations: list[str]

    def solve(self, delivered_cost: np.ndarray, dest_price: np.ndarray,
              supply: np.ndarray, demand: np.ndarray):
        """
        Parameters
        ----------
        delivered_cost : (n_o, n_d) array, c_{ij}
        dest_price     : (n_d,) array, p_j
        supply         : (n_o,) array, s_i
        demand         : (n_d,) array, d_j

        Returns
        -------
        dict with flows (n_o x n_d), total_margin, shadow prices, and a tidy
        flows DataFrame.
        """
        n_o, n_d = delivered_cost.shape
        # margin per lane = p_j - c_{ij}; we MAXIMISE total margin => minimise -margin
        margin = dest_price[None, :] - delivered_cost           # (n_o, n_d)
        c = (-margin).ravel()                                   # linprog minimises

        # constraints: supply (<=) and demand (<=)
        # variables x flattened row-major: index = i*n_d + j
        A_ub = []
        b_ub = []
        # supply rows
        for i in range(n_o):
            row = np.zeros(n_o * n_d)
            row[i * n_d:(i + 1) * n_d] = 1.0
            A_ub.append(row)
            b_ub.append(supply[i])
        # demand cols
        for j in range(n_d):
            row = np.zeros(n_o * n_d)
            row[j::n_d] = 1.0
            A_ub.append(row)
            b_ub.append(demand[j])

        A_ub = np.array(A_ub)
        b_ub = np.array(b_ub)
        bounds = [(0, None)] * (n_o * n_d)

        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
        if not res.success:
            raise RuntimeError(f"LP failed: {res.message}")

        flows = res.x.reshape(n_o, n_d)
        total_margin = float(margin.ravel() @ res.x)

        # dual variables (shadow prices). linprog returns marginals on A_ub.
        duals = res.ineqlin.marginals          # length n_o + n_d, <= 0 convention
        supply_shadow = -duals[:n_o]           # value of one extra tonne supply
        demand_shadow = -duals[n_o:]           # value of one extra tonne demand

        flow_rows = []
        for i, oi in enumerate(self.origins):
            for j, dj in enumerate(self.destinations):
                if flows[i, j] > 1e-6:
                    flow_rows.append(dict(
                        origin=oi, destination=dj,
                        flow=flows[i, j],
                        unit_margin=float(margin[i, j]),
                    ))
        flow_df = pd.DataFrame(flow_rows)

        return dict(
            flows=flows,
            flow_df=flow_df,
            total_margin=total_margin,
            supply_shadow=supply_shadow,
            demand_shadow=demand_shadow,
            margin_matrix=margin,
        )

    # ------------------------------------------------------------------
    def solve_from_arb(self, arb_slice: pd.DataFrame,
                       supply: dict[str, float] | None = None,
                       demand: dict[str, float] | None = None):
        """Convenience wrapper: build matrices from one date's arb rows."""
        o_idx = {o: i for i, o in enumerate(self.origins)}
        d_idx = {d: j for j, d in enumerate(self.destinations)}
        n_o, n_d = len(self.origins), len(self.destinations)

        dc = np.full((n_o, n_d), 1e6)
        pj = np.zeros(n_d)
        for _, row in arb_slice.iterrows():
            if row["origin"] in o_idx and row["destination"] in d_idx:
                i, j = o_idx[row["origin"]], d_idx[row["destination"]]
                dc[i, j] = row["delivered_cost"]
                pj[j] = row["dest_price"]

        s = np.array([(supply or {}).get(o, 100.0) for o in self.origins])
        d = np.array([(demand or {}).get(x, 90.0) for x in self.destinations])
        return self.solve(dc, pj, s, d)
