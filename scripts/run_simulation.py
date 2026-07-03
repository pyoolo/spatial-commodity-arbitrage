#!/usr/bin/env python3
"""
run_simulation.py
=================

End-to-end demo pipeline:

  1. Build the soybean trade network.
  2. Generate synthetic market data (prices, freight, FX, futures).
  3. Compute delivered costs and arbitrage windows for every lane.
  4. Solve the spatial price equilibrium for a snapshot date.
  5. Back-test the arbitrage trading strategy.
  6. Produce interactive maps + a dashboard + a metrics report.

Run:  python scripts/run_simulation.py
Outputs land in ./outputs/
"""

import os
import sys
import json
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from geoarb import (
    build_default_network, SyntheticDGP, DGPConfig,
    ArbitrageEngine, SpatialEquilibrium,
    ArbitrageSimulator, SimConfig, performance_summary,
)
from geoarb.metrics import mean_reversion_check, adf_pvalue
from geoarb import viz

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")
os.makedirs(OUT, exist_ok=True)


def main():
    print("=" * 64)
    print(" GEOGRAPHIC ARBITRAGE — synthetic simulation pipeline")
    print("=" * 64)

    # 1) network
    locations, routes = build_default_network()
    origins = [l.code for l in locations if l.role == "origin"]
    dests = [l.code for l in locations if l.role == "destination"]
    print(f"\n[1] Network: {len(origins)} origins x {len(dests)} destinations "
          f"= {len(routes)} lanes")

    # 2) synthetic data
    dgp = SyntheticDGP(DGPConfig(n_days=750, seed=42), locations, routes)
    data = dgp.generate()
    print(f"[2] Generated {len(data['prices'])} days of synthetic data")

    # cointegration sanity check on two origins
    cc = mean_reversion_check(data["prices"], "BR_SANTOS", "AR_ROSARIO")
    adf = adf_pvalue((data["prices"]["BR_SANTOS"]
                      - data["prices"]["AR_ROSARIO"]).to_numpy())
    print(f"    Mean-reversion {cc['pair']}: AR(1)={cc['ar1_coef']:.3f}, "
          f"half-life={cc['half_life_days']:.1f}d")
    print(f"    ADF stat={adf['adf_stat']:.2f} (5% crit {adf['crit_value_5pct']}); "
          f"reject unit root: {adf['reject_unit_root_5pct']}")

    # 3) arbitrage windows
    engine = ArbitrageEngine(locations, routes, financing_rate=0.06)
    arb = engine.compute(data)
    stats = engine.window_stats(arb)
    print("\n[3] Arbitrage window stats (top lanes):")
    print(stats.head(6).round(2).to_string())

    # 4) spatial equilibrium snapshot
    spe = SpatialEquilibrium(origins, dests)
    snap_date = data["prices"]["date"].iloc[-1]
    snap = arb[arb["date"] == snap_date]
    eq = spe.solve_from_arb(snap)
    print(f"\n[4] Spatial equilibrium @ {pd.Timestamp(snap_date).date()}: "
          f"total margin = {eq['total_margin']:.0f} (per-tonne units)")
    if not eq["flow_df"].empty:
        print("    Equilibrium flows:")
        print(eq["flow_df"].round(2).to_string(index=False))
    # dual variables = equilibrium location prices (the pedagogical payoff)
    print("    Shadow prices (equilibrium location values, USD/tonne):")
    for code, sp in zip(origins, eq["supply_shadow"]):
        print(f"      supply  {code:14s}: {sp:8.2f}")
    for code, dp in zip(dests, eq["demand_shadow"]):
        print(f"      demand  {code:14s}: {dp:8.2f}")

    # 5) trading simulation -- hedged vs unhedged, to expose flat-price risk
    perf_by_mode = {}
    sim_out = None
    for hedged in (True, False):
        sim = ArbitrageSimulator(SimConfig(entry_threshold=4.0,
                                           cargo_size=60_000,
                                           capital=60_000_000,
                                           hedge_with_futures=hedged))
        out = sim.run(arb, data["futures"])
        perf = performance_summary(out["equity"], out["trades"],
                                   starting_capital=60_000_000)
        perf_by_mode["hedged" if hedged else "unhedged"] = perf
        if hedged:
            sim_out = out
    print("\n[5] Strategy performance (hedged vs unhedged):")
    keys = ["total_return_pct", "ann_vol_pct", "sharpe", "max_drawdown_pct",
            "win_rate_pct", "n_trades"]
    print(f"    {'metric':22s} {'hedged':>14s} {'unhedged':>14s}")
    for k in keys:
        h, u = perf_by_mode["hedged"][k], perf_by_mode["unhedged"][k]
        print(f"    {k:22s} {h:14,.2f} {u:14,.2f}")
    perf = perf_by_mode["hedged"]

    # 6) outputs
    print("\n[6] Writing visual + data outputs ...")
    viz.arbitrage_map(arb, date=snap_date, locations=locations, routes=routes,
                      out_path=os.path.join(OUT, "arb_map.html"))
    viz.dashboard(data, arb, sim_out, out_path=os.path.join(OUT, "dashboard.html"))
    viz.margin_heatmap(arb, out_path=os.path.join(OUT, "margin_heatmap.html"))

    arb.to_csv(os.path.join(OUT, "arbitrage_panel.csv"), index=False)
    sim_out["trades"].to_csv(os.path.join(OUT, "trades.csv"), index=False)
    sim_out["equity"].to_csv(os.path.join(OUT, "equity.csv"), index=False)
    stats.to_csv(os.path.join(OUT, "window_stats.csv"))
    with open(os.path.join(OUT, "performance.json"), "w") as f:
        json.dump(perf, f, indent=2, default=str)

    print(f"    Wrote maps + dashboard + data to {OUT}/")
    print("\nDone. Open outputs/dashboard.html and outputs/arb_map.html")


if __name__ == "__main__":
    main()
