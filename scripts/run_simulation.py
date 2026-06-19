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
from geoarb.metrics import cointegration_check
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
    cc = cointegration_check(data["prices"], "BR_SANTOS", "AR_ROSARIO")
    print(f"    Cointegration check {cc['pair']}: AR(1)={cc['ar1_coef']:.3f}, "
          f"half-life={cc['half_life_days']:.1f}d, "
          f"stationary={cc['stationary_like']}")

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

    # 5) trading simulation
    sim = ArbitrageSimulator(SimConfig(entry_threshold=4.0,
                                       cargo_size=60_000,
                                       capital=60_000_000,
                                       hedge_with_futures=True))
    sim_out = sim.run(arb, data["futures"])
    perf = performance_summary(sim_out["equity"], sim_out["trades"],
                               starting_capital=60_000_000)
    print("\n[5] Strategy performance (hedged):")
    for k, v in perf.items():
        print(f"    {k:24s}: {v:,.2f}" if isinstance(v, float) else f"    {k:24s}: {v}")

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
