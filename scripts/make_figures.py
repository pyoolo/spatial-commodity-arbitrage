#!/usr/bin/env python3
"""
make_figures.py
===============

Generate STATIC PNG charts (matplotlib) for the README and docs.

GitHub's markdown preview does NOT render the interactive folium/plotly HTML,
so we render publication-style PNGs here and embed those in the README. The
interactive HTML versions are still produced by scripts/run_simulation.py.

Outputs land in ./assets/  (committed to the repo so the README shows them).
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from geoarb import (build_default_network, SyntheticDGP, DGPConfig,
                    ArbitrageEngine, SpatialEquilibrium,
                    ArbitrageSimulator, SimConfig, performance_summary)
from geoarb.metrics import mean_reversion_check

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS = os.path.join(ROOT, "assets")
os.makedirs(ASSETS, exist_ok=True)

# ---- house style -----------------------------------------------------------
plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 120,
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.titleweight": "bold",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})
GREEN = "#2e8b57"
RED = "#b3402a"
BLUE = "#2c6fbb"
GREY = "#9aa0a6"


def build():
    locations, routes = build_default_network()
    origins = [l.code for l in locations if l.role == "origin"]
    dests = [l.code for l in locations if l.role == "destination"]
    data = SyntheticDGP(DGPConfig(n_days=750, seed=42), locations, routes).generate()
    arb = ArbitrageEngine(locations, routes, financing_rate=0.06).compute(data)
    return locations, routes, origins, dests, data, arb


# ---------------------------------------------------------------------------
def fig_network_map(locations, routes, arb, data):
    """A clean schematic 'map' of the trade network on a lon/lat canvas,
    lanes coloured by whether the window is open on the last date."""
    by_code = {l.code: l for l in locations}
    snap_date = data["prices"]["date"].iloc[-1]
    snap = arb[arb["date"] == snap_date]
    max_abs = max(snap["arb_margin"].abs().max(), 1.0)

    fig, ax = plt.subplots(figsize=(11, 5.5))
    # faint world latitude/longitude guide
    ax.axhline(0, color=GREY, lw=0.6, alpha=0.5)
    for _, row in snap.iterrows():
        o, d = by_code[row["origin"]], by_code[row["destination"]]
        open_ = row["arb_margin"] > 0
        color = GREEN if open_ else GREY
        lw = 0.8 + 4.0 * min(abs(row["arb_margin"]) / max_abs, 1.0)
        ax.plot([o.lon, d.lon], [o.lat, d.lat], color=color,
                lw=lw, alpha=0.75 if open_ else 0.35, zorder=1,
                solid_capstyle="round")
    for l in locations:
        c = "#1f6f43" if l.role == "origin" else RED
        ax.scatter(l.lon, l.lat, s=130, color=c, zorder=3,
                   edgecolor="white", linewidth=1.4)
        ax.annotate(l.name.split(",")[0], (l.lon, l.lat),
                    textcoords="offset points", xytext=(6, 6),
                    fontsize=8, fontweight="bold")
    ax.set_xlim(-100, 135); ax.set_ylim(-45, 62)
    ax.set_xlabel("longitude"); ax.set_ylabel("latitude")
    ax.set_title(f"Soybean trade network — arbitrage windows on "
                 f"{pd.Timestamp(snap_date).date()}  (synthetic)")
    # legend
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], color=GREEN, lw=3, label="window open (margin > 0)"),
        Line2D([0], [0], color=GREY, lw=3, label="window closed"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#1f6f43",
               markersize=10, label="origin (export)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=RED,
               markersize=10, label="destination (import)"),
    ]
    ax.legend(handles=handles, loc="lower left", fontsize=8, framealpha=0.9)
    fig.tight_layout()
    p = os.path.join(ASSETS, "network_map.png")
    fig.savefig(p, bbox_inches="tight"); plt.close(fig)
    return p


def fig_prices_and_spread(data):
    prices = data["prices"]
    fig = plt.figure(figsize=(11, 6))
    gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1], hspace=0.32)

    ax1 = fig.add_subplot(gs[0])
    cols = [c for c in prices.columns if c != "date"]
    for c in cols:
        ax1.plot(prices["date"], prices[c], lw=1.0, label=c)
    ax1.set_title("Synthetic location cash prices (USD/tonne)")
    ax1.set_ylabel("USD/tonne")
    ax1.legend(ncol=4, fontsize=7, loc="upper left")

    ax2 = fig.add_subplot(gs[1])
    spread = prices["BR_SANTOS"] - prices["AR_ROSARIO"]
    ax2.plot(prices["date"], spread, lw=1.0, color=BLUE)
    ax2.axhline(spread.mean(), color=RED, ls="--", lw=1.0, label="mean")
    cc = mean_reversion_check(prices, "BR_SANTOS", "AR_ROSARIO")
    ax2.set_title(f"Cointegrated spread Santos−Rosario  "
                  f"(AR(1)={cc['ar1_coef']:.2f}, half-life={cc['half_life_days']:.1f}d)")
    ax2.set_ylabel("USD/tonne"); ax2.legend(fontsize=8, loc="upper right")

    p = os.path.join(ASSETS, "prices_and_spread.png")
    fig.savefig(p, bbox_inches="tight"); plt.close(fig)
    return p


def fig_margins(arb):
    fig, ax = plt.subplots(figsize=(11, 4.5))
    stats = ArbitrageEngine.window_stats(arb)
    top_lanes = stats.head(4).index.tolist()
    for lane in top_lanes:
        g = arb[arb["lane"] == lane]
        ax.plot(g["date"], g["arb_margin"], lw=1.1, label=lane)
    ax.axhline(0, color="black", ls="--", lw=1.0)
    ax.set_title("Arbitrage margin by lane (USD/tonne) — above 0 = window open")
    ax.set_ylabel("USD/tonne")
    ax.legend(fontsize=8, ncol=2, loc="upper left")
    p = os.path.join(ASSETS, "arbitrage_margins.png")
    fig.savefig(p, bbox_inches="tight"); plt.close(fig)
    return p


def fig_heatmap(arb):
    stats = (arb.groupby(["origin", "destination"])["window_open"]
             .mean().mul(100).reset_index(name="pct_open"))
    pivot = stats.pivot(index="origin", columns="destination", values="pct_open")
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    im = ax.imshow(pivot.values, cmap="Greens", aspect="auto", vmin=0, vmax=100)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=8)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.values[i, j]
            ax.text(j, i, f"{v:.0f}", ha="center", va="center",
                    color="white" if v > 55 else "black", fontsize=9,
                    fontweight="bold")
    ax.set_title("Share of days the arbitrage window is open (%)")
    fig.colorbar(im, ax=ax, label="% days open")
    p = os.path.join(ASSETS, "window_heatmap.png")
    fig.savefig(p, bbox_inches="tight"); plt.close(fig)
    return p


def fig_equity(arb, data):
    fig, ax = plt.subplots(figsize=(11, 4.5))
    summary = {}
    for hedge, color in [(False, RED), (True, GREEN)]:
        sim = ArbitrageSimulator(SimConfig(capital=60e6, hedge_with_futures=hedge))
        out = sim.run(arb, data["futures"])
        perf = performance_summary(out["equity"], out["trades"], 60e6)
        summary[hedge] = perf
        lbl = ("hedged" if hedge else "unhedged")
        ax.plot(out["equity"]["date"], out["equity"]["equity"] / 1e6,
                lw=1.8, color=color,
                label=f"{lbl}  (Sharpe {perf['sharpe']:.1f}, "
                      f"maxDD {perf['max_drawdown_pct']:.1f}%)")
    ax.set_title("Strategy equity curve — hedged vs unhedged (USD millions, synthetic)")
    ax.set_ylabel("equity (USD mm)")
    ax.legend(fontsize=9, loc="upper left")
    p = os.path.join(ASSETS, "equity_curve.png")
    fig.savefig(p, bbox_inches="tight"); plt.close(fig)
    return p, summary


def main():
    locations, routes, origins, dests, data, arb = build()
    print("Generating figures ...")
    print(" ", fig_network_map(locations, routes, arb, data))
    print(" ", fig_prices_and_spread(data))
    print(" ", fig_margins(arb))
    print(" ", fig_heatmap(arb))
    p, summary = fig_equity(arb, data)
    print(" ", p)
    print("\nDone. PNGs written to assets/")


if __name__ == "__main__":
    main()
