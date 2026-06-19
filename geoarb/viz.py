"""
viz.py
======

Visualisation helpers: interactive flow maps (folium) and time-series /
dashboard charts (plotly). All outputs are standalone HTML files.
"""

from __future__ import annotations
import numpy as np
import pandas as pd

from .geography import build_default_network


# ---------------------------------------------------------------------------
# Interactive arbitrage flow map (folium)
# ---------------------------------------------------------------------------
def arbitrage_map(arb_df: pd.DataFrame, date=None, locations=None, routes=None,
                  out_path: str = "outputs/arb_map.html"):
    """Render a world map: origin/destination markers + lanes coloured by
    whether the arbitrage window is open (green) or closed (grey), width
    scaled by the size of the margin.
    """
    import folium

    if locations is None or routes is None:
        locations, routes = build_default_network()
    by_code = {l.code: l for l in locations}

    if date is None:
        date = arb_df["date"].max()
    snap = arb_df[arb_df["date"] == date]

    m = folium.Map(location=[20, 0], zoom_start=2, tiles="cartodbpositron")

    title_html = (
        f'<div style="position: fixed; top: 10px; left: 50px; z-index: 9999;'
        f'background: white; padding: 8px 14px; border-radius: 8px;'
        f'box-shadow: 0 1px 4px rgba(0,0,0,.3); font-family: sans-serif;">'
        f'<b>Geographic Arbitrage — soybean network</b><br>'
        f'<span style="font-size:12px;color:#555">Snapshot: {pd.Timestamp(date).date()} '
        f'&nbsp;|&nbsp; <span style="color:green">●</span> window open '
        f'&nbsp; <span style="color:#999">●</span> closed &nbsp;(synthetic data)</span>'
        f'</div>'
    )
    m.get_root().html.add_child(folium.Element(title_html))

    # markers
    for l in locations:
        color = "#1f6f43" if l.role == "origin" else "#b3402a"
        icon = "ship" if l.role == "origin" else "industry"
        folium.CircleMarker(
            location=[l.lat, l.lon], radius=7, color=color, fill=True,
            fill_opacity=0.9,
            tooltip=f"{l.name} ({l.role})",
        ).add_to(m)

    # lanes
    max_abs = max(snap["arb_margin"].abs().max(), 1.0)
    for _, row in snap.iterrows():
        o, d = by_code[row["origin"]], by_code[row["destination"]]
        open_ = row["arb_margin"] > 0
        color = "#2e8b57" if open_ else "#b0b0b0"
        weight = 1.5 + 5.0 * min(abs(row["arb_margin"]) / max_abs, 1.0)
        popup = (f"{row['origin']} → {row['destination']}<br>"
                 f"margin: {row['arb_margin']:.1f} USD/t<br>"
                 f"delivered: {row['delivered_cost']:.1f} | "
                 f"dest price: {row['dest_price']:.1f}<br>"
                 f"freight: {row['freight']:.1f} | "
                 f"{row['transit_days']:.0f} days at sea")
        folium.PolyLine(
            locations=[[o.lat, o.lon], [d.lat, d.lon]],
            color=color, weight=weight, opacity=0.75 if open_ else 0.4,
            popup=folium.Popup(popup, max_width=260),
        ).add_to(m)

    m.save(out_path)
    return out_path


# ---------------------------------------------------------------------------
# Plotly dashboard (prices, margins, equity)
# ---------------------------------------------------------------------------
def dashboard(data, arb_df, sim_out=None, out_path="outputs/dashboard.html"):
    """A multi-panel interactive plotly dashboard saved to HTML."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    prices = data["prices"]
    locs = {c for c in prices.columns if c != "date"}

    n_rows = 3 if sim_out is not None else 2
    titles = ["Location cash prices (synthetic, USD/t)",
              "Arbitrage margin by lane (USD/t) — >0 = window open"]
    if sim_out is not None:
        titles.append("Strategy equity curve (USD)")

    fig = make_subplots(rows=n_rows, cols=1, shared_xaxes=False,
                        subplot_titles=titles, vertical_spacing=0.09)

    # panel 1: prices
    for c in sorted(locs):
        fig.add_trace(go.Scatter(x=prices["date"], y=prices[c], name=c,
                                 mode="lines", line=dict(width=1.2)),
                      row=1, col=1)

    # panel 2: arb margins per lane
    for lane, g in arb_df.groupby("lane"):
        fig.add_trace(go.Scatter(x=g["date"], y=g["arb_margin"], name=lane,
                                 mode="lines", line=dict(width=1),
                                 showlegend=False),
                      row=2, col=1)
    fig.add_hline(y=0, line_dash="dash", line_color="black", row=2, col=1)

    # panel 3: equity
    if sim_out is not None:
        eq = sim_out["equity"]
        fig.add_trace(go.Scatter(x=eq["date"], y=eq["equity"],
                                 name="equity", mode="lines",
                                 line=dict(width=2, color="#1f6f43")),
                      row=3, col=1)

    fig.update_layout(height=320 * n_rows, template="plotly_white",
                      title="Geographic Arbitrage — Synthetic Simulation Dashboard",
                      legend=dict(orientation="h", y=1.08, font=dict(size=9)))
    fig.write_html(out_path, include_plotlyjs="cdn")
    return out_path


def margin_heatmap(arb_df, out_path="outputs/margin_heatmap.html"):
    """Heatmap of % of days each lane's window is open."""
    import plotly.express as px

    stats = (arb_df.groupby(["origin", "destination"])["window_open"]
             .mean().mul(100).reset_index(name="pct_open"))
    pivot = stats.pivot(index="origin", columns="destination",
                        values="pct_open")
    fig = px.imshow(pivot, text_auto=".0f", color_continuous_scale="Greens",
                    aspect="auto",
                    labels=dict(color="% days open"),
                    title="Share of days the arbitrage window is open (synthetic)")
    fig.update_layout(template="plotly_white", height=450)
    fig.write_html(out_path, include_plotlyjs="cdn")
    return out_path
