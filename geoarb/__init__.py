"""
geoarb — Geographic Arbitrage for Agricultural Commodities
==========================================================

A research toolkit for studying spatial (geographic) arbitrage in physical
commodity markets, using synthetic data. It includes:

* A spatial market model (origins, destinations, freight network).
* A synthetic data-generating process (DGP) with cointegrated price series,
  stochastic freight rates, FX, and basis dynamics.
* A delivered-cost / arbitrage-window engine.
* A spatial price equilibrium (SPE) optimizer (linear transportation problem).
* A back-testable arbitrage trading simulator with risk metrics.
* Interactive maps (folium) and plots (plotly / matplotlib).

The economics follow the Law of One Price and the spatial-arbitrage condition
(Takayama & Judge spatial equilibrium). See paper/ for the rigorous treatment.

Author: pyoolo
License: MIT
"""

from .geography import Location, Route, build_default_network
from .dgp import SyntheticDGP, DGPConfig
from .arbitrage import (
    delivered_cost,
    arbitrage_signal,
    ArbitrageEngine,
)
from .equilibrium import SpatialEquilibrium
from .simulator import ArbitrageSimulator, SimConfig
from .metrics import (
    performance_summary,
    mean_reversion_check,
    cointegration_check,
    adf_pvalue,
)

__version__ = "0.2.0"

__all__ = [
    "Location",
    "Route",
    "build_default_network",
    "SyntheticDGP",
    "DGPConfig",
    "delivered_cost",
    "arbitrage_signal",
    "ArbitrageEngine",
    "SpatialEquilibrium",
    "ArbitrageSimulator",
    "SimConfig",
    "performance_summary",
    "mean_reversion_check",
    "cointegration_check",
    "adf_pvalue",
    "__version__",
]
