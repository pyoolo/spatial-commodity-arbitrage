"""
Tests for the geoarb package. Run with:  pytest -q
"""
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from geoarb import (build_default_network, SyntheticDGP, DGPConfig,
                    ArbitrageEngine, SpatialEquilibrium, ArbitrageSimulator,
                    SimConfig, performance_summary)
from geoarb.geography import haversine_nm
from geoarb.arbitrage import delivered_cost, transit_days
from geoarb.metrics import cointegration_check


def test_network_shape():
    locs, routes = build_default_network()
    n_o = sum(1 for l in locs if l.role == "origin")
    n_d = sum(1 for l in locs if l.role == "destination")
    assert len(routes) == n_o * n_d
    assert all(r.distance_nm > 0 for r in routes)


def test_haversine_known_distance():
    # NYC -> London ~ 3000 nm; allow wide tolerance
    d = haversine_nm(40.71, -74.01, 51.51, -0.13)
    assert 2900 < d < 3300


def test_dgp_reproducible():
    cfg = DGPConfig(n_days=200, seed=7)
    a = SyntheticDGP(cfg).generate()["prices"]
    b = SyntheticDGP(cfg).generate()["prices"]
    pd.testing.assert_frame_equal(a, b)


def test_dgp_cointegration_stationary():
    data = SyntheticDGP(DGPConfig(n_days=750, seed=1)).generate()
    cc = cointegration_check(data["prices"], "BR_SANTOS", "AR_ROSARIO")
    # spread should be mean-reverting (AR1 coef below 1)
    assert cc["ar1_coef"] < 1.0
    assert cc["stationary_like"]


def test_delivered_cost_monotone_in_freight():
    base = delivered_cost(400, 12, 9, 40, 0, fx=1.0, dist_nm=5000)
    higher = delivered_cost(400, 12, 9, 60, 0, fx=1.0, dist_nm=5000)
    assert higher > base


def test_arbitrage_engine_columns():
    data = SyntheticDGP(DGPConfig(n_days=120, seed=3)).generate()
    arb = ArbitrageEngine().compute(data)
    for col in ["arb_margin", "delivered_cost", "window_open", "lane"]:
        assert col in arb.columns
    assert arb["window_open"].dtype == bool


def test_equilibrium_respects_supply_demand():
    origins = ["O1", "O2"]
    dests = ["D1", "D2"]
    spe = SpatialEquilibrium(origins, dests)
    dc = np.array([[10.0, 12.0], [11.0, 9.0]])
    pj = np.array([20.0, 20.0])
    supply = np.array([100.0, 100.0])
    demand = np.array([80.0, 80.0])
    res = spe.solve(dc, pj, supply, demand)
    flows = res["flows"]
    assert (flows.sum(axis=1) <= supply + 1e-6).all()   # supply respected
    assert (flows.sum(axis=0) <= demand + 1e-6).all()   # demand respected
    assert res["total_margin"] > 0


def test_simulator_runs_and_metrics():
    data = SyntheticDGP(DGPConfig(n_days=300, seed=11)).generate()
    arb = ArbitrageEngine().compute(data)
    sim = ArbitrageSimulator(SimConfig(capital=5e7))
    out = sim.run(arb, data["futures"])
    perf = performance_summary(out["equity"], out["trades"],
                               starting_capital=5e7)
    assert "sharpe" in perf
    assert out["equity"].shape[0] == len(data["prices"])


def test_transit_days_positive():
    assert transit_days(5000, 13) > 0
