"""
geography.py
============

Spatial structure of the market: a set of geographic *Locations* (export
origins and import destinations) connected by maritime *Routes*.

All coordinates are real port locations so the maps look correct, but every
*price*, *freight rate* and *flow* produced elsewhere in this package is
SYNTHETIC. Nothing here is investment advice or real market data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal
import math

Role = Literal["origin", "destination"]


@dataclass(frozen=True)
class Location:
    """A node in the spatial market network (an export or import port)."""

    code: str          # short id, e.g. "BR_SANTOS"
    name: str          # human readable, e.g. "Santos, Brazil"
    country: str
    role: Role
    lat: float
    lon: float
    # baseline local cash price level (USD/tonne) used to anchor the DGP
    base_price: float = 400.0
    # local handling / elevation cost at the port (USD/tonne)
    handling_cost: float = 12.0

    def __repr__(self) -> str:
        return f"Location({self.code}, {self.role})"


@dataclass(frozen=True)
class Route:
    """A directed maritime route from an origin to a destination."""

    origin: str            # Location.code
    destination: str       # Location.code
    distance_nm: float     # great-circle nautical miles
    # baseline freight in USD/tonne for this lane (anchors freight DGP)
    base_freight: float = 35.0
    # per-tonne import tariff / duty applied at destination
    tariff: float = 0.0

    @property
    def key(self) -> tuple[str, str]:
        return (self.origin, self.destination)


def haversine_nm(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in nautical miles between two lat/lon points."""
    r_km = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    km = 2 * r_km * math.asin(math.sqrt(a))
    return km / 1.852  # km -> nautical miles


# ---------------------------------------------------------------------------
# Default network: a stylised global soybean trade map.
# Origins: South America + US Gulf.  Destinations: China + EU.
# ---------------------------------------------------------------------------

_DEFAULT_ORIGINS = [
    Location("BR_SANTOS", "Santos, Brazil", "Brazil", "origin",
             -23.96, -46.30, base_price=395.0, handling_cost=14.0),
    Location("BR_PARANAGUA", "Paranagua, Brazil", "Brazil", "origin",
             -25.52, -48.51, base_price=398.0, handling_cost=13.0),
    Location("AR_ROSARIO", "Rosario (Up-River), Argentina", "Argentina", "origin",
             -32.95, -60.65, base_price=388.0, handling_cost=11.0),
    Location("US_NOLA", "New Orleans (US Gulf), USA", "USA", "origin",
             29.95, -90.07, base_price=410.0, handling_cost=10.0),
]

_DEFAULT_DESTINATIONS = [
    Location("CN_QINGDAO", "Qingdao, China", "China", "destination",
             36.07, 120.38, base_price=448.0, handling_cost=9.0),
    Location("CN_NANTONG", "Nantong, China", "China", "destination",
             32.01, 120.86, base_price=450.0, handling_cost=9.0),
    Location("NL_ROTTERDAM", "Rotterdam, Netherlands", "EU", "destination",
             51.95, 4.14, base_price=440.0, handling_cost=8.0),
    Location("ES_BARCELONA", "Barcelona, Spain", "EU", "destination",
             41.35, 2.16, base_price=442.0, handling_cost=8.5),
]

# tariff assumptions (synthetic, USD/tonne) keyed by (origin country, dest country)
_TARIFF_TABLE = {
    ("USA", "China"): 25.0,     # stylised trade-war style duty
    ("Brazil", "China"): 0.0,
    ("Argentina", "China"): 0.0,
    ("Brazil", "EU"): 0.0,
    ("Argentina", "EU"): 0.0,
    ("USA", "EU"): 0.0,
}


def build_default_network() -> tuple[list[Location], list[Route]]:
    """Return (locations, routes) for the default global soybean network.

    Freight baselines scale roughly with distance (a stylised USD/tonne per
    1000 nm rate plus a fixed port-call component).
    """
    locations = _DEFAULT_ORIGINS + _DEFAULT_DESTINATIONS
    by_code = {loc.code: loc for loc in locations}

    routes: list[Route] = []
    rate_per_1000nm = 8.5   # USD/tonne per 1000 nm (synthetic Panamax-ish)
    fixed_component = 12.0  # USD/tonne fixed (port calls, canal, bunkers base)

    for o in _DEFAULT_ORIGINS:
        for d in _DEFAULT_DESTINATIONS:
            dist = haversine_nm(o.lat, o.lon, d.lat, d.lon)
            base_freight = fixed_component + rate_per_1000nm * dist / 1000.0
            tariff = _TARIFF_TABLE.get((o.country, d.country), 0.0)
            routes.append(
                Route(
                    origin=o.code,
                    destination=d.code,
                    distance_nm=round(dist, 1),
                    base_freight=round(base_freight, 2),
                    tariff=tariff,
                )
            )
    return locations, routes


def network_as_frames():
    """Convenience: return (locations_df, routes_df) as pandas DataFrames."""
    import pandas as pd

    locs, routes = build_default_network()
    loc_df = pd.DataFrame(
        [
            dict(code=l.code, name=l.name, country=l.country, role=l.role,
                 lat=l.lat, lon=l.lon, base_price=l.base_price,
                 handling_cost=l.handling_cost)
            for l in locs
        ]
    )
    route_df = pd.DataFrame(
        [
            dict(origin=r.origin, destination=r.destination,
                 distance_nm=r.distance_nm, base_freight=r.base_freight,
                 tariff=r.tariff)
            for r in routes
        ]
    )
    return loc_df, route_df
