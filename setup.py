from setuptools import setup, find_packages

setup(
    name="spatial-commodity-arbitrage",
    version="0.1.0",
    description="Geographic arbitrage for agricultural commodities (synthetic-data research toolkit)",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.24", "pandas>=2.0", "scipy>=1.10",
        "matplotlib>=3.7", "plotly>=5.18", "folium>=0.15",
        "networkx>=3.0", "pulp>=2.7",
    ],
    license="MIT",
)
