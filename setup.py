from setuptools import setup, find_packages

with open("README.md", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="spatial-commodity-arbitrage",
    version="0.2.0",
    description="Geographic arbitrage for agricultural commodities "
                "(synthetic-data research toolkit)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(exclude=("tests", "scripts", "notebooks")),
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.24",
        "pandas>=2.2",
        "scipy>=1.10",
        "matplotlib>=3.7",
        "plotly>=5.18",
        "folium>=0.15",
    ],
    extras_require={
        "dev": ["pytest>=7.0", "jupyter>=1.0", "nbconvert>=7.0"],
    },
    license="MIT",
)
