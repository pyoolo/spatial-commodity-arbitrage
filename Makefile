.PHONY: install dev test sim figures clean clean-nb

install:          ## install runtime deps + package
	pip install -e .

dev:              ## install with dev extras (pytest, jupyter)
	pip install -e ".[dev]"

test:             ## run the test suite
	pytest -q

sim:              ## run the end-to-end pipeline -> outputs/
	python scripts/run_simulation.py

figures:          ## regenerate static PNGs -> assets/
	python scripts/make_figures.py

clean:            ## remove generated outputs and caches
	rm -rf outputs/*.html outputs/*.csv outputs/*.json
	find . -type d -name __pycache__ -exec rm -rf {} +
	rm -rf .pytest_cache *.egg-info

clean-nb:         ## strip notebook outputs (keeps notebooks diff-friendly)
	jupyter nbconvert --clear-output --inplace notebooks/*.ipynb
