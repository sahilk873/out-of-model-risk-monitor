.PHONY: install test lint clean run

install:
	pip install -e ".[dev]"

test:
	python -m pytest tests/ -v --tb=short

coverage:
	python -m pytest tests/ --cov=src/risk_monitor -v --tb=short

lint:
	ruff check src/ tests/

typecheck:
	mypy src/ --ignore-missing-imports

clean:
	rm -rf build/ dist/ *.egg-info/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name '*.pyc' -delete

run-pipeline:
	python scripts/run_pipeline.py

run-all:
	bash scripts/run_all.sh

data-dir:
	mkdir -p data reports/tables reports/figures
