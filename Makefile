.PHONY: install install-dev test coverage lint typecheck clean run run-pipeline run-all data-dir pre-commit ci

VERSION := 0.3.0

install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

test:
	python -m pytest tests/ -v --tb=short

test-quick:
	python -m pytest tests/ -v --tb=short -m "not slow"

coverage:
	python -m pytest tests/ --cov=src/risk_monitor --cov-report=term-missing -v --tb=short

lint:
	ruff check src/ tests/

lint-fix:
	ruff check src/ tests/ --fix

typecheck:
	mypy src/ --ignore-missing-imports

clean:
	rm -rf build/ dist/ *.egg-info/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name '*.pyc' -delete
	rm -rf .mypy_cache/ .ruff_cache/ .pytest_cache/
	rm -f data/cache/*.pkl data/cache/*.json

pre-commit:
	pre-commit run --all-files

ci: lint typecheck test

run-pipeline:
	python scripts/run_pipeline.py

run-pipeline-verbose:
	python scripts/run_pipeline.py --verbose

run-all:
	bash scripts/run_all.sh

data-dir:
	mkdir -p data data/cache reports/tables reports/figures

version:
	@echo $(VERSION)

notebook:
	cd research && jupyter notebook OOM_risk_research.ipynb
