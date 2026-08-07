# Every target below must work with NO GPU and NO network, except `test-integration`
# and `submodule`, which are marked.
#
# Windows users: `make` is usually absent. Use ./tasks.ps1 <target>, which mirrors these.

PY ?= python
PKG := src/rdl

.DEFAULT_GOAL := help

.PHONY: help setup-cpu lint fmt test-unit test-contract test-integration cpu-all \
        submodule repro-dry clean

help:
	@echo "setup-cpu         editable install + cpu extras"
	@echo "lint              ruff + black --check + mypy"
	@echo "fmt               ruff --fix + black (writes)"
	@echo "test-unit         pytest tests/unit        (<10s)"
	@echo "test-contract     pytest tests/contract    (StubLM, <60s)"
	@echo "test-integration  pytest tests/integration (NEEDS NETWORK)"
	@echo "cpu-all           lint + unit + contract   <- the local CPU gate"
	@echo "submodule         git submodule update --init --recursive (NEEDS NETWORK)"
	@echo "repro-dry         print the exact open-unlearning command, run nothing"

setup-cpu:
	$(PY) -m pip install -e ".[cpu,dev]"

lint:
	$(PY) -m ruff check src tests
	$(PY) -m black --check src tests
	$(PY) -m mypy

fmt:
	$(PY) -m ruff check --fix src tests
	$(PY) -m black src tests

test-unit:
	$(PY) -m pytest tests/unit

test-contract:
	$(PY) -m pytest tests/contract

test-integration:
	$(PY) -m pytest tests/integration

# The gate. This must be green on a laptop, offline, before anything touches Colab.
cpu-all: lint test-unit test-contract
	@echo ""
	@echo "CPU GATE PASSED"

submodule:
	git submodule update --init --recursive

repro-dry:
	$(PY) -m rdl.cli run-repro --dry-run --condition configs/conditions/C0.yaml

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
