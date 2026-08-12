# Every target below must work with NO GPU and NO network, except `test-integration`
# and `submodule`, which are marked.
#
# Windows users: `make` is usually absent. Use ./tasks.ps1 <target>, which mirrors these.

PY ?= python
PKG := src/rdl

.DEFAULT_GOAL := help

.PHONY: help setup-cpu lint fmt test-unit test-contract test-graph test-integration \
        cpu-all graph-smoke submodule repro-dry clean

help:
	@echo "setup-cpu         editable install + cpu extras"
	@echo "lint              ruff + black --check + mypy"
	@echo "fmt               ruff --fix + black (writes)"
	@echo "test-unit         pytest tests/unit        (<10s)"
	@echo "test-contract     pytest tests/contract    (StubLM, <60s)"
	@echo "test-graph        pytest graph/defenses/graph_memory/runtime (graph-unlearning-v1)"
	@echo "test-integration  pytest tests/integration (NEEDS NETWORK)"
	@echo "cpu-all           lint + unit + contract + graph  <- the local CPU gate"
	@echo "graph-smoke       full graph pipeline on the stub backend (diagnostic only)"
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

# graph-unlearning-v1: the graph, defence, memory and runtime layers.
test-graph:
	$(PY) -m pytest tests/graph tests/defenses tests/graph_memory tests/runtime

test-integration:
	$(PY) -m pytest tests/integration

# The gate. This must be green on a laptop, offline, before anything touches a GPU.
cpu-all: lint test-unit test-contract test-graph
	@echo ""
	@echo "CPU GATE PASSED"

# End-to-end graph run on the stub backend. No network, no GPU, nothing reportable.
GRAPH_SMOKE_OUT ?= /tmp/rdl-graph-smoke
GRAPH_SMOKE_ARGS := --launch configs/graph/launch.yaml \
	--fixture tests/fixtures/tofu_forget10_sample.json \
	--cohort data/cohorts/graph_unlearning_v1/cpu_stub.json --limit 4

graph-smoke:
	rm -rf $(GRAPH_SMOKE_OUT)
	$(PY) -m rdl.cli graph-plan $(GRAPH_SMOKE_ARGS)
	$(PY) -m rdl.cli graph-run $(GRAPH_SMOKE_ARGS) --challenges direct --output $(GRAPH_SMOKE_OUT)
	$(PY) -m rdl.cli graph-score --run $(GRAPH_SMOKE_OUT)
	$(PY) -m rdl.cli graph-report --run $(GRAPH_SMOKE_OUT) --challenge direct
	$(PY) -m rdl.cli graph-finalize --run $(GRAPH_SMOKE_OUT)
	@echo ""
	@echo "GRAPH SMOKE PASSED (diagnostic only)"

submodule:
	git submodule update --init --recursive

repro-dry:
	$(PY) -m rdl.cli run-repro --dry-run --condition configs/conditions/C0.yaml

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
