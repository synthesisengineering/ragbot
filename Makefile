# Ragbot Makefile
#
# Convenience targets that wrap the most common dev / test / eval flows.
# Designed to be idempotent and free of platform-specific assumptions
# beyond "python3 is on PATH and the requirements are installed."

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PYTHON ?= python3
PIP ?= $(PYTHON) -m pip
PYTEST ?= $(PYTHON) -m pytest

# Where the eval runner writes its markdown scorecard.
EVAL_SCORECARD ?= tests/evals/last-scorecard.md

.DEFAULT_GOAL := help

# `lint` and `typecheck` were listed here without rules to match, so
# `make lint` failed with "No rule to make target". No Python linter or type
# checker is configured in requirements.txt and there is no pyproject.toml, so
# the names were aspirational rather than broken. Dropped rather than stubbed:
# a .PHONY entry with no rule advertises a target that does not exist. The web
# side does have a real linter, wired into CI as `npm run lint`.
.PHONY: help install test test-fast eval eval-quick \
        eval-regressions eval-clean observability-test metrics-curl clean

# ---------------------------------------------------------------------------
# Help
# ---------------------------------------------------------------------------

help:
	@echo "Ragbot Makefile — common targets:"
	@echo ""
	@echo "  make install            Install / upgrade Python dependencies."
	@echo "  make test               Run the full pytest suite."
	@echo "  make test-fast          Run pytest excluding the integration suite."
	@echo "  make observability-test Run just the observability test module."
	@echo "  make eval               Run the offline eval suite, emit scorecard."
	@echo "  make eval-quick         Run only the quick subset of the eval suite."
	@echo "  make eval-regressions   Run only the regression-capture cases."
	@echo "  make eval-clean         Remove the last eval scorecard."
	@echo "  make metrics-curl       Hit /api/metrics on a local running server."
	@echo "  make clean              Remove caches and build artifacts."

# ---------------------------------------------------------------------------
# Install / lint
# ---------------------------------------------------------------------------

install:
	$(PIP) install -r requirements.txt

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

test:
	$(PYTEST) tests/ -v

test-fast:
	$(PYTEST) tests/ -v --ignore=tests/test_models_integration.py

observability-test:
	$(PYTEST) tests/test_observability.py -v

# ---------------------------------------------------------------------------
# Eval suite
#
# The runner is invoked as a module so its package-relative imports work
# correctly regardless of the CWD.
# ---------------------------------------------------------------------------

eval:
	@mkdir -p $(dir $(EVAL_SCORECARD))
	$(PYTHON) -m tests.evals.runner --output $(EVAL_SCORECARD)
	@echo ""
	@echo "Scorecard written to $(EVAL_SCORECARD)"

eval-quick:
	@mkdir -p $(dir $(EVAL_SCORECARD))
	$(PYTHON) -m tests.evals.runner --quick --output $(EVAL_SCORECARD)
	@echo ""
	@echo "Quick scorecard written to $(EVAL_SCORECARD)"

# Regression-only mode. The runner loads every YAML under
# tests/evals/regressions/, runs only those, and writes a focused
# scorecard. Exit code is non-zero if any regression re-emerges.
eval-regressions:
	@mkdir -p $(dir $(EVAL_SCORECARD))
	$(PYTHON) -m tests.evals.runner --regressions-only --output $(EVAL_SCORECARD)
	@echo ""
	@echo "Regression scorecard written to $(EVAL_SCORECARD)"

eval-clean:
	rm -f $(EVAL_SCORECARD)

# ---------------------------------------------------------------------------
# Observability quick-checks
# ---------------------------------------------------------------------------

metrics-curl:
	@echo "==> /api/metrics (Prometheus exposition)"
	@curl -fsS http://localhost:8000/api/metrics || true
	@echo ""
	@echo "==> /api/metrics/cache (60-minute window)"
	@curl -fsS 'http://localhost:8000/api/metrics/cache?window_minutes=60' || true
	@echo ""

# ---------------------------------------------------------------------------
# Housekeeping
# ---------------------------------------------------------------------------

clean:
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	find . -type d -name .pytest_cache -prune -exec rm -rf {} +
	find . -type d -name .mypy_cache -prune -exec rm -rf {} +
