"""Pytest configuration for Ragbot tests.

Test categories:
- Unit tests (test_config.py, test_keystore.py, test_helpers.py): No external dependencies
- API tests (test_api.py): Uses TestClient, no real LLM calls
- Integration tests (test_models_integration.py): Makes real LLM API calls

Run all unit tests:
    pytest tests/ --ignore=tests/test_models_integration.py

Run integration tests (requires API keys):
    pytest tests/test_models_integration.py -v

Run all tests including expensive models:
    TEST_EXPENSIVE_MODELS=1 pytest tests/
"""

import pytest
import os
import sys

# Add src directory to Python path for all tests
src_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src')
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)


def pytest_configure(config):
    """Configure custom markers."""
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests (require API keys)"
    )
    config.addinivalue_line(
        "markers", "expensive: marks tests as expensive (use TEST_EXPENSIVE_MODELS=1 to run)"
    )


# ---------------------------------------------------------------------------
# OpenTelemetry — single session-scoped initialization for the whole suite
# ---------------------------------------------------------------------------
#
# OTEL's global TracerProvider and MeterProvider are process-singletons that
# refuse to be replaced once set (and warn at every re-init attempt). To keep
# the test suite reliable, we initialize ONCE per pytest session, share the
# in-memory exporter across every test that needs spans, and reset captured
# spans between tests via ``exporter.clear()``.
#
# Test files that need access to the shared exporter consume the
# ``in_memory_tracer`` fixture below. The legacy per-file ``_session_tracer``
# fixtures in test_observability.py / test_agent_loop.py / test_agent_capabilities.py
# have been replaced by aliases that resolve to this conftest fixture so the
# whole suite shares one initialization.


@pytest.fixture(scope="session", autouse=True)
def _otel_session_exporter():
    """One-time per-session OTEL initialization with an in-memory exporter.

    Yields the InMemorySpanExporter so per-test fixtures can read captured
    spans. Tears down once at session end via ``shutdown_tracer``.

    AUTOUSE rationale: OTEL's global TracerProvider and MeterProvider are
    set-once singletons — once any code (e.g., the FastAPI lifespan in
    test_api_lifespan.py) calls ``trace.set_tracer_provider``, OTEL
    silently refuses replacement attempts. The session-scoped fixture
    MUST run before any test that fires the lifespan, otherwise the
    conftest's InMemorySpanExporter never makes it onto the global
    provider and span-capturing tests fail with empty span lists.
    autouse=True guarantees this fixture runs at session start
    regardless of which tests pytest collects first.
    """
    # Import lazily so non-OTEL test runs (e.g., test_helpers.py) don't pay
    # the import cost.
    try:
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
            InMemorySpanExporter,
        )
    except ImportError:  # pragma: no cover - OTEL is a hard dep
        pytest.skip("opentelemetry.sdk not available; OTEL tests skipped")

    from synthesis_engine.observability import init_tracer, shutdown_tracer

    exporter = InMemorySpanExporter()
    provider = init_tracer(
        service_name="synthesis_engine_test_suite",
        exporter=exporter,
        force=True,
    )
    assert provider is not None, (
        "init_tracer returned None — opentelemetry-sdk should be installed."
    )
    yield exporter
    shutdown_tracer()


@pytest.fixture
def in_memory_tracer(_otel_session_exporter):
    """Per-test handle on the shared in-memory exporter.

    Clears any previously captured spans so each test observes only the
    spans emitted within its own body. Shared with every test file that
    needs to assert on OTEL spans — see the comment block above.
    """
    _otel_session_exporter.clear()
    return _otel_session_exporter


# ---------------------------------------------------------------------------
# Synthesis identity — hermetic test default
# ---------------------------------------------------------------------------
#
# ``synthesis_engine.identity.get_personal_workspaces()`` reads from
# ``~/.synthesis/identity.yaml`` by default, which on the operator's
# machine declares their personal workspaces (e.g., ``rajiv``). Test runs
# must not pick up that file or the test outcome depends on whose laptop
# pytest runs on. The autouse fixture below points
# ``SYNTHESIS_IDENTITY_CONFIG`` at a non-existent path so tests see an
# empty personal-workspace list by default. Tests that need identity
# behaviour set the env var to a tmp_path file themselves (see
# ``tests/test_identity.py``).


@pytest.fixture(autouse=True)
def _hermetic_synthesis_identity(monkeypatch, tmp_path_factory):
    """Default SYNTHESIS_IDENTITY_CONFIG to a missing path for all tests."""
    bogus = tmp_path_factory.mktemp("no_identity") / "missing.yaml"
    monkeypatch.setenv("SYNTHESIS_IDENTITY_CONFIG", str(bogus))


# ---------------------------------------------------------------------------
# Synthesis runtime state — hermetic test default
# ---------------------------------------------------------------------------
#
# Production task and agent-checkpoint state belongs under ~/.synthesis.
# Tests must never write to that live control plane: it makes outcomes depend
# on filesystem permissions and leaves test task records mixed with real ones.


@pytest.fixture(autouse=True)
def _hermetic_synthesis_runtime_state(monkeypatch, tmp_path):
    """Redirect task and checkpoint state and reset the task singleton."""
    monkeypatch.setenv("SYNTHESIS_TASK_DIR", str(tmp_path / "tasks"))
    monkeypatch.setenv(
        "SYNTHESIS_AGENT_CHECKPOINT_DIR",
        str(tmp_path / "agent-checkpoints"),
    )

    from synthesis_engine.tasks import set_default_manager

    set_default_manager(None)
    yield
    set_default_manager(None)
