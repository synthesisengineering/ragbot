"""Tests for the `ragbot mcp serve` CLI entry point and its production
wiring (synthesis_engine.mcp_server.wiring).

RagbotMCPServer itself and its tool dispatch are already covered by
tests/test_mcp_server.py against fake dependencies. These tests cover the
piece that was missing before this change: the glue that builds a REAL
ServerDependencies from live substrate singletons, and the CLI command
that uses it. No live Postgres/LLM backend is required — dependency
construction is mocked at the substrate-function level.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from unittest.mock import MagicMock, patch

import pytest

_SRC = os.path.join(os.path.dirname(__file__), '..', 'src')
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)


def _load_ragbot_cli():
    """Load src/ragbot.py (the CLI script) as its own module.

    `src/ragbot.py` and the `src/ragbot/` package share a name — a plain
    `import ragbot` resolves to the package (CPython prioritises regular
    packages over same-named modules in the same directory), not the CLI
    script, so `run_mcp_serve` / `create_mcp_parser` would be missing.
    Loading the file directly by path, under a different module name,
    sidesteps the collision. Same pattern documented (but left unused, in
    a since-skipped test module) in tests/test_ragbot.py.
    """
    spec = importlib.util.spec_from_file_location(
        "ragbot_cli_under_test", os.path.join(_SRC, "ragbot.py"),
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# wiring.py — document_getter
# ---------------------------------------------------------------------------


class TestDocumentGetter:
    def test_raises_keyerror_when_vector_store_unavailable(self):
        from synthesis_engine.mcp_server.wiring import _document_getter

        with patch('synthesis_engine.vectorstore.get_vector_store', return_value=None):
            with pytest.raises(KeyError):
                _document_getter("ws-a", "doc-1")

    def test_raises_keyerror_when_no_chunk_matches(self):
        from synthesis_engine.mcp_server.wiring import _document_getter

        fake_vs = MagicMock()
        fake_vs.scroll_documents.return_value = []
        with patch('synthesis_engine.vectorstore.get_vector_store', return_value=fake_vs):
            with pytest.raises(KeyError):
                _document_getter("ws-a", "nope.md")

    def test_reconstructs_content_from_ordered_chunks(self):
        from synthesis_engine.mcp_server.wiring import _document_getter
        from synthesis_engine.vectorstore import SearchHit

        hit1 = SearchHit(
            text="Hello ", score=0.0,
            metadata={"source_file": "doc.md", "char_start": 0, "char_end": 6, "filename": "doc.md"},
        )
        hit2 = SearchHit(
            text="world", score=0.0,
            metadata={"source_file": "doc.md", "char_start": 6, "char_end": 11, "filename": "doc.md"},
        )
        fake_vs = MagicMock()
        # Deliberately out of order — the getter must sort by char_start.
        fake_vs.scroll_documents.return_value = [hit2, hit1]

        with patch('synthesis_engine.vectorstore.get_vector_store', return_value=fake_vs):
            result = _document_getter("ws-a", "doc.md")

        assert result["content"] == "Hello world"
        assert result["metadata"]["filename"] == "doc.md"


# ---------------------------------------------------------------------------
# wiring.py — list_workspaces / routing_policies
# ---------------------------------------------------------------------------


class TestListWorkspacesAndRouting:
    def test_list_workspaces_delegates_to_resolve_repo_index(self):
        from synthesis_engine.mcp_server.wiring import _list_workspaces

        with patch(
            'synthesis_engine.workspaces.resolve_repo_index',
            return_value={"b": "/tmp/b", "a": "/tmp/a"},
        ):
            assert _list_workspaces() == ["a", "b"]

    def test_routing_policy_falls_back_to_default_when_workspace_unknown(self):
        from synthesis_engine.mcp_server.wiring import _routing_policy_for
        from synthesis_engine.policy.routing import RoutingPolicy, Confidentiality

        with patch('synthesis_engine.workspaces.resolve_repo_index', return_value={}):
            policy = _routing_policy_for("unknown-ws")

        assert isinstance(policy, RoutingPolicy)
        assert policy.confidentiality == Confidentiality.PUBLIC


# ---------------------------------------------------------------------------
# wiring.py — build_default_dependencies fails loudly without memory
# ---------------------------------------------------------------------------


class TestBuildDefaultDependencies:
    def test_raises_runtime_error_when_memory_unavailable(self):
        from synthesis_engine.mcp_server.wiring import build_default_dependencies

        with patch('synthesis_engine.memory.get_memory', return_value=None):
            with pytest.raises(RuntimeError, match="Memory backend unavailable"):
                build_default_dependencies()

    def test_wires_server_dependencies_when_memory_available(self):
        from synthesis_engine.mcp_server.wiring import build_default_dependencies
        from synthesis_engine.mcp_server import ServerDependencies

        fake_memory = MagicMock()
        fake_agent_loop = MagicMock()

        with patch('synthesis_engine.memory.get_memory', return_value=fake_memory), \
             patch('synthesis_engine.skills.discover_skills', return_value=[]), \
             patch('synthesis_engine.llm.get_llm_backend', return_value=MagicMock()), \
             patch('synthesis_engine.agent.AgentLoop', return_value=fake_agent_loop), \
             patch('synthesis_engine.agent.FilesystemCheckpointStore', return_value=MagicMock()):
            deps = build_default_dependencies()

        assert isinstance(deps, ServerDependencies)
        assert deps.memory is fake_memory
        assert deps.skill_runtime is not None
        assert callable(deps.list_workspaces)
        assert callable(deps.routing_policies)
        assert callable(deps.document_getter)
        assert deps.agent_run_starter is not None


# ---------------------------------------------------------------------------
# ragbot.py — `ragbot mcp serve` CLI command
# ---------------------------------------------------------------------------


class TestRunMcpServeCLI:
    def test_returns_1_and_prints_error_when_dependencies_fail(self, capsys):
        ragbot_cli = _load_ragbot_cli()

        args = argparse.Namespace()
        with patch(
            'synthesis_engine.mcp_server.wiring.build_default_dependencies',
            side_effect=RuntimeError("no db"),
        ):
            rc = ragbot_cli.run_mcp_serve(args)

        assert rc == 1
        captured = capsys.readouterr()
        assert "Failed to wire MCP server dependencies" in captured.err

    def test_starts_server_and_calls_serve_stdio_when_dependencies_ok(self):
        ragbot_cli = _load_ragbot_cli()

        args = argparse.Namespace()
        fake_deps = MagicMock()
        fake_server_instance = MagicMock()

        with patch(
            'synthesis_engine.mcp_server.wiring.build_default_dependencies',
            return_value=fake_deps,
        ), patch(
            'synthesis_engine.mcp_server.RagbotMCPServer',
            return_value=fake_server_instance,
        ) as mock_server_cls, patch(
            'asyncio.run',
        ) as mock_asyncio_run:
            rc = ragbot_cli.run_mcp_serve(args)

        assert rc == 0
        mock_server_cls.assert_called_once_with(fake_deps)
        # asyncio.run() must be called with the coroutine from serve_stdio().
        assert mock_asyncio_run.call_count == 1
        fake_server_instance.serve_stdio.assert_called_once()

    def test_keyboard_interrupt_during_serve_exits_cleanly(self):
        ragbot_cli = _load_ragbot_cli()

        args = argparse.Namespace()
        fake_deps = MagicMock()

        with patch(
            'synthesis_engine.mcp_server.wiring.build_default_dependencies',
            return_value=fake_deps,
        ), patch(
            'synthesis_engine.mcp_server.RagbotMCPServer',
            return_value=MagicMock(),
        ), patch(
            'asyncio.run', side_effect=KeyboardInterrupt,
        ):
            rc = ragbot_cli.run_mcp_serve(args)

        assert rc == 0


class TestMcpParserWiring:
    def test_mcp_serve_subcommand_is_registered(self):
        """`ragbot mcp serve` must parse to run_mcp_serve without hitting
        argparse errors — regression guard for the subparser wiring in
        main()."""
        ragbot_cli = _load_ragbot_cli()

        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest='command')
        ragbot_cli.create_mcp_parser(subparsers)

        args = parser.parse_args(['mcp', 'serve'])
        assert args.func is ragbot_cli.run_mcp_serve
