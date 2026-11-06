"""Sandboxed code execution for the agent loop.

The agent occasionally needs to run LLM-generated code — to compute a
numeric answer, to transform data, to drive a small tool the planner
synthesised on the fly. Running that code in a plain ``subprocess`` on
the host is forbidden: it is untrusted by construction, and a
prompt-injection vector can reach for the filesystem, the network, or
worse.

This module defines a provider-agnostic :class:`Sandbox` abstract base
and three concrete implementations:

* :class:`DisabledSandbox` — the dev-time default. Every ``execute()``
  call returns an error result that tells the operator exactly how to
  enable a real sandbox. Fail-closed.

* :class:`E2BSandbox` — wraps the `e2b <https://e2b.dev>`_ Python SDK.
  Requires the ``E2B_API_KEY`` environment variable. Each call spins
  up a fresh microVM, uploads the requested files, runs the code,
  captures stdout / stderr / exit code / files written, and tears the
  microVM down. If the SDK is not installed, the class still imports
  but every ``execute()`` returns an error result rather than crashing
  the agent loop.

* :class:`DaytonaSandbox` — uses HTTP requests to a self-hosted Daytona
  instance. Requires ``DAYTONA_API_URL`` (and optionally
  ``DAYTONA_API_TOKEN``). Same lifecycle as E2B. If the ``httpx`` SDK
  is unavailable the class falls back to a no-op error result.

Design notes
============

* The interface is async at the boundary so the agent loop can dispatch
  many sandbox calls in parallel without blocking.

* ``ExecutionResult`` is the only return shape the loop knows about.
  Provider-specific telemetry can ride along in
  ``ExecutionResult.metadata``.

* The SDK imports are deferred inside ``execute()``. The module
  must be import-clean even when neither ``e2b`` nor ``httpx`` is
  installed.

* The default sandbox in :class:`AgentLoop` is :class:`DisabledSandbox`.
  Real sandboxes are opt-in: callers pass an instance to
  ``AgentLoop(sandbox=...)``.
"""

from __future__ import annotations

import abc
import asyncio
import base64
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class ExecutionResult:
    """The single return shape every Sandbox implementation produces.

    Attributes:
        stdout: Captured stdout text.
        stderr: Captured stderr text. For ``DisabledSandbox`` and for
            failures inside a real backend this holds the operator-facing
            error message.
        exit_code: The process exit code. ``-1`` is the convention for
            "sandbox could not even start the process" (disabled, SDK
            missing, configuration error). ``0`` is success; anything
            else is the program's own exit code.
        duration_seconds: Wall-clock time the call took. Includes
            sandbox spin-up and tear-down for the cloud backends.
        files_written: Mapping of ``path -> bytes`` for files the
            program created inside the sandbox. Empty by default — a
            backend opts in to capturing files by passing ``capture_*``
            to its provider SDK.
        provider: A short identifier ("disabled", "e2b", "daytona") so
            the loop's instrumentation can tag the span correctly.
        metadata: Free-form bag for provider-specific telemetry
            (microVM id, request id, billed milliseconds, etc.).
    """

    stdout: str = ""
    stderr: str = ""
    exit_code: int = 0
    duration_seconds: float = 0.0
    files_written: Dict[str, bytes] = field(default_factory=dict)
    provider: str = "unknown"
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def success(self) -> bool:
        """True iff the program ran and returned exit code 0."""
        return self.exit_code == 0

    def to_dict(self) -> Dict[str, Any]:
        """JSON-friendly serialisation.

        ``files_written`` bytes are base64-encoded so the whole result
        survives a checkpoint round-trip through JSON.
        """
        return {
            "stdout": self.stdout,
            "stderr": self.stderr,
            "exit_code": self.exit_code,
            "duration_seconds": self.duration_seconds,
            "files_written": {
                path: base64.b64encode(data).decode("ascii")
                for path, data in self.files_written.items()
            },
            "provider": self.provider,
            "metadata": dict(self.metadata),
        }


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class Sandbox(abc.ABC):
    """Provider-agnostic sandboxed execution surface.

    Implementations spin up an isolated environment (microVM, container,
    chroot — provider's choice), run the supplied code, and return an
    :class:`ExecutionResult`. The base class advertises the language
    set the implementation supports so the agent loop can fail fast if
    a plan asks for an unsupported language.
    """

    #: Short identifier the loop uses for instrumentation labels.
    provider: str = "abstract"

    #: Languages this sandbox supports. ``"python"`` is the universal
    #: default; ``"bash"`` is the second most common.
    supported_languages: tuple = ("python",)

    @abc.abstractmethod
    async def execute(
        self,
        code: str,
        language: str = "python",
        timeout_seconds: int = 30,
        files: Optional[Dict[str, bytes]] = None,
    ) -> ExecutionResult:
        """Run ``code`` inside the sandbox and return an :class:`ExecutionResult`."""

    def _check_language(self, language: str) -> Optional[ExecutionResult]:
        """Return an error result if the language is unsupported, else None."""
        if language not in self.supported_languages:
            return ExecutionResult(
                exit_code=-1,
                stderr=(
                    f"Sandbox {self.provider!r} does not support language "
                    f"{language!r}. Supported: "
                    f"{', '.join(self.supported_languages)}."
                ),
                provider=self.provider,
            )
        return None


# ---------------------------------------------------------------------------
# DisabledSandbox — fail-closed default
# ---------------------------------------------------------------------------


_DISABLED_REASON = (
    "Sandboxed code execution is disabled. To enable it, instantiate one "
    "of the real backends and pass it to AgentLoop:\n"
    "  - E2BSandbox(): requires the e2b SDK (`pip install e2b`) and "
    "the E2B_API_KEY environment variable.\n"
    "  - DaytonaSandbox(): requires httpx (`pip install httpx`) and the "
    "DAYTONA_API_URL environment variable pointing at a self-hosted "
    "Daytona deployment.\n"
    "Running untrusted LLM-generated code in a plain subprocess on the "
    "host is intentionally not supported."
)


class DisabledSandbox(Sandbox):
    """The fail-closed default sandbox.

    Every ``execute()`` call returns an :class:`ExecutionResult` with
    ``exit_code = -1`` and an actionable stderr explaining exactly how
    to opt in to a real sandbox. The agent loop treats this as a step
    failure, which routes through replan just like any other failure —
    but the operator sees the clear "you need to enable a sandbox"
    message in the loop's trace.
    """

    provider = "disabled"
    supported_languages = ("python", "bash", "javascript", "typescript")

    async def execute(
        self,
        code: str,
        language: str = "python",
        timeout_seconds: int = 30,
        files: Optional[Dict[str, bytes]] = None,
    ) -> ExecutionResult:
        return ExecutionResult(
            exit_code=-1,
            stderr=_DISABLED_REASON,
            provider=self.provider,
        )


# ---------------------------------------------------------------------------
# E2BSandbox — e2b.dev microVM backend
# ---------------------------------------------------------------------------


class E2BSandbox(Sandbox):
    """Run code inside an e2b.dev microVM.

    The e2b SDK is imported lazily inside :meth:`execute` so this class
    can be constructed even when the SDK isn't installed. A missing SDK
    or a missing ``E2B_API_KEY`` produces a clear error result rather
    than crashing the agent loop.
    """

    provider = "e2b"
    supported_languages = ("python", "bash", "javascript", "typescript")

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        template: Optional[str] = None,
    ) -> None:
        self._api_key = api_key or os.environ.get("E2B_API_KEY")
        # ``template`` lets advanced callers pin a specific microVM image
        # (e.g., one with extra apt packages pre-installed). None means
        # the SDK default.
        self._template = template

    async def execute(
        self,
        code: str,
        language: str = "python",
        timeout_seconds: int = 30,
        files: Optional[Dict[str, bytes]] = None,
    ) -> ExecutionResult:
        lang_error = self._check_language(language)
        if lang_error is not None:
            return lang_error

        if not self._api_key:
            return ExecutionResult(
                exit_code=-1,
                stderr=(
                    "E2BSandbox: E2B_API_KEY is not set. Export the key "
                    "or pass api_key= to the constructor."
                ),
                provider=self.provider,
            )

        try:
            # Imported lazily so the agent loop can run without the e2b
            # SDK installed. The catch covers ModuleNotFoundError, ImportError,
            # and even runtime ImportError from incompatible versions.
            from e2b import Sandbox as E2BSdkSandbox  # type: ignore  # noqa: WPS433
        except Exception as exc:  # pragma: no cover - SDK absent in unit tests
            return ExecutionResult(
                exit_code=-1,
                stderr=(
                    f"E2BSandbox: the e2b Python SDK is not importable "
                    f"({exc}). Install with `pip install e2b`."
                ),
                provider=self.provider,
            )

        start = time.monotonic()
        try:
            # The SDK is blocking; run it in a thread so the asyncio
            # event loop stays responsive.
            return await asyncio.get_running_loop().run_in_executor(
                None,
                self._run_blocking,
                E2BSdkSandbox,
                code,
                language,
                timeout_seconds,
                files or {},
                start,
            )
        except Exception as exc:  # pragma: no cover - network errors
            duration = time.monotonic() - start
            return ExecutionResult(
                exit_code=-1,
                stderr=f"E2BSandbox: dispatch failed: {exc!r}",
                duration_seconds=duration,
                provider=self.provider,
            )

    def _run_blocking(
        self,
        sdk_class: Any,
        code: str,
        language: str,
        timeout_seconds: int,
        files: Dict[str, bytes],
        start: float,
    ) -> ExecutionResult:  # pragma: no cover - exercised only with real SDK
        # The constructor signature changed across e2b SDK versions; we
        # try the modern kwargs first and fall back to positional.
        try:
            sandbox = sdk_class(
                template=self._template,
                api_key=self._api_key,
                timeout=timeout_seconds,
            )
        except TypeError:
            sandbox = sdk_class(self._template or "base", api_key=self._api_key)

        try:
            for path, data in files.items():
                try:
                    sandbox.files.write(path, data)
                except Exception as exc:
                    logger.warning("E2B upload of %s failed: %s", path, exc)

            if language == "python":
                result = sandbox.run_code(code, timeout=timeout_seconds)
            else:
                # e2b's ``commands.run`` is the language-agnostic shell entry.
                cmd = self._shell_command_for(language, code)
                result = sandbox.commands.run(cmd, timeout=timeout_seconds)

            stdout = getattr(result, "stdout", "") or ""
            stderr = getattr(result, "stderr", "") or ""
            exit_code = int(getattr(result, "exit_code", 0) or 0)
            return ExecutionResult(
                stdout=stdout,
                stderr=stderr,
                exit_code=exit_code,
                duration_seconds=time.monotonic() - start,
                provider=self.provider,
                metadata={"template": self._template or "default"},
            )
        finally:
            try:
                sandbox.kill()
            except Exception:
                pass

    @staticmethod
    def _shell_command_for(language: str, code: str) -> str:  # pragma: no cover - SDK path
        if language == "bash":
            return code
        if language in ("javascript", "typescript"):
            suffix = ".ts" if language == "typescript" else ".js"
            runner = "ts-node" if language == "typescript" else "node"
            return f"cat > /tmp/snippet{suffix} <<'EOF'\n{code}\nEOF\n{runner} /tmp/snippet{suffix}"
        return code


# ---------------------------------------------------------------------------
# DaytonaSandbox — self-hosted Daytona backend
# ---------------------------------------------------------------------------


class DaytonaSandbox(Sandbox):
    """Run code inside a self-hosted Daytona workspace.

    Daytona exposes an HTTP API for creating ephemeral workspaces and
    executing commands inside them. This class drives that API using
    ``httpx`` (imported lazily). The API URL must be supplied via
    ``DAYTONA_API_URL`` or the constructor; an optional bearer token
    comes from ``DAYTONA_API_TOKEN``.

    If ``httpx`` isn't installed, ``execute()`` returns an error result
    so the agent loop's behavior is consistent with the E2B backend.
    """

    provider = "daytona"
    supported_languages = ("python", "bash", "javascript", "typescript")

    def __init__(
        self,
        *,
        api_url: Optional[str] = None,
        api_token: Optional[str] = None,
        image: str = "python:3.12-slim",
    ) -> None:
        self._api_url = (api_url or os.environ.get("DAYTONA_API_URL") or "").rstrip("/")
        self._api_token = api_token or os.environ.get("DAYTONA_API_TOKEN")
        self._image = image

    async def execute(
        self,
        code: str,
        language: str = "python",
        timeout_seconds: int = 30,
        files: Optional[Dict[str, bytes]] = None,
    ) -> ExecutionResult:
        lang_error = self._check_language(language)
        if lang_error is not None:
            return lang_error

        if not self._api_url:
            return ExecutionResult(
                exit_code=-1,
                stderr=(
                    "DaytonaSandbox: DAYTONA_API_URL is not set. Export "
                    "the URL of your self-hosted Daytona deployment or "
                    "pass api_url= to the constructor."
                ),
                provider=self.provider,
            )

        try:
            import httpx  # type: ignore  # noqa: WPS433
        except Exception as exc:  # pragma: no cover - SDK absent in unit tests
            return ExecutionResult(
                exit_code=-1,
                stderr=(
                    f"DaytonaSandbox: httpx is not importable ({exc}). "
                    "Install with `pip install httpx`."
                ),
                provider=self.provider,
            )

        headers: Dict[str, str] = {"Content-Type": "application/json"}
        if self._api_token:
            headers["Authorization"] = f"Bearer {self._api_token}"

        payload = {
            "image": self._image,
            "language": language,
            "code": code,
            "files": {
                path: base64.b64encode(data).decode("ascii")
                for path, data in (files or {}).items()
            },
            "timeout_seconds": timeout_seconds,
        }

        start = time.monotonic()
        try:  # pragma: no cover - network path not exercised in unit tests
            async with httpx.AsyncClient(timeout=timeout_seconds + 30) as client:
                response = await client.post(
                    f"{self._api_url}/v1/execute",
                    json=payload,
                    headers=headers,
                )
                response.raise_for_status()
                body = response.json()
        except Exception as exc:  # pragma: no cover - network errors
            return ExecutionResult(
                exit_code=-1,
                stderr=f"DaytonaSandbox: HTTP call failed: {exc!r}",
                duration_seconds=time.monotonic() - start,
                provider=self.provider,
            )

        files_written = {
            path: base64.b64decode(payload_b64)
            for path, payload_b64 in (body.get("files_written") or {}).items()
        }
        return ExecutionResult(  # pragma: no cover - network path
            stdout=str(body.get("stdout") or ""),
            stderr=str(body.get("stderr") or ""),
            exit_code=int(body.get("exit_code") or 0),
            duration_seconds=time.monotonic() - start,
            files_written=files_written,
            provider=self.provider,
            metadata={"workspace_id": body.get("workspace_id")},
        )


__all__ = [
    "DaytonaSandbox",
    "DisabledSandbox",
    "E2BSandbox",
    "ExecutionResult",
    "Sandbox",
]
