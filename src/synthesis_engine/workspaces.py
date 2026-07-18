"""Workspace discovery and management for Ragbot.

This module handles discovering and loading workspaces from ai-knowledge repos.

Discovery resolution order (first match wins):
    1. --base-path CLI arg     (treated as flat parent containing ai-knowledge-*)
    2. RAGBOT_BASE_PATH env    (treated as flat parent)
    3. ~/.synthesis/console.yaml  (synthesis-console's source list — preferred)
    4. ~/workspaces/*/ai-knowledge-*  (workspace-rooted layout glob fallback)
    5. /app/ai-knowledge       (Docker container default)
    6. ~/ai-knowledge          (legacy flat-parent convention)

This integrates ragbot with synthesis-console: when both are installed and
~/.synthesis/console.yaml exists, ragbot reuses synthesis-console's source list.
Both products remain independently usable.

IMPORTANT: Inheritance configuration is centralized in my-projects.yaml in the
personal repo (per ADR-006). Individual compile-config.yaml files do NOT contain
inheritance information - that would reveal private repo existence in shared repos.
"""

import glob
import os
import re
from pathlib import Path
from typing import Optional, List, Dict, Any
import yaml

from .discovery import SCOPE_WORKSPACES, apply_discovery_filter
from .exceptions import WorkspaceError, WorkspaceNotFoundError
from .inheritance import (
    find_inheritance_config,
    get_inheritance_chain,
    load_inheritance_config,
)
from .models import WorkspaceInfo


# Synthesis-console config path (the integration point).
SYNTHESIS_CONSOLE_CONFIG = Path.home() / ".synthesis" / "console.yaml"

# Convention-based fallback search globs.
WORKSPACE_GLOB_PATTERN = str(Path.home() / "workspaces" / "*" / "ai-knowledge-*")

# Legacy flat-parent fallback paths.
LEGACY_FLAT_PARENT_PATHS = [
    "/app/ai-knowledge",                          # Docker container default
    str(Path.home() / "ai-knowledge"),            # Legacy convention
]


def _derive_workspace_name(repo_path: str) -> str:
    """Derive workspace name from an ai-knowledge-* repo directory name.

    Strips the `ai-knowledge-` prefix and returns the remainder.
        ai-knowledge-personal           → personal
        ai-knowledge-example-client     → example-client
        ai-knowledge-example-private    → example-private
    """
    name = os.path.basename(repo_path.rstrip("/"))
    if name.startswith("ai-knowledge-"):
        return name[len("ai-knowledge-"):]
    return name


# Pattern for extracting the bare workspace identifier from a private repo
# whose suffix encodes ownership (e.g., `<workspace>-<owner>-private` or
# plain `<workspace>-private`). Used to find compiled/{bare_name}/ output.
_PRIVATE_SUFFIX_RE = re.compile(r'(?:-[^-]+)?-private$')


def _bare_workspace_name(name: str) -> str:
    """Strip private-suffix annotations to recover the bare workspace name.

    Examples:
        personal              → personal
        example-client        → example-client
        example-private       → example
        example-someone-private → example
    """
    return _PRIVATE_SUFFIX_RE.sub('', name)


def _read_synthesis_console_sources() -> List[Dict[str, str]]:
    """Read sources from ~/.synthesis/console.yaml.

    Returns a list of {name, root} dicts. The `name` is derived from the
    directory (stripping the `ai-knowledge-` prefix) so it aligns with
    my-projects.yaml's workspace keys, which is ragbot's source of truth for
    inheritance. Synthesis-console's `name` field is a display label for its
    own UI/URLs and is not used as the workspace identifier here.

    Demo sources (demo: true) are skipped.
    """
    if not SYNTHESIS_CONSOLE_CONFIG.exists():
        return []
    try:
        with open(SYNTHESIS_CONSOLE_CONFIG) as f:
            config = yaml.safe_load(f) or {}
    except (yaml.YAMLError, OSError):
        return []

    sources = []
    for src in config.get("sources", []):
        if src.get("demo"):
            continue
        root = src.get("root")
        if not root:
            continue
        root = os.path.expanduser(root)
        # Only include sources whose root is an ai-knowledge-* directory.
        if not os.path.basename(root.rstrip("/")).startswith("ai-knowledge-"):
            continue
        name = _derive_workspace_name(root)
        sources.append({"name": name, "root": root})
    return sources


def _scan_flat_parent(parent: str) -> Dict[str, str]:
    """Walk a flat parent dir for ai-knowledge-* children. Returns {name: path}."""
    parent = os.path.expanduser(parent)
    if not os.path.isdir(parent):
        return {}
    results = {}
    for item in os.listdir(parent):
        if not item.startswith("ai-knowledge-"):
            continue
        repo_path = os.path.join(parent, item)
        if os.path.isdir(repo_path):
            results[_derive_workspace_name(repo_path)] = repo_path
    return results


def resolve_repo_index(base_path: Optional[str] = None) -> Dict[str, str]:
    """Resolve the available ai-knowledge repos.

    Returns a mapping of workspace_name -> absolute repo path. Workspace name
    is derived from the directory (the `ai-knowledge-` prefix is stripped).

    When base_path or RAGBOT_BASE_PATH is set, ONLY that flat parent is
    scanned (override mode, for back-compat and Docker container use).

    Otherwise, the index is the UNION of:
        - ~/.synthesis/console.yaml sources (synthesis-console integration)
        - ~/workspaces/*/ai-knowledge-* glob (workspace-rooted layout)
        - /app/ai-knowledge children (Docker container default)
        - ~/ai-knowledge children (legacy flat-parent convention)

    Private repos (-private suffix or .ai-knowledge-private-owner sentinel)
    are filtered out unless RAGBOT_OWNER_CONTEXT=1.

    Runtimes can override discovery entirely by registering a filter for
    the ``"workspaces"`` scope via
    :func:`synthesis_engine.discovery.set_discovery_filter`. When such a
    filter is active and returns a non-None dict, that dict short-circuits
    every other source. Ragbot uses this to hard-isolate demo mode to
    the bundled workspace.
    """
    # Runtime-registered filter takes precedence. The filter returns the
    # full replacement index (or None to defer to the substrate default).
    override = apply_discovery_filter(SCOPE_WORKSPACES, None)
    if override is not None:
        return override

    owner_context = _is_owner_context()

    # Override mode: explicit base_path arg or RAGBOT_BASE_PATH env.
    # When set, only this flat parent is scanned. Used by Docker and tests.
    override_path = base_path or os.environ.get("RAGBOT_BASE_PATH")
    if override_path:
        return _filter_private(_scan_flat_parent(override_path), owner_context)

    # Union mode: aggregate across all available sources.
    index: Dict[str, str] = {}

    # ~/.synthesis/console.yaml sources
    for src in _read_synthesis_console_sources():
        if os.path.isdir(src["root"]):
            index.setdefault(src["name"], src["root"])

    # ~/workspaces/*/ai-knowledge-* glob
    for repo_path in glob.glob(WORKSPACE_GLOB_PATTERN):
        if os.path.isdir(repo_path):
            index.setdefault(_derive_workspace_name(repo_path), repo_path)

    # Legacy flat parents (Docker, ~/ai-knowledge)
    for parent in LEGACY_FLAT_PARENT_PATHS:
        for name, path in _scan_flat_parent(parent).items():
            index.setdefault(name, path)

    return _filter_private(index, owner_context)


def _filter_private(index: Dict[str, str], owner_context: bool) -> Dict[str, str]:
    """Filter out private repos unless owner context is set."""
    if owner_context:
        return index
    return {
        name: path for name, path in index.items()
        if not _is_private_repo(path, os.path.basename(path))
    }


# Back-compat alias: legacy code references DEFAULT_AI_KNOWLEDGE_PATHS as the
# list of flat-parent search paths. Kept for any external callers.
DEFAULT_AI_KNOWLEDGE_PATHS = LEGACY_FLAT_PARENT_PATHS


# Environment variable for owner-context discovery override.
# When set to "1" or "true", *-private repos will NOT be filtered out.
# This should ONLY be used when the current user is the owner of those private repos.
# Default (unset) behavior: filter out all *-private repos — safe for team contexts.
_OWNER_CONTEXT_ENV = 'RAGBOT_OWNER_CONTEXT'


def _is_owner_context() -> bool:
    """Check if running in owner context (allows -private repos to be discovered)."""
    value = os.environ.get(_OWNER_CONTEXT_ENV, '').lower()
    return value in ('1', 'true', 'yes')


def is_owner_context() -> bool:
    """Public alias for :func:`_is_owner_context`.

    Other modules (e.g. ``rag.py``'s archive-tier indexing path) need to
    check owner context without reaching into a private helper. Kept as
    a thin alias rather than renaming the private one, to avoid
    disturbing existing internal call sites.
    """
    return _is_owner_context()


def _is_private_repo(repo_path: str, repo_name: str) -> bool:
    """Check if a repo is a workspace-private repo per ADR-014.

    A repo is considered private if ANY of:
    - Name ends with '-private' (e.g., ai-knowledge-example-client-person-private)
    - A sentinel file `.ai-knowledge-private-owner` exists at the repo root

    Private repos MUST be filtered from default discovery. They are only
    included when _is_owner_context() returns True (see ADR-014, ADR-020).

    Args:
        repo_path: Absolute path to the repo directory
        repo_name: Basename of the repo (e.g., 'ai-knowledge-example-private')

    Returns:
        True if the repo is private and should be filtered out by default
    """
    if repo_name.endswith('-private'):
        return True
    sentinel = os.path.join(repo_path, '.ai-knowledge-private-owner')
    if os.path.isfile(sentinel):
        return True
    return False


def _build_repo_metadata(workspace_name: str, repo_path: str) -> Optional[Dict[str, Any]]:
    """Build the metadata dict for a single ai-knowledge repo.

    Returns None if the repo has no usable content (no source, no compiled, no config).
    """
    if not os.path.isdir(repo_path):
        return None

    config = {}
    compile_config_path = os.path.join(repo_path, "compile-config.yaml")
    if os.path.isfile(compile_config_path):
        with open(compile_config_path, "r") as f:
            config = yaml.safe_load(f) or {}

    # Compiled output is keyed by the bare workspace name (private-suffix stripped).
    bare_name = _bare_workspace_name(workspace_name)
    compiled_base = os.path.join(repo_path, "compiled", bare_name)
    instructions_dir = os.path.join(compiled_base, "instructions")
    knowledge_dir = os.path.join(compiled_base, "knowledge")

    source_dir = os.path.join(repo_path, "source")
    source_datasets_dir = os.path.join(source_dir, "datasets")
    source_instructions_dir = os.path.join(source_dir, "instructions")
    has_source = os.path.isdir(source_dir)
    # ``has_instructions`` and ``has_datasets`` reflect "is there any source
    # of this content the user can chat against," not "is a pre-compiled
    # bundle on disk." The pre-v3 architecture wrote compiled/{name}/knowledge/
    # but the v3 compiler ships only compiled/{name}/instructions/ — knowledge
    # concatenation moved to CI/CD. Falling back to source/* keeps the UI
    # honest under both layouts.
    has_instructions = (
        os.path.isdir(instructions_dir)
        or os.path.isdir(source_instructions_dir)
    )
    has_knowledge = (
        os.path.isdir(knowledge_dir)
        or os.path.isdir(source_datasets_dir)
    )

    # `personal-archive/` is a separate top-level directory (sibling of
    # `source/`, `compiled/`), holding lower-priority reference/archive
    # content. It is deliberately resolved ONLY in owner context — same
    # sensitivity bar as the -private repos gated by RAGBOT_OWNER_CONTEXT
    # (ADR-014). This check is independent of (and in addition to) the
    # per-repo private-suffix filtering that already happened upstream in
    # resolve_repo_index(): a repo need not be named `*-private` to have
    # a personal-archive/ directory that should stay invisible by default.
    archive_dir = os.path.join(repo_path, "personal-archive")
    has_archive = _is_owner_context() and os.path.isdir(archive_dir)

    if not (has_instructions or has_knowledge or has_source or has_archive or config):
        return None

    return {
        "instructions": instructions_dir if has_instructions else None,
        "datasets": knowledge_dir if has_knowledge else None,
        "archive": archive_dir if has_archive else None,
        "repo_path": repo_path,
        "source_path": source_dir if has_source else None,
        "config": config,
        "has_instructions": has_instructions,
        "has_datasets": has_knowledge,
        "has_archive": has_archive,
        "has_source": has_source,
    }


def discover_ai_knowledge_repos(ai_knowledge_root: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
    """
    Auto-discover ai-knowledge repos.

    When ai_knowledge_root is provided, treats it as a flat parent directory and
    walks for ai-knowledge-* children (legacy mode).

    When ai_knowledge_root is None, uses the full resolution chain:
    ~/.synthesis/console.yaml, then ~/workspaces/*/ai-knowledge-* glob,
    then legacy flat parents.

    Private repos (matching 'ai-knowledge-*-private' or containing a
    `.ai-knowledge-private-owner` sentinel file) are filtered out by default
    per ADR-014. Set the RAGBOT_OWNER_CONTEXT environment variable to '1' to
    include them (owner-context compilation per ADR-020).

    Args:
        ai_knowledge_root: Optional flat parent override. If None, uses the
            full resolution chain (~/.synthesis/console.yaml, etc.).

    Returns:
        Dictionary mapping workspace names to their content and metadata.
    """
    index = resolve_repo_index(ai_knowledge_root)
    discovered = {}
    for workspace_name, repo_path in index.items():
        meta = _build_repo_metadata(workspace_name, repo_path)
        if meta is not None:
            discovered[workspace_name] = meta
    return discovered


def find_ai_knowledge_root() -> Optional[str]:
    """Find the ai-knowledge root directory (legacy flat-parent only).

    NOTE: With the workspace-rooted layout there is no single root — repos are
    distributed across workspaces. This function only succeeds for legacy
    flat-parent setups (e.g., ~/ai-knowledge/, /app/ai-knowledge/). Code that
    needs to enumerate repos should use resolve_repo_index() or
    discover_ai_knowledge_repos() directly.

    Returns:
        Path to flat-parent root, or None if no flat-parent layout exists.
    """
    env_path = os.environ.get("RAGBOT_BASE_PATH")
    if env_path:
        expanded = os.path.expanduser(env_path)
        if os.path.isdir(expanded):
            return expanded
    for candidate in LEGACY_FLAT_PARENT_PATHS:
        if os.path.isdir(candidate):
            return candidate
    return None


def _load_centralized_inheritance(ai_knowledge_root: Optional[str] = None) -> Dict[str, Any]:
    """
    Load inheritance configuration from my-projects.yaml.

    Per ADR-006, inheritance config lives ONLY in the personal repo. This function
    searches all discovered ai-knowledge repos to find the one containing
    my-projects.yaml.

    Args:
        ai_knowledge_root: Optional flat-parent override. Otherwise uses the
            full resolution chain.

    Returns:
        Inheritance configuration dictionary, or empty dict if not found.
    """
    index = resolve_repo_index(ai_knowledge_root)
    for workspace_name, repo_path in index.items():
        config_path = find_inheritance_config(repo_path)
        if config_path:
            try:
                return load_inheritance_config(config_path)
            except Exception:
                pass
    return {}


def _get_inheritance_for_workspace(
    workspace_name: str,
    inheritance_config: Dict[str, Any]
) -> List[str]:
    """
    Get the inheritance chain for a workspace from centralized config.

    Args:
        workspace_name: Name of the workspace
        inheritance_config: Loaded my-projects.yaml config

    Returns:
        List of parent workspace names (not including self)
    """
    if not inheritance_config:
        return []

    try:
        # Get full chain (includes self at end)
        chain = get_inheritance_chain(workspace_name, inheritance_config)
        # Return parents only (exclude self)
        return [p for p in chain if p != workspace_name]
    except (ValueError, KeyError):
        # Workspace not in inheritance config - that's OK, just no inheritance
        return []


def discover_workspaces(ai_knowledge_root: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Discover all workspaces from ai-knowledge repos.

    Uses the full resolution chain (synthesis-console config, workspace glob,
    legacy flat parents). Inheritance is resolved from my-projects.yaml in the
    personal repo (per ADR-006), NOT from individual compile-config.yaml files.

    Args:
        ai_knowledge_root: Optional flat-parent override (legacy mode).

    Returns:
        List of workspace dictionaries.
    """
    ai_knowledge_repos = discover_ai_knowledge_repos(ai_knowledge_root)
    inheritance_config = _load_centralized_inheritance(ai_knowledge_root)

    discovered = []

    for workspace_name, content_info in ai_knowledge_repos.items():
        compile_config = content_info.get('config', {})
        project_config = compile_config.get('project', {})

        display_name = project_config.get('name', workspace_name)
        if display_name == workspace_name:
            display_name = workspace_name.replace('-', ' ').title()

        description = project_config.get('description', f'AI Knowledge workspace: {workspace_name}')

        # Get inheritance from centralized my-projects.yaml (NOT from compile-config.yaml)
        # This respects ADR-006: inheritance config lives only in personal repo
        inherits_from = _get_inheritance_for_workspace(workspace_name, inheritance_config)

        discovered.append({
            'name': display_name,
            'path': None,
            'dir_name': workspace_name,
            'config': {
                'name': display_name,
                'description': description,
                'status': 'active',
                'type': project_config.get('type', 'project'),
                'inherits_from': inherits_from,
            },
            'ai_knowledge': content_info
        })

    discovered.sort(key=lambda x: x['name'])
    return discovered


def resolve_workspace_paths(
    workspace: Dict[str, Any],
    all_workspaces: Optional[List[Dict[str, Any]]] = None,
    resolved_chain: Optional[set] = None
) -> Dict[str, List[str]]:
    """
    Resolve a workspace's content paths including inherited workspaces.

    Args:
        workspace: Workspace dictionary
        all_workspaces: List of all discovered workspaces (for inheritance)
        resolved_chain: Set of already resolved workspace names (prevents cycles)

    Returns:
        Dictionary with 'instructions', 'datasets', and 'archive' as lists
        of paths. 'archive' is only ever populated in owner context (see
        _build_repo_metadata's has_archive gate) — in team/default context
        it is always an empty list.
    """
    if resolved_chain is None:
        resolved_chain = set()

    config = workspace.get('config', {})
    workspace_name = workspace['dir_name']

    if workspace_name in resolved_chain:
        return {'instructions': [], 'datasets': [], 'archive': []}
    resolved_chain.add(workspace_name)

    instructions = []
    datasets = []
    archive = []

    # Resolve inherited workspaces first
    inherits_from = config.get('inherits_from', [])
    if all_workspaces and inherits_from:
        for parent_name in inherits_from:
            parent_workspace = next(
                (w for w in all_workspaces if w['dir_name'] == parent_name),
                None
            )
            if parent_workspace:
                parent_paths = resolve_workspace_paths(
                    parent_workspace, all_workspaces, resolved_chain.copy()
                )
                instructions.extend(parent_paths['instructions'])
                datasets.extend(parent_paths['datasets'])
                archive.extend(parent_paths.get('archive', []))

    # Add ai-knowledge compiled content
    ai_knowledge = workspace.get('ai_knowledge', {})
    if ai_knowledge:
        ai_instructions = ai_knowledge.get('instructions')
        ai_datasets = ai_knowledge.get('datasets')
        ai_archive = ai_knowledge.get('archive')

        if ai_instructions and os.path.isdir(ai_instructions):
            instructions.append(ai_instructions)
        if ai_datasets and os.path.isdir(ai_datasets):
            datasets.append(ai_datasets)
        if ai_archive and os.path.isdir(ai_archive):
            archive.append(ai_archive)

        # Fallback to source content
        source_path = ai_knowledge.get('source_path')
        if source_path and os.path.isdir(source_path):
            if not ai_instructions or not os.path.isdir(ai_instructions):
                source_instructions = os.path.join(source_path, 'instructions')
                if os.path.isdir(source_instructions):
                    instructions.append(source_instructions)
                source_runbooks = os.path.join(source_path, 'runbooks')
                if os.path.isdir(source_runbooks):
                    instructions.append(source_runbooks)

            if not ai_datasets or not os.path.isdir(ai_datasets):
                source_datasets = os.path.join(source_path, 'datasets')
                if os.path.isdir(source_datasets):
                    datasets.append(source_datasets)

    return {'instructions': instructions, 'datasets': datasets, 'archive': archive}


def workspace_to_profile(
    workspace: Dict[str, Any],
    all_workspaces: Optional[List[Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """
    Convert a workspace to profile format.

    Args:
        workspace: Workspace dictionary
        all_workspaces: List of all workspaces for inheritance resolution

    Returns:
        Profile dictionary
    """
    resolved = resolve_workspace_paths(workspace, all_workspaces)

    return {
        'name': workspace['name'],
        'dir_name': workspace['dir_name'],
        'instructions': resolved['instructions'],
        'datasets': resolved['datasets'],
        'archive': resolved.get('archive', []),
        'ai_knowledge': workspace.get('ai_knowledge', {})
    }


def load_workspaces_as_profiles(ai_knowledge_root: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Discover workspaces and convert them to profiles format.

    This is the main entry point for workspace-based profile loading.

    Args:
        ai_knowledge_root: Optional root directory for ai-knowledge repos

    Returns:
        List of profiles
    """
    workspaces = discover_workspaces(ai_knowledge_root)
    profiles = []

    for workspace in workspaces:
        profile = workspace_to_profile(workspace, workspaces)
        profiles.append(profile)

    # Add "none" option
    profiles.append({
        'name': '(none - no workspace)',
        'dir_name': '',
        'instructions': [],
        'datasets': [],
        'archive': []
    })

    return profiles


def get_workspace(name: str, ai_knowledge_root: Optional[str] = None) -> Dict[str, Any]:
    """
    Get a specific workspace by name or dir_name.

    Args:
        name: Workspace name or dir_name
        ai_knowledge_root: Optional root directory

    Returns:
        Workspace dictionary

    Raises:
        WorkspaceNotFoundError: If workspace not found
    """
    workspaces = discover_workspaces(ai_knowledge_root)

    for ws in workspaces:
        if ws['name'] == name or ws['dir_name'] == name:
            return workspace_to_profile(ws, workspaces)

    raise WorkspaceNotFoundError(f"Workspace not found: {name}")


def get_workspace_info(workspace: Dict[str, Any]) -> WorkspaceInfo:
    """
    Convert workspace dict to WorkspaceInfo Pydantic model.

    Args:
        workspace: Workspace dictionary

    Returns:
        WorkspaceInfo model
    """
    config = workspace.get('config', {})
    ai_knowledge = workspace.get('ai_knowledge', {})

    return WorkspaceInfo(
        name=workspace['name'],
        dir_name=workspace['dir_name'],
        description=config.get('description'),
        status=config.get('status', 'active'),
        type=config.get('type', 'project'),
        inherits_from=config.get('inherits_from', []),
        has_instructions=ai_knowledge.get('has_instructions', False),
        has_datasets=ai_knowledge.get('has_datasets', False),
        has_archive=ai_knowledge.get('has_archive', False),
        has_source=ai_knowledge.get('has_source', False),
        repo_path=ai_knowledge.get('repo_path'),
    )


def list_workspace_info(ai_knowledge_root: Optional[str] = None) -> List[WorkspaceInfo]:
    """
    Get list of all workspaces as WorkspaceInfo models.

    Args:
        ai_knowledge_root: Optional root directory

    Returns:
        List of WorkspaceInfo models
    """
    workspaces = discover_workspaces(ai_knowledge_root)
    return [get_workspace_info(ws) for ws in workspaces]


# Mapping from engine name to instruction file
ENGINE_TO_INSTRUCTION_FILE = {
    'anthropic': 'claude.md',
    'openai': 'chatgpt.md',
    'google': 'gemini.md',
}

# Fallback order if preferred file doesn't exist
INSTRUCTION_FALLBACK_ORDER = ['claude.md', 'chatgpt.md', 'gemini.md']


def get_llm_specific_instruction_path(
    workspace_name: str,
    engine: str = 'anthropic',
    ai_knowledge_root: Optional[str] = None
) -> Optional[str]:
    """
    Get the path to the LLM-specific compiled instruction file for a workspace.

    The compiler generates separate instruction files for each LLM platform:
    - claude.md for Anthropic models
    - chatgpt.md for OpenAI models
    - gemini.md for Google Gemini models

    Args:
        workspace_name: Name of the workspace (e.g., 'personal', 'example-client')
        engine: LLM engine name ('anthropic', 'openai', 'google')
        ai_knowledge_root: Optional flat-parent override (legacy mode).

    Returns:
        Path to the instruction file, or None if not found.
    """
    index = resolve_repo_index(ai_knowledge_root)
    repo_path = index.get(workspace_name)
    if not repo_path:
        return None

    # Compiled subdirectory is keyed by the bare workspace name.
    bare_name = _bare_workspace_name(workspace_name)
    instructions_dir = os.path.join(repo_path, "compiled", bare_name, "instructions")

    if not os.path.isdir(instructions_dir):
        return None

    # Get preferred instruction file for this engine
    preferred_file = ENGINE_TO_INSTRUCTION_FILE.get(engine, 'claude.md')
    preferred_path = os.path.join(instructions_dir, preferred_file)

    if os.path.isfile(preferred_path):
        return preferred_path

    # Try fallbacks
    for fallback in INSTRUCTION_FALLBACK_ORDER:
        fallback_path = os.path.join(instructions_dir, fallback)
        if os.path.isfile(fallback_path):
            return fallback_path

    return None
