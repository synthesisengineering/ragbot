#!/usr/bin/env python3
# ragbot.py - https://github.com/synthesisengineering/ragbot

import os
import sys
import argparse
import re
import json
import appdirs
import openai
import anthropic
import litellm
from helpers import load_files, load_config, print_saved_files, chat, load_workspaces_as_profiles
from synthesis_engine.config import normalize_model_id
from synthesis_engine.keystore import get_api_key
from synthesis_engine.workspaces import get_llm_specific_instruction_path, ENGINE_TO_INSTRUCTION_FILE

appname = "ragbot"
appauthor = "Rajiv Pant"

data_dir = appdirs.user_data_dir(appname, appauthor)
sessions_data_dir = os.path.join(data_dir, "sessions")

# Load configuration from engines.yaml
config = load_config('engines.yaml')
engines_config = {engine['name']: engine for engine in config['engines']}
engine_choices = list(engines_config.keys())
default_models = {engine: engines_config[engine]['default_model'] for engine in engine_choices}

model_cost_map = litellm.model_cost


def create_chat_parser(subparsers):
    """Create the chat subcommand parser."""
    chat_parser = subparsers.add_parser(
        'chat',
        help='Chat with an AI assistant (default command)',
        description='Chat with an AI assistant using various models and configurations.'
    )

    input_group = chat_parser.add_mutually_exclusive_group()
    input_group.add_argument(
        "-ls", "--list-saved",
        action="store_true",
        help="List all the currently saved JSON files."
    )

    input_group2 = chat_parser.add_mutually_exclusive_group()
    input_group2.add_argument(
        "-p", "--prompt",
        help="The user's input to generate a response for."
    )
    input_group2.add_argument(
        "-f", "--prompt_file",
        help="The file containing the user's input to generate a response for."
    )
    input_group2.add_argument(
        "-i", "--interactive",
        action="store_true",
        help="Enable interactive assistant chatbot mode."
    )
    input_group2.add_argument(
        "--stdin",
        action="store_true",
        help="Read the user's input from stdin."
    )

    chat_parser.add_argument(
        "-profile", "--profile",
        help="Name of the profile to use."
    )
    chat_parser.add_argument(
        "-c", "--custom_instructions",
        nargs='*', default=[],
        help="Path to the prompt custom instructions file or folder. Can accept multiple values."
    )
    chat_parser.add_argument(
        "-nc", "--nocustom_instructions",
        action="store_true",
        help="Ignore all prompt custom instructions even if they are specified."
    )
    chat_parser.add_argument(
        "--rag",
        action="store_true",
        default=True,
        help="Enable RAG (Retrieval-Augmented Generation) for knowledge retrieval. Default: enabled."
    )
    chat_parser.add_argument(
        "--no-rag",
        action="store_true",
        help="Disable RAG - use instructions only, no knowledge retrieval."
    )
    chat_parser.add_argument(
        "-e", "--engine",
        default=config.get('default', 'openai'),
        choices=engine_choices,
        help="The engine to use for the chat."
    )
    chat_parser.add_argument(
        "-m", "--model",
        help="The model to use for the chat. Defaults to engine's default model. Use 'flagship' to select the engine's most powerful model."
    )
    chat_parser.add_argument(
        "-t", "--temperature",
        type=float, default=None,
        help="The creativity of the response, with higher values being more creative."
    )
    chat_parser.add_argument(
        "-mt", "--max_tokens",
        type=int, default=None,
        help="The maximum number of tokens to generate in the response."
    )
    chat_parser.add_argument(
        "-l", "--load",
        help="Load a previous interactive session from a file."
    )

    # Reasoning / thinking effort. Reads from RAGBOT_THINKING_EFFORT env when
    # unset; per-call value wins. Models that don't advertise thinking
    # silently ignore the value.
    chat_parser.add_argument(
        "--thinking-effort",
        choices=["auto", "off", "minimal", "low", "medium", "high"],
        default=None,
        help="Reasoning effort for the LLM call. Defaults: flagship → medium, "
             "non-flagship → off, models without thinking metadata → ignored. "
             "Override via this flag or RAGBOT_THINKING_EFFORT env var."
    )

    # Cross-workspace search controls.
    chat_parser.add_argument(
        "--workspace", "-w",
        action="append",
        default=None,
        dest="extra_workspaces",
        help="Additional workspace to query alongside the primary one. "
             "Repeatable. When unset, the canonical 'skills' workspace is "
             "auto-included if it has indexed content."
    )
    chat_parser.add_argument(
        "--no-skills",
        action="store_true",
        help="Disable auto-inclusion of the 'skills' workspace in retrieval."
    )

    chat_parser.set_defaults(func=run_chat)
    return chat_parser


def create_compile_parser(subparsers):
    """Create the compile subcommand parser.

    Compiles instructions for AI Knowledge repositories. Knowledge concatenation
    is handled by CI/CD (GitHub Actions), not this CLI. RAG indexing is handled
    by the 'index' subcommand.
    """
    compile_parser = subparsers.add_parser(
        'compile',
        help='Compile instructions for AI Knowledge repositories',
        description='Compile instructions from AI Knowledge repositories for various LLM platforms. '
                    'Knowledge concatenation is handled by CI/CD (GitHub Actions). '
                    'For RAG indexing, use "ragbot index".'
    )

    # Project selection - what to compile
    project_group = compile_parser.add_mutually_exclusive_group()
    project_group.add_argument(
        '--project', '-p',
        help='Project name to compile (e.g., company, client)'
    )
    project_group.add_argument(
        '--repo', '-r',
        help='Path to ai-knowledge-* repository to compile'
    )
    project_group.add_argument(
        '--all-with-inheritance',
        action='store_true',
        help='Compile ALL projects with inheritance into your private repo. '
             'Output goes to ai-knowledge-{user}/compiled/{project}/ for each project.'
    )

    # Output destination - where to put compiled output
    compile_parser.add_argument(
        '--output-repo', '-o',
        help='Path to repo where compiled output should go. Required with --project when using --with-inheritance. '
             'Example: --project {client} --with-inheritance --output-repo ~/ai-knowledge-{person}'
    )

    # Compilation options
    compile_parser.add_argument(
        '--with-inheritance',
        action='store_true',
        help='Include inherited content from parent repos (per my-projects.yaml)'
    )
    compile_parser.add_argument(
        '--llm',
        choices=['claude', 'chatgpt', 'gemini', 'all'],
        default='all',
        help='Target LLM platform (default: all)'
    )

    # Behavior options
    compile_parser.add_argument(
        '--force', '-f',
        action='store_true',
        help='Force recompilation, ignore cache'
    )
    compile_parser.add_argument(
        '--no-llm',
        action='store_true',
        help='Skip LLM compilation, just assemble content'
    )
    compile_parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Verbose output with token counts'
    )
    compile_parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Minimal output'
    )

    # Path options
    compile_parser.add_argument(
        '--base-path',
        default=os.environ.get('RAGBOT_BASE_PATH'),
        help='Optional flat-parent path containing ai-knowledge-* repos. '
             'When unset, repos are discovered via ~/.synthesis/console.yaml '
             'and the workspace-rooted layout.'
    )
    compile_parser.add_argument(
        '--personal-repo',
        help='Path to personal ai-knowledge repo (for inheritance config)'
    )

    compile_parser.set_defaults(func=run_compile)
    return compile_parser


def create_index_parser(subparsers):
    """Create the index subcommand parser.

    Indexes AI Knowledge content into the pgvector store for RAG retrieval.
    Reads source files directly — no intermediate compiled files needed.
    """
    index_parser = subparsers.add_parser(
        'index',
        help='Index AI Knowledge content into vector store for RAG',
        description='Index AI Knowledge content into the pgvector store for RAG retrieval. '
                    'Reads source files directly from ai-knowledge repositories.'
    )

    index_parser.add_argument(
        '--workspace', '-w',
        required=True,
        help='Workspace name to index (e.g., personal, example-client)'
    )
    index_parser.add_argument(
        '--force', '-f',
        action='store_true',
        help='Clear existing index and rebuild from scratch'
    )
    index_parser.add_argument(
        '--tier',
        choices=['datasets', 'archive'],
        default='datasets',
        help="Retrieval tier to index. 'datasets' (default) indexes "
             "source/datasets/ as before. 'archive' indexes personal-archive/ "
             "content into the 'archive-only' tier, which is excluded from "
             "default retrieval and reachable only via rag.search_archive(). "
             "Archive indexing requires RAGBOT_OWNER_CONTEXT=1 — the same "
             "flag that already gates private-repo discovery."
    )
    index_parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Verbose output'
    )

    index_parser.set_defaults(func=run_index)
    return index_parser


def create_skills_parser(subparsers):
    """Create the `skills` subcommand parser.

    List, inspect, index, and run Agent Skills (directories with SKILL.md).
    """
    skills_parser = subparsers.add_parser(
        'skills',
        help='Discover, inspect, index, and run Agent Skills',
        description='Manage Agent Skills (directories with SKILL.md) for RAG '
                    'indexing, inspection, and workspace-scoped execution. '
                    'Skills are discovered from ~/.synthesis/skills, '
                    '~/.claude/skills, plugin caches, and per-workspace '
                    'collections under ~/workspaces/<W>/synthesis-skills-<W>/.'
    )
    skills_subparsers = skills_parser.add_subparsers(dest='skills_command', required=True)

    list_parser = skills_subparsers.add_parser(
        'list',
        help='List discovered skills with their visibility scope.',
        description='List discovered skills. Without --workspace, every skill '
                    'is shown with its scope (universal vs workspace-scoped). '
                    'With --workspace, only skills visible from that workspace '
                    '(via inheritance chain) are shown.',
    )
    list_parser.add_argument('--verbose', '-v', action='store_true',
                             help='Include description and file counts (legacy verbose layout).')
    list_parser.add_argument('--workspace', '-w', default=None,
                             help='Filter to skills visible from this workspace '
                                  '(via the inheritance chain in my-projects.yaml).')
    list_parser.set_defaults(func=run_skills_list)

    info_parser = skills_subparsers.add_parser('info', help='Show details for one skill.')
    info_parser.add_argument('name', help='Skill name (directory or frontmatter name).')
    info_parser.set_defaults(func=run_skills_info)

    index_parser = skills_subparsers.add_parser(
        'index',
        help='Index discovered skills into a vector-store workspace (default: `skills`).',
    )
    index_parser.add_argument('--workspace', '-w', default='skills',
                              help='Target workspace name (default: skills).')
    index_parser.add_argument('--only', action='append', default=None,
                              help='Restrict to a specific skill name (repeatable).')
    index_parser.add_argument('--force', '-f', action='store_true',
                              help='Clear the target workspace before indexing.')
    index_parser.set_defaults(func=run_skills_index)

    run_parser = skills_subparsers.add_parser(
        'run',
        help='Activate a skill and dispatch its first declared tool (or body prompt).',
        description='Run a skill via the workspace-scoped loader. The skill '
                    'must be visible from --workspace (the inheritance chain '
                    'applies). The first declared tool is dispatched through '
                    'the agent loop with inputs supplied via --input KEY=VALUE '
                    'pairs and/or --file. When the skill declares no tools, '
                    'the SKILL.md body is sent to the LLM as a prompt.',
    )
    run_parser.add_argument('name', help='Skill name to activate and run.')
    run_parser.add_argument('--workspace', '-w', default=None,
                            help='Workspace to scope the skill set to. When '
                                 'omitted, the universal skill chain is used.')
    run_parser.add_argument('--input', '-i', action='append', default=None,
                            metavar='KEY=VALUE',
                            help='Tool input. Repeatable. Values are parsed as '
                                 'JSON when possible, else passed as strings.')
    run_parser.add_argument('--file', '-f', default=None,
                            help='Path to a file whose UTF-8 contents are '
                                 'bound to the "file" input.')
    run_parser.add_argument('--model', '-m', default=None,
                            help='Model id override (defaults to engines.yaml default).')
    run_parser.set_defaults(func=run_skills_run)

    return skills_parser


def create_agent_parser(subparsers):
    """Create the ``agent`` subgroup parser.

    Surfaces the agent loop's checkpoint store + replay machinery as a
    first-class CLI so an operator can debug a regression by rerunning a
    specific session from a specific checkpoint. The three commands —
    ``replay``, ``list-sessions``, and ``checkpoints`` — are
    inspection-only; they never start a fresh agent run.
    """
    agent_parser = subparsers.add_parser(
        'agent',
        help='Inspect and replay durable agent-loop sessions',
        description='Inspect and replay durable agent-loop sessions. '
                    'The checkpoint store under '
                    '$SYNTHESIS_AGENT_CHECKPOINT_DIR '
                    '(default ~/.synthesis/agent-checkpoints) holds one '
                    'JSON file per state transition. These commands let '
                    'an operator list recent task ids, inspect the '
                    'checkpoint stream for a task, and deterministically '
                    'replay a task from any checkpoint.'
    )
    agent_sub = agent_parser.add_subparsers(
        dest='agent_command', required=True,
    )

    # --- ragbot agent replay ----------------------------------------------
    replay = agent_sub.add_parser(
        'replay',
        help='Replay an agent session from a checkpoint and report the result.',
        description='Re-drive an agent session from a specific checkpoint '
                    'and print the final state plus a stable hash over '
                    '{current_state, final_answer, step_results} '
                    '(timestamps excluded). When --against-checkpoint M is '
                    'given, the replay state at step M is compared byte-'
                    'for-byte to the original checkpoint M; the verdict '
                    '(IDENTICAL / DIVERGENT) is the determinism check.',
    )
    replay.add_argument('task_id', help='Task id whose checkpoints to replay.')
    replay.add_argument(
        '--from-checkpoint', type=int, default=None, metavar='N',
        help='Checkpoint index to resume from. Defaults to the latest.',
    )
    replay.add_argument(
        '--against-checkpoint', type=int, default=None, metavar='M',
        help='Compare the replay state at step M against the original '
             'checkpoint M and report IDENTICAL or DIVERGENT.',
    )
    replay.add_argument(
        '--show-trace', action='store_true',
        help='Print the post-replay turn_history in addition to the summary.',
    )
    replay.add_argument(
        '--save-output', default=None, metavar='PATH',
        help='Write the full final-state JSON to PATH (pretty-printed).',
    )
    replay.set_defaults(func=run_agent_replay)

    # --- ragbot agent list-sessions ---------------------------------------
    listing = agent_sub.add_parser(
        'list-sessions',
        help='List recent agent sessions, most recent first.',
        description='List task ids in the checkpoint store, ordered by '
                    'the mtime of each task\'s most-recent checkpoint.',
    )
    listing.add_argument(
        '--limit', type=int, default=20,
        help='Maximum number of task ids to print (default: 20).',
    )
    listing.set_defaults(func=run_agent_list_sessions)

    # --- ragbot agent checkpoints -----------------------------------------
    cps = agent_sub.add_parser(
        'checkpoints',
        help='List the checkpoint indices for a task id with one-line summaries.',
        description='List every checkpoint index for the supplied task '
                    'id alongside the state name, iteration count, plan '
                    'length, and one-line synopsis pulled from the most '
                    'recent turn record.',
    )
    cps.add_argument('task_id', help='Task id whose checkpoints to inspect.')
    cps.set_defaults(func=run_agent_checkpoints)

    return agent_parser


def create_mcp_parser(subparsers):
    """Create the `mcp` subcommand parser.

    Exposes RagbotMCPServer (synthesis_engine.mcp_server) — the SERVER
    side of MCP, where Ragbot's own primitives (workspace_search,
    workspace_search_multi, document_get, skill_run, agent_run_start) are
    surfaced to an external MCP client such as Claude Code or Cursor. This
    is distinct from synthesis_engine.mcp, the CLIENT side (Ragbot calling
    OUT to other MCP servers), which is managed via the `/api/mcp/*`
    endpoints instead.

    Only the `serve` subcommand exists today (stdio transport, the
    transport Claude Code and other desktop MCP clients speak for local
    servers). See docs/mcp-server-activation.md for what a user needs to
    do to actually register this with a client — this command starts the
    server; it does not register itself anywhere.
    """
    mcp_parser = subparsers.add_parser(
        'mcp',
        help='Run Ragbot as an MCP server for other agents to call',
        description='Serve RagbotMCPServer, exposing workspace_search, '
                    'workspace_search_multi, document_get, skill_run, and '
                    'agent_run_start as MCP tools to an external client.'
    )
    mcp_subparsers = mcp_parser.add_subparsers(dest='mcp_command', required=True)

    serve_parser = mcp_subparsers.add_parser(
        'serve',
        help='Serve over stdio in the foreground (blocks until the client disconnects).',
    )
    serve_parser.set_defaults(func=run_mcp_serve)

    return mcp_parser


def _agent_checkpoint_store():
    """Construct the default FilesystemCheckpointStore the CLI inspects.

    Honours the same ``SYNTHESIS_AGENT_CHECKPOINT_DIR`` env var that
    :func:`synthesis_engine.agent.checkpoints._default_base_dir` reads,
    so the CLI always inspects the same on-disk layout the agent loop
    writes to.
    """
    from synthesis_engine.agent import FilesystemCheckpointStore
    return FilesystemCheckpointStore()


def _replay_hash(state) -> str:
    """Compute a stable hash over the timestamp-excluded state fields.

    The hash is sha256 over a JSON serialisation that drops every
    field whose value changes between two byte-equivalent runs of the
    same plan: ``turn_history[*].timestamp`` (wall-clock), and any
    other clock-derived value future code adds to metadata. Excluding
    those fields means two deterministic replays with identical fake
    substrates produce the same hash.
    """
    import hashlib

    data = state.to_dict() if hasattr(state, 'to_dict') else dict(state)
    relevant = {
        'current_state': data.get('current_state'),
        'final_answer': data.get('final_answer'),
        'step_results': data.get('step_results') or {},
    }
    encoded = json.dumps(relevant, sort_keys=True, default=str).encode('utf-8')
    return hashlib.sha256(encoded).hexdigest()


def _compare_state_at_checkpoint(replay_state, original_state) -> tuple:
    """Return (verdict, diff_keys) comparing two GraphState dicts.

    The verdict is the string ``"IDENTICAL"`` when the substantive
    fields match (current_state, final_answer, step_results,
    plan-summary), and ``"DIVERGENT"`` otherwise. ``diff_keys`` is the
    list of top-level fields that differ; useful for the CLI's stderr
    output without printing entire state blobs.
    """
    rhash = _replay_hash(replay_state)
    ohash = _replay_hash(original_state)
    if rhash == ohash:
        return "IDENTICAL", []

    r_data = (
        replay_state.to_dict()
        if hasattr(replay_state, 'to_dict')
        else dict(replay_state)
    )
    o_data = (
        original_state.to_dict()
        if hasattr(original_state, 'to_dict')
        else dict(original_state)
    )
    diffs = []
    for key in ('current_state', 'final_answer', 'step_results'):
        if r_data.get(key) != o_data.get(key):
            diffs.append(key)
    return "DIVERGENT", diffs


def run_agent_replay(args):
    """Replay a session from a checkpoint and report the result.

    The CLI never starts a fresh planner-led agent run; it loads a
    checkpoint, re-drives the loop with the same LLM backend the rest
    of Ragbot uses, and prints the final state plus a stable hash.
    Replay determinism depends on the substrates being deterministic;
    the production LLM backend is not deterministic by default, so
    operators typically pair this with a fake backend override in a
    test or with deterministic fixtures pinned via environment.
    """
    import asyncio

    from synthesis_engine.agent import AgentLoop, GraphState
    from synthesis_engine.llm import get_llm_backend

    store = _agent_checkpoint_store()
    indices = asyncio.run(store.list_checkpoints(args.task_id))
    if not indices:
        print(
            f"Error: no checkpoints found for task '{args.task_id}'.",
            file=sys.stderr,
        )
        return 1

    target_idx = args.from_checkpoint
    if target_idx is None:
        target_idx = indices[-1]
    elif target_idx not in indices:
        print(
            f"Error: checkpoint {target_idx} does not exist for task "
            f"'{args.task_id}'. Available: {indices}",
            file=sys.stderr,
        )
        return 1

    # Construct a loop wired to the configured LLM backend. The
    # checkpoint store points at the same on-disk layout we just read
    # from so the replay's transitions land in the same task directory.
    try:
        backend = get_llm_backend()
    except Exception as exc:
        print(
            f"Error: LLM backend not configured: {exc}",
            file=sys.stderr,
        )
        return 1
    loop = AgentLoop(
        llm_backend=backend,
        mcp_client=None,
        checkpoint_store=store,
        default_mcp_server="local",
    )

    try:
        final_state = asyncio.run(loop.replay(args.task_id, target_idx))
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"Error: replay failed: {exc}", file=sys.stderr)
        return 1

    hash_value = _replay_hash(final_state)
    print(f"Task:             {args.task_id}")
    print(f"Replayed from:    checkpoint {target_idx}")
    print(f"Final state:      {final_state.current_state.value}")
    print(f"Final answer:     {final_state.final_answer or '(none)'}")
    print(f"State hash:       {hash_value}")

    if args.against_checkpoint is not None:
        try:
            original = asyncio.run(
                store.load(args.task_id, args.against_checkpoint),
            )
        except FileNotFoundError as exc:
            print(
                f"Error: --against-checkpoint {args.against_checkpoint} "
                f"not found: {exc}",
                file=sys.stderr,
            )
            return 1
        verdict, diff_keys = _compare_state_at_checkpoint(
            final_state, original,
        )
        print(f"Determinism:      {verdict}")
        if verdict == "DIVERGENT":
            print(f"Divergent fields: {', '.join(diff_keys) or '(see hash)'}")

    if args.show_trace:
        print("\nTurn history:")
        for turn in final_state.turn_history:
            print(
                f"  iter={turn.iteration:>3} "
                f"state={turn.state.value:<11} {turn.summary}"
            )

    if args.save_output:
        try:
            payload = json.dumps(
                final_state.to_dict(), indent=2, default=str, sort_keys=False,
            )
            with open(args.save_output, 'w', encoding='utf-8') as fp:
                fp.write(payload)
            print(f"\nFull final state written to {args.save_output}")
        except OSError as exc:
            print(
                f"Error writing --save-output {args.save_output!r}: {exc}",
                file=sys.stderr,
            )
            return 1
    return 0


def run_agent_list_sessions(args):
    """Print recent task ids in mtime order, most recent first."""
    store = _agent_checkpoint_store()
    task_ids = store.list_recent_task_ids(limit=max(args.limit, 0))
    if not task_ids:
        print("No agent sessions found in the checkpoint store.")
        return 0

    print(f"Recent agent sessions ({len(task_ids)}):")
    print()
    print(f"{'TASK_ID':<40}  CHECKPOINTS")
    print("-" * 70)
    import asyncio
    for tid in task_ids:
        indices = asyncio.run(store.list_checkpoints(tid))
        count = len(indices)
        last = indices[-1] if indices else 0
        print(f"{tid:<40}  {count} (latest={last})")
    return 0


def run_agent_checkpoints(args):
    """Print every checkpoint index for ``task_id`` with a one-line summary."""
    import asyncio

    store = _agent_checkpoint_store()
    indices = asyncio.run(store.list_checkpoints(args.task_id))
    if not indices:
        print(
            f"No checkpoints found for task '{args.task_id}'.",
            file=sys.stderr,
        )
        return 1

    print(f"Task: {args.task_id}")
    print(f"Checkpoints: {len(indices)}")
    print()
    print(f"{'IDX':>4}  {'STATE':<12}  {'ITER':>4}  {'PLAN':>4}  SUMMARY")
    print("-" * 78)
    for idx in indices:
        try:
            summary = store.summarise_checkpoint(args.task_id, idx)
        except Exception as exc:
            print(f"{idx:>4}  (unreadable: {exc})")
            continue
        synopsis = summary.get("summary") or ""
        if len(synopsis) > 50:
            synopsis = synopsis[:47] + "..."
        print(
            f"{idx:>4}  {summary['state']:<12}  "
            f"{summary['iteration_count']:>4}  "
            f"{summary['plan_step_count']:>4}  {synopsis}"
        )
    return 0


def create_db_parser(subparsers):
    """Create the db subcommand parser.

    Diagnostics and maintenance for the configured vector store backend.
    """
    db_parser = subparsers.add_parser(
        'db',
        help='Vector store backend diagnostics and maintenance',
        description='Inspect and maintain the pgvector store. Use '
                    '`ragbot db status` to verify connectivity, list '
                    'collections, and confirm migrations are applied.'
    )

    db_subparsers = db_parser.add_subparsers(dest='db_command', required=True)

    status_parser = db_subparsers.add_parser(
        'status',
        help='Show backend health, configured connection, and indexed collections.'
    )
    status_parser.set_defaults(func=run_db_status)

    init_parser = db_subparsers.add_parser(
        'init',
        help='Apply schema migrations (no-op if already applied). Pgvector only.'
    )
    init_parser.set_defaults(func=run_db_init)

    return db_parser


def create_memory_parser(subparsers):
    """Create the `memory` subcommand parser.

    Surfaces the scheduled memory consolidator on the CLI. Two
    sub-commands:

    * ``ragbot memory consolidate`` — invoke the consolidator inline.
    * ``ragbot memory consolidation-history`` — tail recent runs from
      the audit log.

    The consolidator is the "dreaming" pass: it reads a session's
    payload, distils durable facts via an LLM extractor, and writes
    them into the entity graph with provenance pointing back at the
    session and the model id that did the work.
    """
    memory_parser = subparsers.add_parser(
        'memory',
        help='Memory consolidation and history',
        description=(
            'Run the between-session consolidation pass (the "dreaming" '
            'pattern) and inspect its audit history.'
        ),
    )

    memory_subparsers = memory_parser.add_subparsers(
        dest='memory_command', required=True
    )

    consolidate_parser = memory_subparsers.add_parser(
        'consolidate',
        help='Distil sessions into the entity graph.',
        description=(
            'Consolidate one or more sessions into the workspace entity '
            'graph. Filter by session id, by time window, or by idle '
            'threshold; the default is "--idle-hours 4" when no other '
            'filter is supplied.'
        ),
    )
    consolidate_parser.add_argument(
        '--session-id',
        default=None,
        help='Consolidate exactly this session id (single-session mode).',
    )
    consolidate_parser.add_argument(
        '--since',
        default=None,
        help='ISO-8601 lower bound on session checkpoint mtime (inclusive).',
    )
    consolidate_parser.add_argument(
        '--until',
        default=None,
        help='ISO-8601 upper bound on session checkpoint mtime (inclusive).',
    )
    consolidate_parser.add_argument(
        '--idle-hours',
        type=float,
        default=None,
        help=(
            'Consolidate every session whose latest checkpoint is at '
            'least this many hours old. Default 4.0 when no other '
            'filter is supplied.'
        ),
    )
    consolidate_parser.add_argument(
        '--model',
        default=None,
        help='LLM model id for the extractor (engines.yaml id).',
    )
    consolidate_parser.add_argument(
        '--workspace', '-w',
        default=None,
        help='Workspace to scope writes to. Defaults to the session workspace.',
    )
    consolidate_parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Compute counts without writing to the entity graph.',
    )
    consolidate_parser.set_defaults(func=run_memory_consolidate)

    history_parser = memory_subparsers.add_parser(
        'consolidation-history',
        help='Tail recent memory_consolidation audit entries.',
    )
    history_parser.add_argument(
        '--limit',
        type=int,
        default=20,
        help='Maximum entries to print (default 20).',
    )
    history_parser.set_defaults(func=run_memory_consolidation_history)

    return memory_parser


def run_chat(args):
    """Run the chat command."""
    if args.list_saved:
        print_saved_files(data_dir)
        return

    new_session = True if not args.load else False

    # Workspace discovery for the chat command:
    #   - Honour RAGBOT_DATA_ROOT (legacy flat-parent override) when set.
    #   - Honour RAGBOT_BASE_PATH (the canonical override used elsewhere).
    #   - Otherwise fall through to the full discovery chain
    #     (~/.synthesis/console.yaml, ~/workspaces/*/ai-knowledge-*, legacy fallbacks).
    data_root = os.getenv('RAGBOT_DATA_ROOT') or os.getenv('RAGBOT_BASE_PATH')
    profiles = load_workspaces_as_profiles(data_root)

    workspace_name = None
    workspace_dir_name = None

    if args.profile:
        selected_profile_data = next((profile for profile in profiles if profile['name'] == args.profile or profile.get('dir_name') == args.profile), None)
        if not selected_profile_data:
            available_workspaces = [p['name'] for p in profiles if p['name'] != '(none - no workspace)']
            print(f"Error: Workspace '{args.profile}' not found.")
            print(f"Available workspaces: {', '.join(available_workspaces)}")
            sys.exit(1)
        workspace_name = args.profile
        workspace_dir_name = selected_profile_data.get('dir_name', args.profile)

    # Handle custom instructions
    # If user provides explicit -c paths, load them manually
    # Otherwise, let core.py auto-load LLM-specific instructions based on model
    custom_instructions = ""
    auto_load_instructions = True  # Let core.py handle it

    if not args.nocustom_instructions:
        if args.custom_instructions:
            # User provided explicit instruction paths - use them instead of auto-loading
            custom_instruction_paths = [p for p in args.custom_instructions if p.strip() != '']
            if custom_instruction_paths:
                custom_instructions, custom_instructions_files = load_files(
                    file_paths=custom_instruction_paths, file_type="custom_instructions"
                )
                auto_load_instructions = False  # Don't auto-load, user provided explicit instructions
                print("Custom instructions being used:")
                for file in custom_instructions_files:
                    print(f" - {file}")

    # Show what will be auto-loaded if no explicit instructions provided
    if auto_load_instructions and workspace_dir_name:
        llm_instruction_file = get_llm_specific_instruction_path(workspace_dir_name, args.engine)
        if llm_instruction_file:
            llm_name = ENGINE_TO_INSTRUCTION_FILE.get(args.engine, 'claude.md').replace('.md', '')
            print(f"LLM-specific instructions will be auto-loaded for {args.engine} engine")
            print(f" - {llm_instruction_file}")
        else:
            print("No custom instructions files available.")
    elif auto_load_instructions:
        print("No custom instructions files available.")

    # Determine RAG usage
    use_rag = args.rag and not args.no_rag
    if use_rag and workspace_dir_name:
        print(f"RAG enabled for workspace: {workspace_dir_name}")
    elif use_rag:
        print("RAG enabled (no workspace selected - RAG requires a workspace)")
        use_rag = False
    else:
        print("RAG disabled - using instructions only")

    # Resolve additional workspaces (cross-workspace retrieval).
    #   --workspace W (repeatable): explicit list (auto-include disabled).
    #   --no-skills:                  explicit opt-out of skills auto-include.
    #   neither:                      auto-include policy applies in get_relevant_context.
    extra_workspaces_arg = getattr(args, 'extra_workspaces', None)
    no_skills = getattr(args, 'no_skills', False)
    if extra_workspaces_arg is not None:
        # Explicit list given; respect --no-skills filter.
        additional_workspaces = [w for w in extra_workspaces_arg if w and (not no_skills or w != 'skills')]
    elif no_skills:
        # User opts out without specifying alternates.
        additional_workspaces = []
    else:
        additional_workspaces = None  # let get_relevant_context auto-include skills

    thinking_effort = getattr(args, 'thinking_effort', None)

    history = []

    if args.load:
        filename = args.load.strip()
        full_path = os.path.join(sessions_data_dir, filename)
        with open(full_path, 'r') as f:
            history = json.load(f)
        print(f"Continuing previously saved session from file: {filename}")

    model = args.model
    if model is None:
        model = default_models[args.engine]
    elif model == "flagship":
        flagship_model = next(
            (m for m in engines_config[args.engine]['models'] if m.get('is_flagship')),
            None
        )
        if flagship_model:
            model = flagship_model['name']
        else:
            print(f"Warning: No flagship model defined for engine '{args.engine}'. Using default.")
            model = default_models[args.engine]

    api_key = get_api_key(args.engine)
    if api_key:
        engines_config[args.engine]['api_key'] = api_key

        if args.engine == 'openai':
            openai.api_key = api_key
        elif args.engine == 'anthropic':
            anthropic.api_key = api_key
        elif args.engine == 'google':
            os.environ['GEMINI_API_KEY'] = api_key

    selected_model = next((item for item in engines_config[args.engine]['models'] if item['name'] == model), None)

    if model in model_cost_map:
        model_data = model_cost_map[model]
    else:
        model_data = {}

    if selected_model:
        default_temperature = selected_model.get("temperature", 0.75)
        max_output_tokens = selected_model.get("max_output_tokens") or model_data.get("max_output_tokens") or selected_model.get("default_max_tokens") or 4096
        default_max_tokens = selected_model.get("default_max_tokens", min(max_output_tokens, 4096))
        max_input_tokens = selected_model.get("max_input_tokens") or model_data.get("max_input_tokens") or 128000
    else:
        default_temperature = 0.75
        max_output_tokens = 4096
        default_max_tokens = 4096
        max_input_tokens = 128000

    max_tokens = args.max_tokens or default_max_tokens
    temperature = args.temperature or default_temperature

    if max_tokens > max_output_tokens:
        print(f"Warning: Requested max_tokens ({max_tokens}) exceeds model's maximum output limit ({max_output_tokens})")
        print(f"Setting max_tokens to {max_output_tokens}")
        max_tokens = max_output_tokens

    supports_system_role = selected_model.get('supports_system_role', True)

    print(f"Using AI engine {args.engine} with model {model}")
    print(f"Creativity temperature setting: {temperature}")
    print(f"Max tokens setting: {max_tokens} (model max output: {max_output_tokens})")

    # Convert engine/model to litellm format via the shared normalizer.
    litellm_model = normalize_model_id(args.engine, model)

    # Stream callback for interactive mode
    def print_chunk(chunk):
        print(chunk, end='', flush=True)

    if args.interactive:
        print("Entering interactive mode. Conversation history is maintained between turns.")
        while True:
            prompt = input("\nEnter prompt below. /quit to exit or /save file_name.json to save conversation.\n> ")
            if prompt.lower() == "/quit":
                break
            elif prompt.lower().startswith("/save "):
                filename = prompt[6:].strip()
                full_path = os.path.join(sessions_data_dir, filename)
                os.makedirs(os.path.dirname(full_path), exist_ok=True)
                with open(full_path, 'w') as f:
                    json.dump(history, f)
                print(f"Conversation saved to {full_path}")
                continue
            history.append({"role": "user", "content": prompt})
            print("Ragbot.AI: ", end='', flush=True)
            reply = chat(
                prompt=prompt,
                custom_instructions=custom_instructions,
                model=litellm_model,
                max_tokens=max_tokens,
                max_input_tokens=max_input_tokens,
                temperature=temperature,
                history=history,
                supports_system_role=supports_system_role,
                stream=True,
                stream_callback=print_chunk,
                workspace_name=workspace_dir_name,
                use_rag=use_rag,
                auto_load_instructions=auto_load_instructions,
                thinking_effort=thinking_effort,
                additional_workspaces=additional_workspaces,
            )
            print()  # Newline after streaming
            history.append({"role": "assistant", "content": reply})
    else:
        prompt = None
        if args.prompt:
            prompt = args.prompt
        elif args.prompt_file:
            with open(args.prompt_file, 'r') as f:
                prompt = f.read().strip()
        elif args.stdin:
            stdin = sys.stdin.readlines()
            if stdin:
                prompt = "".join(stdin).strip()

        if prompt is None:
            print("Error: No prompt provided. Please provide a prompt using -p, -f, or -i option.")
            sys.exit(1)

        history.append({"role": "user", "content": prompt})
        reply = chat(
            prompt=prompt,
            custom_instructions=custom_instructions,
            model=litellm_model,
            max_tokens=max_tokens,
            max_input_tokens=max_input_tokens,
            temperature=temperature,
            history=history,
            supports_system_role=supports_system_role,
            stream=False,
            workspace_name=workspace_dir_name,
            use_rag=use_rag,
            auto_load_instructions=auto_load_instructions
        )
        pattern = re.compile(r"OUTPUT ?= ?\"\"\"((\n|.)*?)\"\"\"", re.MULTILINE)
        is_structured = pattern.search(reply)
        if is_structured:
            reply = is_structured[1].strip()
        print(reply)


def _resolve_workspace_for_skills(workspace_arg):
    """Resolve the effective workspace name for skill filtering.

    Honours the explicit ``--workspace`` flag first, then falls back to the
    demo workspace name when RAGBOT_DEMO=1 is set, then None. None means
    "show every skill regardless of scope" — the unfiltered list path.
    """
    if workspace_arg:
        return workspace_arg
    if os.environ.get('RAGBOT_DEMO', '').strip().lower() in ('1', 'true', 'yes', 'on'):
        from ragbot.demo import DEMO_WORKSPACE_NAME
        return DEMO_WORKSPACE_NAME
    return None


def _format_scope_tag(scope):
    """Render a SkillScope as a short tag for tabular output.

    ``universal`` for skills visible everywhere, ``workspace:<name>`` for
    single-workspace scopes, ``workspaces:<a,b>`` for multi-workspace scopes.
    """
    if scope.universal:
        return "universal"
    if len(scope.workspaces) == 1:
        return f"workspace:{scope.workspaces[0]}"
    return f"workspaces:{','.join(scope.workspaces)}"


def run_skills_list(args):
    """Print discovered skills.

    When ``--workspace`` is supplied, results are filtered through
    ``get_skills_for_workspace`` so only skills visible from that workspace
    (via the inheritance chain) appear. When omitted, every skill is shown
    with its scope tag so the operator can see which skills are workspace-
    restricted and which are universal.
    """
    from synthesis_engine.skills import discover_skills, get_skills_for_workspace

    workspace = _resolve_workspace_for_skills(getattr(args, 'workspace', None))
    if workspace is not None:
        skills = get_skills_for_workspace(workspace)
        scope_header = f"visible from workspace '{workspace}'"
    else:
        skills = discover_skills()
        scope_header = "all workspaces"

    if not skills:
        if workspace is not None:
            print(f"No skills visible from workspace '{workspace}'. "
                  f"Searched ~/.synthesis/skills, ~/.claude/skills, plugin caches, "
                  f"and per-workspace collections.")
        else:
            print("No skills discovered. Searched ~/.synthesis/skills, "
                  "~/.claude/skills, plugin caches, and per-workspace collections.")
        return 0

    # Legacy verbose layout (preserved for back-compat with the previous
    # `ragbot skills list -v` output shape that tests may pin to).
    if getattr(args, 'verbose', False):
        print(f"Discovered {len(skills)} skills ({scope_header}):")
        for s in skills:
            desc_short = (s.description or '').strip().replace('\n', ' ')[:80]
            extras = []
            if s.references:
                extras.append(f"{len(s.references)} refs")
            if s.scripts:
                extras.append(f"{len(s.scripts)} scripts")
            if s.other_files:
                extras.append(f"{len(s.other_files)} other")
            extras_str = f" ({', '.join(extras)})" if extras else ""
            version_str = f" v{s.version}" if s.version else ""
            scope_tag = _format_scope_tag(s.scope)
            print(f"  {s.name}{version_str} [{scope_tag}]{extras_str}")
            if desc_short:
                print(f"    {desc_short}")
        return 0

    # Tabular default layout: name, scope, description (truncated to 70),
    # and the on-disk source path. Columns are width-fitted to the longest
    # entry so the output stays scannable across narrow and wide terminals.
    rows = []
    for s in skills:
        desc = (s.description or '').strip().replace('\n', ' ')
        if len(desc) > 70:
            desc = desc[:67] + "..."
        rows.append((s.name, _format_scope_tag(s.scope), desc, s.path))

    name_w = max(len("NAME"), max(len(r[0]) for r in rows))
    scope_w = max(len("SCOPE"), max(len(r[1]) for r in rows))
    desc_w = max(len("DESCRIPTION"), max(len(r[2]) for r in rows))

    print(f"Discovered {len(skills)} skills ({scope_header}):")
    print()
    header = f"{'NAME':<{name_w}}  {'SCOPE':<{scope_w}}  {'DESCRIPTION':<{desc_w}}  SOURCE"
    print(header)
    print("-" * len(header))
    for name, scope, desc, path in rows:
        print(f"{name:<{name_w}}  {scope:<{scope_w}}  {desc:<{desc_w}}  {path}")
    return 0


def run_skills_info(args):
    """Show full details for one skill."""
    from synthesis_engine.skills import discover_skills

    target = args.name
    skills = discover_skills()
    skill = next((s for s in skills if s.name == target), None)
    if skill is None:
        print(f"Skill not found: {target}", file=sys.stderr)
        print(f"Run `ragbot skills list` to see all discovered skills.", file=sys.stderr)
        return 1

    print(f"Name:        {skill.name}")
    print(f"Path:        {skill.path}")
    print(f"Version:     {skill.version or '(unset)'}")
    print(f"Description: {skill.description or '(none)'}")
    print(f"Triggers:    {', '.join(skill.triggers) if skill.triggers else '(none)'}")
    print(f"Files:       {len(skill.files)}")
    print()
    for f in skill.files:
        chars = len(f.content) if f.is_text else 0
        print(f"  {f.kind.value:18s} {f.relative_path:50s} {chars} chars")
    return 0


def run_skills_index(args):
    """Index discovered skills into a vector-store workspace."""
    from rag import index_skills, get_index_status

    result = index_skills(
        workspace_name=args.workspace,
        only=args.only,
        force=args.force,
    )
    if 'error' in result:
        print(f"Error: {result['error']}", file=sys.stderr)
        return 1

    print(f"Backend:        {result.get('backend', '?')}")
    print(f"Workspace:      {result.get('workspace', '?')}")
    print(f"Skills indexed: {result.get('skills_indexed', 0)}")
    print(f"Chunks indexed: {result.get('chunks_indexed', 0)}")
    skipped = result.get('skipped') or []
    if skipped:
        print(f"Skipped (no indexable text): {', '.join(skipped)}")
    indexed_total, count = get_index_status(args.workspace)
    print(f"Workspace now contains {count} chunks (indexed={indexed_total}).")
    return 0


def _parse_input_kv_pairs(pairs):
    """Parse ``--input KEY=VALUE`` pairs into a dict.

    Values are JSON-decoded when possible so callers can pass numbers,
    lists, and booleans without quoting gymnastics. A bare token that
    fails JSON parsing is preserved as a plain string — the common case
    of ``--input topic=climate change`` keeps working.
    """
    result = {}
    for raw in pairs or []:
        if '=' not in raw:
            raise ValueError(
                f"Invalid --input value {raw!r}; expected KEY=VALUE.")
        key, value = raw.split('=', 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Invalid --input value {raw!r}; key is empty.")
        try:
            result[key] = json.loads(value)
        except (ValueError, json.JSONDecodeError):
            result[key] = value
    return result


def run_skills_run(args):
    """Activate a workspace-scoped skill and dispatch its first tool.

    Resolution order:

    1. Filter skills via ``get_skills_for_workspace(workspace)`` when
       --workspace (or RAGBOT_DEMO) supplies one; otherwise the
       universal-only view from ``discover_skills``.
    2. Refuse with a clear error when the named skill is not in the
       filtered set — this is the workspace-visibility contract.
    3. Activate via :class:`SkillLoader`. If the skill declares one or
       more tools, dispatch the first tool through the agent loop with
       the user's inputs. Otherwise send the SKILL.md body as the prompt.

    The agent loop is wired to the configured LLM backend
    (``synthesis_engine.llm.get_llm_backend()``) and a permissive
    permission registry so the CLI works out-of-the-box in single-user
    deployments. Production deployments that need tighter gates wire
    their own registry through the API surface.
    """
    from synthesis_engine.skills import (
        discover_skills,
        get_skills_for_workspace,
    )
    from synthesis_engine.skills.loader import SkillLoader, SkillNotFoundError

    workspace = _resolve_workspace_for_skills(getattr(args, 'workspace', None))
    if workspace is not None:
        visible = get_skills_for_workspace(workspace)
    else:
        visible = discover_skills()

    loader = SkillLoader(visible)
    if not loader.has_skill(args.name):
        ws_label = workspace or "(universal-only chain)"
        print(
            f"Error: skill '{args.name}' is not visible from workspace "
            f"'{ws_label}'.",
            file=sys.stderr,
        )
        visible_names = [s.name for s in visible]
        if visible_names:
            print(
                f"Visible skills: {', '.join(visible_names)}",
                file=sys.stderr,
            )
        else:
            print("No skills are visible from this workspace.", file=sys.stderr)
        return 1

    try:
        inputs = _parse_input_kv_pairs(getattr(args, 'input', None))
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    if args.file:
        try:
            with open(args.file, 'r', encoding='utf-8') as fp:
                inputs['file'] = fp.read()
        except OSError as exc:
            print(f"Error reading --file {args.file!r}: {exc}", file=sys.stderr)
            return 2

    try:
        activated = loader.activate(args.name)
    except SkillNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    return _dispatch_activated_skill(activated, inputs, args.model)


def _dispatch_activated_skill(activated, inputs, model_override):
    """Run an :class:`ActivatedSkill` through the agent loop and print the answer.

    First-tool semantics: the skill's first declared :class:`SkillTool`
    becomes the dispatch target; the user inputs map to that tool's
    parameters. When the skill declares no tools, the SKILL.md body is
    rendered as the task prompt and the LLM runs in plain-completion mode.
    """
    import asyncio
    from synthesis_engine.agent import (
        AgentLoop,
        FilesystemCheckpointStore,
        PermissionRegistry,
        PermissionResult,
    )
    from synthesis_engine.config import get_default_model
    from synthesis_engine.llm import get_llm_backend

    skill = activated.skill
    tools = activated.tools
    model = model_override or get_default_model()

    if tools:
        target_tool = tools[0]
        task = (
            f"Use the '{target_tool.name}' tool from skill '{skill.name}' "
            f"with the supplied inputs.\n\n"
            f"Tool description: {target_tool.description or '(none)'}\n"
            f"Inputs: {json.dumps(inputs, sort_keys=True, default=str)}\n\n"
            f"Skill body:\n{activated.body_markdown}"
        )
    else:
        target_tool = None
        body = activated.body_markdown or skill.description or skill.name
        if inputs:
            task = (
                f"{body}\n\n---\n\nInputs: "
                f"{json.dumps(inputs, sort_keys=True, default=str)}"
            )
        else:
            task = body

    # When the skill has tools, run through the full agent loop so the
    # planner-execute-evaluate cycle can dispatch them with permission
    # gates. When the skill has no tools, the LLM-direct path is simpler
    # and faster — no planner JSON to assemble.
    if target_tool is not None:
        backend = get_llm_backend()
        registry = PermissionRegistry()
        registry.register(
            "*", lambda _ctx: PermissionResult.allow(
                "cli-permissive-skills-run"
            ),
        )
        checkpoint_root = os.path.join(data_dir, 'agent_checkpoints')
        loop = AgentLoop(
            llm_backend=backend,
            mcp_client=None,
            permission_registry=registry,
            checkpoint_store=FilesystemCheckpointStore(
                base_dir=checkpoint_root
            ),
            default_mcp_server="local",
        )
        try:
            final_state = asyncio.run(loop.run(task))
        except Exception as exc:
            print(f"Agent loop failed: {exc}", file=sys.stderr)
            return 1
        answer = final_state.final_answer or "(no final answer produced)"
        print(answer)
        return 0

    # No-tools path: render the body through the LLM directly.
    from synthesis_engine.llm import LLMRequest

    backend = get_llm_backend()
    request = LLMRequest(
        model=model,
        messages=[{"role": "user", "content": task}],
    )
    try:
        response = backend.complete(request)
    except Exception as exc:
        print(f"LLM call failed: {exc}", file=sys.stderr)
        return 1
    print(response.text)
    return 0


def run_mcp_serve(args):
    """Run `ragbot mcp serve` — start RagbotMCPServer over stdio, foreground.

    Wires a real ServerDependencies from live substrate singletons (see
    synthesis_engine.mcp_server.wiring.build_default_dependencies) and
    blocks on stdin/stdout until the client disconnects or the process
    receives an interrupt. Stdio is process-local: whatever spawned this
    process (Claude Code, Cursor, etc.) is the trust boundary — no bearer
    token is consulted, matching RagbotMCPServer.serve_stdio()'s contract.

    This command does not register itself with any client — see
    docs/mcp-server-activation.md for what a user needs to do to actually
    point Claude Code (or another MCP client) at this command.
    """
    import asyncio

    try:
        from synthesis_engine.mcp_server import RagbotMCPServer
        from synthesis_engine.mcp_server.wiring import build_default_dependencies
    except Exception as e:
        print(f"MCP server unavailable: {e}", file=sys.stderr)
        return 1

    try:
        dependencies = build_default_dependencies()
    except Exception as e:
        print(f"Failed to wire MCP server dependencies: {e}", file=sys.stderr)
        return 1

    server = RagbotMCPServer(dependencies)
    print(
        "Ragbot MCP server starting (stdio transport). "
        "Press Ctrl-C or close the client connection to stop.",
        file=sys.stderr,
    )
    try:
        asyncio.run(server.serve_stdio())
    except KeyboardInterrupt:
        pass
    return 0


def run_db_status(args):
    """Print backend health and indexed collections.

    In demo mode the listing is filtered to only the bundled demo
    workspace, so screenshots taken with ``RAGBOT_DEMO=1`` cannot leak
    real workspace names that happen to exist on the same vector store.
    """
    from synthesis_engine.vectorstore import get_vector_store
    from ragbot.demo import is_demo_mode, DEMO_WORKSPACE_NAME

    vs = get_vector_store()
    if vs is None:
        print("Vector backend: unavailable.")
        print("Check RAGBOT_VECTOR_BACKEND and RAGBOT_DATABASE_URL.")
        return 1

    health = vs.healthcheck()
    print(f"Backend: {vs.backend_name}")
    print(f"Healthy: {health.get('ok', False)}")
    for k, v in health.items():
        if k in ('backend', 'ok'):
            continue
        print(f"  {k}: {v}")

    collections = vs.list_collections()
    if is_demo_mode():
        # Show only demo-scoped collections. Hide everything else so
        # screenshots cannot leak real workspace names that happen to
        # exist on the same vector store.
        from ragbot.demo import DEMO_SKILLS_WORKSPACE_NAME
        allowed = {DEMO_WORKSPACE_NAME, DEMO_SKILLS_WORKSPACE_NAME}
        collections = [c for c in collections if c in allowed]
    print(f"\nIndexed collections ({len(collections)}):")
    if not collections:
        print("  (none)")
    for name in collections:
        info = vs.get_collection_info(name) or {}
        print(f"  {name:30s} chunks={info.get('count', 0)}")
    return 0


def run_db_init(args):
    """Apply schema migrations explicitly (idempotent)."""
    from synthesis_engine.vectorstore import get_vector_store

    vs = get_vector_store()
    if vs is None:
        print("Vector backend: unavailable.")
        return 1

    if vs.backend_name != 'pgvector':
        print(f"`db init` is a no-op for {vs.backend_name}; nothing to do.")
        return 0

    # Pgvector backend runs migrations on construction; explicit re-trigger
    # by calling init_collection(workspace='_init_') is harmless and confirms
    # the schema is applied.
    ok = vs.init_collection('_init_check_', vector_size=384)
    print("Schema migrations applied." if ok else "Migration failed (see logs).")
    return 0 if ok else 1


def run_memory_consolidate(args):
    """Drive the scheduled memory consolidator from the CLI.

    Routes to the consolidator's three entry points based on which
    filter the operator supplied. Output is a human-readable tabular
    report — sessions consolidated, entities added, relations added,
    duration. The same report is parseable by downstream tooling: the
    column order is stable and the column headers are documented.
    """
    import asyncio as _asyncio

    from synthesis_engine.memory import MemoryConsolidator, get_memory

    memory = get_memory()
    if memory is None:
        print(
            "Memory backend unavailable. Set RAGBOT_DATABASE_URL "
            "and ensure pgvector is reachable.",
            file=sys.stderr,
        )
        return 1

    consolidator = MemoryConsolidator(memory)

    async def _run():
        if args.session_id:
            return await consolidator.consolidate_session(
                args.session_id,
                model_id=args.model,
                workspace=args.workspace,
                dry_run=args.dry_run,
            )
        if args.since or args.until:
            return await consolidator.consolidate_batch(
                since_iso=args.since,
                until_iso=args.until,
                model_id=args.model,
                workspace=args.workspace,
                dry_run=args.dry_run,
            )
        # Default path: idle-hours mode (4.0 unless overridden).
        threshold = args.idle_hours if args.idle_hours is not None else 4.0
        return await consolidator.consolidate_recent_idle(
            idle_threshold_hours=threshold,
            model_id=args.model,
            workspace=args.workspace,
            dry_run=args.dry_run,
        )

    try:
        result = _asyncio.run(_run())
    except Exception as exc:
        print(f"Consolidation error: {exc}", file=sys.stderr)
        return 1

    _print_consolidation_result(result, dry_run=args.dry_run)
    return 0


def _print_consolidation_result(result, *, dry_run: bool) -> None:
    """Render a ConsolidationReport or BatchReport as a tabular block."""
    # Single-session vs batch: ConsolidationReport carries a
    # ``session_id``; BatchReport carries ``per_session``.
    if hasattr(result, 'per_session'):
        # BatchReport
        print(
            f"Consolidation batch (dry_run={result.dry_run}) "
            f"model={result.model_id} duration={result.duration_seconds:.2f}s"
        )
        print(
            f"  sessions_consolidated={result.sessions_consolidated} "
            f"sessions_skipped={result.sessions_skipped} "
            f"sessions_errored={result.sessions_errored}"
        )
        print(
            f"  entities_added={result.total_entities_added} "
            f"relations_added={result.total_relations_added}"
        )
        if result.per_session:
            print()
            header = (
                "session_id",
                "outcome",
                "entities_added",
                "relations_added",
                "duration_s",
            )
            rows = []
            for r in result.per_session:
                outcome = (
                    "errored"
                    if r.error
                    else ("skipped" if r.skipped else "consolidated")
                )
                rows.append(
                    (
                        r.session_id[:24],
                        outcome,
                        str(r.entities_added),
                        str(r.relations_added),
                        f"{r.duration_seconds:.2f}",
                    )
                )
            widths = [
                max(len(header[i]), *(len(row[i]) for row in rows))
                for i in range(len(header))
            ]
            fmt = "  " + "  ".join(f"{{:<{w}}}" for w in widths)
            print(fmt.format(*header))
            for row in rows:
                print(fmt.format(*row))
        return

    # ConsolidationReport
    outcome = (
        "errored"
        if result.error
        else ("skipped" if result.skipped else "consolidated")
    )
    print(
        f"Consolidation (dry_run={dry_run}) session={result.session_id} "
        f"model={result.model_id} outcome={outcome} "
        f"duration={result.duration_seconds:.2f}s"
    )
    print(
        f"  entities_added={result.entities_added} "
        f"relations_added={result.relations_added} "
        f"entities_existing={result.entities_existing} "
        f"relations_existing={result.relations_existing}"
    )
    if result.skip_reason:
        print(f"  skip_reason={result.skip_reason}")
    if result.error:
        print(f"  error={result.error}")


def run_memory_consolidation_history(args):
    """Tail recent ``memory_consolidation`` audit entries."""
    from synthesis_engine.memory import read_consolidation_history

    try:
        entries = read_consolidation_history(limit=args.limit)
    except Exception as exc:
        print(f"History read error: {exc}", file=sys.stderr)
        return 1

    if not entries:
        print("No memory_consolidation audit entries found.")
        return 0

    header = (
        "timestamp",
        "session_id",
        "workspace",
        "model_id",
        "entities_added",
        "relations_added",
    )
    rows = []
    for e in entries:
        meta = e.get("metadata") or {}
        ws_list = e.get("workspaces") or []
        rows.append(
            (
                str(e.get("timestamp_iso") or "")[:19],
                str(meta.get("session_id", ""))[:24],
                str(ws_list[0] if ws_list else "")[:20],
                str(e.get("model_id") or "")[:32],
                str(meta.get("entities_added", 0)),
                str(meta.get("relations_added", 0)),
            )
        )
    widths = [
        max(len(header[i]), *(len(row[i]) for row in rows))
        for i in range(len(header))
    ]
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    print(fmt.format(*header))
    for row in rows:
        print(fmt.format(*row))
    return 0


def run_compile(args):
    """Run the compile command (instructions only).

    Knowledge concatenation is handled by CI/CD (GitHub Actions).
    RAG indexing is handled by the 'index' subcommand.
    """
    import time
    from compiler.config import load_compile_config, validate_config, get_project_name
    from compiler import compile_project, compile_all_with_inheritance
    from compiler.manifest import format_manifest_summary

    def get_personal_repo_path_local(base_path):
        """Get the path to the user's personal ai-knowledge repo."""
        from compiler import get_personal_repo_path
        return get_personal_repo_path(base_path)

    def find_project_repo(project_name, base_path):
        """Find the repository path for a project name."""
        from synthesis_engine.workspaces import resolve_repo_index
        repo_path = resolve_repo_index(base_path).get(project_name)
        if repo_path and os.path.exists(repo_path):
            return repo_path
        raise FileNotFoundError(
            f"Repository not found for project '{project_name}'. "
            f"Check ~/.synthesis/console.yaml or pass --base-path."
        )

    def compile_repo(repo_path):
        """Compile instructions for a single repository."""
        start_time = time.time()

        if not args.quiet:
            print(f"Compiling instructions: {repo_path}")

        try:
            config = load_compile_config(repo_path)
            errors = validate_config(config)
            if errors:
                print("Configuration errors:", file=sys.stderr)
                for error in errors:
                    print(f"  - {error}", file=sys.stderr)
                return 1
        except FileNotFoundError as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1

        project_name = get_project_name(config)

        llm_map = {
            'claude': ['anthropic'],
            'chatgpt': ['openai'],
            'gemini': ['google'],
            'all': None
        }
        target_platforms = llm_map.get(args.llm)

        # Determine output repo path
        output_repo = None
        with_inheritance = getattr(args, 'with_inheritance', False)
        if hasattr(args, 'output_repo') and args.output_repo:
            output_repo = os.path.expanduser(args.output_repo)
        elif with_inheritance:
            output_repo = get_personal_repo_path_local(args.base_path)

        try:
            result = compile_project(
                config=config,
                platforms=target_platforms,
                personalized=with_inheritance,
                force=args.force,
                use_llm=not args.no_llm,
                instructions_only=True,  # Always instructions-only
                verbose=args.verbose,
                personal_repo_path=args.personal_repo,
                base_path=args.base_path,
                output_repo_path=output_repo
            )
        except Exception as e:
            print(f"Compilation error: {e}", file=sys.stderr)
            if args.verbose:
                import traceback
                traceback.print_exc()
            return 1

        elapsed = time.time() - start_time

        if args.quiet:
            print(f"Compiled {project_name} in {elapsed:.2f}s")
        else:
            print(format_manifest_summary(result.get('manifest', {})))
            print(f"\nCompleted in {elapsed:.2f}s")

        return 0

    # Determine what to compile
    if hasattr(args, 'all_with_inheritance') and args.all_with_inheritance:
        personal_repo = get_personal_repo_path_local(args.base_path)

        if not personal_repo or not os.path.exists(personal_repo):
            print(f"Error: Personal repo not found. Set default_workspace in ~/.synthesis/ragbot.yaml", file=sys.stderr)
            return 1

        if not args.quiet:
            print("=" * 70)
            print("COMPILING ALL INSTRUCTIONS WITH INHERITANCE")
            print("=" * 70)
            print(f"Output repo: {personal_repo}")
            print(f"All output will go to: {personal_repo}/compiled/")
            print("=" * 70)

        llm_map = {
            'claude': ['anthropic'],
            'chatgpt': ['openai'],
            'gemini': ['google'],
            'all': None
        }
        target_platforms = llm_map.get(args.llm)

        result = compile_all_with_inheritance(
            output_repo_path=personal_repo,
            base_path=args.base_path,
            verbose=args.verbose,
            platforms=target_platforms,
            force=args.force,
            use_llm=not args.no_llm,
            instructions_only=True,  # Always instructions-only
        )

        if result['errors']:
            print("\nErrors:", file=sys.stderr)
            for error in result['errors']:
                print(f"  - {error}", file=sys.stderr)

        print(f"\n{'='*70}")
        print(f"COMPILATION COMPLETE")
        print(f"{'='*70}")
        print(f"Projects compiled: {result['total_compiled']}")
        print(f"Output location: {personal_repo}/compiled/")

        print("\nCompiled projects:")
        for name, proj_result in result['projects'].items():
            if 'error' in proj_result:
                print(f"  ✗ {name}: {proj_result['error']}")
            else:
                chain = proj_result.get('inheritance_chain', [])
                chain_str = ' → '.join(chain) if chain else 'baseline'
                print(f"  ✓ {name} ({chain_str})")

        return 1 if result['errors'] else 0

    elif args.project:
        try:
            repo_path = find_project_repo(args.project, args.base_path)
        except FileNotFoundError as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1
        return compile_repo(repo_path)

    elif args.repo:
        return compile_repo(args.repo)

    else:
        cwd = os.getcwd()
        if os.path.exists(os.path.join(cwd, 'compile-config.yaml')):
            return compile_repo(cwd)
        else:
            print("Error: No project specified. Use --project, --repo, or --all-with-inheritance.")
            print("       Or run from an ai-knowledge-* directory with compile-config.yaml")
            return 1


def run_index(args):
    """Run the index command — index AI Knowledge content into vector store.

    Reads source files directly from ai-knowledge repositories and indexes
    them into pgvector for RAG retrieval. No intermediate compiled files needed.

    ``--tier archive`` indexes personal-archive/ content into the
    'archive-only' retrieval tier instead of the default 'topic-specific'
    tier used for source/datasets/. The two tiers are independently
    force-clearable: `--force` only wipes the tier being indexed in this
    invocation, never the other tier's content in the same workspace.
    """
    import time

    workspace_name = args.workspace
    tier = getattr(args, 'tier', 'datasets')

    if args.verbose:
        print(f"Indexing workspace: {workspace_name} (tier={tier})")
        if args.force:
            print(f"Force mode: clearing existing '{tier}' index first")

    start_time = time.time()

    try:
        if tier == 'archive':
            from rag import index_workspace_archive_by_name
            indexed = index_workspace_archive_by_name(workspace_name, force=args.force)
        else:
            from rag import index_workspace_by_name
            indexed = index_workspace_by_name(workspace_name, force=args.force)
    except Exception as e:
        print(f"Indexing error: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1

    elapsed = time.time() - start_time

    print(f"Indexed {indexed} documents for workspace '{workspace_name}' "
          f"(tier={tier}) in {elapsed:.2f}s")
    return 0


def main():
    parser = argparse.ArgumentParser(
        prog='ragbot',
        description="Ragbot.AI - An augmented brain and AI assistant. Learn more at https://ragbot.ai"
    )

    # Top-level --demo flag: equivalent to setting RAGBOT_DEMO=1 for the
    # rest of the process. Activates the bundled demo workspace and skill,
    # hard-isolating from any real workspaces or skills on the host.
    parser.add_argument(
        '--demo',
        action='store_true',
        help='Run in demo mode using the bundled sample workspace and skill.'
    )

    subparsers = parser.add_subparsers(
        title='commands',
        description='Available commands',
        dest='command'
    )

    # Create subcommand parsers
    create_chat_parser(subparsers)
    create_compile_parser(subparsers)
    create_index_parser(subparsers)
    create_skills_parser(subparsers)
    create_db_parser(subparsers)
    create_memory_parser(subparsers)
    create_agent_parser(subparsers)
    create_mcp_parser(subparsers)

    # Parse arguments
    args = parser.parse_args()

    # If --demo was passed at the top level (or on a subcommand), set the
    # env var early so every later import that reads RAGBOT_DEMO sees it.
    if getattr(args, 'demo', False):
        os.environ['RAGBOT_DEMO'] = '1'

    # If no command specified, default to chat behavior for backward compatibility
    if args.command is None:
        # Re-parse with 'chat' as the default command
        # This maintains backward compatibility with existing usage like:
        #   ragbot -p "prompt"
        #   ragbot -i
        sys.argv.insert(1, 'chat')
        args = parser.parse_args()
        if getattr(args, 'demo', False):
            os.environ['RAGBOT_DEMO'] = '1'

    # Demo mode: ensure the bundled workspace is indexed before any
    # subcommand runs. First-run setup; idempotent on subsequent runs.
    if os.environ.get('RAGBOT_DEMO', '').strip().lower() in ('1', 'true', 'yes', 'on'):
        _ensure_demo_indexed()

    # Run the appropriate command
    if hasattr(args, 'func'):
        return args.func(args)
    else:
        parser.print_help()
        return 1


def _ensure_demo_indexed() -> None:
    """First-run auto-indexing for demo mode.

    Indexes the bundled demo workspace and demo skill into the configured
    vector backend (typically pgvector) under the canonical 'demo'
    workspace name. Idempotent: if the workspace already has chunks,
    this is a no-op.
    """
    try:
        from ragbot.demo import DEMO_WORKSPACE_NAME, demo_workspace_path
        from rag import (
            get_index_status,
            index_content,
            index_skills,
            init_collection,
        )
    except Exception:
        # Imports might fail in narrow contexts (e.g., db status). Skip
        # silently — discovery still hard-isolates either way.
        return

    workspace_path = demo_workspace_path()
    if workspace_path is None:
        return

    # Skip if already indexed.
    indexed, count = get_index_status(DEMO_WORKSPACE_NAME)
    if indexed and count > 0:
        return

    init_collection(DEMO_WORKSPACE_NAME)
    source_dir = workspace_path / 'source'
    for category in ('datasets', 'runbooks', 'instructions'):
        category_dir = source_dir / category
        if category_dir.is_dir():
            try:
                index_content(DEMO_WORKSPACE_NAME, [str(category_dir)], content_type=category)
            except Exception as exc:  # noqa: BLE001
                print(f"Demo: skipped {category} indexing ({exc})", file=sys.stderr)

    # Index the bundled demo skill into a demo-scoped workspace so the
    # demo's cross-workspace fan-out can NOT retrieve real skill content
    # that happens to share the host vector store.
    try:
        from ragbot.demo import DEMO_SKILLS_WORKSPACE_NAME
        index_skills(workspace_name=DEMO_SKILLS_WORKSPACE_NAME, force=False)
    except Exception:
        pass


if __name__ == "__main__":
    sys.exit(main() or 0)
