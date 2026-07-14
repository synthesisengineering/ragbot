"""
AI Knowledge Compiler — Instructions Only

Compiles instructions from AI Knowledge repositories for various LLM platforms
(Claude, ChatGPT, Gemini, etc.).

Knowledge concatenation is handled by CI/CD (GitHub Actions).
RAG indexing is handled by `ragbot index` (reads source directly).

Key concept: The output repo determines what content is included—not who
runs the compiler. See docs/compilation-guide.md for details.

Example usage:
    from ragbot.compiler import compile_project
    from ragbot.compiler.config import load_compile_config

    config = load_compile_config("/path/to/ai-knowledge-{project}")
    result = compile_project(config, instructions_only=True)

Public API:
- compile_project() - Compile a single project's instructions
- compile_all_with_inheritance() - Compile all projects into an output repo
- Config: load_compile_config, resolve_model, validate_config
- Assembler: assemble_content, count_tokens
- Instructions: compile_instructions, format_for_platform
- Cache: load_cache, save_cache, is_file_changed
- Manifest: generate_manifest, save_manifest
"""

import os
import shutil
import time
import fnmatch
from pathlib import Path
from typing import Optional, List, Dict, Any

import yaml

# Import library modules
from .config import (
    load_compile_config,
    load_engines_config,
    resolve_model,
    validate_config,
    get_project_name,
    get_output_dir,
    get_source_path,
    get_include_patterns,
    get_exclude_patterns,
    get_token_budget,
    get_targets,
    get_default_compiler,
)

from .assembler import (
    find_files,
    read_file,
    assemble_content,
    count_tokens,
    merge_content,
    apply_context_filter,
    format_knowledge_file,
    check_token_budget,
    merge_assembled_content
)

from .cache import (
    compute_hash,
    compute_file_hash,
    load_cache,
    save_cache,
    is_file_changed,
    is_compilation_valid,
    update_file_cache,
    update_compilation_cache,
    get_cache_path
)

from synthesis_engine.inheritance import (
    load_inheritance_config,
    find_inheritance_config,
    get_inheritance_chain,
    resolve_dependencies,
    ensure_repos_available,
    get_repo_source_path
)

from .instructions import (
    compile_instructions,
    format_for_platform,
    get_platform_constraints,
    passthrough_instructions,
    assemble_instructions_content
)

from .manifest import (
    generate_manifest,
    add_target_to_manifest,
    save_manifest,
    load_manifest,
    format_manifest_summary
)

# vectors.py removed — RAG indexing reads source directly via rag.py


def _resolve_base_path() -> Optional[str]:
    """Resolve the legacy base path for ai-knowledge repos, if one exists.

    Returns the flat-parent path that contains ai-knowledge-* directories, if
    such a layout is in use. Returns None when repos are scattered across
    workspaces (the modern layout), in which case callers should use
    resolve_repo_index() / get_repo_path() instead.

    Resolution order:
    1. RAGBOT_BASE_PATH environment variable
    2. /app/ai-knowledge (Docker default)
    3. ~/ai-knowledge (legacy convention)
    """
    env_path = os.environ.get('RAGBOT_BASE_PATH')
    if env_path:
        expanded = os.path.expanduser(env_path)
        if os.path.isdir(expanded):
            return expanded
    for candidate in ('/app/ai-knowledge', os.path.expanduser('~/ai-knowledge')):
        if os.path.isdir(candidate):
            return candidate
    return None


def get_repo_path(workspace_name: str, base_path: Optional[str] = None) -> Optional[str]:
    """Look up the absolute path of an ai-knowledge repo by workspace name.

    Uses ragbot.workspaces.resolve_repo_index for resolution, which handles
    both the workspace-rooted layout and the legacy flat-parent layout.
    """
    from synthesis_engine.workspaces import resolve_repo_index
    return resolve_repo_index(base_path).get(workspace_name)


def get_personal_repo_path(base_path: Optional[str] = None) -> Optional[str]:
    """
    Get the path to the personal repo using the user's configured default workspace.

    Reads default_workspace from ~/.synthesis/ragbot.yaml, then resolves it
    against the discovered repo index. Falls back to searching all discovered
    repos for the one containing my-projects.yaml.

    Args:
        base_path: Optional flat-parent override (legacy mode).

    Returns:
        Path to the personal repo, or None if not found.
    """
    from synthesis_engine.keystore import get_default_workspace
    from synthesis_engine.workspaces import resolve_repo_index

    index = resolve_repo_index(base_path)

    # First, try the user's configured default workspace.
    default_workspace = get_default_workspace()
    if default_workspace and default_workspace in index:
        repo_path = index[default_workspace]
        if find_inheritance_config(repo_path):
            return repo_path

    # Fallback: any discovered repo that contains my-projects.yaml.
    for repo_path in index.values():
        if find_inheritance_config(repo_path):
            return repo_path

    return None


def load_context_definition(context_name: str, source_path: str) -> Optional[Dict]:
    """
    Load a context definition YAML file.

    Args:
        context_name: Name of the context (e.g., 'writing-mode')
        source_path: Path to the source directory

    Returns:
        Parsed context definition or None if not found
    """
    # Look for context file in source/contexts/
    contexts_dir = os.path.join(source_path, 'contexts')

    # Try with and without .yaml extension
    possible_paths = [
        os.path.join(contexts_dir, f'{context_name}.yaml'),
        os.path.join(contexts_dir, f'{context_name}.yml'),
        os.path.join(contexts_dir, context_name),
    ]

    for path in possible_paths:
        if os.path.exists(path):
            with open(path, 'r') as f:
                return yaml.safe_load(f)

    return None


def apply_context_to_assembled(assembled: Dict, context_def: Dict, source_path: str) -> Dict:
    """
    Apply context filtering to assembled content.

    Args:
        assembled: Assembled content dict from assemble_content()
        context_def: Context definition from load_context_definition()
        source_path: Path to source directory (for resolving relative paths)

    Returns:
        Filtered assembled content dict
    """
    include_rules = context_def.get('include', {})
    exclude_rules = context_def.get('exclude', [])

    # Build include patterns by category
    include_patterns = {}
    for category, patterns in include_rules.items():
        if isinstance(patterns, list):
            include_patterns[category] = patterns
        elif isinstance(patterns, str):
            include_patterns[category] = [patterns]

    # Convert exclude rules to patterns
    exclude_patterns = []
    if isinstance(exclude_rules, list):
        exclude_patterns = exclude_rules
    elif isinstance(exclude_rules, str):
        exclude_patterns = [exclude_rules]

    result = {
        'files': [],
        'by_category': {},
        'total_tokens': 0
    }

    for file_info in assembled.get('files', []):
        category = file_info.get('category', 'other')
        rel_path = file_info.get('relative_path', '')

        # Check if file matches any exclude pattern
        excluded = False
        for pattern in exclude_patterns:
            # Normalize pattern - handle both with and without category prefix
            if fnmatch.fnmatch(rel_path, pattern) or fnmatch.fnmatch(rel_path, f'*/{pattern}'):
                excluded = True
                break
            # Also check if the category/path matches
            full_pattern = f'{category}/{pattern}' if not pattern.startswith(category) else pattern
            if fnmatch.fnmatch(rel_path, full_pattern):
                excluded = True
                break

        if excluded:
            continue

        # Check if file matches include patterns for its category
        category_patterns = include_patterns.get(category, [])
        if category_patterns:
            # If include patterns exist for this category, file must match one
            matched = False
            for pattern in category_patterns:
                # Handle glob patterns
                if fnmatch.fnmatch(rel_path, pattern) or fnmatch.fnmatch(rel_path, f'{category}/{pattern}'):
                    matched = True
                    break
                # Also check just the filename
                filename = os.path.basename(rel_path)
                if fnmatch.fnmatch(filename, pattern):
                    matched = True
                    break

            if not matched:
                continue

        # File passes all filters
        result['files'].append(file_info)
        result['by_category'].setdefault(category, []).append(file_info)
        result['total_tokens'] += file_info.get('tokens', 0)

    return result


def assemble_inherited_content(
    project_name: str,
    inheritance_config: Dict,
    base_path: Optional[str] = None,
    include_patterns: List[str] = None,
    exclude_patterns: List[str] = None,
    verbose: bool = False
) -> Dict:
    """
    Assemble content from a project and all its inherited parents.

    Args:
        project_name: Name of the project to assemble
        inheritance_config: Loaded my-projects.yaml
        base_path: Optional flat-parent override (legacy mode). If None, the
            modern resolve_repo_index() is used as fallback when a project's
            local_path is missing.
        include_patterns: File patterns to include
        exclude_patterns: File patterns to exclude
        verbose: Print verbose output

    Returns:
        Merged assembled content dict with content from all inherited repos
    """
    # Get inheritance chain (parents first, project last)
    chain = get_inheritance_chain(project_name, inheritance_config)

    if verbose:
        print(f"Inheritance chain for {project_name}: {' → '.join(chain)}")

    assembled_list = []

    # Lazy-built repo index for fallback lookups.
    repo_index = None

    for proj_name in chain:
        # Get project config from inheritance config
        proj_config = inheritance_config.get('projects', {}).get(proj_name, {})
        local_path = proj_config.get('local_path', '')

        # Expand ~ in path
        if local_path.startswith('~'):
            local_path = os.path.expanduser(local_path)

        # Fallback chain: my-projects.yaml local_path → flat parent (legacy) → repo index.
        if not local_path or not os.path.exists(local_path):
            if base_path:
                candidate = os.path.join(base_path, f'ai-knowledge-{proj_name}')
                if os.path.exists(candidate):
                    local_path = candidate
            if not local_path or not os.path.exists(local_path):
                if repo_index is None:
                    from synthesis_engine.workspaces import resolve_repo_index
                    repo_index = resolve_repo_index(base_path)
                local_path = repo_index.get(proj_name, local_path)

        if not os.path.exists(local_path):
            if verbose:
                print(f"  Warning: Repo not found for {proj_name}: {local_path}")
            continue

        source_path = os.path.join(local_path, 'source')
        if not os.path.exists(source_path):
            if verbose:
                print(f"  Warning: Source path not found: {source_path}")
            continue

        if verbose:
            print(f"  Assembling from {proj_name}: {source_path}")

        # Assemble content from this repo
        assembled = assemble_content(
            source_path,
            include_patterns or ['**/*'],
            exclude_patterns or []
        )

        # Tag each file with its source repo
        for file_info in assembled.get('files', []):
            file_info['source_repo'] = proj_name
            file_info['source_repo_path'] = local_path

        assembled_list.append(assembled)

        if verbose:
            print(f"    Found {len(assembled['files'])} files, {assembled['total_tokens']:,} tokens")

    # Merge all assembled content (earlier repos have lower priority)
    return merge_assembled_content(assembled_list)



def assemble_skills_for_compilation(skills_config: dict, verbose: bool = False) -> dict:
    """Assemble Agent Skill content for the compiler.

    Returns a structure shaped like ``assemble_content``'s output so it can
    be merged with local source content and fed into the same downstream
    compile pipeline.

    Schema for ``skills_config`` (mirrors the ``sources.skills`` block in
    compile-config.yaml)::

        enabled: bool                  # default True if section is present
        roots: List[str]               # extra discovery roots beyond defaults
        include: List[str]             # name globs to include (default: all)
        exclude: List[str]             # name globs to exclude
        include_references: bool       # default True
        include_scripts_inline: bool   # default False — list scripts by name only

    Each skill yields:
        * one file_info per SKILL.md (category=instructions)
        * one file_info per included reference (category=instructions)
        * (optional) one file_info per inlined script (category=runbooks)
        * (always) one synthetic 'inventory' file_info that lists scripts
          and other artifacts by name (category=runbooks). Lets the LLM know
          what's available without bloating context with executable code.
    """

    # Local import — `skills` module lives under the runtime package.
    import sys as _sys
    _src_path = os.path.join(os.path.dirname(__file__), '..')
    if _src_path not in _sys.path:
        _sys.path.insert(0, _src_path)
    from synthesis_engine.skills import discover_skills  # type: ignore
    from compiler.assembler import count_tokens  # type: ignore

    if not skills_config or skills_config.get('enabled') is False:
        return {'files': [], 'by_category': {'instructions': [], 'runbooks': []}, 'total_tokens': 0}

    extra_roots = skills_config.get('roots') or []
    include_patterns = skills_config.get('include') or []
    exclude_patterns = skills_config.get('exclude') or []
    include_references = skills_config.get('include_references', True)
    include_scripts_inline = skills_config.get('include_scripts_inline', False)

    skills = discover_skills(extra=[os.path.expanduser(r) for r in extra_roots])

    if include_patterns:
        skills = [s for s in skills if any(fnmatch.fnmatch(s.name, p) for p in include_patterns)]
    if exclude_patterns:
        skills = [s for s in skills if not any(fnmatch.fnmatch(s.name, p) for p in exclude_patterns)]

    if verbose:
        print(f"Assembling {len(skills)} skill(s) for compilation")

    result = {
        'files': [],
        'by_category': {'instructions': [], 'runbooks': []},
        'total_tokens': 0,
    }

    def _push(content: str, category: str, relative_path: str, source_path: str):
        tokens = count_tokens(content)
        info = {
            'path': source_path,
            'relative_path': relative_path,
            'content': content,
            'tokens': tokens,
            'category': category,
        }
        result['files'].append(info)
        result['by_category'].setdefault(category, []).append(info)
        result['total_tokens'] += tokens

    for skill in skills:
        # 1. SKILL.md → instructions. Front-load skill metadata as an H1.
        skill_md_header = (
            f"# Skill: {skill.name}\n\n"
            f"**Description:** {skill.description}\n\n"
            f"**Source:** {skill.path}\n\n"
        )
        if skill.version:
            skill_md_header += f"**Version:** {skill.version}\n\n"
        _push(
            skill_md_header + skill.body,
            'instructions',
            f'skills/{skill.name}/SKILL.md',
            skill.skill_md_path,
        )

        # 2. References → instructions (if enabled).
        if include_references:
            for ref in skill.references:
                if not ref.is_text or not ref.content:
                    continue
                ref_header = (
                    f"# Skill {skill.name} — Reference: {ref.relative_path}\n\n"
                )
                _push(
                    ref_header + ref.content,
                    'instructions',
                    f'skills/{skill.name}/{ref.relative_path}',
                    ref.absolute_path,
                )

        # 3. Scripts: optional inline body, always include an inventory.
        if include_scripts_inline:
            for script in skill.scripts:
                if not script.is_text or not script.content:
                    continue
                _push(
                    f"# Skill {skill.name} — Script: {script.relative_path}\n\n"
                    f"```\n{script.content}\n```\n",
                    'runbooks',
                    f'skills/{skill.name}/{script.relative_path}',
                    script.absolute_path,
                )

        if skill.scripts or skill.other_files:
            inv_lines = [f"# Skill {skill.name} — File Inventory\n",
                         f"This skill ships the following non-instruction artifacts.",
                         f"They are available on disk at: `{skill.path}`.\n"]
            if skill.scripts:
                inv_lines.append("## Scripts\n")
                for s in skill.scripts:
                    inv_lines.append(f"- `{s.relative_path}` ({len(s.content)} chars)")
                inv_lines.append("")
            if skill.other_files:
                inv_lines.append("## Other artifacts\n")
                for o in skill.other_files:
                    inv_lines.append(f"- `{o.relative_path}`")
                inv_lines.append("")
            _push(
                '\n'.join(inv_lines),
                'runbooks',
                f'skills/{skill.name}/_inventory.md',
                skill.path,
            )

    return result


def compile_project(config: dict,
                    platforms: list = None,
                    personalized: bool = False,
                    context: str = None,
                    force: bool = False,
                    use_llm: bool = True,
                    instructions_only: bool = False,
                    verbose: bool = False,
                    personal_repo_path: str = None,
                    base_path: str = None,
                    output_repo_path: str = None,
                    target_project: str = None) -> dict:
    """
    Compile instructions for an AI Knowledge project.

    Output structure:
        compiled/{project}/
        └── instructions/
            ├── claude.md
            ├── chatgpt.md
            └── gemini.md

    Knowledge concatenation is handled by CI/CD (GitHub Actions).
    RAG indexing is handled by `ragbot index` (reads source directly).

    PRIVACY RULES:
    - If personalized=True, output MUST go to output_repo_path (your private repo)
    - Personalized compilations merge private content and must NEVER go to shared repos

    This function:
    1. Assembles content from source directories
    2. Optionally merges inherited content (if personalized)
    3. Applies context filtering (if specified)
    4. Compiles instructions for each target LLM (claude.md, chatgpt.md, gemini.md)
    5. Creates a manifest

    Args:
        config: Loaded compile-config.yaml (from load_compile_config)
        platforms: List of platforms to compile for (None = all)
        personalized: Whether to include inherited content
        context: Optional context filter to apply (single context)
        force: Force recompilation, ignore cache
        use_llm: Whether to use LLMs for instruction compilation
        instructions_only: Only compile instructions, skip knowledge
        verbose: Print verbose output
        personal_repo_path: Path to personal repo (for inheritance config)
        base_path: Base path where ai-knowledge-* repos are located
        output_repo_path: Path to repo where output should be written (REQUIRED if personalized)
        target_project: Name of project to compile (if different from config's project)

    Returns:
        Dictionary with:
        - manifest: Compilation manifest
        - output_dir: Path to compiled output
        - compiled_files: List of output file paths
        - errors: List of any errors encountered
        - inheritance_chain: List of repos in inheritance order (if personalized)
    """
    start_time = time.time()

    # Use target_project if specified, otherwise get from config
    project_name = target_project or get_project_name(config)
    repo_path = config.get('_repo_path', '.')
    source_path = get_source_path(config)

    # Determine base_path for ai-knowledge repos
    if not base_path:
        base_path = _resolve_base_path()

    # =========================================================================
    # PRIVACY SAFEGUARD: Validate output destination for personalized builds
    # =========================================================================
    # Personalized compilations merge private content (personal info, etc.)
    # and MUST go to the user's private repo, never to shared/client repos.
    #
    # See: projects/active/ai-knowledge-architecture/architecture.md
    # =========================================================================
    if personalized and not output_repo_path:
        # If personalized but no explicit output repo, default to personal repo
        if not personal_repo_path:
            # Get personal repo path using user's configured default workspace
            personal_repo_path = get_personal_repo_path(base_path)
            if not personal_repo_path:
                raise ValueError(
                    f"Cannot find personal repo. Either:\n"
                    f"1. Set default_workspace in ~/.synthesis/ragbot.yaml, or\n"
                    f"2. Ensure my-projects.yaml exists in one of the ai-knowledge-* "
                    f"directories under {base_path}"
                )
        output_repo_path = personal_repo_path

    # Determine output directory: compiled/{project}/
    if output_repo_path:
        # Output goes to specified repo (used for personalized compilations)
        base_output = os.path.join(output_repo_path, 'compiled')
    else:
        # Output goes to source repo (baseline compilation)
        base_output = get_output_dir(config)  # Gets the repo's compiled/ directory

    output_dir = os.path.join(base_output, project_name)

    result = {
        'manifest': {},
        'output_dir': output_dir,
        'compiled_files': [],
        'errors': [],
        'inheritance_chain': []
    }

    # Load engines config for model resolution
    try:
        engines_config = load_engines_config()
    except FileNotFoundError:
        engines_config = {'engines': []}
        result['errors'].append("Warning: engines.yaml not found, using defaults")

    # Load cache
    cache_path = get_cache_path(repo_path)
    cache = load_cache(cache_path) if not force else {'files': {}, 'compilations': {}}

    # Assemble source content
    include_patterns = get_include_patterns(config)
    exclude_patterns = get_exclude_patterns(config)

    # Handle personalized compilation (merge inherited content)
    inheritance_config = None
    if personalized:
        # Find personal repo path if not provided
        if not personal_repo_path:
            # Get personal repo path using user's configured default workspace
            personal_repo_path = get_personal_repo_path(base_path)

        inheritance_config_path = None
        if personal_repo_path:
            inheritance_config_path = find_inheritance_config(personal_repo_path)
        if inheritance_config_path:
            try:
                inheritance_config = load_inheritance_config(inheritance_config_path)

                if verbose:
                    print(f"Loading inheritance config from {inheritance_config_path}")

                # Assemble content from all inherited repos
                assembled = assemble_inherited_content(
                    project_name,
                    inheritance_config,
                    base_path,
                    include_patterns,
                    exclude_patterns,
                    verbose
                )

                # Record inheritance chain
                result['inheritance_chain'] = get_inheritance_chain(project_name, inheritance_config)

            except Exception as e:
                result['errors'].append(f"Inheritance config error: {e}")
                if verbose:
                    print(f"Inheritance error: {e}, falling back to local-only")
                assembled = assemble_content(source_path, include_patterns, exclude_patterns)
        else:
            if verbose:
                print("No my-projects.yaml found, using local content only")
            assembled = assemble_content(source_path, include_patterns, exclude_patterns)
    else:
        if verbose:
            print(f"Assembling content from {source_path}")
        assembled = assemble_content(source_path, include_patterns, exclude_patterns)

    # Merge in Agent Skills if the compile-config opts in.
    skills_config = (config.get('sources') or {}).get('skills')
    if skills_config:
        skill_assembled = assemble_skills_for_compilation(skills_config, verbose=verbose)
        if skill_assembled['files']:
            assembled['files'].extend(skill_assembled['files'])
            for cat, items in skill_assembled['by_category'].items():
                assembled['by_category'].setdefault(cat, []).extend(items)
            assembled['total_tokens'] += skill_assembled['total_tokens']
            if verbose:
                print(
                    f"Added {len(skill_assembled['files'])} skill file(s), "
                    f"{skill_assembled['total_tokens']:,} tokens"
                )

    if verbose:
        print(f"Total: {len(assembled['files'])} files, {assembled['total_tokens']:,} tokens")

    # Apply single context filter if specified
    if context:
        context_def = load_context_definition(context, source_path)
        if context_def:
            if verbose:
                print(f"Applying context filter: {context}")
            assembled = apply_context_to_assembled(assembled, context_def, source_path)
            if verbose:
                print(f"After filter: {len(assembled['files'])} files, {assembled['total_tokens']:,} tokens")
        else:
            result['errors'].append(f"Context definition not found: {context}")

    # Check token budget
    budget = get_token_budget(config)
    budget_check = check_token_budget(assembled, budget)
    if not budget_check['within_budget']:
        result['errors'].append(
            f"Token budget exceeded: {assembled['total_tokens']:,} > {budget:,}"
        )

    # Clean up and create output directory: compiled/{project}/
    # Remove the project's output directory if it exists to ensure clean output
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
        if verbose:
            print(f"Cleaned existing output directory: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    compiled_files = []
    targets = get_targets(config)

    # Filter targets by platform if specified
    if platforms:
        targets = [t for t in targets if t.get('platform') in platforms]

    # Map platform names to output filenames
    platform_to_filename = {
        'anthropic': 'claude.md',
        'openai': 'chatgpt.md',
        'google': 'gemini.md',
        'grok': 'grok.md'
    }

    # Create instructions directory: compiled/{project}/instructions/
    instructions_dir = os.path.join(output_dir, 'instructions')
    os.makedirs(instructions_dir, exist_ok=True)

    # Compile instructions for each target LLM
    instructions_files = assembled['by_category'].get('instructions', [])

    for target in targets:
        target_name = target.get('name', 'unknown')
        platform = target.get('platform', 'anthropic')
        model_tier = target.get('model_tier', 'flagship')

        if verbose:
            print(f"Compiling instructions for: {platform}")

        # Resolve model
        try:
            model = resolve_model(engines_config, platform, model_tier)
        except ValueError:
            model = None
            result['errors'].append(f"Could not resolve model for {platform}")

        # Compile instructions
        if instructions_files:
            instructions_content = assemble_instructions_content(instructions_files)

            if use_llm and model:
                try:
                    compiled = compile_instructions(
                        instructions_content, platform, model
                    )
                except Exception as e:
                    result['errors'].append(f"LLM compilation failed for {platform}: {e}")
                    compiled = passthrough_instructions(instructions_content, platform)
            else:
                compiled = passthrough_instructions(instructions_content, platform)

            # Save compiled instructions as {platform}.md
            # e.g., compiled/{project}/instructions/claude.md
            filename = platform_to_filename.get(platform, f'{platform}.md')
            instructions_path = os.path.join(instructions_dir, filename)

            with open(instructions_path, 'w') as f:
                f.write(compiled)

            compiled_files.append(instructions_path)

            if verbose:
                print(f"  Wrote: {instructions_path}")

    # Knowledge concatenation is handled by CI/CD (GitHub Actions).
    # RAG indexing is handled by `ragbot index` (reads source directly).

    # Generate manifest
    compilation_time = time.time() - start_time
    manifest = generate_manifest(
        project_name=project_name,
        source_files=assembled['files'],
        compiled_files=compiled_files,
        config=config,
        compilation_time=compilation_time
    )

    # Add inheritance info to manifest
    if result['inheritance_chain']:
        manifest['inheritance'] = {
            'chain': result['inheritance_chain'],
            'personalized': True
        }

    # Add target info to manifest
    for target in targets:
        platform = target.get('platform', 'anthropic')
        filename = platform_to_filename.get(platform, f'{platform}.md')
        target_files = [f for f in compiled_files if filename in f]

        add_target_to_manifest(
            manifest, platform, platform,
            target_files, assembled['total_tokens'],
            compiled_with_llm=use_llm
        )

    # Save manifest
    manifest_path = save_manifest(manifest, output_dir)
    compiled_files.append(manifest_path)

    # Update cache
    for file_info in assembled['files']:
        update_file_cache(cache, file_info['path'])
    save_cache(cache, cache_path)

    result['manifest'] = manifest
    result['compiled_files'] = compiled_files

    return result


def compile_all_projects(base_path: Optional[str] = None, **kwargs) -> dict:
    """
    Compile all AI Knowledge projects discovered via the repo index.

    Args:
        base_path: Optional flat-parent override (legacy mode). When None,
            uses resolve_repo_index() to find repos across ~/workspaces/* and
            ~/.synthesis/console.yaml sources.
        **kwargs: Arguments passed to compile_project

    Returns:
        Dictionary with results for each project
    """
    from synthesis_engine.workspaces import resolve_repo_index

    results = {}
    index = resolve_repo_index(base_path)

    if not index:
        return {'error': "No ai-knowledge repos discovered"}

    for project_name, repo_path in index.items():
        config_path = os.path.join(repo_path, 'compile-config.yaml')
        if not os.path.exists(config_path):
            continue

        try:
            config = load_compile_config(repo_path)
            result = compile_project(config, base_path=base_path, **kwargs)
            results[project_name] = result
        except Exception as e:
            results[project_name] = {'error': str(e)}

    return results


def compile_all_with_inheritance(
    output_repo_path: str,
    base_path: str = None,
    verbose: bool = False,
    **kwargs
) -> dict:
    """
    Compile all projects with inheritance into a specified output repo.

    The output repo determines what content is included—not who runs the compiler.
    Anyone with write access can compile into a repo. Content included depends
    solely on the output repo's position in the inheritance tree.

    See: docs/compilation-guide.md

    Example:
        # Compile into personal repo (includes all personal content)
        compile_all_with_inheritance(
            output_repo_path='~/ai-knowledge/ai-knowledge-{person}'
        )

        # Compile into company repo (no personal content)
        compile_all_with_inheritance(
            output_repo_path='~/ai-knowledge/ai-knowledge-{company}'
        )

        # Compile into client repo (baseline only)
        compile_all_with_inheritance(
            output_repo_path='~/ai-knowledge/ai-knowledge-{client}'
        )

    Output structure:
        {output_repo}/compiled/
        ├── {project1}/    ← content based on output repo's inheritance position
        ├── {project2}/
        └── ...

    Args:
        output_repo_path: Path to repo where compiled output goes (determines content)
        base_path: Base path containing all ai-knowledge-* repos
        verbose: Print verbose output
        **kwargs: Additional arguments passed to compile_project

    Returns:
        Dictionary with:
        - projects: Dict of project_name -> compilation result
        - output_repo: Path to output repo
        - total_compiled: Number of projects compiled
        - errors: List of any errors
    """
    # base_path is optional in the modern layout; resolve_repo_index() handles
    # the workspace-rooted layout and synthesis-console sources directly.
    if not base_path:
        base_path = _resolve_base_path()  # may be None

    # Expand ~ in output path
    if output_repo_path.startswith('~'):
        output_repo_path = os.path.expanduser(output_repo_path)

    results = {
        'projects': {},
        'output_repo': output_repo_path,
        'total_compiled': 0,
        'errors': []
    }

    # Find personal repo using user's config or by searching for my-projects.yaml
    # The inheritance config (my-projects.yaml) lives ONLY in personal repos (per ADR-006)
    personal_repo_path = get_personal_repo_path(base_path)

    if not personal_repo_path:
        # Try output repo as fallback (in case it IS the personal repo)
        config_path = find_inheritance_config(output_repo_path)
        if config_path:
            personal_repo_path = output_repo_path

    if not personal_repo_path:
        results['errors'].append(
            f"Cannot find personal repo. Either:\n"
            f"1. Set default_workspace in ~/.config/ragbot/config.yaml, or\n"
            f"2. Ensure my-projects.yaml exists in one of the ai-knowledge-* "
            f"directories under {base_path}"
        )
        return results

    inheritance_config_path = find_inheritance_config(personal_repo_path)

    try:
        inheritance_config = load_inheritance_config(inheritance_config_path)
    except Exception as e:
        results['errors'].append(f"Failed to load inheritance config: {e}")
        return results

    projects = inheritance_config.get('projects', {})

    if verbose:
        print(f"Found {len(projects)} projects in my-projects.yaml")
        print(f"All with-inheritance output will go to: {output_repo_path}/compiled/")

    # Lazy-built repo index for fallback lookups.
    repo_index = None

    # Compile each project with inheritance into the output repo
    for project_name, project_config in projects.items():
        if verbose:
            print(f"\n{'='*60}")
            print(f"Compiling: {project_name} (with-inheritance)")
            print('='*60)

        # Get the source repo path for this project
        local_path = project_config.get('local_path', '')
        if local_path.startswith('~'):
            local_path = os.path.expanduser(local_path)

        # Fallback chain: local_path → flat parent (legacy) → repo index.
        if not local_path or not os.path.exists(local_path):
            if base_path:
                candidate = os.path.join(base_path, f'ai-knowledge-{project_name}')
                if os.path.exists(candidate):
                    local_path = candidate
            if not local_path or not os.path.exists(local_path):
                if repo_index is None:
                    from synthesis_engine.workspaces import resolve_repo_index
                    repo_index = resolve_repo_index(base_path)
                local_path = repo_index.get(project_name, local_path)

        if not os.path.exists(local_path):
            results['errors'].append(f"Repo not found for {project_name}: {local_path}")
            results['projects'][project_name] = {'error': 'Repo not found'}
            continue

        config_path = os.path.join(local_path, 'compile-config.yaml')
        if not os.path.exists(config_path):
            results['errors'].append(f"No compile-config.yaml for {project_name}")
            results['projects'][project_name] = {'error': 'No compile-config.yaml'}
            continue

        try:
            config = load_compile_config(local_path)

            # Compile with inheritance, output to specified repo
            result = compile_project(
                config=config,
                personalized=True,  # Enable inheritance chain resolution
                verbose=verbose,
                personal_repo_path=personal_repo_path,  # For inheritance config lookup
                base_path=base_path,
                output_repo_path=output_repo_path,  # CRITICAL: Output goes to specified repo
                target_project=project_name,
                **kwargs
            )

            results['projects'][project_name] = result
            results['total_compiled'] += 1

            if verbose:
                chain = result.get('inheritance_chain', [])
                if chain:
                    print(f"  Inheritance: {' → '.join(chain)}")
                print(f"  Output: {result.get('output_dir')}")

        except Exception as e:
            results['errors'].append(f"Failed to compile {project_name}: {e}")
            results['projects'][project_name] = {'error': str(e)}

    return results


# Version
__version__ = '0.2.0'
