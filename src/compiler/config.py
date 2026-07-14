"""
Config Parser for AI Knowledge Compiler

Loads and validates compile-config.yaml files, resolves model tiers
to actual model IDs using engines.yaml.

Library API:
- load_compile_config(path) -> dict
- load_engines_config(path) -> dict
- resolve_model(engines_config, platform, tier) -> str
- validate_config(config) -> list[str] (returns validation errors)
"""

import os
import yaml
from pathlib import Path
from typing import Optional


def load_yaml(path: str) -> dict:
    """Load a YAML file and return its contents."""
    with open(path, 'r') as f:
        return yaml.safe_load(f) or {}


def load_compile_config(repo_path: str) -> dict:
    """
    Load compile-config.yaml from a repository.

    Args:
        repo_path: Path to the ai-knowledge-* repository

    Returns:
        Parsed configuration dictionary

    Raises:
        FileNotFoundError: If compile-config.yaml doesn't exist
    """
    config_path = os.path.join(repo_path, 'compile-config.yaml')
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"No compile-config.yaml found at {config_path}")

    config = load_yaml(config_path)
    config['_repo_path'] = repo_path
    config['_config_path'] = config_path
    return config


def load_engines_config(engines_path: Optional[str] = None) -> dict:
    """
    Load engines.yaml configuration.

    Args:
        engines_path: Path to engines.yaml. If None, searches in standard locations.

    Returns:
        Engines configuration dictionary
    """
    if engines_path and os.path.exists(engines_path):
        return load_yaml(engines_path)

    # Search standard locations
    search_paths = [
        os.path.join(os.path.dirname(__file__), '..', 'engines.yaml'),  # src/engines.yaml
        os.path.join(os.path.dirname(__file__), '..', '..', 'engines.yaml'),  # ragbot/engines.yaml
        'engines.yaml',  # current directory
    ]

    for path in search_paths:
        if os.path.exists(path):
            return load_yaml(path)

    raise FileNotFoundError("Could not find engines.yaml in any standard location")


def resolve_model(engines_config: dict, platform: str, tier: str = 'flagship') -> str:
    """
    Resolve a model tier to an actual model ID.

    Args:
        engines_config: Loaded engines.yaml configuration
        platform: Platform name (anthropic, openai, google)
        tier: Model tier (flagship, judgment, routine, bulk).
              'flagship' selects the model marked is_flagship: true;
              the other three match the model's `tier` field.

    Returns:
        Model ID string (e.g., 'claude-fable-5')

    Raises:
        ValueError: If platform or tier not found
    """
    engines = engines_config.get('engines', [])

    # Find the engine by platform name
    engine = None
    for e in engines:
        if e.get('name') == platform:
            engine = e
            break

    if not engine:
        raise ValueError(f"Platform '{platform}' not found in engines.yaml")

    models = engine.get('models', [])

    # If looking for flagship, find the model with is_flagship: true
    if tier == 'flagship':
        for model in models:
            if model.get('is_flagship'):
                return model['name']
        # Fallback to first judgment-tier model
        for model in models:
            if model.get('tier') == 'judgment':
                return model['name']
        # Fallback to first model
        if models:
            return models[0]['name']
        raise ValueError(f"No flagship model found for platform '{platform}'")

    # Find model by tier
    for model in models:
        if model.get('tier') == tier:
            return model['name']

    raise ValueError(f"No model with tier '{tier}' found for platform '{platform}'")


def get_model_info(engines_config: dict, platform: str, model_name: str) -> dict:
    """
    Get full model information from engines.yaml.

    Args:
        engines_config: Loaded engines.yaml configuration
        platform: Platform name
        model_name: Model ID

    Returns:
        Model configuration dictionary
    """
    engines = engines_config.get('engines', [])

    for engine in engines:
        if engine.get('name') == platform:
            for model in engine.get('models', []):
                if model.get('name') == model_name:
                    return model

    return {}


def validate_config(config: dict) -> list:
    """
    Validate a compile configuration.

    Args:
        config: Parsed compile-config.yaml

    Returns:
        List of validation error messages (empty if valid)
    """
    errors = []

    # Check required fields
    if 'project' not in config:
        errors.append("Missing required 'project' section")
    else:
        if 'name' not in config['project']:
            errors.append("Missing required 'project.name' field")

    if 'sources' not in config:
        errors.append("Missing required 'sources' section")
    else:
        if 'local' not in config['sources']:
            errors.append("Missing required 'sources.local' section")

    if 'compilation' not in config:
        errors.append("Missing required 'compilation' section")
    else:
        compilation = config['compilation']
        if 'targets' not in compilation:
            errors.append("Missing required 'compilation.targets' section")
        else:
            for i, target in enumerate(compilation['targets']):
                if 'name' not in target:
                    errors.append(f"Target {i}: missing 'name' field")
                if 'platform' not in target:
                    errors.append(f"Target {i}: missing 'platform' field")
                if 'model_tier' not in target and 'model' not in target:
                    errors.append(f"Target {i}: missing 'model_tier' or 'model' field")

    return errors


def get_project_name(config: dict) -> str:
    """Get the project name from config."""
    return config.get('project', {}).get('name', 'unknown')


def get_output_dir(config: dict) -> str:
    """Get the output directory from config."""
    repo_path = config.get('_repo_path', '.')
    output_dir = config.get('project', {}).get('output_dir', './compiled')
    return os.path.join(repo_path, output_dir)


def get_source_path(config: dict) -> str:
    """Get the local source path from config."""
    repo_path = config.get('_repo_path', '.')
    source_path = config.get('sources', {}).get('local', {}).get('path', './source')
    return os.path.join(repo_path, source_path)


def get_include_patterns(config: dict) -> list:
    """Get include patterns from config."""
    return config.get('sources', {}).get('local', {}).get('include', ['**/*'])


def get_exclude_patterns(config: dict) -> list:
    """Get exclude patterns from config."""
    return config.get('sources', {}).get('local', {}).get('exclude', [])


def get_token_budget(config: dict) -> int:
    """Get the default token budget from config."""
    return config.get('compilation', {}).get('default_token_budget', 100000)


def get_targets(config: dict) -> list:
    """Get compilation targets from config."""
    return config.get('compilation', {}).get('targets', [])


def get_default_compiler(config: dict) -> dict:
    """Get the default compiler settings from config."""
    return config.get('compilation', {}).get('default_compiler', {
        'engine': 'anthropic',
        'model_tier': 'flagship'
    })


