"""
LLM Instruction Compiler for AI Knowledge Compiler

Compiles instructions using each target platform's flagship model.
Each LLM compiles its own optimized instructions for best results:
- Claude compiles Claude instructions
- GPT compiles ChatGPT instructions
- Gemini compiles Gemini instructions

This ensures each platform's instructions are optimized by a model that
understands that platform's conventions and capabilities.

Model names are NEVER hardcoded here - they come from engines.yaml.
This module only knows about platform names (anthropic, openai, google).

Library API:
- compile_instructions(content, platform, model, api_key) -> str
- format_for_platform(instructions, platform) -> str
- get_platform_constraints(platform) -> dict
"""

import os
import sys
from typing import Optional

# Add parent directory to path for ragbot imports
_parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

from synthesis_engine.keystore import get_api_key
from .config import load_engines_config, resolve_model

# Import LLM clients
try:
    import anthropic
except ImportError:
    anthropic = None

try:
    import openai
except ImportError:
    openai = None

# Google Gen AI SDK (the supported successor to google-generativeai). The
# legacy package is kept around as a transitive dep of older tools; this
# code path uses the new SDK exclusively to avoid the EOL deprecation
# warning emitted at import time.
try:
    from google import genai as google_genai  # type: ignore
except ImportError:
    google_genai = None  # type: ignore


# Platform-specific constraints
PLATFORM_CONSTRAINTS = {
    'anthropic': {
        'name': 'Claude',
        'max_instruction_tokens': 8000,
        'format': 'markdown',
        'supports_xml': True,
        'notes': 'Claude works well with XML tags for structure'
    },
    'openai': {
        'name': 'ChatGPT',
        'max_instruction_tokens': 8000,
        'format': 'markdown',
        'supports_xml': False,
        'notes': 'ChatGPT prefers concise markdown'
    },
    'google': {
        'name': 'Gemini',
        'max_instruction_tokens': 4000,
        'format': 'markdown',
        'supports_xml': False,
        'notes': 'Gemini has stricter limits, be concise'
    },
    'grok': {
        'name': 'Grok',
        'max_instruction_tokens': 4000,
        'format': 'markdown',
        'supports_xml': False,
        'notes': 'Similar to ChatGPT format'
    }
}


def get_platform_constraints(platform: str) -> dict:
    """
    Get constraints for a specific platform.

    Args:
        platform: Platform name (anthropic, openai, google, grok)

    Returns:
        Dictionary with platform constraints
    """
    return PLATFORM_CONSTRAINTS.get(platform, {
        'name': platform,
        'max_instruction_tokens': 4000,
        'format': 'markdown',
        'supports_xml': False,
        'notes': ''
    })


def get_compilation_prompt(platform: str) -> str:
    """
    Get the system prompt for instruction compilation.

    Args:
        platform: Target platform

    Returns:
        Compilation prompt string
    """
    constraints = get_platform_constraints(platform)

    return f"""You are an expert at optimizing AI custom instructions for {constraints['name']}.

Your task is to take the provided source instructions and optimize them for use as custom instructions in {constraints['name']}.

Guidelines:
1. Preserve all essential behavioral rules, identity, and preferences
2. Optimize for clarity and token efficiency
3. Use {constraints['format']} format
4. Keep within approximately {constraints['max_instruction_tokens']} tokens
5. {'Use XML tags for structure where helpful' if constraints['supports_xml'] else 'Use markdown headers for structure'}
6. Remove redundant or verbose phrasing while keeping the meaning
7. Organize logically: identity first, then behavioral rules, then preferences

{constraints['notes']}

Output ONLY the optimized instructions. Do not include explanations or meta-commentary."""


def compile_with_anthropic(content: str, model: str, api_key: str) -> str:
    """
    Compile instructions using Claude.

    Args:
        content: Source instructions content
        model: Model ID (e.g., 'claude-opus-4-5-20251101')
        api_key: Anthropic API key

    Returns:
        Compiled instructions
    """
    if anthropic is None:
        raise ImportError("anthropic package not installed")

    client = anthropic.Anthropic(api_key=api_key)

    message = client.messages.create(
        model=model,
        max_tokens=8192,
        system=get_compilation_prompt('anthropic'),
        messages=[
            {
                "role": "user",
                "content": f"Please optimize these instructions for Claude:\n\n{content}"
            }
        ]
    )

    return message.content[0].text


def compile_with_openai(content: str, model: str, api_key: str) -> str:
    """
    Compile instructions using OpenAI.

    Args:
        content: Source instructions content
        model: Model ID from engines.yaml
        api_key: OpenAI API key

    Returns:
        Compiled instructions
    """
    if openai is None:
        raise ImportError("openai package not installed")

    client = openai.OpenAI(api_key=api_key)

    response = client.chat.completions.create(
        model=model,
        max_tokens=8192,
        messages=[
            {
                "role": "system",
                "content": get_compilation_prompt('openai')
            },
            {
                "role": "user",
                "content": f"Please optimize these instructions for ChatGPT:\n\n{content}"
            }
        ]
    )

    return response.choices[0].message.content


def compile_with_google(content: str, model: str, api_key: str) -> str:
    """
    Compile instructions using Gemini.

    Args:
        content: Source instructions content
        model: Model ID from engines.yaml (may have 'gemini/' prefix)
        api_key: Google API key

    Returns:
        Compiled instructions
    """
    if google_genai is None:
        raise ImportError("google-genai package not installed (pip install google-genai)")

    # Strip the litellm-style 'gemini/' prefix; google-genai expects bare ids.
    if model.startswith('gemini/'):
        model = model[len('gemini/'):]

    client = google_genai.Client(api_key=api_key)

    prompt = f"""{get_compilation_prompt('google')}

Please optimize these instructions for Gemini:

{content}"""

    response = client.models.generate_content(model=model, contents=prompt)

    # The SDK exposes the consolidated string via the .text accessor.
    return response.text or ""


def compile_for_unsupported_platform(content: str, target_platform: str,
                                      model: str, api_key: str) -> str:
    """
    Use Claude to compile instructions for platforms without direct API support.

    Args:
        content: Source instructions content
        target_platform: The platform we're optimizing for (e.g., 'grok', 'perplexity')
        model: Claude model to use
        api_key: Anthropic API key

    Returns:
        Compiled instructions optimized for the target platform
    """
    if anthropic is None:
        raise ImportError("anthropic package not installed")

    constraints = get_platform_constraints(target_platform)

    prompt = f"""You are an expert at optimizing AI custom instructions for different platforms.

Your task is to take the provided source instructions and optimize them for use as custom instructions in {constraints['name']}.

Guidelines:
1. Preserve all essential behavioral rules, identity, and preferences
2. Optimize for clarity and token efficiency
3. Use {constraints['format']} format
4. Keep within approximately {constraints['max_instruction_tokens']} tokens
5. {'Use XML tags for structure where helpful' if constraints['supports_xml'] else 'Use markdown headers for structure'}
6. Remove redundant or verbose phrasing while keeping the meaning
7. Organize logically: identity first, then behavioral rules, then preferences

{constraints['notes']}

Note: You are Claude, but you are optimizing these instructions for {constraints['name']}, so format them appropriately for that platform's conventions and capabilities.

Output ONLY the optimized instructions. Do not include explanations or meta-commentary."""

    client = anthropic.Anthropic(api_key=api_key)

    message = client.messages.create(
        model=model,
        max_tokens=8192,
        system=prompt,
        messages=[
            {
                "role": "user",
                "content": f"Please optimize these instructions for {constraints['name']}:\n\n{content}"
            }
        ]
    )

    return message.content[0].text


def compile_instructions(content: str, platform: str, model: str,
                         api_key: Optional[str] = None,
                         fallback_platform: str = None,
                         fallback_model: str = None) -> str:
    """
    Compile instructions using the specified platform's model.

    Args:
        content: Source instructions content
        platform: Platform name (anthropic, openai, google, grok, perplexity)
        model: Model ID
        api_key: API key (uses environment variable if not provided)
        fallback_platform: Platform to use if target platform not supported
        fallback_model: Model to use with fallback platform

    Returns:
        Compiled instructions optimized for the platform

    Raises:
        ValueError: If platform not supported
        ImportError: If required SDK not installed
    """
    # Platforms that need a fallback (no direct API support)
    fallback_platforms = {'grok', 'perplexity'}

    # If this platform needs fallback, use Claude to generate optimized instructions
    if platform in fallback_platforms:
        # Use Claude to generate instructions optimized for the target platform
        fallback_platform = fallback_platform or 'anthropic'
        # Get default model from engines.yaml - never hardcode model names
        if not fallback_model:
            engines_config = load_engines_config()
            fallback_model = resolve_model(engines_config, fallback_platform, 'routine')
        fallback_key = get_api_key('anthropic')

        if not fallback_key:
            raise ValueError(f"No API key for fallback platform (anthropic) to compile {platform} instructions")

        # Custom prompt for generating instructions for unsupported platforms
        return compile_for_unsupported_platform(content, platform, fallback_model, fallback_key)

    # Get API key from keystore if not provided
    if api_key is None:
        api_key = get_api_key(platform)

    if not api_key:
        raise ValueError(f"No API key provided for {platform}")

    if platform == 'anthropic':
        return compile_with_anthropic(content, model, api_key)
    elif platform == 'openai':
        return compile_with_openai(content, model, api_key)
    elif platform == 'google':
        return compile_with_google(content, model, api_key)
    else:
        raise ValueError(f"Unsupported platform: {platform}")


def format_for_platform(instructions: str, platform: str) -> str:
    """
    Format instructions for a specific platform's UI.

    Args:
        instructions: Compiled instructions
        platform: Target platform

    Returns:
        Formatted instructions string
    """
    constraints = get_platform_constraints(platform)

    # Add platform-specific header
    header = f"# Custom Instructions for {constraints['name']}\n\n"

    # Clean up any excessive whitespace
    lines = instructions.split('\n')
    cleaned_lines = []
    prev_empty = False

    for line in lines:
        is_empty = not line.strip()
        if is_empty and prev_empty:
            continue
        cleaned_lines.append(line)
        prev_empty = is_empty

    return header + '\n'.join(cleaned_lines)


def passthrough_instructions(content: str, platform: str) -> str:
    """
    Format instructions without LLM compilation (for testing or when API unavailable).

    Args:
        content: Source instructions content
        platform: Target platform

    Returns:
        Formatted instructions (not LLM-optimized)
    """
    constraints = get_platform_constraints(platform)

    header = f"""# Custom Instructions for {constraints['name']}
# Note: These instructions were assembled without LLM optimization

"""
    return header + content


def assemble_instructions_content(files: list) -> str:
    """
    Assemble instruction files into a single content string.

    Args:
        files: List of file info dicts with 'content' and 'relative_path'

    Returns:
        Assembled content string
    """
    parts = []

    for file_info in files:
        rel_path = file_info.get('relative_path', '')
        content = file_info.get('content', '')

        if content.strip():
            # Add a header with the file path
            parts.append(f"## From: {rel_path}\n\n{content}")

    return '\n\n---\n\n'.join(parts)
