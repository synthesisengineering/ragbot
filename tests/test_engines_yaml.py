"""Schema tests for engines.yaml — Ragbot's single source of truth for LLM config.

These tests assert the engines.yaml file parses cleanly and that the v3.4
open-weights model additions (Llama 4, Qwen3.6, DeepSeek V3.2, Mistral 3
family) are wired in with the schema fields the rest of Ragbot relies on.

The schema-level assertions read engines.yaml directly via PyYAML so they
remain valid even if the synthesis_engine loader is mid-refactor in a
parallel branch. A second class wires the same assertions through
``synthesis_engine.config.load_engines_config`` when the loader is
importable, so the tests cover both the YAML structure and the loader's
consumption of it.
"""

import os
import sys
from typing import Any, Dict, List

import pytest
import yaml


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENGINES_YAML = os.path.join(REPO_ROOT, "engines.yaml")
SRC_DIR = os.path.join(REPO_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)


# Models added in the Ragbot v3.4 open-weights expansion. Each entry is
# ``(model_name, provider, expected_tier, expected_min_context_window)``.
# These are the load-bearing schema invariants — if any one of these
# entries disappears from engines.yaml, a v3.4 release is incomplete.
V3_4_LOCAL_MODELS: List[tuple] = [
    ("llama4:scout", "ollama", "routine", 1_000_000),
    ("llama4:maverick", "ollama", "judgment", 1_000_000),
    ("qwen3.6:27b", "ollama", "routine", 200_000),
    ("qwen3.6:35b-a3b", "ollama", "judgment", 200_000),
    ("deepseek-v3.2:671b", "ollama", "judgment", 128_000),
    ("mistral-large-3:675b", "ollama", "judgment", 200_000),
    ("mistral-medium-3.5:128b", "ollama", "judgment", 200_000),
    ("mistral-small-4:119b", "ollama", "routine", 200_000),
]


# Existing Gemma 4 entries that v3.4 augmented with MLX-backend metadata.
# These should retain their original schema AND gain the new fields.
V3_3_GEMMA_MODELS: List[str] = [
    "gemma4:e4b",
    "gemma4:26b",
    "gemma4:31b",
]


# Required schema fields for every new local-inference entry. Missing any
# of these breaks model picker rendering, LiteLLM routing, or the
# sizing-matrix link.
LOCAL_INFERENCE_REQUIRED_FIELDS = {
    "mlx_supported",
    "ollama_tag",
    "llama_cpp_quants",
}


# Valid tier values. Shared vocabulary with the synthesis-model-tiers
# skill's tiers.yaml (judgment / routine / bulk — labels name the work,
# not the artifact) — one vocabulary, two files, enforced by
# TestTierRoleConsistency below.
VALID_TIERS = {"judgment", "routine", "bulk"}


# Valid thinking-mode values, used to validate `thinking.modes` and
# `thinking.mode` when present.
VALID_THINKING_MODES = {
    "minimal",
    "low",
    "medium",
    "high",
    "adaptive",
    "hybrid",
    "thinking",
    "instruct",
    "extended",
}


def _load_yaml() -> Dict[str, Any]:
    """Read engines.yaml directly via PyYAML."""
    with open(ENGINES_YAML, "r") as f:
        config = yaml.safe_load(f)
    assert config is not None, "engines.yaml is empty"
    return config


def _find_provider(config: Dict[str, Any], provider_name: str) -> Dict[str, Any]:
    """Return the provider block from the parsed engines.yaml."""
    for engine in config["engines"]:
        if engine["name"] == provider_name:
            return engine
    raise AssertionError(f"Provider {provider_name!r} not found in engines.yaml")


def _find_model(provider_block: Dict[str, Any], model_name: str) -> Dict[str, Any]:
    """Return the model block by name within a provider."""
    for model in provider_block.get("models", []):
        if model["name"] == model_name:
            return model
    raise AssertionError(
        f"Model {model_name!r} not found in provider "
        f"{provider_block['name']!r}"
    )


class TestEnginesYamlStructure:
    """Top-level schema invariants for engines.yaml."""

    def test_engines_yaml_exists(self):
        """engines.yaml exists at the repo root."""
        assert os.path.exists(ENGINES_YAML), (
            f"engines.yaml not found at {ENGINES_YAML}"
        )

    def test_engines_yaml_parses(self):
        """engines.yaml is valid YAML."""
        config = _load_yaml()
        assert isinstance(config, dict)
        assert "engines" in config
        assert isinstance(config["engines"], list)

    def test_required_top_level_keys(self):
        """Top-level keys engines.yaml must carry."""
        config = _load_yaml()
        for key in ("engines", "default", "temperature_settings"):
            assert key in config, f"Missing top-level key {key!r}"

    def test_each_engine_has_name_and_models(self):
        """Each engine has a name and a non-empty models list."""
        config = _load_yaml()
        for engine in config["engines"]:
            assert "name" in engine
            assert "models" in engine
            assert isinstance(engine["models"], list)
            assert len(engine["models"]) > 0, (
                f"Engine {engine['name']!r} has no models"
            )

    def test_default_provider_exists(self):
        """The top-level `default` provider must be a configured engine."""
        config = _load_yaml()
        engine_names = {e["name"] for e in config["engines"]}
        assert config["default"] in engine_names


class TestV34LocalModelEntries:
    """Each v3.4-added open-weights model is present with the required schema."""

    @pytest.mark.parametrize(
        "model_name,provider,expected_tier,expected_min_context",
        V3_4_LOCAL_MODELS,
    )
    def test_v34_model_present(
        self,
        model_name: str,
        provider: str,
        expected_tier: str,
        expected_min_context: int,
    ):
        """Model is registered under its provider with the expected tier."""
        config = _load_yaml()
        provider_block = _find_provider(config, provider)
        model = _find_model(provider_block, model_name)

        # Tier drives the picker badge (bulk / routine / judgment).
        assert model.get("tier") == expected_tier, (
            f"{model_name}: tier should be {expected_tier!r}, "
            f"got {model.get('tier')!r}"
        )

        # Context window must be at least the documented minimum.
        ctx = model.get("max_input_tokens")
        assert ctx is not None, f"{model_name}: missing max_input_tokens"
        assert ctx >= expected_min_context, (
            f"{model_name}: max_input_tokens={ctx} below "
            f"expected {expected_min_context}"
        )

    @pytest.mark.parametrize(
        "model_name,provider,_,__",
        V3_4_LOCAL_MODELS,
    )
    def test_v34_model_has_required_local_inference_block(
        self, model_name: str, provider: str, _: str, __: int
    ):
        """local_inference block is present with the required fields."""
        config = _load_yaml()
        provider_block = _find_provider(config, provider)
        model = _find_model(provider_block, model_name)

        local_inf = model.get("local_inference")
        assert local_inf is not None, (
            f"{model_name}: missing local_inference block"
        )
        missing = LOCAL_INFERENCE_REQUIRED_FIELDS - set(local_inf.keys())
        assert not missing, (
            f"{model_name}: local_inference missing fields {missing}"
        )

    @pytest.mark.parametrize(
        "model_name,provider,_,__",
        V3_4_LOCAL_MODELS,
    )
    def test_v34_model_has_license(
        self, model_name: str, provider: str, _: str, __: int
    ):
        """Each local-inference model declares its license."""
        config = _load_yaml()
        provider_block = _find_provider(config, provider)
        model = _find_model(provider_block, model_name)
        assert "license" in model, f"{model_name}: missing license"
        assert isinstance(model["license"], str)
        assert len(model["license"]) > 0

    @pytest.mark.parametrize(
        "model_name,provider,_,__",
        V3_4_LOCAL_MODELS,
    )
    def test_v34_model_has_backend_notes_for_mlx(
        self, model_name: str, provider: str, _: str, __: int
    ):
        """Each local-inference model declares MLX backend notes."""
        config = _load_yaml()
        provider_block = _find_provider(config, provider)
        model = _find_model(provider_block, model_name)
        notes = model.get("backend_notes")
        assert notes is not None, f"{model_name}: missing backend_notes"
        assert "mlx" in notes, f"{model_name}: backend_notes missing mlx entry"
        assert isinstance(notes["mlx"], str)
        assert len(notes["mlx"]) > 10  # non-trivial note

    def test_v34_thinking_mode_valid_when_present(self):
        """Where `thinking` is declared, its modes/mode use valid values."""
        config = _load_yaml()
        for engine in config["engines"]:
            for model in engine["models"]:
                thinking = model.get("thinking")
                if not thinking:
                    continue
                if "mode" in thinking:
                    assert thinking["mode"] in VALID_THINKING_MODES, (
                        f"{model['name']}: invalid thinking.mode "
                        f"{thinking['mode']!r}"
                    )
                if "modes" in thinking:
                    for m in thinking["modes"]:
                        assert m in VALID_THINKING_MODES, (
                            f"{model['name']}: invalid thinking.modes entry "
                            f"{m!r}"
                        )


class TestGemmaMlxAugmentations:
    """v3.4 augmented existing Gemma 4 entries with MLX-backend metadata."""

    @pytest.mark.parametrize("model_name", V3_3_GEMMA_MODELS)
    def test_gemma_still_present(self, model_name: str):
        """Existing Gemma entries are not removed."""
        config = _load_yaml()
        ollama = _find_provider(config, "ollama")
        model = _find_model(ollama, model_name)
        assert model is not None

    @pytest.mark.parametrize("model_name", V3_3_GEMMA_MODELS)
    def test_gemma_has_mlx_backend_notes(self, model_name: str):
        """v3.4 backfills MLX backend notes onto the Gemma 4 family."""
        config = _load_yaml()
        ollama = _find_provider(config, "ollama")
        model = _find_model(ollama, model_name)
        notes = model.get("backend_notes")
        assert notes is not None, f"{model_name}: missing backend_notes"
        assert "mlx" in notes, f"{model_name}: backend_notes missing mlx"

    @pytest.mark.parametrize("model_name", V3_3_GEMMA_MODELS)
    def test_gemma_has_local_inference_block(self, model_name: str):
        """Gemma entries also gain the local_inference block."""
        config = _load_yaml()
        ollama = _find_provider(config, "ollama")
        model = _find_model(ollama, model_name)
        local_inf = model.get("local_inference")
        assert local_inf is not None, (
            f"{model_name}: missing local_inference"
        )
        assert local_inf.get("mlx_supported") is True


class TestEnginesYamlIntegrity:
    """Cross-cutting invariants that prevent silent regressions."""

    def test_no_duplicate_model_names_within_provider(self):
        """Each provider's models have unique names (LiteLLM relies on it)."""
        config = _load_yaml()
        for engine in config["engines"]:
            names = [m["name"] for m in engine["models"]]
            assert len(names) == len(set(names)), (
                f"Duplicate model name in provider {engine['name']!r}: {names}"
            )

    def test_ollama_provider_has_no_api_key_required(self):
        """The ollama provider doesn't require an api_key_name."""
        config = _load_yaml()
        ollama = _find_provider(config, "ollama")
        # Either no api_key_name field, or a falsy value
        assert not ollama.get("api_key_name"), (
            "Ollama is a local-server provider; should not declare api_key_name"
        )

    def test_default_model_for_ollama_exists(self):
        """The ollama provider's default_model is in its models list."""
        config = _load_yaml()
        ollama = _find_provider(config, "ollama")
        default = ollama.get("default_model")
        assert default is not None
        names = {m["name"] for m in ollama["models"]}
        assert default in names, (
            f"Ollama default_model {default!r} not in models list {names}"
        )

    def test_every_local_model_max_output_positive(self):
        """Every local model declares a positive max_output_tokens."""
        config = _load_yaml()
        ollama = _find_provider(config, "ollama")
        for model in ollama["models"]:
            mot = model.get("max_output_tokens")
            assert mot is not None and mot > 0, (
                f"{model['name']}: max_output_tokens must be positive, got {mot!r}"
            )

    def test_every_local_model_default_max_lte_max_output(self):
        """default_max_tokens stays at or below max_output_tokens."""
        config = _load_yaml()
        ollama = _find_provider(config, "ollama")
        for model in ollama["models"]:
            default_max = model.get("default_max_tokens")
            max_output = model.get("max_output_tokens")
            if default_max is None:
                continue
            assert default_max <= max_output, (
                f"{model['name']}: default_max_tokens={default_max} > "
                f"max_output_tokens={max_output}"
            )

    def test_v34_count_growth(self):
        """v3.4 added at least 8 new entries to the ollama provider."""
        config = _load_yaml()
        ollama = _find_provider(config, "ollama")
        # 3 pre-existing Gemma + 8 new = 11 minimum.
        assert len(ollama["models"]) >= 11, (
            f"Expected ollama provider to have ≥11 models post-v3.4, "
            f"got {len(ollama['models'])}"
        )


def _find_tiers_yaml() -> str:
    """Locate the synthesis-model-tiers tiers.yaml, or return ''.

    Search order: SYNTHESIS_TIERS_FILE env override, then the standard
    skill-install locations. Returns '' when not found so the consistency
    tests skip on machines (and public CI) without the skill installed.
    """
    env_path = os.environ.get("SYNTHESIS_TIERS_FILE")
    if env_path and os.path.exists(env_path):
        return env_path
    candidates = [
        os.path.expanduser("~/.synthesis/skills/synthesis-model-tiers/tiers.yaml"),
        os.path.expanduser("~/.claude/skills/synthesis-model-tiers/tiers.yaml"),
        os.path.expanduser("~/.agents/skills/synthesis-model-tiers/tiers.yaml"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return ""


class TestTierVocabulary:
    """Every model declares a tier from the shared role vocabulary."""

    def test_every_model_has_valid_tier(self):
        """Each model's tier is one of frontier / efficient / light."""
        config = _load_yaml()
        for engine in config["engines"]:
            for model in engine["models"]:
                tier = model.get("tier")
                assert tier in VALID_TIERS, (
                    f"{engine['name']}/{model['name']}: tier must be one of "
                    f"{sorted(VALID_TIERS)}, got {tier!r}"
                )

    def test_no_legacy_category_field(self):
        """The pre-v3.6 `category` field must not reappear."""
        config = _load_yaml()
        for engine in config["engines"]:
            for model in engine["models"]:
                assert "category" not in model, (
                    f"{engine['name']}/{model['name']}: legacy `category` "
                    f"field present — use `tier` (judgment/routine/bulk)"
                )


class TestTierRoleConsistency:
    """engines.yaml and synthesis-model-tiers tiers.yaml must agree.

    tiers.yaml is the cross-provider ordered-preference table (which model
    plays which role); engines.yaml is the capability catalog. Both use the
    same tier vocabulary. These tests fail loudly when the two drift:
    a model named in tiers.yaml that is missing from engines.yaml, or
    listed under a different role than its catalog tier.

    The catalog may carry MORE models than the role lists bless (e.g.,
    Mac-Studio-class local models kept for bigger hardware) — that
    direction is allowed. The reverse is not.
    """

    @pytest.fixture(scope="class")
    def tiers_config(self):
        path = _find_tiers_yaml()
        if not path:
            pytest.skip(
                "synthesis-model-tiers tiers.yaml not installed "
                "(set SYNTHESIS_TIERS_FILE to override)"
            )
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        assert data and "providers" in data, f"{path}: malformed tiers.yaml"
        return data

    def test_every_engine_provider_covered(self, tiers_config):
        """Each engines.yaml provider has a role block in tiers.yaml."""
        config = _load_yaml()
        tier_providers = set(tiers_config["providers"].keys())
        for engine in config["engines"]:
            assert engine["name"] in tier_providers, (
                f"Provider {engine['name']!r} configured in engines.yaml "
                f"but has no block in tiers.yaml — add its role lists"
            )

    def test_role_list_models_exist_with_matching_tier(self, tiers_config):
        """Every model in a tiers.yaml role list exists in engines.yaml
        under that provider, with `tier` equal to the role it is listed
        under."""
        config = _load_yaml()
        engines_by_name = {e["name"]: e for e in config["engines"]}

        for provider, roles in tiers_config["providers"].items():
            engine = engines_by_name.get(provider)
            if engine is None:
                # tiers.yaml may cover providers this product doesn't
                # configure; nothing to cross-check for those.
                continue
            catalog = {m["name"]: m for m in engine["models"]}
            for role in sorted(VALID_TIERS):
                for model_name in roles.get(role, []) or []:
                    assert model_name in catalog, (
                        f"tiers.yaml {provider}.{role} lists "
                        f"{model_name!r}, which is not in engines.yaml — "
                        f"the two files must be updated together"
                    )
                    actual = catalog[model_name].get("tier")
                    assert actual == role, (
                        f"{provider}/{model_name}: tiers.yaml lists it "
                        f"under {role!r} but engines.yaml says tier="
                        f"{actual!r}"
                    )

    def test_role_lists_nonempty_for_cloud_providers(self, tiers_config):
        """Each provider block in tiers.yaml fills all three roles."""
        for provider, roles in tiers_config["providers"].items():
            for role in sorted(VALID_TIERS):
                models = roles.get(role)
                assert models, (
                    f"tiers.yaml {provider}: role {role!r} is empty or "
                    f"missing — every provider block fills all three roles"
                )


class TestEnginesLoaderIntegration:
    """Wire the schema assertions through synthesis_engine.config when available.

    The substrate is undergoing parallel refactor in other branches; if the
    loader is temporarily unimportable, these tests skip rather than fail,
    so the schema-level tests above remain useful as the canonical contract.
    """

    @pytest.fixture(scope="class")
    def loader_fns(self):
        try:
            from synthesis_engine.config import (  # type: ignore
                load_engines_config,
                get_all_models,
                get_providers,
            )
        except ImportError as exc:
            pytest.skip(
                f"synthesis_engine.config loader unavailable "
                f"(parallel refactor in flight): {exc}"
            )
        return {
            "load_engines_config": load_engines_config,
            "get_all_models": get_all_models,
            "get_providers": get_providers,
        }

    def test_loader_parses_v34_yaml(self, loader_fns):
        """The loader parses the v3.4 engines.yaml without error."""
        config = loader_fns["load_engines_config"](force_reload=True)
        assert isinstance(config, dict)
        assert "engines" in config

    def test_loader_surfaces_all_ollama_models(self, loader_fns):
        """get_all_models() includes the v3.4 ollama additions."""
        models = loader_fns["get_all_models"]()
        assert "ollama" in models
        ollama_names = {m["name"] for m in models["ollama"]}
        for model_name, _, _, _ in V3_4_LOCAL_MODELS:
            assert model_name in ollama_names, (
                f"Loader did not surface {model_name!r}"
            )

    def test_loader_marks_ollama_models_local(self, loader_fns):
        """Every ollama-provider model is marked is_local=True."""
        models = loader_fns["get_all_models"]()
        for model in models.get("ollama", []):
            assert model.get("is_local") is True, (
                f"{model['name']}: is_local should be True for ollama provider"
            )
