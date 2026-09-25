"""Apply verified catalog contracts without substituting a model or effort.

The catalog owns identifiers, allowed modes and provider request policy. This
module owns transport translation; adding a model never requires an ID check.
"""
from typing import Any
from collections.abc import Mapping
from copy import deepcopy
from threading import RLock

_REGISTRATION_LOCK = RLock()


def model_contract(model: str) -> tuple[dict[str, Any], dict[str, Any]]:
    from ..config import get_model_info, get_all_models
    info = get_model_info(model) or {}
    if not info and "/" not in model:
        matches = [entry for entries in get_all_models().values() for entry in entries
                   if entry["id"].partition("/")[2] == model]
        if len(matches) > 1:
            raise ValueError("Native model identifier is ambiguous; qualify its provider")
        info = matches[0] if matches else {}
    return info, info.get("request_parameters") or {}


def resolve_effort(info: dict[str, Any], requested: str | None) -> str:
    meta = info.get("thinking") or {}
    effort = str(requested).strip().lower() if requested is not None else "auto"
    if effort in ("auto", "default"):
        effort = meta.get("default")
    if effort not in meta.get("modes", []):
        raise ValueError(f"Unsupported reasoning effort {requested!r} for {info.get('id')}; allowed: {meta.get('modes', [])}")
    return effort


def apply_request_contract(request, kwargs: dict[str, Any], *, native_model: bool = False) -> dict[str, Any]:
    info, contract = model_contract(request.model)
    if not contract:
        return kwargs
    effort = resolve_effort(info, request.reasoning_effort)
    extra = request.extra or {}
    # The free-form transport extension cannot replace identity or an explicit
    # effort. A conflict is actionable; it is never repaired by lowering effort.
    expected_model = request.model.partition("/")[2] if native_model and "/" in request.model else request.model
    if kwargs.get("model") != expected_model:
        raise ValueError("Model override conflicts with request identity")
    if "reasoning_effort" in extra and extra["reasoning_effort"] != effort:
        raise ValueError("Conflicting reasoning effort override")
    token_limit = contract.get("token_limit", "max_tokens")
    if contract.get("token_limit") or contract.get("effort") == "output_config":
        key = token_limit
        for token_key in ("max_tokens", "max_completion_tokens"):
            if token_key in extra and extra[token_key] != request.max_tokens:
                raise ValueError("Conflicting output-token limit override")
            kwargs.pop(token_key, None)
        kwargs[key] = request.max_tokens
    if contract.get("effort") == "output_config":
        supplied = extra.get("output_config") or {}
        if not isinstance(supplied, Mapping):
            raise ValueError("output_config must be an object")
        if "effort" in supplied and supplied["effort"] != effort:
            raise ValueError("Conflicting output_config effort override")
        for thinking in (request.thinking, extra.get("thinking")):
            if thinking is not None and thinking != {"type": "adaptive"}:
                raise ValueError("This model requires adaptive thinking without a token budget")
        kwargs["thinking"] = {"type": "adaptive"}
        kwargs["output_config"] = {**supplied, "effort": effort}
        kwargs.pop("reasoning_effort", None)
    else:
        if request.thinking is not None or "thinking" in extra:
            raise ValueError("This model requires its declared reasoning-effort control")
        kwargs["reasoning_effort"] = effort
    sampling = contract.get("sampling")
    if sampling == "unsupported" or (sampling == "non_reasoning_only" and effort != "none"):
        for key in ("temperature", "top_p", "top_k", "top_logprobs", "logprobs"):
            kwargs.pop(key, None)
    tool_policy = contract.get("tool_calling")
    if (kwargs.get("tools") or kwargs.get("functions")) and (
        tool_policy == "responses_only" or (tool_policy == "non_reasoning_only" and effort != "none")
    ):
        raise ValueError("This model/effort requires the Responses API for tool calling; no effort was changed")
    return validate_final_body(request, kwargs, contract, effort, token_limit, info["id"].partition("/")[2])


def validate_final_body(request, kwargs, contract, effort, token_limit, wire_model):
    """Validate the SDK's final shallow extra_body merge, preserving extensions.

    Both SDKs merge extra_body after typed arguments. Validating only kwargs
    would let this later layer replace an explicit model, effort or budget.
    """
    body = kwargs.get("extra_body")
    if body is None:
        return kwargs
    if not isinstance(body, Mapping):
        raise ValueError("extra_body must be an object")
    body = deepcopy(dict(body))
    if "extra_body" in body:
        raise ValueError("Nested transport extra_body is not a provider body extension")
    for key, expected in (("model", wire_model), (token_limit, request.max_tokens)):
        if key in body and body[key] != expected:
            raise ValueError(f"extra_body.{key} conflicts with the explicit request")
    for key in {"max_tokens", "max_completion_tokens", "max_output_tokens"} - {token_limit}:
        if key in body:
            raise ValueError(f"extra_body.{key} is not this model's token-limit field")
    if contract.get("effort") == "output_config":
        if "reasoning_effort" in body:
            raise ValueError("Anthropic effort belongs in output_config")
        if "output_config" in body:
            supplied = body["output_config"]
            if not isinstance(supplied, Mapping) or ("effort" in supplied and supplied["effort"] != effort):
                raise ValueError("extra_body.output_config conflicts with explicit effort")
            # SDK merging is shallow. Preserve canonical effort when an extension
            # adds a sibling output option without repeating the effort field.
            body["output_config"] = {**kwargs["output_config"], **supplied, "effort": effort}
        if "thinking" in body:
            supplied = body["thinking"]
            if not isinstance(supplied, Mapping) or supplied.get("type", "adaptive") != "adaptive" or "budget_tokens" in supplied:
                raise ValueError("extra_body.thinking conflicts with adaptive thinking")
            body["thinking"] = {**supplied, "type": "adaptive"}
    else:
        if "reasoning_effort" in body and body["reasoning_effort"] != effort:
            raise ValueError("extra_body.reasoning_effort conflicts with explicit effort")
        if "thinking" in body or "output_config" in body:
            raise ValueError("extra_body must use the model's declared reasoning control")
    sampling = contract.get("sampling")
    if sampling == "unsupported" or (sampling == "non_reasoning_only" and effort != "none"):
        if set(body) & {"temperature", "top_p", "top_k", "top_logprobs", "logprobs"}:
            raise ValueError("extra_body sampling controls are unsupported at this effort")
    tool_policy = contract.get("tool_calling")
    if (body.get("tools") or body.get("functions")) and (
        tool_policy == "responses_only" or (tool_policy == "non_reasoning_only" and effort != "none")
    ):
        raise ValueError("This model/effort requires the Responses API for tool calling; no effort was changed")
    kwargs["extra_body"] = body
    return kwargs


def register_litellm_capabilities(model: str) -> None:
    """Supply verified capability fields absent from a bundled dependency map.

    Prices are neither inferred nor registered here. Existing provider pricing
    remains owned by LiteLLM; a missing price is not a claim of zero cost.
    """
    info, contract = model_contract(model)
    if not contract:
        return
    import litellm
    model = info["id"]
    provider, _, _ = model.partition("/")
    meta = info["thinking"]
    capabilities = {
        "litellm_provider": provider,
        "supports_reasoning": True,
    }
    for effort in meta["modes"]:
        capabilities[f"supports_{effort}_reasoning_effort"] = True
    if contract.get("effort") == "output_config":
        capabilities["supports_output_config"] = True
        capabilities["supports_adaptive_thinking"] = True
        capabilities["thinking_always_on"] = bool(meta.get("always_on"))
    # Explicit provider-qualified registration cannot change a different
    # provider's similarly named model, or any client/user model selection.
    # register_model may resolve the qualified name to its canonical bare alias
    # and synthesize unrelated null fields. Retain exactly the preexisting
    # metadata at both possible keys, adding only these verified capabilities.
    # Serializing our registration calls also prevents competing requests from
    # restoring one another's intermediate normalization result.
    native = model.partition("/")[2]
    keys = {model, native}
    with _REGISTRATION_LOCK:
        before = {key: deepcopy(litellm.model_cost[key]) for key in keys if key in litellm.model_cost}
        if model in before and before[model].get("litellm_provider", provider) != provider:
            raise ValueError("Capability registration resolves to a different provider")
        litellm.register_model({model: capabilities}, persist_across_reloads=False)
        for key in keys:
            current = litellm.model_cost.get(key)
            if current is None or current == before.get(key):
                continue
            previous = before.get(key, {})
            if previous.get("litellm_provider", provider) != provider:
                # Never claim a similarly named model belonging to another provider.
                for prior_key in keys:
                    if prior_key in before:
                        litellm.model_cost[prior_key] = before[prior_key]
                    else:
                        litellm.model_cost.pop(prior_key, None)
                raise ValueError("Capability registration resolved to a different provider")
            litellm.model_cost[key] = {**previous, **capabilities}
