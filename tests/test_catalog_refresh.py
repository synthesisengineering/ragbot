"""Offline acceptance of the model catalog through actual consumer modules.

Every cloud model in engines.yaml (verified 2026-11-06; sources in
docs/model-catalog-2026-11-06.yaml) is checked through the loader, the chat
core, both LLM backends, and the real provider SDKs serializing against a mock
transport. Nothing here makes a network call.
"""
import os
from pathlib import Path
import pytest
import yaml
from synthesis_engine import config
from synthesis_engine.llm.base import LLMRequest
from synthesis_engine.llm.direct_backend import DirectBackend
from synthesis_engine.llm.litellm_backend import _build_completion_kwargs
from ragbot.core import _resolve_thinking_for_model

MODELS = {
 'openai/gpt-6-astra': ('judgment', 922000, 128000),
 'openai/gpt-6.1-sol': ('routine', 922000, 128000),
 'openai/gpt-6.1-terra': ('routine', 922000, 128000),
 'openai/gpt-6-luna': ('bulk', 922000, 128000),
 'anthropic/claude-fable-5-1': ('judgment', 1000000, 128000),
 'anthropic/claude-opus-5-5': ('judgment', 1000000, 128000),
 'anthropic/claude-sonnet-5-5': ('routine', 1000000, 128000),
 'anthropic/claude-haiku-5-5': ('bulk', 1000000, 128000),
 'gemini/gemini-4-argon': ('judgment', 1048576, 65536),
 'gemini/gemini-3.8-flash': ('routine', 1048576, 65536),
 'gemini/gemini-3.5-flash-lite': ('bulk', 1048576, 65536),
}
ANTHROPIC = [m for m in MODELS if m.startswith('anthropic/')]
GOOGLE_LEVELS = {
 'gemini/gemini-4-argon': ['low', 'medium', 'high'],
 'gemini/gemini-3.8-flash': ['low', 'medium', 'high'],
 'gemini/gemini-3.5-flash-lite': ['minimal', 'low', 'medium', 'high'],
}

# Superseded releases removed on 2026-11-06. Their ids must no longer resolve.
REMOVED = [
 'openai/gpt-6-sol', 'openai/gpt-5.6-sol', 'openai/gpt-5.6-terra', 'openai/gpt-5.6-luna',
 'anthropic/claude-fable-5', 'anthropic/claude-opus-4-8', 'anthropic/claude-sonnet-5',
 'anthropic/claude-haiku-4-5-20251001',
 'gemini/gemini-3.1-pro-preview', 'gemini/gemini-3-flash-preview', 'gemini/gemini-3.1-flash-lite-preview',
 'ollama_chat/qwen3.6:35b-a3b', 'ollama_chat/qwen3.6:27b', 'ollama_chat/llama4:scout',
 'ollama_chat/llama4:maverick', 'ollama_chat/deepseek-v3.2:671b', 'ollama_chat/mistral-large-3:675b',
 'ollama_chat/mistral-small-4:119b',
]


@pytest.mark.parametrize('model,expected', MODELS.items())
def test_actual_loader_capabilities(model, expected):
 info = config.get_model_info(model)
 assert info is not None
 assert (info['tier'], info['context_window'], info['max_output_tokens']) == expected
 assert info['supports_thinking'] and not info['is_local']
 assert info['thinking']['strict'] is True
 assert info['default_max_tokens'] == 8192


@pytest.mark.parametrize('model', REMOVED)
def test_superseded_models_are_gone(model):
 assert config.get_model_info(model) is None


def test_preferred_role_resolution():
 tiers = yaml.safe_load(Path(os.environ.get('SYNTHESIS_TIERS_FILE', Path(__file__).parent / 'fixtures/model-tier-preferences-2026-11-06.yaml')).read_text())
 for provider, block in tiers['providers'].items():
  for role in ['judgment', 'routine', 'bulk']:
   assert config.get_model_by_tier(provider, role) == config.normalize_model_id(provider, block[role][0])


def test_provider_defaults():
 assert {p: config.get_provider_config(p)['default_model'] for p in config.get_providers()} == {
  'openai': 'gpt-6.1-sol', 'anthropic': 'claude-sonnet-5-5', 'google': 'gemini/gemini-3.8-flash', 'ollama': 'gemma4:31b'}
 assert config.get_default_model() == 'anthropic/claude-sonnet-5-5'


@pytest.mark.parametrize('model', list(MODELS))
def test_explicit_supported_effort_survives_core(model):
 effort = 'xhigh' if model.startswith(('openai/', 'anthropic/')) else 'high'
 assert _resolve_thinking_for_model(model, effort)['reasoning_effort'] == effort


@pytest.mark.parametrize('effort', ['minimal', 'off', 'ultra', 'typo'])
def test_unsupported_effort_is_not_silently_replaced(effort):
 with pytest.raises(ValueError): _resolve_thinking_for_model('openai/gpt-6-astra', effort)


def test_gpt_6_1_sol_has_no_none_effort():
 # GPT-6.1 Sol dropped the `none` level that GPT-6 Sol had.
 with pytest.raises(ValueError): _resolve_thinking_for_model('openai/gpt-6.1-sol', 'none')


@pytest.mark.parametrize('effort', ['xhigh', 'max', 'none'])
def test_openai_request_builders_preserve_supported_effort(effort):
 model = 'openai/gpt-6.1-terra'
 req = LLMRequest(model=model, messages=[], temperature=0.6, max_tokens=9137, reasoning_effort=effort)
 outputs = [_build_completion_kwargs(req), DirectBackend._build_openai_kwargs(object.__new__(DirectBackend), req)]
 for out in outputs:
  assert out['max_completion_tokens'] == 9137 and 'max_tokens' not in out
  assert out['reasoning_effort'] == effort
  assert ('temperature' in out) == (effort == 'none')


@pytest.mark.parametrize('model', ANTHROPIC)
def test_anthropic_request_builders_preserve_effort(model):
 req = LLMRequest(model=model, messages=[], temperature=0.6, reasoning_effort='xhigh')
 outputs = [_build_completion_kwargs(req), DirectBackend._build_anthropic_kwargs(object.__new__(DirectBackend), req)]
 for out in outputs:
  assert out['output_config']['effort'] == 'xhigh'
  assert out['thinking'] == {'type': 'adaptive'}
  assert 'temperature' not in out and 'reasoning_effort' not in out


@pytest.mark.parametrize('model,effort', [('gemini/gemini-4-argon', 'medium'), ('gemini/gemini-3.8-flash', 'high'), ('gemini/gemini-3.5-flash-lite', 'minimal')])
def test_google_direct_builder_uses_level_not_budget(model, effort):
 backend = object.__new__(DirectBackend)
 name, contents, cfg = backend._build_google_request(LLMRequest(model=model, messages=[], temperature=0.6, reasoning_effort=effort))
 assert name == model.removeprefix('gemini/')
 assert str(cfg.thinking_config.thinking_level).lower().split('.')[-1] == effort
 assert cfg.thinking_config.thinking_budget is None
 assert cfg.temperature is None


def test_model_endpoints_preserve_catalog_modes(monkeypatch):
 import asyncio, importlib.util
 path = Path(config.__file__).parents[1] / 'api/routers/models.py'
 spec = importlib.util.spec_from_file_location('catalog_model_endpoint', path)
 module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
 monkeypatch.setattr(module, 'check_api_keys', lambda: {p: False for p in config.get_providers()})
 monkeypatch.setattr(module, 'get_available_models', config.get_all_models)
 outputs = [asyncio.run(module.list_all_models())['models'], [m.model_dump() for m in asyncio.run(module.list_models()).models]]
 for models in outputs:
  info = next(m for m in models if m['id'] == 'openai/gpt-6-astra')
  assert info['thinking']['modes'] == ['low', 'medium', 'high', 'xhigh', 'max']


@pytest.mark.parametrize('model', ['openai/gpt-6-astra', 'anthropic/claude-fable-5-1', 'gemini/gemini-4-argon'])
def test_transport_rejects_unsupported_effort(model):
 req = LLMRequest(model=model, messages=[], reasoning_effort='off')
 with pytest.raises(ValueError): _build_completion_kwargs(req)
 backend = object.__new__(DirectBackend)
 build = backend._build_openai_kwargs if model.startswith('openai') else backend._build_anthropic_kwargs if model.startswith('anthropic') else backend._build_google_request
 with pytest.raises(ValueError): build(req)


@pytest.mark.parametrize('extra', [{'reasoning_effort': 'low'}, {'model': 'other-model'}, {'max_tokens': 17}, {'max_completion_tokens': 17}])
def test_freeform_cannot_replace_request_identity_effort_or_limit(extra):
 req = LLMRequest(model='openai/gpt-6.1-sol', messages=[], max_tokens=8192, reasoning_effort='xhigh', extra=extra)
 with pytest.raises(ValueError): _build_completion_kwargs(req)
 with pytest.raises(ValueError): DirectBackend._build_openai_kwargs(object.__new__(DirectBackend), req)


TOOLS = {'tools': [{'type': 'function', 'function': {'name': 'example', 'parameters': {'type': 'object'}}}]}


@pytest.mark.parametrize('model', ['openai/gpt-6-astra', 'openai/gpt-6.1-sol', 'openai/gpt-6.1-terra'])
def test_tool_requests_never_lower_effort_for_unsupported_endpoint(model):
 req = LLMRequest(model=model, messages=[], reasoning_effort='xhigh', extra=TOOLS)
 with pytest.raises(ValueError, match='Responses API'): _build_completion_kwargs(req)


def test_luna_takes_chat_completions_tools_only_at_effort_none():
 _build_completion_kwargs(LLMRequest(model='openai/gpt-6-luna', messages=[], reasoning_effort='none', extra=TOOLS))
 with pytest.raises(ValueError, match='Responses API'):
  _build_completion_kwargs(LLMRequest(model='openai/gpt-6-luna', messages=[], reasoning_effort='low', extra=TOOLS))


@pytest.mark.parametrize('model', [m.removeprefix('anthropic/') for m in ANTHROPIC])
def test_real_anthropic_sdk_serializes_exact_effort_without_network(model):
 import anthropic, httpx2 as httpx, json
 bodies = []
 def respond(request):
  bodies.append(json.loads(request.content))
  return httpx.Response(200, json={'id': 'fixture', 'type': 'message', 'role': 'assistant', 'model': model, 'content': [{'type': 'text', 'text': 'fixture'}], 'stop_reason': 'end_turn', 'usage': {'input_tokens': 1, 'output_tokens': 1}})
 with httpx.Client(transport=httpx.MockTransport(respond)) as transport:
  client = anthropic.Anthropic(api_key='fixture-noncredential', base_url='https://fixture.invalid', http_client=transport)
  request = LLMRequest(model='anthropic/' + model, messages=[{'role': 'user', 'content': 'fixture'}], temperature=0.6, reasoning_effort='xhigh')
  client.messages.create(**DirectBackend._build_anthropic_kwargs(object.__new__(DirectBackend), request))
 assert bodies[0]['output_config'] == {'effort': 'xhigh'}
 assert bodies[0]['thinking'] == {'type': 'adaptive'}
 assert 'temperature' not in bodies[0]


def test_real_openai_sdk_serializes_exact_effort_without_network():
 import openai, httpx, json
 bodies = []
 def respond(request):
  bodies.append(json.loads(request.content))
  return httpx.Response(200, json={'id': 'fixture', 'object': 'chat.completion', 'created': 0, 'model': 'gpt-6.1-sol', 'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': 'fixture'}, 'finish_reason': 'stop'}]})
 with httpx.Client(transport=httpx.MockTransport(respond)) as transport:
  client = openai.OpenAI(api_key='fixture-noncredential', base_url='https://fixture.invalid', http_client=transport)
  request = LLMRequest(model='openai/gpt-6.1-sol', messages=[{'role': 'user', 'content': 'fixture'}], reasoning_effort='xhigh')
  client.chat.completions.create(**DirectBackend._build_openai_kwargs(object.__new__(DirectBackend), request))
 assert bodies[0]['reasoning_effort'] == 'xhigh'
 assert bodies[0]['max_completion_tokens'] == request.max_tokens
 assert 'temperature' not in bodies[0]


@pytest.mark.parametrize('model', [m.removeprefix('anthropic/') for m in ANTHROPIC])
def test_actual_litellm_anthropic_transform_retains_effort(model):
 import litellm
 from litellm.llms.anthropic.chat.transformation import AnthropicConfig
 req = LLMRequest(model='anthropic/' + model, messages=[], reasoning_effort='xhigh')
 kwargs = _build_completion_kwargs(req)
 data = {}; options = {'output_config': kwargs['output_config']}
 old = litellm.drop_params
 try:
  litellm.drop_params = False
  AnthropicConfig()._apply_output_config(data, model, options)
 finally: litellm.drop_params = old
 assert data['output_config'] == {'effort': 'xhigh'}


@pytest.mark.parametrize('model', list(MODELS))
def test_litellm_contract_does_not_invent_prices(model):
 import litellm
 from synthesis_engine.llm.model_parameters import register_litellm_capabilities
 # Record the actual raw map, not get_model_info's synthetic zero defaults.
 before = {key: {k: v for k, v in row.items() if ('cost' in k or 'price' in k) and v is not None} for key, row in litellm.model_cost.items() if isinstance(row, dict)}
 register_litellm_capabilities(model)
 for key, row in litellm.model_cost.items():
  if not isinstance(row, dict): continue
  prices = {k: v for k, v in row.items() if ('cost' in k or 'price' in k) and v is not None}
  assert prices == before.get(key, {}), (key, prices, before.get(key, {}))


def test_model_missing_from_litellms_map_still_takes_its_effort():
 """GPT-6.1 Terra is absent from LiteLLM 1.108.1's bundled model map.

 Ragbot registers the catalog's verified capabilities, so a request at an
 effort LiteLLM would otherwise refuse for an unknown model still passes,
 with no price invented.
 """
 import litellm
 from litellm.utils import get_optional_params
 keys = ['openai/gpt-6.1-terra', 'gpt-6.1-terra']
 saved = {k: litellm.model_cost.pop(k) for k in keys if k in litellm.model_cost}
 try:
  kwargs = _build_completion_kwargs(LLMRequest(model='openai/gpt-6.1-terra', messages=[], reasoning_effort='xhigh'))
  assert kwargs['reasoning_effort'] == 'xhigh' and kwargs['drop_params'] is False
  registered = litellm.model_cost['openai/gpt-6.1-terra']
  assert registered['supports_xhigh_reasoning_effort'] is True
  assert not any(('cost' in k or 'price' in k) and v for k, v in registered.items())
  params = get_optional_params(model='gpt-6.1-terra', custom_llm_provider='openai', reasoning_effort='xhigh', drop_params=False)
  assert params['reasoning_effort'] == 'xhigh'
 finally:
  for k in keys: litellm.model_cost.pop(k, None)
  litellm.model_cost.update(saved)


def test_new_contract_cannot_silently_drop_effort():
 assert _build_completion_kwargs(LLMRequest(model='openai/gpt-6-astra', messages=[], reasoning_effort='xhigh'))['drop_params'] is False


@pytest.mark.parametrize('model,effort', [(model, effort) for model, efforts in GOOGLE_LEVELS.items() for effort in efforts])
def test_actual_litellm_google_transform_retains_exact_level(model, effort):
 from litellm.llms.gemini.chat.transformation import GoogleAIStudioGeminiConfig
 kwargs = _build_completion_kwargs(LLMRequest(model=model, messages=[], temperature=0.6, reasoning_effort=effort))
 assert 'temperature' not in kwargs
 mapped = GoogleAIStudioGeminiConfig().map_openai_params({'reasoning_effort': kwargs['reasoning_effort']}, {}, model.removeprefix('gemini/'), False)
 assert mapped['thinkingConfig']['thinkingLevel'] == effort


@pytest.mark.parametrize('extra', [{'reasoning_effort': 'low'}, {'temperature': 0.5}])
def test_google_direct_does_not_silently_ignore_freeform_options(extra):
 req = LLMRequest(model='gemini/gemini-3.8-flash', messages=[], reasoning_effort='high', extra=extra)
 with pytest.raises(ValueError, match='Unsupported direct Google'):
  DirectBackend._build_google_request(object.__new__(DirectBackend), req)
