"""Offline final-body and capability-only regression checks."""
import copy,json
import pytest
from synthesis_engine.llm.base import LLMRequest
from synthesis_engine.llm.direct_backend import DirectBackend
from synthesis_engine.llm.litellm_backend import _build_completion_kwargs

CASES=[('openai/gpt-6-sol','openai','gpt-6-sol','max_completion_tokens'),('anthropic/claude-fable-5-1','anthropic','claude-fable-5-1','max_tokens')]

def serialize(provider,kwargs):
 bodies=[]
 if provider=='openai':
  import openai,httpx
  def respond(req):
   bodies.append(json.loads(req.content));return httpx.Response(200,json={'id':'fixture','object':'chat.completion','created':0,'model':'gpt-6-sol','choices':[]})
  with httpx.Client(transport=httpx.MockTransport(respond)) as http:
   with openai.OpenAI(api_key='synthetic',base_url='https://fixture.invalid',http_client=http) as client:client.chat.completions.create(**kwargs)
 else:
  import anthropic,httpx2 as httpx
  def respond(req):
   bodies.append(json.loads(req.content));return httpx.Response(200,json={'id':'fixture','type':'message','role':'assistant','model':'claude-fable-5-1','content':[],'stop_reason':'end_turn','usage':{'input_tokens':1,'output_tokens':1}})
  with httpx.Client(transport=httpx.MockTransport(respond)) as http:
   with anthropic.Anthropic(api_key='synthetic',base_url='https://fixture.invalid',http_client=http) as client:client.messages.create(**kwargs)
 return bodies[0]

@pytest.mark.parametrize('model,provider,native,limit',CASES)
@pytest.mark.parametrize('bare',[False,True])
def test_serialized_legitimate_extensions_preserve_contract(model,provider,native,limit,bare):
 body={'metadata':{'extension':'fixture'},'provider_extension':{'nested':True},'model':native,limit:8192}
 if provider=='anthropic':body['output_config']={'format':{'type':'json_schema','schema':{'type':'object'}}};body['thinking']={'display':'summarized'}
 else:body['reasoning_effort']='xhigh'
 extra={'extra_body':body,'extra_headers':{'X-Fixture':'preserved'}};before=copy.deepcopy(extra)
 req=LLMRequest(model=native if bare else model,messages=[{'role':'user','content':'fixture'}],reasoning_effort='xhigh',max_tokens=8192,extra=extra)
 backend=object.__new__(DirectBackend)
 kwargs=(backend._build_openai_kwargs if provider=='openai' else backend._build_anthropic_kwargs)(req)
 result=serialize(provider,kwargs)
 assert result['model']==native and result[limit]==8192 and result['metadata']==body['metadata'] and result['provider_extension']==body['provider_extension']
 if provider=='anthropic':assert result['output_config']['effort']=='xhigh' and result['output_config']['format']==body['output_config']['format'] and result['thinking']=={'type':'adaptive','display':'summarized'}
 else:assert result['reasoning_effort']=='xhigh'
 assert extra==before

@pytest.mark.parametrize('model,provider,native,limit',CASES)
def test_litellm_matching_native_identity_extension_allowed(model,provider,native,limit):
 result=_build_completion_kwargs(LLMRequest(model=model,messages=[],reasoning_effort='xhigh',extra={'extra_body':{'model':native,'metadata':{'fixture':True}}}))
 assert result['model']==model and result['extra_body']['model']==native

@pytest.mark.parametrize('model,provider,native,limit',CASES)
@pytest.mark.parametrize('kind',['effort','model','tokens','sampling','tools','nested','nonobject','alternate_limit'])
@pytest.mark.parametrize('builder',['direct','litellm'])
def test_final_body_conflicts_rejected_before_serialization(model,provider,native,limit,kind,builder):
 body={'effort':({'output_config':{'effort':'low'}} if provider=='anthropic' else {'reasoning_effort':'low'}),'model':{'model':'other'},'tokens':{limit:1},'sampling':{'temperature':0.9},'tools':({'thinking':{'type':'disabled'}} if provider=='anthropic' else {'tools':[{'type':'function'}]}),'nested':{'extra_body':{'model':'other'}},'nonobject':[],'alternate_limit':{'max_output_tokens':1}}[kind]
 req=LLMRequest(model=model,messages=[],reasoning_effort='xhigh',extra={'extra_body':body})
 backend=object.__new__(DirectBackend)
 build=_build_completion_kwargs if builder=='litellm' else backend._build_openai_kwargs if provider=='openai' else backend._build_anthropic_kwargs
 with pytest.raises(ValueError):build(req)

@pytest.mark.parametrize('model,provider,native,limit',CASES)
def test_bare_native_identifiers_do_not_escape_contract(model,provider,native,limit):
 req=LLMRequest(model=native,messages=[],reasoning_effort='xhigh',extra={'extra_body':{'model':'other'}})
 backend=object.__new__(DirectBackend)
 with pytest.raises(ValueError):(backend._build_openai_kwargs if provider=='openai' else backend._build_anthropic_kwargs)(req)

@pytest.mark.parametrize('model',['openai/gpt-6-astra','openai/gpt-6-sol','openai/gpt-6-luna','anthropic/claude-fable-5-1','anthropic/claude-opus-5-5','gemini/gemini-3.8-flash','gemini/gemini-3.5-flash-lite'])
def test_registration_preserves_every_unrelated_field_and_all_prices(model):
 import litellm
 from synthesis_engine.llm.model_parameters import register_litellm_capabilities
 backup=json.loads((__import__('pathlib').Path(litellm.__file__).parent/'model_prices_and_context_window_backup.json').read_text())
 litellm.model_cost.clear();litellm.model_cost.update(copy.deepcopy(backup));before=copy.deepcopy(litellm.model_cost)
 register_litellm_capabilities(model)
 aliases={model,model.partition('/')[2]};allowed={'litellm_provider','supports_reasoning','supports_output_config','supports_adaptive_thinking','thinking_always_on'}|{f'supports_{v}_reasoning_effort' for v in ['none','minimal','low','medium','high','xhigh','max']}
 for name,row in litellm.model_cost.items():
  old=before.get(name,{})
  if name not in aliases:assert row==old
  else:
   assert {k:v for k,v in row.items() if k not in allowed}=={k:v for k,v in old.items() if k not in allowed}
   assert row.get('supports_reasoning') is True
 before_repeat=copy.deepcopy(litellm.model_cost);register_litellm_capabilities(model);assert litellm.model_cost==before_repeat

def test_registration_foreign_alias_collision_has_no_side_effect(monkeypatch):
 import litellm
 from synthesis_engine.llm.model_parameters import register_litellm_capabilities
 monkeypatch.setitem(litellm.model_cost,'gpt-6-sol',{'litellm_provider':'another-provider','max_input_tokens':37,'input_cost_per_token':0.123})
 before=copy.deepcopy(litellm.model_cost)
 try:register_litellm_capabilities('openai/gpt-6-sol')
 except ValueError:assert litellm.model_cost==before
 else:
  assert litellm.model_cost['gpt-6-sol']==before['gpt-6-sol']
  assert litellm.model_cost['openai/gpt-6-sol']['litellm_provider']=='openai'
