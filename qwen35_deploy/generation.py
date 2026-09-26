"""Serial generation with explicit tested request limits."""
import time
import torch
from qwen35_deploy.operators.int8 import Int8Experts


@torch.inference_mode()
def generate(model,tokenizer,cache,prompt,max_new_tokens,max_context,prefill_chunk=0,retain_prefill_experts=False):
    request_started=time.perf_counter()
    if not isinstance(prompt,str) or not prompt:
        raise ValueError('prompt must be a nonempty string')
    if type(max_new_tokens) is not int or max_new_tokens<1:
        raise ValueError('max_new_tokens must be a positive integer')
    if retain_prefill_experts and (not prefill_chunk or not hasattr(cache,'seed_enabled')):
        raise ValueError('Retention requires the installed seeded cache and chunked profile')
    text=tokenizer.apply_chat_template([{'role':'user','content':prompt}],
        tokenize=False,add_generation_prompt=True,enable_thinking=False)
    ids=tokenizer(text,return_tensors='pt').input_ids
    prompt_tokens=ids.shape[-1]
    if getattr(cache,'frequency_seed_deployment',False) and (prompt_tokens>4096 or max_new_tokens>33):
        raise ValueError('Frequency-seed experiment supports up to 4096 prompt and 33 output tokens')
    if prefill_chunk and (prompt_tokens>8192 or max_new_tokens>256):
        raise ValueError('Experimental chunked profile supports up to 8192 prompt and 256 output tokens')
    if prompt_tokens+max_new_tokens>max_context:
        raise ValueError(f'{prompt_tokens}+{max_new_tokens} exceeds configured context {max_context}')
    ids=ids.cuda()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    before=(cache.hits,cache.misses,cache.transfer_bytes)
    seed_before=(getattr(cache,'seeded_experts',0),getattr(cache,'seed_d2d_bytes',0))
    started=time.perf_counter()
    past=None
    tokens=[]
    durations=[]
    eos=tokenizer.eos_token_id
    eos=set(eos if isinstance(eos,list) else [eos])
    for _ in range(max_new_tokens):
        step=time.perf_counter()
        if not tokens and prefill_chunk:
            for begin in range(0,prompt_tokens,prefill_chunk):
                if retain_prefill_experts:cache.seed_enabled=begin+prefill_chunk>=prompt_tokens
                out=model(input_ids=ids[:,begin:begin+prefill_chunk],past_key_values=past,
                          use_cache=True,logits_to_keep=1)
                past=out.past_key_values
        else:
            out=model(input_ids=ids,past_key_values=past,use_cache=True,logits_to_keep=1)
        past=out.past_key_values
        ids=out.logits[:,-1].argmax(-1,keepdim=True)
        token=int(ids.item())
        token_ready=time.perf_counter()
        durations.append(token_ready-step)
        if not tokens:first_token_ready=token_ready
        tokens.append(token)
        if token in eos:break
    seconds=time.perf_counter()-started
    text=tokenizer.decode(tokens,skip_special_tokens=True)
    return {'text':text,'token_ids':tokens,
        'prompt_tokens':prompt_tokens,'generated_tokens':len(tokens),
        'prefill_chunk_tokens':prefill_chunk or prompt_tokens,
        'prefill_coverage_scope':'last_chunk' if prefill_chunk else 'whole_prompt',
        'request_seconds':time.perf_counter()-request_started,'model_seconds':seconds,
        'first_token_ready_seconds':first_token_ready-request_started,
        'prefill_seconds':durations[0],'prefill_tokens_per_second':prompt_tokens/durations[0],
        'decode_tokens_per_second':(len(tokens)-1)/sum(durations[1:]) if len(tokens)>1 else None,
        'step_seconds':durations,'peak_cuda_bytes':torch.cuda.max_memory_allocated(),
        'expert_cache_hits':cache.hits-before[0],'expert_cache_misses':cache.misses-before[1],
        'expert_transfer_bytes':cache.transfer_bytes-before[2],
        'retained_experts':getattr(cache,'seeded_experts',0)-seed_before[0],
        'retention_d2d_bytes':getattr(cache,'seed_d2d_bytes',0)-seed_before[1],
        'prefill_expert_coverage':next((m.prefill_pipeline.layer_stats for m in model.modules()
            if isinstance(m,Int8Experts) and m.prefill_pipeline is not None),[]),
        'scope':'single serial non-streaming response; internal first-token readiness; cold capture included when needed; not acceptance benchmark'}
