"""Text inference baseline with CPU INT8 expert bank and bounded GPU LRU."""
import json
from pathlib import Path

import torch
from accelerate import init_empty_weights
from accelerate.utils import set_module_tensor_to_device
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModelForImageTextToText
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeTextRotaryEmbedding

from qwen35_deploy.convert import sha
from qwen35_deploy.operators.int8 import Int8Linear, Int8Experts, ExpertCache


def replace(model, name, module):
    parent, _, child = name.rpartition('.')
    setattr(model.get_submodule(parent), child, module)


class Embedding(torch.nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.register_buffer('weight', weight)

    def forward(self, ids):
        return torch.nn.functional.embedding(ids.cpu(), self.weight).to(ids.device)


def load(root, source, cache_gib=7, verify=True):
    decode_direct=True;grouped_decode=True;async_descriptor=True
    prefill_pipeline=True;large_prefill_tile=True;packed_experts=False;target_context=4129
    if packed_experts and not prefill_pipeline:raise ValueError('Packed experts require pipeline')
    manifest = json.loads((root / 'manifest.json').read_text())
    assert manifest['complete']
    source = Path(source)
    if sha(source / 'config.json') != manifest['identity']['source_config_sha256']:
        raise RuntimeError('Source configuration changed')
    config = AutoConfig.from_pretrained(source, local_files_only=True)
    expected={'num_hidden_layers':40,'hidden_size':2048,'num_experts':256,
              'num_experts_per_tok':8,'moe_intermediate_size':512}
    if any(getattr(config.text_config,k,None)!=v for k,v in expected.items()):
        raise ValueError('Unsupported model geometry')
    if config.text_config.layer_types.count('full_attention')!=10:
        raise ValueError('Unsupported attention layer layout')
    config._attn_implementation = 'sdpa'
    with init_empty_weights(include_buffers=True):
        model = AutoModelForImageTextToText.from_config(config, torch_dtype=torch.bfloat16)
    cache = ExpertCache(int((8 if cache_gib is None else cache_gib) * 2**30))
    expert_banks = {}
    for r in manifest['records']:
        path = root / r['file']
        if path.stat().st_size != r['bytes'] or (verify and sha(path) != r['sha256']):
            raise RuntimeError('Corrupt artifact: ' + str(path))
        data = load_file(path, device='cpu')
        name = r['name']
        if '.mlp.experts.' in name:
            expert_banks[name] = data
        elif 'embed_tokens' in name:
            replace(model, name.removesuffix('.weight'), Embedding(data['weight']))
        elif r['format'] == 'int8_g128_fp32scale':
            replace(model, name.removesuffix('.weight'), Int8Linear(data['q'].cuda(), data['scale'].cuda()))
        else:
            set_module_tensor_to_device(model, name, 'cuda', value=data['weight'])
    for layer in range(config.text_config.num_hidden_layers):
        prefix = f'model.language_model.layers.{layer}.mlp.experts'
        replace(model, prefix, Int8Experts(layer, expert_banks[prefix+'.gate_up_proj'],
                                           expert_banks[prefix+'.down_proj'], cache))
        model.get_submodule(prefix).decode_direct = decode_direct
        model.get_submodule(prefix).grouped_decode = grouped_decode
        model.get_submodule(prefix).async_descriptor = async_descriptor
    # Modules now own the CPU banks. Drop duplicate mmap references before
    # constructing pinned copies, so each old mapping can be released promptly.
    expert_banks.clear()
    rope = Qwen3_5MoeTextRotaryEmbedding(config.text_config, device='cuda')
    model.model.language_model.rotary_emb = rope
    # Keep lazy vision parameters from deciding the text model's device.
    text_model = model.model._modules.pop('language_model')
    model.model._modules = {'language_model': text_model, **model.model._modules}
    missing = [n for n,p in model.named_parameters() if p.is_meta and
               not (n.startswith('model.visual.') or n.startswith('mtp.'))]
    if missing:
        raise RuntimeError(f'Unloaded text tensors: {missing}')
    for module in model.modules():
        if isinstance(module,Int8Linear):
            module.large_prefill_tile=large_prefill_tile
    if prefill_pipeline:
        from qwen35_deploy.runtime.prefill import PrefillPipeline
        modules=[model.get_submodule(f'model.language_model.layers.{i}.mlp.experts') for i in range(config.text_config.num_hidden_layers)]
        pipeline=PrefillPipeline(modules,cache,packed=packed_experts)
        model.model.language_model.register_forward_pre_hook(pipeline.forward_pre_hook,with_kwargs=True)
        for module in modules:
            module.prefill_pipeline=pipeline
    if cache_gib is None:
        c=config.text_config
        full_layers=sum(t=='full_attention' for t in c.layer_types)
        kv_bytes=target_context*full_layers*2*c.num_key_value_heads*c.head_dim*2
        recurrent_bytes=(c.num_hidden_layers-full_layers)*c.linear_num_value_heads*c.linear_key_head_dim*c.linear_value_head_dim*4
        free,_=torch.cuda.mem_get_info()
        capacity=min(8*2**30,free-kv_bytes-recurrent_bytes-2**30)
        cache.max_bytes=int(capacity//(256*2**20)*(256*2**20))
        if cache.max_bytes<256*2**20:
            raise MemoryError('Insufficient GPU headroom for requested context and expert cache')
    model.eval()
    return model, cache, source
