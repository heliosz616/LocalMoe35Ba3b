"""Explicit startup capture of existing GDN graphs using private zero states.

Does not run MoE, read a prompt, populate the expert cache, or create request KV.
Preparation cost must be included in ready latency, never silently discarded.
"""
import time
import torch
from qwen35_deploy.operators.gdn_graph import GraphGDN,StaticCache


def private_inputs(module):
    original=module.original
    weight=original.conv1d.weight
    hidden=torch.zeros((1,1,original.hidden_size),device=weight.device,dtype=weight.dtype)
    conv=torch.zeros((1,original.conv_dim,original.conv_kernel_size),device=weight.device,dtype=weight.dtype)
    recurrent=torch.zeros((1,original.num_v_heads,original.head_k_dim,original.head_v_dim),
                          device=weight.device,dtype=torch.float32)
    return hidden,conv,recurrent


@torch.inference_mode()
def prepare(model):
    modules=[m for m in model.modules() if isinstance(m,GraphGDN)]
    if len(modules)!=30 or any(m.graph is not None for m in modules):
        raise ValueError('Prepare exactly 30 freshly installed GDN graphs before inference')
    torch.cuda.synchronize();started=time.perf_counter();rows=[]
    for module in modules:
        hidden,conv,recurrent=private_inputs(module)
        private=StaticCache(module.original.layer_idx,conv,recurrent)
        module(hidden,cache_params=private)
        assert module.graph is not None
        module.replays=0
        rows.append({'layer':module.original.layer_idx,'hidden_shape':list(hidden.shape),
                     'conv_shape':list(conv.shape),'recurrent_shape':list(recurrent.shape),
                     'conv_dtype':str(conv.dtype),'recurrent_dtype':str(recurrent.dtype)})
    torch.cuda.synchronize()
    return {'seconds':time.perf_counter()-started,'graphs':rows,
            'scope':'private zero-state graph capture; no prompt, expert-cache priming or request KV'}
