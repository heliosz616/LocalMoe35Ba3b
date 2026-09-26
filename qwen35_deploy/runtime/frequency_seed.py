"""Experimental causal prefill-frequency seeding, inside the existing arena.

GPU arithmetic/copy-event order matches GPUPlan. No future decode routing is used.
Ordinary full-bank prefill only; do not compose with sparse-prefill overrides.
"""
import types
from collections import Counter
from qwen35_deploy.operators.int8 import Int8Experts
from qwen35_deploy.runtime.gpu_routes import GPUPlan,routed_moe


def select_seeds(routes,quota):
    last=list(routes[-1])
    if quota<len(last) or len(set(last))!=len(last):raise ValueError('Invalid seed quota/routes')
    counts=Counter(e for row in routes for e in row)
    candidates=sorted((e for e in counts if e not in last),key=lambda e:(-counts[e],e))
    # Most frequent and last-token experts are most recent in the decode LRU.
    return list(reversed(candidates[:quota-len(last)]))+last


class FrequencySeeds:
    def __init__(self,pipeline,quota):
        self.original=pipeline.run;self.enabled=True;self.quota=quota

    def run(self,pipeline,layer,x,ids,weights):
        if not self.enabled or not pipeline.cache.seed_enabled:
            return self.original(layer,x,ids,weights)
        if layer==0 and pipeline.buffers is None:pipeline.begin()
        cpu=ids.cpu();bank=pipeline.buffers[layer%2]
        out=routed_moe(x,weights,ids,bank,pipeline.large_tile,pipeline.ready[layer%2])
        seeds=select_seeds(cpu.tolist(),self.quota)
        pipeline.cache.seed(layer,seeds,bank)
        pipeline.layer_stats.append({'layer':layer,'active_experts':cpu.unique().numel(),
            'total_experts':bank[0].shape[0],'routed_tokens':ids.numel(),'tokens':x.shape[0],
            'frequency_seed_count':len(seeds)})
        pipeline.release[layer%2].record();pipeline.copy(layer+2)
        if layer==len(pipeline.host)-1:
            pipeline.buffers=None;pipeline.cache.finish_prefill()
        return out


def install(model,cache):
    modules=[m for m in model.modules() if isinstance(m,Int8Experts)]
    pipelines={id(m.prefill_pipeline):m.prefill_pipeline for m in modules}
    if len(pipelines)!=1 or None in pipelines.values() or not hasattr(cache,'seed'):
        raise ValueError('Requires existing GPU-plan seeded slab pipeline')
    pipeline=next(iter(pipelines.values()));templates=pipeline.host[0]
    owner=getattr(getattr(pipeline.run,'__func__',None),'__self__',None)
    if not isinstance(owner,GPUPlan) or owner.pipeline is not pipeline or pipeline.packed_host is not None:
        raise ValueError('Install directly after ordinary GPUPlan, before any other prefill override')
    if pipeline.cache is not cache or cache.entries or cache.bytes:
        raise ValueError('Install on the active empty cache before inference')
    expert_bytes=sum(t[0].numel()*t.element_size() for t in templates)
    banks=2*templates[0].shape[0]
    quota=(cache.reserved_bytes//expert_bytes-banks)//len(modules)
    if quota<model.config.text_config.num_experts_per_tok:raise ValueError('Insufficient seed space')
    quota=min(quota,templates[0].shape[0]);cache.top_k=quota
    controller=FrequencySeeds(pipeline,quota)
    pipeline.run=types.MethodType(controller.run,pipeline)
    return controller
