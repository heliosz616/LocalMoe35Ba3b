"""Experimental last-prefill-token expert retention inside the existing arena.

Single dispatch stream/thread, ordinary expert layout only. Seed copies execute
on the consumer stream before the bank release event, so prefetch cannot overwrite
their sources. No routing prediction is used for computation: misses still fetch
the exact routed expert as usual.
"""
import types
import torch
from qwen35_deploy.runtime.slab_cache import SlabExpertCache
from qwen35_deploy.operators.int8 import Int8Experts
from qwen35_deploy.runtime.prefill import routed_moe


class SeededSlabCache(SlabExpertCache):
    def __init__(self,max_bytes,layers,top_k):
        super().__init__(max_bytes)
        self.layers=layers;self.top_k=top_k
        self.seed_enabled=False
        self.seeded_experts=0;self.seed_d2d_bytes=0

    def prefill_buffers(self,templates):
        banks=super().prefill_buffers(templates)
        if not self.seed_enabled:return banks
        per_expert=[t[0] for t in templates]
        layout=tuple((tuple(t.shape),t.dtype) for t in per_expert)
        size=sum(t.numel()*t.element_size() for t in per_expert)
        if self.layout is None:
            self.layout=layout;self.slot_bytes=size;self.capacity=self.reserved_bytes//size
            self.max_bytes=self.capacity*size
            self.slot_views=[self.views(i*size,per_expert)[0] for i in range(self.capacity)]
            self.raw_slots=[self.slab[i*size:(i+1)*size] for i in range(self.capacity)]
        if layout!=self.layout:raise ValueError('Seeded expert geometry changed')
        bank_bytes=sum(t.numel()*t.element_size() for t in templates)
        self.seed_start=(2*bank_bytes+size-1)//size
        if self.seed_start+self.layers*self.top_k>self.capacity:
            raise MemoryError('Arena cannot hold prefill banks plus bounded seeds')
        self.next_slot=self.seed_start
        return banks

    def seed(self,layer,experts,bank):
        if not self.seed_enabled:return
        if len(experts)>self.top_k or len(set(experts))!=len(experts):raise ValueError('Invalid seed route')
        for e in experts:
            key=(layer,e)
            if key in self.entries:raise ValueError('Duplicate prefill seed')
            slot=self.next_slot;self.next_slot+=1
            if not self.seed_start<=slot<self.capacity:raise MemoryError('Seed would overlap prefill bank or quota')
            values=self.slot_views[slot]
            for dst,src in zip(values,bank):dst.copy_(src[e],non_blocking=True)
            self.entries[key]=values;self.bytes+=self.slot_bytes
            self.seeded_experts+=1;self.seed_d2d_bytes+=self.slot_bytes

    def finish_prefill(self):
        if self.seed_enabled:
            # The banks may now become ordinary decode slots. Subsequent misses
            # use the inherited copy-stream wait to guard their prior consumers.
            self.free_slots=list(range(self.seed_start))


def seeded_run(self,layer,x,ids,weights):
    if not self.cache.seed_enabled:return self.unseeded_run(layer,x,ids,weights)
    if layer==0 and self.buffers is None:self.begin()
    cpu_ids=ids.cpu()  # Reused by routed_moe; no second device-to-host routing read.
    stats={'layer':layer}
    bank=self.buffers[layer%2]
    result=routed_moe(x,weights,cpu_ids,bank,large_tile=self.large_tile,
                      stats=stats,ready_event=self.ready[layer%2])
    self.cache.seed(layer,cpu_ids[-1].tolist(),bank)
    self.layer_stats.append(stats)
    self.release[layer%2].record()  # Includes the seed's D2D source reads.
    self.copy(layer+2)
    if layer==len(self.host)-1:
        self.buffers=None;self.cache.finish_prefill()
    return result


def install(model,old):
    if old.entries or old.bytes:raise ValueError('Install before inference')
    config=model.config.text_config
    if config.num_experts_per_tok>8:raise ValueError('At most eight routed experts supported')
    modules=[m for m in model.modules() if isinstance(m,Int8Experts)]
    pipelines={id(m.prefill_pipeline):m.prefill_pipeline for m in modules}
    if len(pipelines)!=1 or None in pipelines.values():raise ValueError('One shared prefill pipeline required')
    pipeline=next(iter(pipelines.values()))
    if pipeline.packed_host is not None:raise ValueError('Seed experiment requires ordinary bank layout')
    cache=SeededSlabCache(old.max_bytes,len(modules),config.num_experts_per_tok)
    cache.queued_transfers=old.queued_transfers
    pipeline.cache=cache;pipeline.unseeded_run=pipeline.run
    pipeline.run=types.MethodType(seeded_run,pipeline)
    for module in modules:module.cache=cache
    return cache
