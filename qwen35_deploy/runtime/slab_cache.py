"""One fixed VRAM arena reused by decode LRU slots and prefill bank buffers.

Uniform expert geometry, >=8 slots, single CUDA-dispatch thread. Payloads and
arithmetic are unchanged. Copy-stream waits guard slot reuse after queued
consumers; ascending LRU touches protect an in-flight group of at most 8 experts.
"""
import torch
from qwen35_deploy.operators.int8 import ExpertCache,Int8Experts


class SlabExpertCache(ExpertCache):
    def __init__(self,max_bytes):
        super().__init__(max_bytes)
        self.reserved_bytes=max_bytes
        self.slab=None
        self.layout=None
        self.slot_bytes=None
        self.next_slot=0
        self.free_slots=[]
        self.group_transfers=False
        self._group_active=False
        self._group_guarded=False
        self._group_ready=None
        self.packed_host={}

    def get_many(self,requests):
        if not self.group_transfers:return super().get_many(requests)
        if self._group_active:raise RuntimeError('Nested expert transfer group')
        if len(requests)>8:raise ValueError('At most 8 experts per transfer group')
        self._group_active=True
        self._group_guarded=False
        self._group_ready=None
        try:
            return super().get_many(requests)
        finally:
            # All copies share one ordered stream. The last event covers every
            # miss; hits were made ready by the preceding completed group.
            if self._group_ready is not None:
                torch.cuda.current_stream().wait_event(self._group_ready)
            self._group_active=False

    def _wait_ready(self,event):
        if self._group_active:self._group_ready=event
        else:super()._wait_ready(event)

    def ensure_slab(self):
        if self.slab is None:
            self.slab=torch.empty(self.reserved_bytes,dtype=torch.uint8,device='cuda')
            self.slab.record_stream(self.stream)

    def views(self,offset,templates):
        result=[]
        for t in templates:
            n=t.numel()*t.element_size()
            if offset%t.element_size():raise ValueError('Unaligned slab tensor')
            if offset+n>self.reserved_bytes:raise MemoryError('Slab quota exceeded')
            result.append(self.slab[offset:offset+n].view(t.dtype).view(t.shape))
            offset+=n
        return result,offset

    def get(self,key,sources):
        self.consumer_stream=torch.cuda.current_stream()
        return super().get(key,sources)

    def _on_evict(self,key,values):
        self.free_slots.append((values[0].data_ptr()-self.slab.data_ptr())//self.slot_bytes)

    def _load_values(self,key,staging):
        self.ensure_slab()
        layout=tuple((tuple(t.shape),t.dtype) for t in staging)
        size=sum(t.numel()*t.element_size() for t in staging)
        if self.layout is None:
            self.layout=layout;self.slot_bytes=size
            self.capacity=self.reserved_bytes//size
            if self.capacity<8:raise ValueError('Slab cache requires at least 8 expert slots')
            self.max_bytes=self.capacity*size
            # Construct tensor views once, not twelve slice/view operations on
            # every cache miss in the decode hot path.
            self.slot_views=[self.views(i*size,staging)[0] for i in range(self.capacity)]
            self.raw_slots=[self.slab[i*size:(i+1)*size] for i in range(self.capacity)]
        if layout!=self.layout:raise ValueError('Slab expert geometry must be uniform')
        if self.free_slots:
            slot=self.free_slots.pop()
        else:
            slot=self.next_slot;self.next_slot+=1
        if slot>=self.capacity:raise MemoryError('No slab slot available')
        result=self.slot_views[slot]
        # Guard old contents still read by previously queued consumer kernels.
        if not self._group_active or not self._group_guarded:
            self.stream.wait_stream(self.consumer_stream)
            self._group_guarded=True
        if self.packed_host:
            self.raw_slots[slot].copy_(self.packed_host[key],non_blocking=True)
        else:
            for dst,src in zip(result,staging):dst.copy_(src,non_blocking=True)
        return result

    def prefill_buffers(self,templates):
        if self.entries or self.bytes:raise RuntimeError('Clear decode entries before prefill reuse')
        self.ensure_slab()
        self.next_slot=0;self.free_slots=[]
        first,end=self.views(0,templates)
        second,_=self.views(end,templates)
        return [first,second]


def install(model,old):
    if old.entries or old.bytes:raise ValueError('Install slab before inference')
    if model.config.text_config.num_experts_per_tok>8:
        raise ValueError('This slab lookup lifetime contract supports at most 8 routed experts')
    cache=SlabExpertCache(old.max_bytes)
    cache.queued_transfers=old.queued_transfers
    for module in model.modules():
        if isinstance(module,Int8Experts):
            module.cache=cache
            if module.prefill_pipeline is not None:module.prefill_pipeline.cache=cache
            pipeline=module.prefill_pipeline
            if pipeline is not None and pipeline.packed_host is not None:
                blob=pipeline.packed_host[module.layer]
                count=module.gate_up['q'].shape[0]
                size=blob.numel()//count
                cache.packed_host.update({(module.layer,e):blob[e*size:(e+1)*size] for e in range(count)})
    return cache
