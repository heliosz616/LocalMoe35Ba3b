"""Capture unmodified HF GDN decode arithmetic; keep private static states.

Single dispatch thread only. Prefill, padding masks and rollback caches use the
original forward. No fused/custom recurrence or changed precision is involved.
"""
from types import SimpleNamespace
import torch


class StaticCache:
    def __init__(self, index, conv, recurrent):
        self.index=index
        self.layers={index:SimpleNamespace(record_past=False,
            conv_states={0:conv.clone()},recurrent_states={0:recurrent.clone()})}

    def has_previous_state(self,index):
        assert index==self.index
        return True

    def update_recurrent_state(self,state,index):
        target=self.layers[index].recurrent_states[0]
        target.copy_(state)
        return target


class GraphGDN(torch.nn.Module):
    def __init__(self,original,pool):
        super().__init__()
        self.original=original
        self.graph=None
        self.signature=None
        self.replays=0
        # Safe only for this runtime's fixed layer order / nonconcurrent calls.
        # Persistent input/state buffers are allocated outside the shared pool.
        self.pool=pool

    def forward(self,hidden_states,cache_params=None,attention_mask=None,**kwargs):
        # Decoder forwards this flag, but GDN state behavior is determined by
        # cache_params; the original implementation does not inspect it.
        kwargs.pop('use_cache',None)
        index=self.original.layer_idx
        if (hidden_states.shape[:2]!=(1,1) or cache_params is None or
            attention_mask is not None or not cache_params.has_previous_state(index) or
            cache_params.layers[index].record_past or kwargs):
            return self.original(hidden_states,cache_params=cache_params,
                                 attention_mask=attention_mask,**kwargs)
        real=cache_params.layers[index]
        conv,recurrent=real.conv_states[0],real.recurrent_states[0]
        signature=(hidden_states.shape,hidden_states.dtype,hidden_states.device,
                   conv.shape,conv.dtype,recurrent.shape,recurrent.dtype)
        if signature!=self.signature:
            self.graph=None
            self.static_input=hidden_states.clone()
            self.static_cache=StaticCache(index,conv,recurrent)
            stream=torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                # Warm libraries/Triton before capture; only private states mutate.
                for _ in range(3):
                    self.original(self.static_input,cache_params=self.static_cache)
            torch.cuda.current_stream().wait_stream(stream)
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph,pool=self.pool):
                self.static_output=self.original(self.static_input,cache_params=self.static_cache)
            self.graph=graph
            self.signature=signature
        static=self.static_cache.layers[index]
        self.static_input.copy_(hidden_states)
        static.conv_states[0].copy_(conv)
        static.recurrent_states[0].copy_(recurrent)
        self.graph.replay()
        conv.copy_(static.conv_states[0])
        recurrent.copy_(static.recurrent_states[0])
        self.replays+=1
        # Preserve ordinary tensor lifetime semantics across future replays.
        return self.static_output.clone()


def install(model):
    pool=torch.cuda.graph_pool_handle()
    names=[name for name,module in model.named_modules()
           if module.__class__.__name__=='Qwen3_5MoeGatedDeltaNet']
    for name in names:
        parent,_,child=name.rpartition('.')
        setattr(model.get_submodule(parent),child,GraphGDN(model.get_submodule(name),pool))
    return len(names)
