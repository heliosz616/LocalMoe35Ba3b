"""Experimental GPU metadata preparation; original INT8 arithmetic/reduction.

FreeToken's moe_align_block_size motivates a bounded padded GPU launch. Here we
retain the existing unpadded stable route order and skip surplus tiles, so the
current exact-rounding kernel and expert layout remain usable without conversion.
No routing metadata is uploaded from CPU. The pipeline still reads IDs to CPU
for coverage/last-token cache retention, isolating the H2D change first.
"""
import types
import torch
import triton
import triton.language as tl
from qwen35_deploy.runtime.prefill import _routed, _reduce
from qwen35_deploy.operators.int8 import Int8Experts


@triton.jit
def _table(OFFSETS, ENDS, TABLE, E:tl.constexpr, BM:tl.constexpr):
    tile=tl.program_id(0)
    total=tl.load(ENDS+E-1)
    expert=0;start=0;stop=0
    if tile<total:
        lo=0;hi=E
        while lo<hi:
            mid=(lo+hi)//2
            end=tl.load(ENDS+mid)
            right=end<=tile
            lo=tl.where(right,mid+1,lo)
            hi=tl.where(right,hi,mid)
        expert=lo
        prior=tl.load(ENDS+expert-1,expert>0,0).to(tl.int32)
        start=tl.load(OFFSETS+expert).to(tl.int32)+(tile-prior)*BM
        stop=tl.load(OFFSETS+expert+1).to(tl.int32)
    tl.store(TABLE+tile*3,expert)
    tl.store(TABLE+tile*3+1,start)
    tl.store(TABLE+tile*3+2,stop)


@triton.jit
def _nonempty(X,Q,S,POS,TILES,Y,N:tl.constexpr,K:tl.constexpr,GATHER:tl.constexpr,
              QES:tl.constexpr,SES:tl.constexpr,BM:tl.constexpr,BN:tl.constexpr):
    tile=tl.program_id(0)
    if tl.load(TILES+tile*3+1)<tl.load(TILES+tile*3+2):
        _routed(X,Q,S,POS,TILES,Y,N,K,GATHER,QES,SES,BM,BN)


def plan(ids,experts,bm):
    if ids.device.type!='cuda' or ids.ndim!=2 or not ids.is_contiguous():
        raise ValueError('Expected contiguous CUDA [tokens,top_k] routing IDs')
    # IDs come from the router's top-k over exactly `experts` columns.
    flat=ids.reshape(-1);routes=flat.numel();top=ids.shape[1]
    sorted_ids,original=torch.sort(flat,stable=True)
    offsets=torch.searchsorted(sorted_ids,torch.arange(experts+1,device=ids.device,dtype=ids.dtype))
    counts=offsets[1:]-offsets[:-1]
    ends=torch.cumsum((counts+bm-1)//bm,dim=0)
    max_tiles=triton.cdiv(routes,bm)+experts
    table=torch.empty((max_tiles,3),device=ids.device,dtype=torch.int64)
    _table[(max_tiles,)](offsets,ends,table,experts,bm)
    slots=ids.argsort(dim=1,stable=True)
    inverse=torch.empty_like(original)
    inverse.scatter_(0,original,torch.arange(routes,device=ids.device))
    ordered=inverse[torch.arange(ids.shape[0],device=ids.device)[:,None]*top+slots]
    return original//top,table,ordered,slots


def routed_moe(x,weights,ids,banks,large_tile=False,ready_event=None):
    bm,bn=(32,64) if large_tile else (16,32)
    pos,table,ordered,slots=plan(ids,banks[0].shape[0],bm)
    routes=ids.numel();qg,sg,qd,sd=banks
    gu=torch.empty((routes,qg.shape[1]),device=x.device,dtype=x.dtype)
    if ready_event is not None:torch.cuda.current_stream().wait_event(ready_event)
    _nonempty[(len(table),triton.cdiv(gu.shape[1],bn))](
        x,qg,sg,pos,table,gu,gu.shape[1],x.shape[1],True,qg.stride(0),sg.stride(0),bm,bn)
    gate,up=gu.chunk(2,-1)
    hidden=(torch.nn.functional.silu(gate)*up).contiguous()
    down=torch.empty((routes,qd.shape[1]),device=x.device,dtype=x.dtype)
    _nonempty[(len(table),triton.cdiv(down.shape[1],bn))](
        hidden,qd,sd,pos,table,down,down.shape[1],hidden.shape[1],False,qd.stride(0),sd.stride(0),bm,bn)
    out=torch.empty_like(x)
    _reduce[(x.shape[0],triton.cdiv(x.shape[1],256))](
        down,ordered,slots,weights,out,x.shape[1],ids.shape[1],enable_fp_fusion=False)
    return out


class GPUPlan:
    def __init__(self,pipeline):
        self.enabled=True;self.pipeline=pipeline;self.original=pipeline.run

    def run(self,pipeline,layer,x,ids,weights):
        if not self.enabled:return self.original(layer,x,ids,weights)
        if layer==0 and pipeline.buffers is None:pipeline.begin()
        cpu=ids.cpu()
        bank=pipeline.buffers[layer%2]
        out=routed_moe(x,weights,ids,bank,pipeline.large_tile,pipeline.ready[layer%2])
        if getattr(pipeline.cache,'seed_enabled',False):
            pipeline.cache.seed(layer,cpu[-1].tolist(),bank)
        pipeline.layer_stats.append({'layer':layer,'active_experts':cpu.unique().numel(),
            'total_experts':bank[0].shape[0],'routed_tokens':ids.numel(),'tokens':x.shape[0]})
        pipeline.release[layer%2].record()
        pipeline.copy(layer+2)
        if layer==len(pipeline.host)-1:
            pipeline.buffers=None
            if getattr(pipeline.cache,'seed_enabled',False):pipeline.cache.finish_prefill()
        return out


def install(model):
    pipelines={id(m.prefill_pipeline):m.prefill_pipeline for m in model.modules() if isinstance(m,Int8Experts)}
    if len(pipelines)!=1 or None in pipelines.values():raise ValueError('One prefill pipeline required')
    pipeline=next(iter(pipelines.values()))
    controller=GPUPlan(pipeline)
    pipeline.run=types.MethodType(controller.run,pipeline)
    return controller
