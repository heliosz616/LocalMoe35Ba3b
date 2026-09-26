"""FreeToken-inspired full-layer double buffering and routed INT8 GEMM.

Reference: FreeToken offload_cache begin/prefetch/wait/release protocol.
The INT8 g128 format and sorted-rounding-preserving reduction are local.
"""
import torch
import triton
import triton.language as tl
from qwen35_deploy.operators.routing import plan_routes


@triton.jit
def _routed(X,Q,S,POS,TILES,Y,N:tl.constexpr,K:tl.constexpr,GATHER:tl.constexpr,
            QES:tl.constexpr,SES:tl.constexpr,BM:tl.constexpr=16,BN:tl.constexpr=32):
    tile=tl.program_id(0)
    e=tl.load(TILES+tile*3)
    start=tl.load(TILES+tile*3+1)
    stop=tl.load(TILES+tile*3+2)
    row=start+tl.arange(0,BM)
    n=tl.program_id(1)*BN+tl.arange(0,BN)
    k=tl.arange(0,128)
    if GATHER:
        xr=tl.load(POS+row,row<stop,0)
    else:
        xr=row
    acc=tl.full((BM,BN),0,tl.float32)
    for b in range(K//128):
        kk=b*128+k
        x=tl.load(X+xr[:,None]*K+kk[None,:],row[:,None]<stop,0)
        q=tl.load(Q+e*QES+n[None,:]*K+kk[:,None],n[None,:]<N,0).to(tl.float32)
        scale=tl.load(S+e*SES+n*(K//128)+b,n<N,0)
        w=(q*scale[None,:]).to(x.dtype)
        acc+=tl.dot(x,w)
    tl.store(Y+row[:,None]*N+n[None,:],acc,(row[:,None]<stop)&(n[None,:]<N))


@triton.jit
def _reduce(V,ORDER,SLOTS,W,Y,N:tl.constexpr,TOP:tl.constexpr):
    token=tl.program_id(0)
    col=tl.program_id(1)*256+tl.arange(0,256)
    acc=tl.full((256,),0,tl.float32)
    for i in range(TOP):
        route=tl.load(ORDER+token*TOP+i)
        slot=tl.load(SLOTS+token*TOP+i)
        weight=tl.load(W+token*TOP+slot).to(tl.float32)
        val=tl.load(V+route*N+col,col<N,0).to(tl.float32)
        val=(val*weight).to(Y.dtype.element_ty).to(tl.float32)
        acc=(acc+val).to(Y.dtype.element_ty).to(tl.float32)
    tl.store(Y+token*N+col,acc,col<N)


def routed_moe(x,weights,ids,banks,large_tile=False,stats=None,ready_event=None):
    bm,bn=(32,64) if large_tile else (16,32)
    cpu=ids.cpu()
    positions,segments=plan_routes(cpu,banks[0].shape[0])
    if stats is not None:
        stats.update(active_experts=len(segments),total_experts=banks[0].shape[0],
                     routed_tokens=cpu.numel(),tokens=x.shape[0])
    tiles=[(e,start,stop) for e,begin,stop in segments for start in range(begin,stop,bm)]
    table=torch.tensor(tiles,dtype=torch.int64,device=x.device)
    pos=positions[0].to(x.device)
    routes=cpu.numel()
    slots=cpu.argsort(dim=1,stable=True)
    original=positions[0]*cpu.shape[1]+positions[1]
    inverse=torch.empty_like(original)
    inverse[original]=torch.arange(routes)
    ordered=inverse[torch.arange(cpu.shape[0])[:,None]*cpu.shape[1]+slots]
    ordered,slots=ordered.to(x.device),slots.to(x.device)
    qg,sg,qd,sd=banks
    gu=torch.empty((routes,qg.shape[1]),device=x.device,dtype=x.dtype)
    # CPU routing and its small metadata uploads do not consume expert weights.
    # Delay this dependency until the first weight-reading kernel, allowing that
    # preparation to overlap the bank's in-flight H2D copy.
    if ready_event is not None:torch.cuda.current_stream().wait_event(ready_event)
    _routed[(len(tiles),triton.cdiv(gu.shape[1],bn))](
        x,qg,sg,pos,table,gu,gu.shape[1],x.shape[1],True,qg.stride(0),sg.stride(0),BM=bm,BN=bn)
    gate,up=gu.chunk(2,-1)
    hidden=(torch.nn.functional.silu(gate)*up).contiguous()
    down=torch.empty((routes,qd.shape[1]),device=x.device,dtype=x.dtype)
    _routed[(len(tiles),triton.cdiv(down.shape[1],bn))](
        hidden,qd,sd,pos,table,down,down.shape[1],hidden.shape[1],False,qd.stride(0),sd.stride(0),BM=bm,BN=bn)
    out=torch.empty_like(x)
    _reduce[(x.shape[0],triton.cdiv(x.shape[1],256))](
        down,ordered,slots,weights,out,x.shape[1],cpu.shape[1],enable_fp_fusion=False)
    return out


class PrefillPipeline:
    """Two full-layer device banks, persistent pinned CPU weights.

    Prefill temporarily clears the decode LRU. Buffers are released after
    the final layer so decode can reclaim its full cache budget. Ready and
    release events prevent overwriting weights still consumed by GEMM.
    """
    def __init__(self,modules,cache,packed=False):
        self.cache=cache
        self.host=[]
        self.packed_host=[] if packed else None
        self.stream=torch.cuda.Stream()
        self.buffers=None
        self.large_tile=False
        self.layer_stats=[]
        for module in modules:
            if packed:raise ValueError('Packed layout is not included in this release')
            module.gate_up={k:v.pin_memory() for k,v in module.gate_up.items()}
            module.down={k:v.pin_memory() for k,v in module.down.items()}
            self.host.append([module.gate_up['q'],module.gate_up['scale'],
                              module.down['q'],module.down['scale']])
        self.pinned_bytes=sum(v.numel()*v.element_size() for b in self.host for v in b)
        self.ready=[torch.cuda.Event(),torch.cuda.Event()]
        self.release=[torch.cuda.Event(),torch.cuda.Event()]

    def copy(self,layer):
        if layer>=len(self.host):
            return
        slot=layer%2
        with torch.cuda.stream(self.stream):
            if layer>=2:
                self.stream.wait_event(self.release[slot])
            pairs=([(self.packed_buffers[slot][0],self.packed_host[layer])] if self.packed_host is not None
                   else zip(self.buffers[slot],self.host[layer]))
            for dest,src in pairs:
                dest.copy_(src,non_blocking=True)
                self.cache.transfer_bytes+=src.numel()*src.element_size()
            self.ready[slot].record()

    def begin(self):
        # Clear decode weights before the first token mixer's large workspace,
        # not only when execution finally reaches the first expert operator.
        self.cache.entries.clear()
        self.cache.bytes=0
        self.layer_stats=[]
        factory=getattr(self.cache,'prefill_buffers',None)
        templates=[self.packed_host[0]] if self.packed_host is not None else self.host[0]
        self.buffers=(factory(templates) if factory else
                      [[torch.empty_like(t,device='cuda') for t in templates] for _ in range(2)])
        self.stream.wait_stream(torch.cuda.current_stream())
        for bank in self.buffers:
            for v in bank:
                v.record_stream(self.stream)
        self.copy(0)
        self.copy(1)

    def forward_pre_hook(self,module,args,kwargs):
        ids=kwargs.get('input_ids')
        embeds=kwargs.get('inputs_embeds')
        if ids is None and embeds is None and args:
            ids=args[0]
        tokens=embeds.shape[-2] if embeds is not None else (ids.shape[-1] if ids is not None else 0)
        if tokens>1:
            self.begin()

    def run(self,layer,x,ids,weights):
        if layer==0 and self.buffers is None:
            self.begin()
        stats={'layer':layer}
        result=routed_moe(x,weights,ids,self.buffers[layer%2],large_tile=self.large_tile,
                          stats=stats,ready_event=self.ready[layer%2])
        self.layer_stats.append(stats)
        self.release[layer%2].record()
        self.copy(layer+2)
        if layer==len(self.host)-1:
            self.buffers=None
        return result
