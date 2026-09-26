"""Direct W8A16 g128 operators and bounded host-backed expert cache."""
from collections import OrderedDict, deque

import torch
import triton
import triton.language as tl


@triton.jit
def _linear(X, Q, S, Y, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
            BM: tl.constexpr = 16, BN: tl.constexpr = 32, BK: tl.constexpr = 128):
    m = tl.program_id(0) * BM + tl.arange(0, BM)
    n = tl.program_id(1) * BN + tl.arange(0, BN)
    k = tl.arange(0, BK)
    acc = tl.full((BM, BN), 0, tl.float32)
    for b in range(K // BK):
        kk = b * BK + k
        x = tl.load(X + m[:, None] * K + kk[None, :], m[:, None] < M, 0)
        q = tl.load(Q + n[None, :] * K + kk[:, None], n[None, :] < N, 0).to(tl.float32)
        s = tl.load(S + n * (K // BK) + b, n < N, 0)
        w = (q * s[None, :]).to(x.dtype)
        acc += tl.dot(x, w)
    tl.store(Y + m[:, None] * N + n[None, :], acc, (m[:, None] < M) & (n[None, :] < N))


def linear(x, q, scale, large_tile=False):
    if not x.is_cuda or x.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError('W8A16 requires CUDA FP16/BF16 activations')
    n, k = q.shape
    if k % 128 or x.shape[-1] != k or scale.shape != (n, k // 128):
        raise ValueError('Invalid INT8 group geometry')
    x = x.contiguous()
    m = x.numel() // k
    out = torch.empty((*x.shape[:-1], n), device=x.device, dtype=x.dtype)
    bm,bn=(32,64) if large_tile and m>=128 else (16,32)
    _linear[(triton.cdiv(m,bm),triton.cdiv(n,bn))](x,q,scale,out,m,n,k,BM=bm,BN=bn)
    return out


class Int8Linear(torch.nn.Module):
    def __init__(self, q, scale):
        super().__init__()
        self.register_buffer('q', q)
        self.register_buffer('scale', scale)
        self.large_prefill_tile=False

    def forward(self, x):
        return linear(x, self.q, self.scale, self.large_prefill_tile)


class ExpertCache:
    """Global byte-budgeted LRU. Cache stores only requested expert pairs.

    Copies use a dedicated CUDA stream; events order consumers and eviction.
    A single pinned staging pair is retained until its H2D copy completes.
    This initial path has host routing synchronization; profiling will guide
    batching/prefetch improvements after end-to-end correctness is established.
    """
    def __init__(self, max_bytes):
        self.max_bytes = max_bytes
        self.entries = OrderedDict()
        self.bytes = self.hits = self.misses = self.transfer_bytes = 0
        self.stream = torch.cuda.Stream()
        self.staging = None
        self.staging_event = None
        # Experimental bounded queue; off until full-model validation.
        self.queued_transfers = False
        self.pending_copies = deque()
        self.max_pending_copies = 8
        self.copy_host_waits = 0
        self.peak_pending_copies = 0

    def _on_evict(self,key,values):
        pass

    def _load_values(self,key,staging):
        return [v.to('cuda',non_blocking=True) for v in staging]

    def _wait_ready(self,event):
        torch.cuda.current_stream().wait_event(event)

    def get_many(self,requests):
        return [self.get(key,sources) for key,sources in requests]

    def get(self, key, sources):
        if key in self.entries:
            self.hits += 1
            self.entries.move_to_end(key)
            return self.entries[key]
        self.misses += 1
        size = sum(v.numel() * v.element_size() for v in sources)
        if size > self.max_bytes:
            raise MemoryError('Expert exceeds cache budget')
        if self.queued_transfers:
            # Keep strong references even for transient pinned allocations.
            # CUDA event completion, not Python reachability, ends their use.
            while self.pending_copies and self.pending_copies[0][0].query():
                self.pending_copies.popleft()
            if len(self.pending_copies) >= self.max_pending_copies:
                self.pending_copies[0][0].synchronize()
                self.copy_host_waits += 1
                self.pending_copies.popleft()
        elif self.staging_event is not None:
            self.staging_event.synchronize()
            self.copy_host_waits += 1
        self.staging = [v.pin_memory() for v in sources]
        while self.bytes + size > self.max_bytes:
            old_key, old = self.entries.popitem(last=False)
            self._on_evict(old_key,old)
            self.bytes -= sum(v.numel() * v.element_size() for v in old)
        with torch.cuda.stream(self.stream):
            result = self._load_values(key,self.staging)
            self.staging_event = torch.cuda.Event()
            self.staging_event.record()
        if self.queued_transfers:
            self.pending_copies.append((self.staging_event, self.staging))
            self.peak_pending_copies = max(self.peak_pending_copies, len(self.pending_copies))
        self._wait_ready(self.staging_event)
        for v in result:
            v.record_stream(torch.cuda.current_stream())
        self.entries[key] = result
        self.bytes += size
        self.transfer_bytes += size
        return result


class Int8Experts(torch.nn.Module):
    def __init__(self, layer, gate_up, down, cache):
        super().__init__()
        # Host mmap tensors intentionally are not registered as GPU buffers.
        self.layer, self.gate_up, self.down, self.cache = layer, gate_up, down, cache
        self.decode_direct = False
        self.grouped_decode = False
        self.prefill_pipeline = None
        self.async_descriptor = False
        self.descriptor_stage = None
        self.descriptor_event = None

    def upload_descriptors(self, descriptors, device):
        if not self.async_descriptor:
            return torch.tensor(descriptors, dtype=torch.int64, device=device)
        # One staging buffer per layer. Do not overwrite it until its previous
        # DMA completed, including callers that enqueue consecutive forwards.
        if self.descriptor_event is not None and not self.descriptor_event.query():
            self.descriptor_event.synchronize()
        shape=(len(descriptors),5)
        if self.descriptor_stage is None or tuple(self.descriptor_stage.shape)!=shape:
            self.descriptor_stage=torch.empty(shape,dtype=torch.int64,pin_memory=True)
        self.descriptor_stage.copy_(torch.tensor(descriptors,dtype=torch.int64))
        table=self.descriptor_stage.to(device,non_blocking=True)
        self.descriptor_event=torch.cuda.Event()
        self.descriptor_event.record(torch.cuda.current_stream())
        return table

    def forward(self, x, top_k_index, top_k_weights):
        if self.prefill_pipeline is not None and x.shape[0]>1:
            return self.prefill_pipeline.run(self.layer,x,top_k_index,top_k_weights)
        out = torch.zeros_like(x)
        # One host routing read, not a sync for each expert's hit test.
        ids = top_k_index.cpu()
        if self.grouped_decode and x.shape[0] == 1:
            resident=[]
            descriptors=[]
            ordered=sorted((e,s) for s,e in enumerate(ids[0].tolist()))
            if getattr(self.cache,'group_transfers',False):
                requests=[((self.layer,e),
                    [self.gate_up['q'][e],self.gate_up['scale'][e],
                     self.down['q'][e],self.down['scale'][e]]) for e,slot in ordered]
                resident=self.cache.get_many(requests)
                descriptors=[[v.data_ptr() for v in values]+[slot]
                             for (_,slot),values in zip(ordered,resident)]
            else:
                for e,slot in ordered:
                    values=self.cache.get((self.layer,e),
                        [self.gate_up['q'][e],self.gate_up['scale'][e],
                         self.down['q'][e],self.down['scale'][e]])
                    resident.append(values)
                    descriptors.append([v.data_ptr() for v in values]+[slot])
            table=self.upload_descriptors(descriptors,x.device)
            intermediate=self.down['q'].shape[-1]
            gu=grouped_linear(x,table,len(descriptors),2*intermediate,0,True)
            gate,up=gu.chunk(2,-1)
            values=grouped_linear(torch.nn.functional.silu(gate)*up,table,
                                  len(descriptors),x.shape[-1],2,False)
            _ordered_reduce[(triton.cdiv(x.shape[-1],256),)](
                values,table,top_k_weights,out,x.shape[-1],len(descriptors),enable_fp_fusion=False)
            return out
        if self.decode_direct and x.shape[0] == 1:
            # Preserve ascending expert accumulation order of the general path.
            for e, slot in sorted((e,s) for s,e in enumerate(ids[0].tolist())):
                qg,sg,qd,sd = self.cache.get((self.layer,e),
                    [self.gate_up['q'][e],self.gate_up['scale'][e],
                     self.down['q'][e],self.down['scale'][e]])
                gate,up = linear(x,qg,sg).chunk(2,-1)
                y = linear(torch.nn.functional.silu(gate)*up,qd,sd)
                # Keep a rank-2 FP32 view: a 0-d scalar changes PyTorch's
                # dtype promotion and rounds the routing coefficient early.
                out.add_((y*top_k_weights[:,slot:slot+1]).to(out.dtype))
            return out
        for e in torch.unique(ids).tolist():
            token, slot = torch.where(ids == e)
            token, slot = token.to(x.device), slot.to(x.device)
            qg, sg, qd, sd = self.cache.get((self.layer, e),
                [self.gate_up['q'][e], self.gate_up['scale'][e],
                 self.down['q'][e], self.down['scale'][e]])
            gate, up = linear(x[token], qg, sg).chunk(2, -1)
            y = linear(torch.nn.functional.silu(gate) * up, qd, sd)
            y = y * top_k_weights[token, slot, None]
            out.index_add_(0, token, y.to(out.dtype))
        return out


@triton.jit
def _grouped_linear(X,T,Y,N:tl.constexpr,K:tl.constexpr,FIELD:tl.constexpr,SHARED:tl.constexpr):
    e=tl.program_id(0)
    n=tl.program_id(1)*32+tl.arange(0,32)
    m=tl.arange(0,16)
    k=tl.arange(0,128)
    Q=tl.load(T+e*5+FIELD).to(tl.pointer_type(tl.int8))
    S=tl.load(T+e*5+FIELD+1).to(tl.pointer_type(tl.float32))
    xbase=0 if SHARED else e*K
    acc=tl.full((16,32),0,tl.float32)
    for b in range(K//128):
        kk=b*128+k
        x=tl.load(X+xbase+m[:,None]*K+kk[None,:],m[:,None]==0,0)
        q=tl.load(Q+n[None,:]*K+kk[:,None],n[None,:]<N,0).to(tl.float32)
        s=tl.load(S+n*(K//128)+b,n<N,0)
        w=(q*s[None,:]).to(x.dtype)
        acc+=tl.dot(x,w)
    tl.store(Y+e*N+m[:,None]*N+n[None,:],acc,(m[:,None]==0)&(n[None,:]<N))


def grouped_linear(x,table,experts,n,field,shared):
    x=x.contiguous()
    out=torch.empty((experts,n),dtype=x.dtype,device=x.device)
    _grouped_linear[(experts,triton.cdiv(n,32))](x,table,out,n,x.shape[-1],field,shared)
    return out


@triton.jit
def _ordered_reduce(V,T,W,Y,N:tl.constexpr,E:tl.constexpr):
    n=tl.program_id(0)*256+tl.arange(0,256)
    acc=tl.full((256,),0,tl.float32)
    for e in range(E):
        slot=tl.load(T+e*5+4)
        weight=tl.load(W+slot).to(tl.float32)
        value=tl.load(V+e*N+n,n<N,0).to(tl.float32)
        value=(value*weight).to(Y.dtype.element_ty).to(tl.float32)
        acc=(acc+value).to(Y.dtype.element_ty).to(tl.float32)
    tl.store(Y+n,acc,n<N)
