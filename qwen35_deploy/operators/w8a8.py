"""Experimental grouped INT8 decode; numerically different from deployed W8A16."""
import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def _quant(X,Q,S,K:tl.constexpr):
    row=tl.program_id(0);group=tl.program_id(1);k=group*128+tl.arange(0,128)
    x=tl.load(X+row*K+k).to(tl.float32)
    scale=tl.maximum(tl.max(tl.abs(x),0),1.e-10)*(1./127.)
    q=libdevice.nearbyint(tl.div_rn(x,scale)).to(tl.int8)
    tl.store(Q+row*K+k,q);tl.store(S+row*(K//128)+group,scale)


def quantize(x):
    x=x.contiguous();k=x.shape[-1]
    if not x.is_cuda or x.ndim!=2 or k%128:raise ValueError('Expected CUDA rows with K divisible by 128')
    q=torch.empty_like(x,dtype=torch.int8);s=torch.empty((x.shape[0],k//128),device=x.device,dtype=torch.float32)
    _quant[(x.shape[0],k//128)](x,q,s,k,enable_fp_fusion=False)
    return q,s


@triton.jit
def _linear(X,XS,T,Y,N:tl.constexpr,K:tl.constexpr,FIELD:tl.constexpr,SHARED:tl.constexpr):
    e=tl.program_id(0);n=tl.program_id(1)*32+tl.arange(0,32)
    m=tl.arange(0,16);k=tl.arange(0,128)
    Q=tl.load(T+e*5+FIELD).to(tl.pointer_type(tl.int8))
    S=tl.load(T+e*5+FIELD+1).to(tl.pointer_type(tl.float32))
    row=0 if SHARED else e
    acc=tl.full((16,32),0,tl.float32)
    for b in range(K//128):
        kk=b*128+k
        x=tl.load(X+row*K+m[:,None]*K+kk[None,:],m[:,None]==0,0)
        w=tl.load(Q+n[None,:]*K+kk[:,None],n[None,:]<N,0)
        dot=tl.dot(x,w).to(tl.float32)
        xs=tl.load(XS+row*(K//128)+b)
        ws=tl.load(S+n*(K//128)+b,n<N,0)
        acc=acc+(dot*xs)*ws[None,:]
    tl.store(Y+e*N+m[:,None]*N+n[None,:],acc,(m[:,None]==0)&(n[None,:]<N))


def grouped_linear(x,table,experts,n,field,shared):
    q,s=quantize(x)
    out=torch.empty((experts,n),device=x.device,dtype=x.dtype)
    _linear[(experts,triton.cdiv(n,32))](q,s,table,out,n,x.shape[-1],field,shared,enable_fp_fusion=False)
    return out
