"""Stable CPU routing plan for one batched metadata transfer per MoE layer."""
import torch


def plan_routes(ids, experts):
    if ids.device.type!='cpu' or ids.ndim!=2 or ids.numel()==0:
        raise ValueError('Expected nonempty CPU [tokens,top_k] IDs')
    if int(ids.min())<0 or int(ids.max())>=experts:
        raise ValueError('Out-of-range expert ID')
    flat=ids.reshape(-1)
    order=torch.argsort(flat,stable=True)
    counts=torch.bincount(flat,minlength=experts).tolist()
    positions=torch.stack((order//ids.shape[1],order%ids.shape[1]))
    segments=[]
    start=0
    for e,count in enumerate(counts):
        if count:
            segments.append((e,start,start+count))
            start+=count
    return positions,segments
