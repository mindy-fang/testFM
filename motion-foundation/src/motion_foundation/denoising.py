from dataclasses import replace
import torch
from torch import nn
from .contracts import Sample
from .model import time_features
from .objectives import LossResult


class ObservationDenoising(nn.Module):
    def __init__(self, dim, heads=4, noise_std=.5, max_queries=512):
        super().__init__()
        self.query_time=nn.Linear(5,dim)
        self.query_context=nn.Linear(dim,dim,bias=False)
        self.attention=nn.MultiheadAttention(dim,heads,batch_first=True)
        self.predict=nn.Sequential(nn.Linear(dim,dim),nn.GELU(),nn.Linear(dim,1))
        self.noise_std=noise_std
        self.max_queries=max_queries

    def compute(self,model,sample,generator=None):
        sample.validate()
        corrupted=[];clean={};noise={}
        for s in sample.streams:
            x=s.normalized()
            eps=torch.randn(x.shape,device=x.device,dtype=x.dtype,generator=generator)*self.noise_std
            clean[s.stream_id]=x
            noise[s.stream_id]=eps
            corrupted.append(replace(s,values=torch.where(s.observed,x+eps,torch.zeros_like(x)),transform='asinh_reference_v1'))
        # Only independently corrupted exports enter the encoder; clean values are loss targets only.
        corrupted_sample=Sample(corrupted)
        rep=model(corrupted_sample)
        terms={};identity={};constant={};count=0
        for s in corrupted:
            bundle=rep.local[s.stream_id]
            h=rep.fused[rep.stream_groups[s.stream_id]]
            origin=rep.times[rep.stream_groups[s.stream_id]][0]
            indices=s.observed.nonzero()
            if len(indices)>self.max_queries:
                order=torch.randperm(len(indices),device=indices.device,generator=generator)[:self.max_queries]
                indices=indices[order]
            row,col=indices.unbind(1)
            semantics=torch.stack([model.frontends['scalar'].semantic(c,h.device) for c in s.channels])
            sums=h.new_zeros(len(s.channels),h.shape[-1]);counts=h.new_zeros(len(s.channels),1)
            route=torch.tensor([r[0] for r in bundle.routes],device=h.device)
            sums.index_add_(0,route,bundle.features)
            counts.index_add_(0,route,torch.ones(len(route),1,device=h.device,dtype=h.dtype))
            q=self.query_time(time_features((s.mapped_times()[row]-origin).to(h.dtype)))+semantics[col]+self.query_context((sums/counts.clamp_min(1))[col])
            context,_=self.attention(q[None],h[None],h[None],need_weights=False)
            pred=self.predict(context[0]).flatten()
            target=clean[s.stream_id][row,col]
            err=nn.functional.smooth_l1_loss(pred,target,reduction='none')
            identity_err=nn.functional.smooth_l1_loss(s.values[row,col],target,reduction='none')
            # A visible noisy per-channel mean is a stronger reference than a global zero predictor.
            mean=(s.values*s.observed).sum(0)/s.observed.sum(0).clamp_min(1)
            constant_err=nn.functional.smooth_l1_loss(mean[col],target,reduction='none')
            for quantity in sorted({c.quantity for c in s.channels}):
                selected=torch.tensor([s.channels[int(c)].quantity==quantity for c in col.cpu()],device=h.device)
                if selected.any():
                    terms.setdefault(quantity,[]).append(err[selected].mean())
                    identity.setdefault(quantity,[]).append(float(identity_err[selected].mean().detach()))
                    constant.setdefault(quantity,[]).append(float(constant_err[selected].mean().detach()))
            count+=len(indices)
        means={q:torch.stack(v).mean() for q,v in terms.items()}
        loss=torch.stack(list(means.values())).mean()
        result=LossResult(loss,count,1,{q:float(v.detach()) for q,v in means.items()})
        baselines=dict(identity={q:sum(v)/len(v) for q,v in identity.items()},channel_mean={q:sum(v)/len(v) for q,v in constant.items()})
        return result,baselines
