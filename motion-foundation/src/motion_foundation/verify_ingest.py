import argparse
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
import json
from pathlib import Path
import random
import numpy as np
import torch
from .adapters import sha256
from .contracts import Sample
from .denoising import ObservationDenoising
from .model import FoundationV13
from .native_data import NativeCorpus


def inspect(row):
    if sha256(row['path'])!=row['sha256']:
        raise ValueError('cache hash mismatch:'+row['source'])
    with np.load(row['path'],allow_pickle=False) as z:
        x,obs,t=z['values'],z['observed'],z['times']
        assert x.shape==obs.shape==(row['rows'],row['channels'])
        assert len(t)==len(x) and np.isfinite(t).all() and (np.diff(t)>0).all()
        assert obs.dtype==bool and np.isfinite(x[obs]).all()
        for a,b in row['spans']:
            assert 0<=a<b<=len(t) and b-a>=2 and np.diff(t[a:b]).max()<=.500001
        for g in row['groups']:
            assert len(g['cols'])==len(g['channels']) and len(set(g['cols']))==len(g['cols'])
            assert min(g['cols'])>=0 and max(g['cols'])<x.shape[1]
    return row['dataset']


def gradients(corpus):
    model=FoundationV13();objective=ObservationDenoising(128);rng=random.Random(137)
    checked=Counter()
    for dataset in sorted(corpus.by_dataset):
        i=next(i for i,r in enumerate(corpus.rows) if r['dataset']==dataset)
        full=corpus.load(i,window_s=4.,rng=rng,max_streams=0)
        for s in full.streams:
            model.zero_grad();objective.zero_grad()
            result,_=objective.compute(model,Sample([s]));result.loss.backward()
            grad=sum(float(p.grad.abs().sum()) for p in model.temporal.parameters() if p.grad is not None)
            grads=[p.grad for p in list(model.parameters())+list(objective.parameters()) if p.grad is not None]
            assert torch.isfinite(result.loss) and grad>0 and grads and all(torch.isfinite(g).all() for g in grads)
            checked[dataset]+=1
        print('GRADIENT',dataset,'device_streams',checked[dataset],flush=True)
    dense=[(max(len(g['cols']) for g in r['groups']),i) for i,r in enumerate(corpus.rows)]
    channels,i=max(dense)
    if channels * 20 > model.config.max_tokens:
        sample=corpus.load(i,window_s=4.,rng=rng,max_streams=1)
        model.zero_grad();objective.zero_grad();loss,_=objective.compute(model,sample);loss.loss.backward()
        assert corpus.last_crop['reason']=='token_budget'
        print('DENSE_FULL_CHANNELS',corpus.last_crop,flush=True)
    for dataset in sorted(corpus.by_dataset):
        for _ in range(10):
            sample,_=corpus.sample(rng,dataset=dataset,window_s=4.,max_streams=1)
            for s in sample.streams:assert float(torch.diff(s.times).max())<=.500001
    print('RANDOM_CONTINUOUS_CROPS',10*len(corpus.by_dataset),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);args=p.parse_args()
    torch.set_num_threads(4)
    data=json.loads(Path(args.manifest).read_text())
    corpus=NativeCorpus(args.manifest)
    for row in data['rows']:
        if not Path(row['path']).is_absolute():
            row['path']=str(Path(args.manifest).parent / row['path'])
    counts=Counter()
    with ThreadPoolExecutor(max_workers=4) as pool:
        for i,dataset in enumerate(pool.map(inspect,[r for split in ('train','validation','test','quarantine') for r in data['rows'] if r['split']==split]),1):
            counts[dataset]+=1
            if i%1000==0:print('CACHE_VERIFIED',i,flush=True)
    gradients(corpus)
    print('VERIFIED',json.dumps(dict(records=sum(counts.values()),counts=dict(counts))),flush=True)


if __name__=='__main__':main()
