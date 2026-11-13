from collections import OrderedDict
from dataclasses import replace
import json
from pathlib import Path
import random
import numpy as np
import torch
from .adapters import sha256
from .contracts import Channel, Sample, StreamRecord


class NativeCorpus:
    def __init__(self,path,split='train',cache_size=8,patch_s=.2,token_budget=16000):
        self.patch_s=patch_s;self.token_budget=token_budget;self.last_crop=None
        self.path=Path(path);self.fingerprint=sha256(path)
        payload=json.loads(self.path.read_text())
        if payload['schema']!='v13-native-corpus-1':raise ValueError('native manifest version')
        for row in payload['rows']:
            if not Path(row['path']).is_absolute():
                row['path']=str(self.path.parent / row['path'])
        self.rows=[r for r in payload['rows'] if r['split']==split]
        if not self.rows:raise ValueError('empty partition')
        self.cache=OrderedDict();self.cache_size=cache_size;self.verified=set()
        self.by_dataset={}
        subject_splits={};source_splits={};hash_splits={}
        for r in payload['rows']:
            for value,mapping in [(r['subject'],subject_splits),(r['source'],source_splits),(r['sha256'],hash_splits)]:
                if value in mapping and mapping[value]!=r['split']:raise ValueError('cross-split identity/source/hash')
                mapping[value]=r['split']
        for i,r in enumerate(self.rows):self.by_dataset.setdefault(r['dataset'],{}).setdefault(r['subject'],[]).append(i)

    def load(self,i,start_s=None,window_s=2.,rng=None,max_streams=2,device='cpu'):
        rng=rng or random
        r=self.rows[i];path=r['path']
        if path not in self.verified:
            if sha256(path)!=r['sha256']:raise ValueError('native cache fingerprint mismatch')
            self.verified.add(path)
        if path not in self.cache:
            with np.load(path,allow_pickle=False) as z:self.cache[path]=(z['values'],z['observed'],z['times'])
            if len(self.cache)>self.cache_size:self.cache.popitem(last=False)
        else:self.cache.move_to_end(path)
        x,observed,t=self.cache[path]
        groups=r['groups']
        if max_streams and len(groups)>max_streams:groups=rng.sample(groups,max_streams)
        max_channels=max(len(g['cols']) for g in groups)
        max_patches=self.token_budget//max_channels
        if max_patches<1:raise ValueError('device exceeds local channel budget')
        requested_window_s=window_s
        window_s=min(window_s,max_patches*self.patch_s-1e-6)
        self.last_crop=dict(requested_window_s=requested_window_s,effective_window_s=window_s,retained_channels=max_channels,reason='token_budget' if window_s<requested_window_s else 'requested_window')
        spans=[(a,b) for a,b in r['spans'] if t[b-1]-t[a]>=min(window_s,1.)]
        if not spans:raise ValueError('no eligible continuous span')
        if start_s is None:
            a,b=rng.choice(spans);start_s=rng.uniform(float(t[a]),max(float(t[a]),float(t[b-1])-window_s))
        else:
            candidates=[(a,b) for a,b in spans if t[a]<=start_s<=t[b-1]]
            if not candidates:raise ValueError('requested start outside continuous eligible spans')
            a,b=candidates[0]
        take=(t>=start_s)&(t<min(start_s+window_s,float(t[b-1])+1e-9))
        self.last_crop['actual_duration_s']=float(t[take][-1]-t[take][0]) if take.any() else 0.
        streams=[]
        for j,g in enumerate(groups):
            cols=g['cols'];obs=observed[take][:,cols]
            if not obs.any():continue
            streams.append(StreamRecord(values=torch.tensor(x[take][:,cols],device=device),times=torch.tensor(t[take],device=device),observed=torch.tensor(obs,device=device),channels=tuple(Channel(**c) for c in g['channels']),stream_id='device-'+str(r['groups'].index(g)),clock=r['source']+':independent',source=r['source'],subject=r['subject'],split=r['split'],modality=g['modality'],time_evidence=r['time_evidence'],transform=r['transform'],dependency_s=r['dependency_s'],channel_independent=False))
        return Sample(streams).validate()

    def sample(self,rng,window_s=2.,device='cpu',dataset=None,max_streams=2):
        dataset=dataset or rng.choice(sorted(self.by_dataset))
        subject=rng.choice(sorted(self.by_dataset[dataset]))
        i=rng.choice(self.by_dataset[dataset][subject])
        return self.load(i,window_s=window_s,rng=rng,max_streams=max_streams,device=device),dataset
