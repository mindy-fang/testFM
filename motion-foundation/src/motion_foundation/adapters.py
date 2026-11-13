from __future__ import annotations

import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from .contracts import Channel, Registry, Sample, StreamRecord, SCHEMA

ADAPTERS = Registry()


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def read_npz(descriptor):
    if descriptor.get('schema') != SCHEMA:
        raise ValueError('NPZ descriptor schema mismatch')
    path = Path(descriptor['path'])
    if sha256(path) != descriptor['sha256']:
        raise ValueError('source fingerprint mismatch')
    with np.load(path, allow_pickle=False) as z:
        values = torch.tensor(z['values'], dtype=torch.float32)
        observed = torch.tensor(z['observed'], dtype=torch.bool)
        if 'times' in z:
            times = torch.tensor(z['times'], dtype=torch.float64)
        else:
            fs = descriptor.get('verified_fs')
            if fs is None or not np.isfinite(fs) or fs <= 0 or not descriptor.get('continuity_evidence'):
                raise ValueError('missing timestamps require verified frequency and continuity evidence')
            if descriptor.get('time_evidence') != 'verified_relative_grid':
                raise ValueError('constructed relative grid must be declared')
            times = torch.arange(len(values), dtype=torch.float64) / fs
    keys = ('stream_id', 'clock', 'source', 'subject', 'split', 'modality', 'time_evidence')
    kwargs = {k: descriptor[k] for k in keys}
    for k in ('transform', 'dependency_s', 'channel_independent', 'time_scale', 'time_offset', 'fusion_ref', 'sync_error_s', 'sync_evidence'):
        if k in descriptor:
            kwargs[k] = descriptor[k]
    channels = tuple(Channel(**c) for c in descriptor['channels'])
    return [StreamRecord(values, times, observed, channels, **kwargs).validate()]


ADAPTERS.register('npz-v13', read_npz)


class ManifestDataset:
    def __init__(self, path, split='train'):
        path = Path(path)
        self.root = path.parent
        payload = json.loads(path.read_text())
        if payload.get('schema') != SCHEMA:
            raise ValueError('manifest schema mismatch')
        self.rows = [r for r in payload['rows'] if r['split'] == split]
        if not self.rows:
            raise ValueError('empty dataset partition')
        identities = {}
        for r in payload['rows']:
            for kind, value in [('subject', r['subject']), ('source', r['source']), ('hash', r['sha256'])]:
                key = kind, value
                if key in identities and identities[key] != r['split']:
                    raise ValueError('source/subject crosses split')
                identities[key] = r['split']
        self.fingerprint = sha256(path)

    def __len__(self):
        return len(self.rows)

    def sample(self, rng):
        datasets = sorted({r.get('dataset', 'manifest') for r in self.rows})
        dataset = rng.choice(datasets)
        candidates = [i for i, r in enumerate(self.rows) if r.get('dataset', 'manifest') == dataset]
        subject = rng.choice(sorted({self.rows[i]['subject'] for i in candidates}))
        return self[rng.choice([i for i in candidates if self.rows[i]['subject'] == subject])]

    def __getitem__(self, i):
        row = dict(self.rows[i])
        if not Path(row['path']).is_absolute():
            row['path'] = str(self.root / row['path'])
        return Sample(ADAPTERS.get(row['adapter'])(row)).validate()
