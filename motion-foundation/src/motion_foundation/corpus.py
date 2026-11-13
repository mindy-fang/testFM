from __future__ import annotations

import hashlib
import json
from pathlib import Path
import numpy as np
from .adapters import sha256
from .contracts import Channel, UNITS


def assign(subject):
    value = int(hashlib.sha256(('v13-corpus-20261113:' + subject).encode()).hexdigest()[:8], 16) % 10
    return 'validation' if value == 0 else 'test' if value == 1 else 'train'


class Builder:
    def __init__(self, out):
        self.out = Path(out)
        self.out.mkdir(exist_ok=True)
        self.rows, self.audit, self.errors, self.seen = [], [], [], set()

    def put(self, dataset, subject, source, values, times, groups, *, observed=None,
            split=None, transform='physical', dependency=None, time_evidence='measured',
            evidence='', source_sha='', spans=None):
        key = hashlib.sha256((dataset + ':' + source).encode()).hexdigest()
        if key in self.seen:
            raise ValueError('duplicate source: ' + source)
        x, t = np.asarray(values, dtype=np.float32), np.asarray(times, dtype=np.float64)
        if x.ndim != 2 or t.ndim != 1 or len(t) != len(x) or len(x) < 2:
            raise ValueError('invalid source dimensions')
        if not np.isfinite(t).all():
            raise ValueError('nonfinite source time')
        obs = np.isfinite(x) if observed is None else np.asarray(observed, dtype=bool) & np.isfinite(x)
        if obs.shape != x.shape:
            raise ValueError('observed shape mismatch')
        if split not in (None, 'train', 'validation', 'test', 'quarantine'):
            raise ValueError('unknown split')
        if time_evidence not in ('measured', 'verified_relative_grid', 'constructed_grid'):
            raise ValueError('unknown time evidence')
        usable = []
        for group in groups:
            cols = group['cols']
            if len(cols) != len(group['channels']) or len(set(cols)) != len(cols):
                raise ValueError('invalid group columns')
            if not cols or min(cols) < 0 or max(cols) >= x.shape[1]:
                raise ValueError('group outside source columns')
            for desc in group['channels']:
                channel = Channel(**desc)
                if channel.quantity not in UNITS or channel.unit != UNITS[channel.quantity]:
                    raise ValueError('unknown physical quantity/unit')
            keep = [(c, d) for c, d in zip(cols, group['channels']) if obs[:, c].any()]
            if keep:
                usable.append(dict(cols=[c for c, _ in keep], channels=[d for _, d in keep], modality=group['modality']))
        if not usable:
            return
        if (np.diff(t) <= 0).any():
            cuts = np.r_[0, np.flatnonzero(np.diff(t) <= 0) + 1, len(t)]
            for j, (a, b) in enumerate(zip(cuts[:-1], cuts[1:])):
                if b - a >= 2:
                    self.put(dataset, subject, source + ':monotonic-part-' + str(j), x[a:b], t[a:b], groups,
                             observed=obs[a:b], split=split, transform=transform, dependency=dependency,
                             time_evidence=time_evidence, evidence=evidence + '; split at non-increasing time', source_sha=source_sha)
            self.seen.add(key)
            return
        if spans is None:
            cuts = np.r_[0, np.flatnonzero(np.diff(t) > .5) + 1, len(t)]
            spans = [[int(a), int(b)] for a, b in zip(cuts[:-1], cuts[1:]) if b-a >= 2 and t[b-1]-t[a] >= 1.]
        for a, b in spans:
            if not 0 <= a < b <= len(t) or b-a < 2 or np.diff(t[a:b]).max() > .5 + 1e-9:
                raise ValueError('invalid continuous span')
        if not spans:
            return
        dest = self.out / (key + '.npz')
        safe = np.where(obs, x, 0)
        if dest.exists():
            with np.load(dest, allow_pickle=False) as z:
                if not (np.array_equal(z['values'], safe) and np.array_equal(z['observed'], obs) and np.array_equal(z['times'], t)):
                    raise ValueError('cache collision')
        else:
            np.savez_compressed(dest, values=safe, observed=obs, times=t)
        self.rows.append(dict(dataset=dataset, subject=subject, source=source, path=dest.name,
                              sha256=sha256(dest), source_sha256=source_sha, split=split or assign(subject),
                              groups=usable, spans=spans, transform=transform, dependency_s=dependency,
                              time_evidence=time_evidence, evidence=evidence, rows=len(x), channels=x.shape[1],
                              duration_s=sum(float(t[b-1]-t[a]) for a, b in spans)))
        self.seen.add(key)

    def attempt(self, name, fn):
        before = len(self.rows)
        try:
            fn()
        except Exception as exc:
            self.errors.append(dict(dataset=name, error=repr(exc)))
        self.audit.append(dict(dataset=name, records=len(self.rows)-before))
        self.save()

    def save(self):
        payload = dict(schema='v13-native-corpus-1', rows=self.rows, audit=self.audit, errors=self.errors,
                       objective_policy='denoising does not authorize unknown-dependency temporal masking')
        (self.out / 'manifest.json').write_text(json.dumps(payload, indent=2), encoding='utf-8')
        return self.out / 'manifest.json'
