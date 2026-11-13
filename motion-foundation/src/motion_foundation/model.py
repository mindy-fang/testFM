from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import torch
from torch import nn
from .contracts import Registry, Sample, TokenBundle, RepresentationBundle, support_intervals


@dataclass
class ModelConfig:
    dim: int = 128
    heads: int = 4
    layers: int = 2
    patch_s: float = 0.2
    subqueries: int = 3
    group_tokens: int = 2
    max_tokens: int = 16000
    max_gap_s: float = 0.5
    max_sync_error_s: float = 0.01
    sites: tuple[str, ...] = ('unknown', 'foot', 'shank', 'thigh', 'sacrum', 'trunk', 'lower_back', 'wrist', 'ankle', 'sternum', 'forehead')

    def __post_init__(self):
        self.sites = tuple(self.sites)
        if self.dim < 8 or self.dim % self.heads or self.layers < 1 or self.patch_s <= 0 or self.subqueries < 1 or self.group_tokens < 1 or self.max_tokens < 1 or self.max_gap_s <= 0 or self.max_sync_error_s < 0:
            raise ValueError('invalid model configuration')
        if not self.sites or self.sites[0] != 'unknown' or len(set(self.sites)) != len(self.sites):
            raise ValueError('site vocabulary must be unique and start with unknown')


def time_features(t):
    return torch.stack((t, torch.sin(t), torch.cos(t), torch.sin(t * 5), torch.cos(t * 5)), -1)


class ScalarFrontend(nn.Module):
    def __init__(self, config):
        super().__init__()
        d = config.dim
        self.config = config
        self.point = nn.Sequential(nn.Linear(3, d), nn.GELU(), nn.Linear(d, d))
        self.quantity = nn.Embedding(3, d)
        self.axis = nn.Embedding(4, d)
        self.site = nn.Embedding(len(config.sites), d)
        self.side = nn.Embedding(3, d)
        self.query = nn.Parameter(torch.randn(config.subqueries, d) / math.sqrt(d))
        self.combine = nn.Linear(config.subqueries * d + 2, d)
        self.clock = nn.Linear(5, d)

    @staticmethod
    def supports(stream):
        try:
            stream.validate()
            if stream.modality not in ('pressure', 'imu'):
                return False, 'scalar frontend supports pressure/IMU only'
            return True, 'validated scalar sensor contract'
        except ValueError as exc:
            return False, str(exc)

    def semantic(self, channel, device):
        qi = {'pressure': 0, 'acc': 1, 'gyro': 2}[channel.quantity]
        ai = {'scalar': 0, 'x': 1, 'y': 2, 'z': 3}[channel.axis]
        si = self.config.sites.index(channel.site) if channel.site in self.config.sites else 0
        li = {'unknown': 0, 'left': 1, 'right': 2}[channel.side]
        return self.quantity.weight[qi] + self.axis.weight[ai] + self.site.weight[si] + self.side.weight[li]

    def forward(self, stream, visible, origin):
        cfg, device = self.config, stream.values.device
        t = stream.mapped_times().to(device)
        x = stream.normalized()
        channels = len(stream.channels)
        n = int(torch.floor((t[-1] - origin) / cfg.patch_s).item()) + 1
        if n * channels > cfg.max_tokens:
            raise ValueError('local token budget exceeded; crop or explicitly subselect streams')
        row, col = (visible & stream.observed).nonzero(as_tuple=True)
        if not len(row):
            return None
        patch = torch.floor((t - origin) / cfg.patch_s).long()
        code = patch[row] * channels + col
        order = torch.argsort(code, stable=True)
        row, col, code = row[order], col[order], code[order]
        unique, group, counts = torch.unique_consecutive(code, return_inverse=True, return_counts=True)
        vt = t[row]
        dt = torch.cat((vt[:1] * 0, vt[1:] - vt[:-1]))
        dt[torch.cat((torch.ones(1, device=device, dtype=torch.bool), group[1:] != group[:-1]))] = 0
        a = origin + patch[row] * cfg.patch_s
        point_values = torch.stack((x[row, col], (vt - a).to(x.dtype), dt.to(x.dtype)), -1)
        semantic = torch.stack([self.semantic(c, device) for c in stream.channels])
        point = self.point(point_values) + semantic[col]
        score = point @ self.query.T / math.sqrt(cfg.dim)
        index = group[:, None].expand(-1, cfg.subqueries)
        maximum = score.new_full((len(unique), cfg.subqueries), -torch.inf)
        maximum.scatter_reduce_(0, index, score, reduce='amax', include_self=True)
        weight = torch.exp(score - maximum[group])
        denominator = score.new_zeros((len(unique), cfg.subqueries)).index_add(0, group, weight)
        weight = weight / denominator[group]
        pooled = point.new_zeros((len(unique), cfg.subqueries, cfg.dim))
        pooled.index_add_(0, group, weight[:, :, None] * point[:, None, :])
        ks, cs = unique // channels, unique % channels
        rows_per_patch = torch.bincount(patch, minlength=n)
        fraction = counts.to(x.dtype) / rows_per_patch[ks]
        aux = torch.stack((fraction, torch.log1p(counts.to(x.dtype))), -1)
        centers = origin + (ks + .5) * cfg.patch_s
        features = self.combine(torch.cat((pooled.flatten(1), aux), -1)) + self.clock(time_features((centers - origin).to(x.dtype)))
        bounds = torch.stack((origin + ks * cfg.patch_s, origin + (ks + 1) * cfg.patch_s), -1)
        routes = list(zip(cs.tolist(), ks.tolist()))
        tt, rr, gg = t.detach().cpu().tolist(), row.cpu().tolist(), group.cpu().tolist()
        supports, dependencies = [], []
        by_group = [[] for _ in routes]
        for r, g in zip(rr, gg):
            by_group[g].append(r)
        for indices in by_group:
            spans, first, last = [], indices[0], indices[0]
            for r in indices[1:]:
                if r != last + 1:
                    spans.append((tt[first], tt[last])); first = r
                last = r
            spans.append((tt[first], tt[last]))
            supports.append(tuple(spans))
            radius = stream.dependency_s
            dependencies.append(None if radius is None else tuple((lo - radius * stream.time_scale, hi + radius * stream.time_scale) for lo, hi in spans))
        return TokenBundle(features, centers, bounds, stream.stream_id, stream.clock, routes, supports, dependencies, fraction)

    def forward_reference(self, stream, visible, origin):
        cfg, device = self.config, stream.values.device
        t = stream.mapped_times().to(device)
        visible = visible & stream.observed
        x = stream.normalized()
        n = int(torch.floor((t[-1] - origin) / cfg.patch_s).item()) + 1
        if n * len(stream.channels) > cfg.max_tokens:
            raise ValueError('local token budget exceeded; crop or explicitly subselect streams')
        features, centers, intervals, routes, supports, dependencies, fractions = [], [], [], [], [], [], []
        for k in range(n):
            a, b = origin + k * cfg.patch_s, origin + (k + 1) * cfg.patch_s
            within = (t >= a) & (t < b)
            if not within.any():
                continue
            for c, desc in enumerate(stream.channels):
                mask = within & visible[:, c]
                if not mask.any():
                    continue
                vt = t[mask]
                dt = torch.cat((vt[:1] * 0, vt[1:] - vt[:-1]))
                values = torch.stack((x[mask, c], (vt - a).to(x.dtype), dt.to(x.dtype)), -1)
                h = self.point(values) + self.semantic(desc, device)
                weights = torch.softmax(self.query @ h.T / math.sqrt(cfg.dim), -1)
                pooled = (weights @ h).flatten()
                fraction = mask.sum().to(x.dtype) / within.sum()
                aux = torch.stack((fraction, torch.log1p(mask.sum().to(x.dtype))))
                center = (a + b) * .5
                h = self.combine(torch.cat((pooled, aux))) + self.clock(time_features((center - origin).to(x.dtype)))
                features.append(h); centers.append(center); intervals.append(torch.stack((a, b)))
                routes.append((c, k)); fractions.append(fraction)
                support = support_intervals(t, mask)
                supports.append(support)
                radius = stream.dependency_s
                dependencies.append(None if radius is None else tuple((lo - radius * stream.time_scale, hi + radius * stream.time_scale) for lo, hi in support))
        if not features:
            return None
        return TokenBundle(torch.stack(features), torch.stack(centers), torch.stack(intervals), stream.stream_id, stream.clock, routes, supports, dependencies, torch.stack(fractions))


class FoundationV13(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = config or ModelConfig()
        c, d = self.config, self.config.dim
        self.frontends = nn.ModuleDict({'scalar': ScalarFrontend(c)})
        self.frontend_registry = Registry()
        self.frontend_registry.register('pressure', self.frontends['scalar'])
        self.frontend_registry.register('imu', self.frontends['scalar'])
        self.group_query = nn.Parameter(torch.randn(c.group_tokens, d) / math.sqrt(d))
        self.group_attention = nn.MultiheadAttention(d, c.heads, batch_first=True)
        layer = nn.TransformerEncoderLayer(d, c.heads, d * 3, dropout=0., batch_first=True, norm_first=True)
        self.temporal = nn.TransformerEncoder(layer, c.layers, enable_nested_tensor=False)
        self.context_attention = nn.MultiheadAttention(d, c.heads, batch_first=True)
        self.norm = nn.LayerNorm(d)

    def register_frontend(self, modality, module):
        if modality in self.frontend_registry.items:
            raise ValueError('duplicate frontend')
        self.frontends[modality] = module
        self.frontend_registry.register(modality, module)

    def group_key(self, stream):
        if stream.fusion_ref is None:
            return 'independent:' + stream.stream_id
        if stream.sync_error_s > self.config.max_sync_error_s:
            raise ValueError('mapping error exceeds fusion policy')
        return 'mapped:' + stream.fusion_ref

    def encode(self, sample: Sample, visible=None):
        sample.validate()
        visible = visible or {}
        if set(visible) - {s.stream_id for s in sample.streams}:
            raise ValueError('unknown visibility stream')
        groups = {}
        for s in sample.streams:
            supported, reason = self.frontend_registry.get(s.modality).supports(s)
            if not supported:
                raise ValueError(reason)
            mapped = s.mapped_times()
            if (mapped[1:] - mapped[:-1]).max() > self.config.max_gap_s:
                raise ValueError('long timestamp gap: crop continuous intervals before encode')
            groups.setdefault(self.group_key(s), []).append(s)
        output = RepresentationBundle()
        for key, streams in groups.items():
            origin = min(s.mapped_times()[0] for s in streams)
            bundles = []
            for s in streams:
                vis = visible.get(s.stream_id, s.observed)
                if vis.shape != s.observed.shape or vis.dtype != torch.bool or (vis & ~s.observed).any():
                    raise ValueError('visibility must be a subset of observed')
                bundle = self.frontend_registry.get(s.modality)(s, vis, origin)
                output.stream_groups[s.stream_id] = key
                if bundle is not None:
                    bundles.append(bundle); output.local[s.stream_id] = bundle
            if not bundles:
                raise ValueError('fusion group has no visible input')
            count = sum(len(b.features) for b in bundles)
            if count > self.config.max_tokens:
                raise ValueError('fusion token budget exceeded')
            # Per-stream pooling prevents dense devices from winning only by token count.
            per_stream = {}
            for bundle in bundles:
                rows = {}
                for k in sorted({route[1] for route in bundle.routes}):
                    indices = [i for i, route in enumerate(bundle.routes) if route[1] == k]
                    values = bundle.features[indices].unsqueeze(0)
                    q = self.group_query.unsqueeze(0)
                    z, _ = self.group_attention(q, values, values, need_weights=False)
                    rows[k] = z[0].mean(0)
                per_stream[bundle.stream_ref] = rows
                output.streams[bundle.stream_ref] = torch.stack(list(rows.values()))
            ks = sorted({k for rows in per_stream.values() for k in rows})
            modalities = {s.stream_id: s.modality for s in streams}
            zrows = []
            for k in ks:
                per_modality = {}
                for stream_id, rows in per_stream.items():
                    if k in rows:
                        per_modality.setdefault(modalities[stream_id], []).append(rows[k])
                zrows.append(torch.stack([torch.stack(values).mean(0) for values in per_modality.values()]).mean(0))
            z = torch.stack(zrows)
            h = self.norm(self.temporal(z.unsqueeze(0))[0])
            output.fused[key] = h
            output.times[key] = torch.stack([origin + (k + .5) * self.config.patch_s for k in ks])
            output.receptive[key] = (float(min(s.mapped_times()[0] for s in streams)), float(max(s.mapped_times()[-1] for s in streams)))
            for b in bundles:
                context, _ = self.context_attention(b.features.unsqueeze(0), h.unsqueeze(0), h.unsqueeze(0), need_weights=False)
                output.contextual_local[b.stream_ref] = b.features + context[0]
        return output

    def forward(self, sample, visible=None):
        return self.encode(sample, visible)

    def config_dict(self):
        return asdict(self.config)


class TaskHead(nn.Module):
    def __init__(self, dim, outputs):
        super().__init__()
        self.readout = nn.Linear(dim, outputs)

    def forward(self, representation, level='pooled'):
        if level == 'pooled':
            values = representation.pooled()
        elif level in ('fused', 'contextual_local', 'streams'):
            values = getattr(representation, level)
        else:
            raise ValueError('unsupported task readout')
        return {k: self.readout(v) for k, v in values.items()}
