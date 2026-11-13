from __future__ import annotations

from dataclasses import dataclass
import torch
from torch import nn
from .contracts import Sample, Registry
from .model import time_features


@dataclass
class Eligibility:
    enabled: bool
    reason: str
    candidates: int


@dataclass
class MaskPlan:
    visible: dict[str, torch.Tensor]
    targets: dict[str, torch.Tensor]
    objective: str


@dataclass
class LossResult:
    loss: torch.Tensor | None
    valid_targets: int
    sample_count: int
    by_quantity: dict[str, float]
    skipped: tuple[str, ...] = ()


class MaskedReconstruction(nn.Module):
    def __init__(self, dim, heads=4):
        super().__init__()
        self.query_time = nn.Linear(5, dim)
        self.query_context = nn.Linear(dim, dim, bias=False)
        self.attention = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.predict = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, 1))
        self.registry = Registry()
        self.registry.register('temporal', self.assess_temporal)
        self.registry.register('axis', self.assess_axis)

    @staticmethod
    def assess_temporal(stream):
        enabled = stream.time_evidence in ('measured', 'verified_relative_grid') and stream.dependency_s is not None
        return Eligibility(enabled, 'known temporal dependency' if enabled else 'temporal dependency/time evidence incomplete', int(stream.observed.sum()) if enabled else 0)

    @staticmethod
    def assess_axis(stream):
        enabled = stream.modality == 'imu' and stream.channel_independent and len(stream.channels) > 1
        return Eligibility(enabled, 'independent axes' if enabled else 'axis holdout requires independent IMU channels', int(stream.observed.sum()) if enabled else 0)

    def plan(self, sample: Sample, mode='temporal'):
        sample.validate()
        assess = self.registry.get(mode)
        visible, targets = {}, {}
        for s in sample.streams:
            vis, target = s.observed.clone(), torch.zeros_like(s.observed)
            if assess(s).enabled:
                if mode == 'temporal':
                    t = s.times
                    a = t[0] + (t[-1] - t[0]) * .35
                    b = t[0] + (t[-1] - t[0]) * .60
                    center = (t >= a) & (t < b)
                    target[center] = s.observed[center]
                    # Hide the entire known dependency neighborhood, not only target rows.
                    expanded = (t >= a - 2 * s.dependency_s) & (t <= b + 2 * s.dependency_s)
                    vis[expanded] = False
                    for c in range(len(s.channels)):
                        if not vis[:, c].any():
                            target[:, c] = False
                else:
                    counts = s.observed.sum(0)
                    choices = (counts > 0).nonzero(as_tuple=False).flatten()
                    if len(choices) > 1:
                        c = int(choices[0])
                        target[:, c] = s.observed[:, c]
                        vis[:, c] = False
            if not vis.any():
                vis, target = s.observed.clone(), torch.zeros_like(s.observed)
            visible[s.stream_id], targets[s.stream_id] = vis, target
        return MaskPlan(visible, targets, mode)

    def loss(self, model, sample, plan):
        if set(plan.visible) != {s.stream_id for s in sample.streams} or set(plan.targets) != set(plan.visible):
            raise ValueError('mask plan streams differ from sample')
        assessed = self.registry.get(plan.objective)
        total = 0
        for s in sample.streams:
            target = plan.targets[s.stream_id]
            vis = plan.visible[s.stream_id]
            if target.shape != s.observed.shape or target.dtype != torch.bool or vis.shape != target.shape or vis.dtype != torch.bool:
                raise ValueError('invalid target/visibility mask')
            if (target & (~s.observed | vis)).any() or (vis & ~s.observed).any() or (target.any() and not assessed(s).enabled):
                raise ValueError('illegal target')
            if target.any() and plan.objective == 'temporal':
                for c in range(len(s.channels)):
                    indices = target[:, c].nonzero(as_tuple=False).flatten()
                    if not len(indices):
                        continue
                    if not vis[:, c].any():
                        raise ValueError('temporal target needs visible channel context')
                    distances = (s.times[vis.any(1)][:, None] - s.times[indices][None, :]).abs()
                    if (distances <= 2 * s.dependency_s).any():
                        raise ValueError('visible input intersects target dependency')
            if target.any() and plan.objective == 'axis':
                for c in range(len(s.channels)):
                    if target[:, c].any() and vis[:, c].any():
                        raise ValueError('axis objective requires full-axis holdout')
            total += int(target.sum())
        if total == 0:
            return LossResult(None, 0, 0, {}, ('no legal targets',))
        for target_stream in sample.streams:
            if plan.objective != 'temporal' or not plan.targets[target_stream.stream_id].any():
                continue
            target_rows = plan.targets[target_stream.stream_id].any(1)
            tt = target_stream.mapped_times()[target_rows]
            for context_stream in sample.streams:
                if model.group_key(context_stream) != model.group_key(target_stream):
                    continue
                if context_stream.dependency_s is None:
                    raise ValueError('temporal target has fused input with unknown dependency')
                visible_rows = plan.visible[context_stream.stream_id].any(1)
                vt = context_stream.mapped_times()[visible_rows]
                radius = target_stream.dependency_s * target_stream.time_scale + context_stream.dependency_s * context_stream.time_scale
                if len(vt) and ((vt[:, None] - tt[None, :]).abs() <= radius).any():
                    raise ValueError('fused input intersects target dependency; use joint safe mask')
        representation = model(sample, plan.visible)
        losses = {}
        for s in sample.streams:
            mask = plan.targets[s.stream_id]
            if not mask.any():
                continue
            h = representation.fused[representation.stream_groups[s.stream_id]]
            bundle = representation.local.get(s.stream_id)
            origin = representation.times[representation.stream_groups[s.stream_id]][0]
            stream_losses = {}
            for c, channel in enumerate(s.channels):
                indices = mask[:, c].nonzero(as_tuple=False).flatten()
                if not len(indices):
                    continue
                qtime = (s.mapped_times()[indices] - origin).to(h.dtype)
                q = self.query_time(time_features(qtime)) + model.frontends['scalar'].semantic(channel, h.device)
                if bundle is not None:
                    local_indices = [i for i, route in enumerate(bundle.routes) if route[0] == c]
                    if local_indices:
                        q = q + self.query_context(bundle.features[local_indices].mean(0))
                context, _ = self.attention(q.unsqueeze(0), h.unsqueeze(0), h.unsqueeze(0), need_weights=False)
                prediction = self.predict(context[0]).flatten()
                clean = s.normalized()[indices, c]
                err = nn.functional.smooth_l1_loss(prediction, clean, reduction='none')
                stream_losses.setdefault(channel.quantity, []).append(err)
            for quantity, errors in stream_losses.items():
                losses.setdefault(quantity, []).append(torch.cat(errors).mean())
        if not losses:
            return LossResult(None, 0, 0, {}, ('no decodable targets',))
        means = {k: torch.stack(v).mean() for k, v in losses.items()}
        return LossResult(torch.stack(list(means.values())).mean(), total, 1, {k: float(v.detach()) for k, v in means.items()})


class TrainingEngine:
    def __init__(self, model, objective):
        self.model, self.objective = model, objective

    def compute(self, samples, modes=None, weights=None):
        if not samples:
            return LossResult(None, 0, 0, {}, ('empty batch',))
        if any(s.split != 'train' for sample in samples for s in sample.streams):
            raise ValueError('training input contains held-out/quarantine records')
        modes = modes or ['temporal'] * len(samples)
        weights = weights if weights is not None else [1.] * len(samples)
        if len(modes) != len(samples) or len(weights) != len(samples) or any(not 0 < w < float('inf') for w in weights):
            raise ValueError('invalid microbatch schedule')
        active, skipped, quantities, count = [], [], {}, 0
        for sample, mode, weight in zip(samples, modes, weights):
            plan = self.objective.plan(sample, mode)
            result = self.objective.loss(self.model, sample, plan)
            if result.loss is None:
                skipped.extend(result.skipped)
                continue
            active.append((result.loss, weight)); count += result.valid_targets
            for k, value in result.by_quantity.items():
                quantities.setdefault(k, []).append(value)
        if not active:
            return LossResult(None, 0, 0, {}, tuple(skipped))
        loss = sum(v * w for v, w in active) / sum(w for _, w in active)
        return LossResult(loss, count, len(active), {k: sum(v) / len(v) for k, v in quantities.items()}, tuple(skipped))

    def step(self, samples, optimizer, modes=None, weights=None):
        optimizer.zero_grad(set_to_none=True)
        result = self.compute(samples, modes, weights)
        if result.loss is None:
            return result
        if not torch.isfinite(result.loss):
            raise ValueError('nonfinite loss')
        result.loss.backward()
        parameters = [p for group in optimizer.param_groups for p in group['params'] if p.grad is not None]
        if any(not torch.isfinite(p.grad).all() for p in parameters):
            optimizer.zero_grad(set_to_none=True)
            raise ValueError('nonfinite gradients')
        optimizer.step()
        return result
