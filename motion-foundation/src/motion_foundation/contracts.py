from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Callable
import math
import torch

SCHEMA = 'motion-stream-v13.1'
UNITS = {'pressure': 'kPa', 'acc': 'm/s^2', 'gyro': 'rad/s'}
SCALES = {'pressure': 100.0, 'acc': 9.80665, 'gyro': 1.0}


class Registry:
    def __init__(self):
        self.items = {}

    def register(self, name: str, value: Callable):
        if name in self.items:
            raise ValueError('duplicate plugin: ' + name)
        self.items[name] = value
        return value

    def get(self, name):
        if name not in self.items:
            raise ValueError('unregistered plugin: ' + name)
        return self.items[name]


@dataclass(frozen=True)
class Channel:
    quantity: str
    unit: str
    axis: str = 'scalar'
    site: str = 'unknown'
    side: str = 'unknown'


@dataclass
class StreamRecord:
    values: torch.Tensor
    times: torch.Tensor
    observed: torch.Tensor
    channels: tuple[Channel, ...]
    stream_id: str
    clock: str
    source: str
    subject: str
    split: str
    modality: str
    time_evidence: str
    transform: str = 'physical'
    dependency_s: float | None = None
    channel_independent: bool = False
    time_scale: float = 1.0
    time_offset: float = 0.0
    fusion_ref: str | None = None
    sync_error_s: float | None = None
    sync_evidence: str = ''

    def validate(self):
        if not self.modality:
            raise ValueError('missing modality; frontend registry checks support')
        if self.values.ndim != 2 or self.observed.shape != self.values.shape:
            raise ValueError('values/observed must be [time, channel]')
        if self.values.shape[1] != len(self.channels) or not self.channels:
            raise ValueError('channel descriptor mismatch')
        if self.observed.dtype != torch.bool or self.times.ndim != 1 or len(self.times) != len(self.values):
            raise ValueError('invalid observed/time shape')
        if len(self.times) < 2 or not torch.isfinite(self.times).all() or not (self.times[1:] > self.times[:-1]).all():
            raise ValueError('time must be finite, strictly increasing, and nonempty')
        if not torch.isfinite(self.values[self.observed]).all():
            raise ValueError('nonfinite observed values')
        if self.time_evidence not in ('measured', 'verified_relative_grid', 'constructed_grid'):
            raise ValueError('unverified time grid')
        if self.transform not in ('physical', 'asinh_reference_v1'):
            raise ValueError('unknown signal transform')
        if self.split not in ('train', 'validation', 'test', 'quarantine'):
            raise ValueError('unknown split')
        if not all((self.stream_id, self.clock, self.source, self.subject)):
            raise ValueError('missing provenance or conservative identity')
        for c in self.channels:
            if c.quantity not in UNITS or c.unit != UNITS[c.quantity]:
                raise ValueError('unregistered quantity/unit')
            if self.modality == 'pressure' and (c.quantity != 'pressure' or c.axis != 'scalar'):
                raise ValueError('pressure descriptor mismatch')
            if self.modality == 'imu' and (c.quantity not in ('acc', 'gyro') or c.axis not in ('x', 'y', 'z')):
                raise ValueError('IMU descriptor mismatch')
            if c.side not in ('left', 'right', 'unknown'):
                raise ValueError('unknown laterality')
        if self.modality == 'imu' and len({(c.quantity, c.axis) for c in self.channels}) != len(self.channels):
            raise ValueError('one IMU device per stream; duplicate axis')
        if not math.isfinite(self.time_scale) or self.time_scale <= 0 or not math.isfinite(self.time_offset):
            raise ValueError('invalid clock mapping')
        if self.dependency_s is not None and (not math.isfinite(self.dependency_s) or self.dependency_s < 0):
            raise ValueError('invalid dependency radius')
        if self.fusion_ref is not None and (not self.sync_evidence or self.sync_error_s is None or not math.isfinite(self.sync_error_s) or self.sync_error_s < 0):
            raise ValueError('fusion requires explicit mapping evidence/error')
        return self

    def mapped_times(self):
        return self.times * self.time_scale + self.time_offset

    def normalized(self):
        safe = torch.where(self.observed, self.values, torch.zeros_like(self.values))
        if self.transform == 'asinh_reference_v1':
            return safe
        scales = safe.new_tensor([SCALES[c.quantity] for c in self.channels])
        return torch.asinh(safe / scales)

    def crop(self, start_s, end_s):
        mask = (self.times >= start_s) & (self.times < end_s)
        return replace(self, values=self.values[mask], times=self.times[mask], observed=self.observed[mask]).validate()


@dataclass
class Sample:
    streams: list[StreamRecord]

    def validate(self):
        if not self.streams or len({s.stream_id for s in self.streams}) != len(self.streams):
            raise ValueError('nonempty unique streams required')
        if len({(s.subject, s.split) for s in self.streams}) != 1:
            raise ValueError('cannot fuse different subjects/splits')
        for s in self.streams:
            s.validate()
        return self


def support_intervals(times, mask):
    indices = mask.nonzero(as_tuple=False).flatten().tolist()
    if not indices:
        return ()
    groups, a, b = [], indices[0], indices[0]
    for i in indices[1:]:
        if i != b + 1:
            groups.append((float(times[a]), float(times[b])))
            a = i
        b = i
    groups.append((float(times[a]), float(times[b])))
    return tuple(groups)


@dataclass
class TokenBundle:
    features: torch.Tensor
    times: torch.Tensor
    intervals: torch.Tensor
    stream_ref: str
    clock_ref: str
    routes: list[tuple[int, int]]
    observed_support: list[tuple]
    dependency_support: list[tuple]
    visible_fraction: torch.Tensor


@dataclass
class RepresentationBundle:
    local: dict[str, TokenBundle] = field(default_factory=dict)
    streams: dict[str, torch.Tensor] = field(default_factory=dict)
    fused: dict[str, torch.Tensor] = field(default_factory=dict)
    times: dict[str, torch.Tensor] = field(default_factory=dict)
    receptive: dict[str, tuple[float, float]] = field(default_factory=dict)
    stream_groups: dict[str, str] = field(default_factory=dict)
    contextual_local: dict[str, torch.Tensor] = field(default_factory=dict)

    def pooled(self):
        return {k: v.mean(0) for k, v in self.fused.items()}
