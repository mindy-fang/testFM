from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import tempfile
import time
import torch
import numpy as np
from .contracts import Channel, Sample, StreamRecord, SCHEMA
from .model import FoundationV13, ModelConfig, TaskHead
from .objectives import MaskedReconstruction, TrainingEngine, MaskPlan
from .runtime import save_package, load_package, transfer_parameters
from .adapters import read_npz, ManifestDataset, sha256


def signal(modality='imu', channels=3, fs=50., stream_id='a', **kwargs):
    n = int(fs * 1.4)
    times = torch.arange(n, dtype=torch.float64) / fs
    desc = tuple(Channel('pressure', 'kPa') for _ in range(channels)) if modality == 'pressure' else tuple(Channel('acc' if i < 3 else 'gyro', 'm/s^2' if i < 3 else 'rad/s', 'xyz'[i % 3], kwargs.pop('site', 'unknown') if i == 0 else 'unknown') for i in range(channels))
    values = torch.stack([torch.sin(times.float() * (i + 1)) + .2 * i for i in range(channels)], 1)
    return StreamRecord(values, times, torch.ones_like(values, dtype=torch.bool), desc, stream_id, 'clock:' + stream_id, 'synthetic:' + stream_id, 'synthetic:subject', 'train', modality, 'measured', dependency_s=0., channel_independent=True, **kwargs)


def raises(fn):
    try:
        fn()
    except (ValueError, FileExistsError):
        return
    raise AssertionError('expected explicit refusal')


def close(a, b):
    torch.testing.assert_close(a, b, atol=3e-6, rtol=3e-5)


def run():
    torch.set_num_threads(2)
    torch.manual_seed(1311)
    model = FoundationV13(ModelConfig(dim=32, heads=4, layers=1)).eval()
    objective = MaskedReconstruction(32, 4).eval()
    checks, real = [], []
    start = time.perf_counter()
    for modality, count, fs in [('imu', 1, 30.), ('imu', 3, 50.), ('imu', 6, 100.), ('pressure', 1, 29.97), ('pressure', 7, 50.), ('pressure', 16, 100.), ('pressure', 53, 51.2)]:
        s = signal(modality, count, fs)
        r = model(Sample([s])); assert all(torch.isfinite(h).all() for h in r.fused.values())
        assert len(r.fused) == 1 and r.contextual_local[s.stream_id].shape == r.local[s.stream_id].features.shape
        if count == 1 and modality == 'imu':
            g = replace(s, channels=(Channel('gyro', 'rad/s', 'z'),))
            assert model(Sample([g])).pooled()
    checks.append('dynamic pressure/IMU-only/acc-only/gyro-only/noninteger frequency')
    s = signal('pressure', 7)
    perm = torch.randperm(7)
    p = replace(s, values=s.values[:, perm], observed=s.observed[:, perm], channels=tuple(s.channels[i] for i in perm))
    close(next(iter(model(Sample([s])).pooled().values())), next(iter(model(Sample([p])).pooled().values())))
    base = model(Sample([s])).pooled()
    padded = replace(s, values=torch.cat((s.values, torch.full((len(s.times), 1), float('nan'))), 1), observed=torch.cat((s.observed, torch.zeros((len(s.times), 1), dtype=torch.bool)), 1), channels=s.channels + (Channel('pressure', 'kPa'),))
    close(next(iter(base.values())), next(iter(model(Sample([padded])).pooled().values())))
    checks.append('pressure permutation and invalid channel padding invariance')
    s = signal('imu', 6)
    sample = Sample([s]); plan = objective.plan(sample)
    altered = s.values.clone(); altered[~plan.visible['a']] = float('nan')
    hidden = replace(s, values=altered, observed=s.observed & plan.visible['a'])
    close(next(iter(model(sample, plan.visible).pooled().values())), next(iter(model(Sample([hidden])).pooled().values())))
    clean_loss = objective.loss(model, sample, plan)
    assert clean_loss.valid_targets > 0 and torch.isfinite(clean_loss.loss)
    clean_loss.loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.temporal.parameters())
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in list(model.parameters()) + list(objective.parameters()))
    model.zero_grad(set_to_none=True); objective.zero_grad(set_to_none=True)
    checks.append('hidden values isolated; finite backward reaches shared temporal trunk')
    a, b = signal('imu', 3, stream_id='a'), signal('imu', 3, fs=100., stream_id='b')
    independent = model(Sample([a, b])); assert len(independent.fused) == 2
    changed = replace(b, values=b.values * 8)
    close(independent.fused['independent:a'], model(Sample([a, changed])).fused['independent:a'])
    synced = [replace(x, fusion_ref='trial', sync_error_s=.001, sync_evidence='synthetic known mapping') for x in [a, b]]
    assert len(model(Sample(synced)).fused) == 1
    raises(lambda: model(Sample([replace(synced[0], sync_error_s=.1)])))
    raises(lambda: model(Sample([replace(a, fusion_ref='x')])) )
    shifted = replace(b, time_offset=2.)
    result = model(Sample([a, shifted])); assert result.times['independent:b'][0] >= 2
    checks.append('independent clock isolation, explicit mixed-rate fusion and error-bound refusal')
    s = signal('imu', 3)
    engine = TrainingEngine(model, objective)
    unknown = replace(s, dependency_s=None)
    assert engine.compute([Sample([unknown])]).loss is None
    axis = objective.plan(Sample([replace(s, time_evidence='constructed_grid', dependency_s=None)]), 'axis')
    assert axis.targets['a'].any()
    assert not objective.plan(Sample([signal('pressure', 3)]), 'axis').targets['a'].any()
    raises(lambda: engine.compute([Sample([replace(s, split='test')])]))
    raises(lambda: model(Sample([replace(s, observed=torch.zeros_like(s.observed))])))
    raises(lambda: model(Sample([replace(s, times=s.times.flip(0))])))
    raises(lambda: model(Sample([replace(s, channels=(Channel('acc', 'g', 'x'),) * 3)])))
    raises(lambda: model(Sample([replace(s, modality='video')])))
    raises(lambda: model(Sample([replace(s, times=s.times + (s.times > .6) * 2)])))
    checks.append('eligibility, no-target handling, held-out guard, malformed input refusal')
    out = model(Sample([s])); assert TaskHead(32, 4)(out)['independent:a'].shape == (4,)
    assert TaskHead(32, 2)(out, 'contextual_local')['a'].shape[-1] == 2
    engine = TrainingEngine(model, objective)
    x, y = Sample([signal('imu', 3, fs=30.)]), Sample([signal('pressure', 5, fs=100.)])
    lx, ly = engine.compute([x]).loss, engine.compute([y]).loss
    weighted = engine.compute([x, y], weights=[1., 3.]).loss
    close(weighted, (lx + 3 * ly) / 4)
    checks.append('classification/dense heads and heterogeneous gradient accumulation weights')
    jitter = signal('imu', 3, fs=51.2)
    jitter.times[1:-1] += .001 * torch.sin(torch.arange(len(jitter.times) - 2, dtype=torch.float64))
    jitter.observed[(jitter.times > .4) & (jitter.times < .65)] = False
    jitter.values[~jitter.observed] = float('nan')
    assert model(Sample([jitter])).pooled()
    fir = replace(s, dependency_s=.06)
    safe_plan = objective.plan(Sample([fir]))
    assert objective.loss(model, Sample([fir]), safe_plan).loss is not None
    unsafe = MaskPlan({s.stream_id: s.observed & ~safe_plan.targets[s.stream_id]}, safe_plan.targets, 'temporal')
    raises(lambda: objective.loss(model, Sample([fir]), unsafe))
    optimizer = torch.optim.AdamW(list(model.parameters()) + list(objective.parameters()))
    snapshots = [p.detach().clone() for p in model.parameters()]
    skipped = engine.step([Sample([unknown])], optimizer)
    assert skipped.loss is None and not optimizer.state
    assert all(torch.equal(a, b) for a, b in zip(snapshots, model.parameters()))
    checks.append('jitter/empty patch, dependency-safe masks and no-target optimizer skip')
    with tempfile.TemporaryDirectory(prefix='v13_selfcheck_') as temp:
        path = Path(temp) / 'roundtrip.pt'
        save_package(path, model, objective, optimizer=optimizer, data_fingerprint='synthetic', sampler_state={'attempts': 0, 'updates': 0})
        loaded, _, metadata = load_package(path, expected_fingerprint='synthetic'); loaded.eval()
        close(next(iter(model(x).pooled().values())), next(iter(loaded(x).pooled().values())))
        raises(lambda: save_package(path, model))
        raises(lambda: load_package(path, expected_fingerprint='other'))
        migration = transfer_parameters(model.state_dict(), loaded, {})
        assert not migration['copied'] and migration['unused']
        npz = Path(temp) / 'sample.npz'
        np.savez(npz, values=s.values.numpy(), observed=s.observed.numpy(), times=s.times.numpy())
        descriptor = dict(schema=SCHEMA, adapter='npz-v13', path=str(npz), sha256=sha256(npz), channels=[vars(c) for c in s.channels], stream_id=s.stream_id, clock=s.clock, source=s.source, subject=s.subject, split='train', modality='imu', time_evidence='measured', dependency_s=0., channel_independent=True)
        assert read_npz(descriptor)[0].values.shape == s.values.shape
        manifest = Path(temp) / 'manifest.json'; manifest.write_text(json.dumps(dict(schema=SCHEMA, rows=[descriptor])))
        dataset = ManifestDataset(manifest)
        assert len(dataset[0].streams) == 1
        import random
        assert len(dataset.sample(random.Random(13)).streams) == 1
        no_time = Path(temp) / 'no_time.npz'
        np.savez(no_time, values=s.values.numpy(), observed=s.observed.numpy())
        no_time_desc = dict(descriptor, path=str(no_time), sha256=sha256(no_time))
        raises(lambda: read_npz(no_time_desc))
        relative = read_npz(dict(no_time_desc, time_evidence='verified_relative_grid', verified_fs=50., continuity_evidence='synthetic fixed grid'))
        close(relative[0].times, s.times)
        bad = dict(descriptor, split='test'); manifest.write_text(json.dumps(dict(schema=SCHEMA, rows=[descriptor, bad])))
        raises(lambda: ManifestDataset(manifest))
    checks.append('model package roundtrip, strict plugin/fingerprint and NPZ manifest adapter')
    from .denoising import ObservationDenoising
    denoising = ObservationDenoising(32)
    sample = Sample([signal('pressure', 7)])
    result, _ = denoising.compute(model, sample)
    result.loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.temporal.parameters())
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in list(model.parameters()) + list(denoising.parameters()))
    checks.append('observation denoising finite backward through shared trunk')
    default = FoundationV13()
    default_rep = default(Sample([signal('imu', 3)]))
    assert next(iter(default_rep.pooled().values())).shape == (128,)
    checks.append('default 128-dim model path')
    return dict(check_groups=checks, groups_passed=len(checks), real_cache_checks=real, model_parameters=sum(p.numel() for p in default.parameters()), objective_parameters=sum(p.numel() for p in MaskedReconstruction(128).parameters()), cpu_seconds=time.perf_counter() - start, optimizer_steps=0, device='cpu', limitations=['synthetic CPU checks only; real corpus validation is separate', 'no V13 pretrained weights', 'video not implemented', 'training recovery not verified'])


def main():
    print(json.dumps(run(), indent=2), flush=True)


if __name__ == '__main__':
    main()
