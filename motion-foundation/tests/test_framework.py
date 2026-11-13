from dataclasses import asdict, replace
import importlib
import json
from pathlib import Path
import random
import numpy as np
import pytest
import torch
from motion_foundation import FoundationV13, ModelConfig, Sample
from motion_foundation.corpus import Builder
from motion_foundation.native_data import NativeCorpus
from motion_foundation.selfcheck import run, signal
from motion_foundation.denoising import ObservationDenoising
from motion_foundation.train_mixed import checkpoint
from motion_foundation.registry import build_model
from motion_foundation.datasets.merge import merge


def make_corpus(path, split='train', subject='synthetic:a', count=3):
    stream = signal('pressure', count)
    stream.values = stream.values + sum(subject.encode()) / 1000.
    builder = Builder(path)
    group = dict(cols=list(range(count)), channels=[asdict(c) for c in stream.channels], modality='pressure')
    builder.put('synthetic', subject, subject, stream.values.numpy(), stream.times.numpy(), [group],
                split=split, dependency=0., evidence='synthetic', time_evidence='measured')
    return builder.save()


def test_structural_checks():
    result = run()
    assert result['groups_passed'] >= 9 and result['optimizer_steps'] == 0


def test_relative_manifest_and_backward(tmp_path):
    manifest = make_corpus(tmp_path / 'corpus')
    assert not Path(json.loads(manifest.read_text())['rows'][0]['path']).is_absolute()
    corpus = NativeCorpus(manifest)
    sample, _ = corpus.sample(random.Random(1))
    model = FoundationV13(ModelConfig(dim=32, layers=1))
    loss, _ = ObservationDenoising(32).compute(model, sample)
    loss.loss.backward()
    assert torch.isfinite(loss.loss)
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.temporal.parameters())


def test_split_guard(tmp_path):
    manifest = make_corpus(tmp_path / 'a')
    payload = json.loads(manifest.read_text())
    payload['rows'].append({**payload['rows'][0], 'split': 'test'})
    manifest.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match='cross-split'):
        NativeCorpus(manifest)


def test_writer_rejects_collision(tmp_path):
    manifest = make_corpus(tmp_path / 'a')
    row = json.loads(manifest.read_text())['rows'][0]
    stream = signal('pressure', 3)
    builder = Builder(manifest.parent)
    with pytest.raises(ValueError, match='collision'):
        builder.put('synthetic', 'synthetic:a', 'synthetic:a', stream.values.numpy()+1, stream.times.numpy(), row['groups'], split='train')


def test_merge_frozen_splits(tmp_path):
    a = make_corpus(tmp_path / 'a')
    b = make_corpus(tmp_path / 'b', 'validation', 'synthetic:b')
    out = merge([a, b], tmp_path / 'merged.json')
    assert len(NativeCorpus(out, 'train').rows) == 1
    assert len(NativeCorpus(out, 'validation').rows) == 1
    with pytest.raises(ValueError, match='duplicate'):
        merge([a, a], tmp_path / 'duplicate.json')


def test_native_dense_budget(tmp_path):
    manifest = make_corpus(tmp_path / 'dense', count=1094)
    corpus = NativeCorpus(manifest)
    sample = corpus.load(0, window_s=4., max_streams=0)
    assert len(sample.streams[0].channels) == 1094
    assert corpus.last_crop['reason'] == 'token_budget'


def test_denoising_checkpoint_safe_load(tmp_path):
    from argparse import Namespace
    corpus = NativeCorpus(make_corpus(tmp_path / 'a'))
    model = FoundationV13(ModelConfig(dim=32, layers=1))
    objective = ObservationDenoising(32)
    optimizer = torch.optim.AdamW(list(model.parameters()) + list(objective.parameters()))
    path = tmp_path / 'untrained.pt'
    checkpoint(path, model, objective, optimizer, corpus, 0, {}, Namespace(seed=13))
    payload = torch.load(path, weights_only=True)
    assert payload['step'] == 0 and not payload['optimizer']['state']
    assert payload['format'] == 'motion-v13-denoising-1'


def test_config_and_reference_frontend(tmp_path):
    config = tmp_path / 'model.json'
    config.write_text(json.dumps({'name':'v13','options':{'dim':32,'layers':1}}))
    model = build_model(config)
    stream = signal('pressure', 7, 51.2)
    frontend = model.frontends['scalar']
    origin = stream.times[0]
    a = frontend(stream, stream.observed, origin)
    b = frontend.forward_reference(stream, stream.observed, origin)
    assert a.routes == b.routes
    torch.testing.assert_close(a.features, b.features, atol=3e-6, rtol=3e-5)


def test_ingest_imports_without_machine_paths():
    for name in ('audit_inputs', 'build_corpus', 'remaining_ingest', 'extend_corpus'):
        importlib.import_module('motion_foundation.datasets.' + name)


def test_training_requires_explicit_flag(tmp_path):
    import os
    import subprocess
    import sys
    for module in ('train', 'train_mixed'):
        result = subprocess.run([sys.executable, '-m', 'motion_foundation.' + module,
                                 '--manifest', str(tmp_path/'missing.json'), '--out', str(tmp_path/'output')],
                                capture_output=True, text=True, env=os.environ.copy())
        assert result.returncode == 2 and 'training' in result.stderr.lower()
        assert not (tmp_path/'output').exists()


def test_hidden_values_do_not_change_encoding():
    model = FoundationV13(ModelConfig(dim=32,layers=1))
    stream = signal()
    missing = stream.observed.clone()
    missing[10:20] = False
    a = replace(stream, observed=missing)
    b = replace(a, values=torch.where(missing, stream.values, torch.full_like(stream.values, float('nan'))))
    torch.testing.assert_close(model(Sample([a])).pooled()['independent:a'], model(Sample([b])).pooled()['independent:a'])
