from __future__ import annotations

import argparse
from pathlib import Path
import random
import torch
from .adapters import ManifestDataset
from .model import FoundationV13, ModelConfig
from .objectives import MaskedReconstruction, TrainingEngine
from .runtime import save_package, load_package


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--steps', type=int, default=0)
    p.add_argument('--microbatch', type=int, default=1)
    p.add_argument('--window-s', type=float, default=2.)
    p.add_argument('--objective', choices=('temporal', 'axis'), default='temporal')
    p.add_argument('--device', default='cpu')
    p.add_argument('--seed', type=int, default=13)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--resume', type=Path)
    p.add_argument('--execute-training', action='store_true')
    args = p.parse_args()
    if not args.execute_training:
        p.error('training is disabled without --execute-training')
    if args.steps < 1 or args.microbatch < 1 or args.window_s <= 0 or args.lr <= 0:
        p.error('positive steps, microbatch, window and lr required')
    if not args.out.parent.is_dir() or args.out.exists():
        p.error('output must be a fresh file with an existing parent')
    random.seed(args.seed); torch.manual_seed(args.seed)
    dataset = ManifestDataset(args.manifest, split='train')
    if args.resume:
        model, objective, payload = load_package(args.resume, args.device, dataset.fingerprint)
        if payload['optimizer'] is None or payload['sampler_state'] is None:
            raise ValueError('checkpoint lacks training recovery state')
        state = payload['sampler_state']
        for key in ('microbatch', 'window_s', 'objective', 'lr'):
            if state['schedule'][key] != getattr(args, key):
                raise ValueError('resume schedule differs: ' + key)
        random.setstate(payload['python_rng']); torch.set_rng_state(payload['torch_rng'].cpu())
        if payload['cuda_rng']:
            torch.cuda.set_rng_state_all([x.cpu() for x in payload['cuda_rng']])
    else:
        model = FoundationV13(ModelConfig()).to(args.device)
        objective = MaskedReconstruction(model.config.dim, model.config.heads).to(args.device)
        state = dict(attempts=0, updates=0, schedule={k: getattr(args, k) for k in ('microbatch', 'window_s', 'objective', 'lr')})
    optimizer = torch.optim.AdamW(list(model.parameters()) + list(objective.parameters()), lr=args.lr)
    if args.resume:
        optimizer.load_state_dict(payload['optimizer'])
    engine = TrainingEngine(model, objective)
    model.train(); objective.train()
    for _ in range(args.steps):
        samples = []
        for _ in range(args.microbatch):
            sample = dataset.sample(random)
            if any(s.fusion_ref is not None for s in sample.streams):
                raise ValueError('CLI sampler requires independent streams; joint mapped crops need an explicit sampler')
            for s in sample.streams:
                lo, hi = float(s.times[0]), float(s.times[-1])
                start = random.uniform(lo, max(lo, hi - args.window_s))
                cropped = s.crop(start, start + args.window_s)
                s.values, s.observed, s.times = cropped.values.to(args.device), cropped.observed.to(args.device), cropped.times.to(args.device)
            samples.append(sample)
        result = engine.step(samples, optimizer, modes=[args.objective] * len(samples))
        state['attempts'] += 1; state['updates'] += int(result.loss is not None)
        print(dict(attempt=state['attempts'], updates=state['updates'], targets=result.valid_targets, loss=None if result.loss is None else float(result.loss.detach())), flush=True)
    save_package(args.out, model, objective, optimizer, dataset.fingerprint, state)


if __name__ == '__main__':
    main()
