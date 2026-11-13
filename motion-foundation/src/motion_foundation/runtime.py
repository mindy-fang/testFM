from __future__ import annotations

import random
from pathlib import Path
import torch
from .contracts import SCHEMA
from .model import FoundationV13, ModelConfig
from .objectives import MaskedReconstruction

PACKAGE = 'motion-foundation-v13.1'


def save_package(path, model, objective=None, optimizer=None, data_fingerprint='', sampler_state=None):
    path = Path(path)
    if path.exists():
        raise FileExistsError('refusing to overwrite model package')
    if set(model.frontend_registry.items) != {'pressure', 'imu'}:
        raise ValueError('custom frontend requires a versioned package loader')
    payload = dict(format=PACKAGE, schema=SCHEMA, config=model.config_dict(), model=model.state_dict(),
                   objective=objective.state_dict() if objective is not None else None,
                   optimizer=optimizer.state_dict() if optimizer is not None else None,
                   torch_rng=torch.get_rng_state(), python_rng=random.getstate(),
                   cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else [],
                   data_fingerprint=data_fingerprint, sampler_state=sampler_state,
                   plugins=['pressure', 'imu', 'temporal', 'axis'],
                   limitations=['bidirectional', 'video frontend not implemented', 'no trained V13 weights', 'unverified temporal dependencies disable temporal targets'])
    torch.save(payload, path)


def load_package(path, device='cpu', expected_fingerprint=None):
    payload = torch.load(path, map_location=device, weights_only=True)
    if payload.get('format') != PACKAGE or payload.get('schema') != SCHEMA:
        raise ValueError('incompatible model package')
    if payload.get('plugins') != ['pressure', 'imu', 'temporal', 'axis']:
        raise ValueError('plugin contract mismatch')
    if expected_fingerprint is not None and payload['data_fingerprint'] != expected_fingerprint:
        raise ValueError('data fingerprint mismatch')
    model = FoundationV13(ModelConfig(**payload['config'])).to(device)
    model.load_state_dict(payload['model'], strict=True)
    objective = MaskedReconstruction(model.config.dim, model.config.heads).to(device)
    if payload['objective'] is not None:
        objective.load_state_dict(payload['objective'], strict=True)
    return model, objective, payload


def transfer_parameters(old_state, model, approved_mapping):
    own = model.state_dict()
    copied, used = [], set()
    for old_name, specification in approved_mapping.items():
        new_name, semantic_evidence = specification
        if not semantic_evidence or old_name not in old_state or new_name not in own:
            raise ValueError('explicit semantic evidence and parameter names required')
        if old_state[old_name].shape != own[new_name].shape:
            raise ValueError('incompatible parameter shape')
        own[new_name] = old_state[old_name].detach().clone()
        copied.append(new_name); used.add(old_name)
    model.load_state_dict(own, strict=True)
    return dict(copied=copied, new_initialization=[k for k in own if k not in copied], unused=[k for k in old_state if k not in used])
