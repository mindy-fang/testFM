from dataclasses import asdict
import json
from pathlib import Path
from .contracts import Registry
from .model import FoundationV13, ModelConfig

MODELS = Registry()
MODELS.register('v13', lambda options: FoundationV13(ModelConfig(**options)))


def build_model(path=None):
    specification = json.loads(Path(path).read_text()) if path else {'name': 'v13', 'options': {}}
    if set(specification) != {'name', 'options'}:
        raise ValueError('model specification requires name and options only')
    return MODELS.get(specification['name'])(specification['options'])


def default_specification():
    return {'name': 'v13', 'options': asdict(ModelConfig())}
