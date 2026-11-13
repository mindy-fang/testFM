from motion_foundation import FoundationV13, ModelConfig
from motion_foundation.registry import MODELS
from motion_foundation.selfcheck import signal
from motion_foundation import Sample


def make_small(options):
    return FoundationV13(ModelConfig(**{'dim': 64, 'layers': 1, **options}))


MODELS.register('v13_small', make_small)


if __name__ == '__main__':
    model = MODELS.get('v13_small')({})
    representation = model.encode(Sample([signal()]))
    print({name: tuple(value.shape) for name, value in representation.fused.items()})
