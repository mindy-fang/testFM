from pathlib import Path
import bz2
import collections
import io
import json
import pickle
import subprocess
import tarfile
import zipfile
import numpy as np
import scipy.io
import scipy.sparse

import os
import shutil

ROOT = Path(os.environ.get('MOTION_LEGACY_ROOT', 'legacy_data'))
DATA = Path(os.environ.get('MOTION_DATA_ROOT', 'data'))


def unrar_executable():
    executable = os.environ.get('MOTION_UNRAR') or shutil.which('unrar')
    if not executable:
        raise RuntimeError('RAR input requires unrar on PATH or MOTION_UNRAR')
    return executable


class PassiveObject:
    pass


class ArrayUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        allowed = {
            ('numpy', 'ndarray'): np.ndarray,
            ('numpy', 'dtype'): np.dtype,
            ('numpy.core.multiarray', '_reconstruct'): np._core.multiarray._reconstruct,
            ('numpy._core.multiarray', '_reconstruct'): np._core.multiarray._reconstruct,
            ('numpy.core.multiarray', 'scalar'): np._core.multiarray.scalar,
            ('numpy._core.multiarray', 'scalar'): np._core.multiarray.scalar,
            ('numpy.core.numeric', '_frombuffer'): np._core.numeric._frombuffer,
            ('numpy._core.numeric', '_frombuffer'): np._core.numeric._frombuffer,
            ('scipy.sparse._csr', 'csr_matrix'): scipy.sparse.csr_matrix,
            ('scipy.sparse.csr', 'csr_matrix'): scipy.sparse.csr_matrix,
            ('footsim.classes', 'Stimulus'): PassiveObject,
            ('footsim.surface', 'Surface'): PassiveObject,
        }
        if (module, name) not in allowed:
            raise ValueError('unsupported serialized global: ' + module + '.' + name)
        return allowed[module, name]


def restricted_load(data):
    return ArrayUnpickler(io.BytesIO(data)).load()


def object_npz(data):
    result = {}
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        for name in z.namelist():
            f = io.BytesIO(z.read(name))
            version = np.lib.format.read_magic(f)
            header = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
            shape, order, dtype = header(f)
            if not dtype.hasobject:
                result[name[:-4]] = np.load(io.BytesIO(z.read(name)), allow_pickle=False)
                continue
            value = restricted_load(f.read())
            result[name[:-4]] = value.item() if isinstance(value, np.ndarray) and value.shape == () else value
    return result


def rar_list(path):
    p = subprocess.run([unrar_executable(), 'lb', str(path)], capture_output=True, check=True)
    return p.stdout.decode().splitlines()


def rar_read(path, member):
    return subprocess.run([unrar_executable(), 'p', '-inul', str(path), member], capture_output=True, check=True).stdout


def main():
    with zipfile.ZipFile(DATA / 'whip/dataset.zip') as z:
        member = next(n for n in z.namelist() if n.endswith('/walk.tar'))
        with tarfile.open(fileobj=io.BytesIO(z.read(member))) as t:
            d = object_npz(t.extractfile('insoles/readings.npz').read())
            for side, fields in d.items():
                print('WHIP', side, {k: (np.shape(v), str(getattr(v, 'dtype', '')), np.asarray(v).flatten()[:3].tolist()) for k, v in fields.items()})
        print('WHIP_DOC', [(n, z.read(n)[:2000].decode(errors='replace')) for n in z.namelist() if n.endswith('.txt')])
    p = next((DATA / 'tekscan_everyday/osfstorage/preprocessed_data').rglob('*.pbz2'))
    try:
        d = restricted_load(bz2.decompress(p.read_bytes()))
        print('TEK_KEYS', [(k, type(v).__name__, np.shape(v) if isinstance(v, np.ndarray) else list(v.__dict__) if isinstance(v, PassiveObject) else '') for k, v in d.items()])
        for k, v in d.items():
            if isinstance(v, PassiveObject):
                print('TEK_OBJECT', k, [(a, np.shape(b), str(b)[:100] if np.isscalar(b) else '') for a, b in v.__dict__.items()])
    except Exception as e:
        print('TEK_ERROR', repr(e))
    arc = DATA / "xsensor_locomotion/PARTICIPANT'S DATA CODE.rar"
    members = rar_list(arc)
    ppri = next(n for n in members if '/PPRI' in n and n.endswith('.zip')) if any('/PPRI' in n for n in members) else next(n for n in members if 'PLANTAR' in n and n.endswith('.zip'))
    with zipfile.ZipFile(io.BytesIO(rar_read(arc, ppri))) as z:
        print('XSENSOR_PPRI', z.namelist()[:8])
        for n in z.namelist():
            if n.lower().endswith(('.csv', '.txt')):
                with z.open(n) as f:
                    print('XSENSOR_HEAD', n, f.read(3000).decode(errors='replace'))
                break
    with zipfile.ZipFile(DATA / 'second_skin/OpenSource_Dataset.zip') as z:
        n = next(n for n in z.namelist() if '/EMG/' in n and n.endswith('.mat'))
        d = scipy.io.loadmat(io.BytesIO(z.read(n)), simplify_cells=True)
        print('EMG_MAT', n, [(k, type(v).__name__, np.shape(v), list(v)[:12] if isinstance(v, dict) else '') for k, v in d.items() if not k.startswith('_')])
    p = next((DATA / 'tekscan_offloading').rglob('*.fsx'))
    b = p.read_bytes()
    print('FSX_TEXT', b[:6000].decode(errors='replace')[:5000])


if __name__ == '__main__':
    main()
