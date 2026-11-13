import argparse
import bz2
import hashlib
import json
import multiprocessing
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import numpy as np
from ..adapters import sha256
from .audit_inputs import DATA, restricted_load
from .build_corpus import Builder, pressure


def worker_init():
    if hasattr(os, 'nice'):
        os.nice(5)
    from threadpoolctl import threadpool_limits
    threadpool_limits(limits=1)


def convert_one(job):
    filename, out = job
    p = Path(filename)
    start = time.monotonic()
    blob = p.read_bytes()
    read_s = time.monotonic() - start
    decoded = restricted_load(bz2.decompress(blob))
    decode_s = time.monotonic() - start - read_s
    raw = np.asarray(decoded['Raw data'])
    if raw.ndim != 3:
        raise ValueError('Tekscan raw cube')
    fs = float(decoded['Stimulus'].fs)
    if fs != 100:
        raise ValueError('Tekscan declared 100Hz mismatch')
    x = raw.reshape(-1, raw.shape[-1]).T
    rel = p.relative_to(DATA / 'tekscan_everyday/osfstorage/preprocessed_data')
    subject = 'tekscan_everyday:' + rel.parts[0]
    side = rel.parts[-2] if rel.parts[-2] in ('left', 'right') else 'unknown'
    builder = Builder(out)
    builder.put('tekscan_everyday', subject, str(rel), x, np.arange(len(x)) / fs,
                [dict(cols=list(range(x.shape[1])), channels=pressure(x.shape[1], side), modality='pressure')],
                time_evidence='verified_relative_grid',
                evidence='published methods DOI10.1098/rsif.2023.0052 kPa and100Hz; stored Raw data cube, no phase-normalized steps; preprocessing dependency unknown',
                source_sha=hashlib.sha256(blob).hexdigest())
    return builder.rows, dict(read_s=read_s, decode_s=decode_s, total_s=time.monotonic()-start)


def parallel_convert(paths, out, workers, builder=None):
    start = time.monotonic()
    rows, timings, failures = [], [], []
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn'), initializer=worker_init) as pool:
        pending = {pool.submit(convert_one, (str(p), str(out))): p for p in paths}
        for i, future in enumerate(as_completed(pending), 1):
            try:
                result, timing = future.result()
                rows.extend(result)
                timings.append(timing)
                if builder is not None:
                    builder.rows.extend(result)
            except Exception as exc:
                failure = dict(source=str(pending[future]), error=repr(exc))
                failures.append(failure)
                if builder is not None:
                    builder.errors.append(failure)
            if builder is not None and (i % 10 == 0 or i == len(paths)):
                builder.save()
                print('TEKSCAN_PARALLEL', json.dumps(dict(completed=i, pending_total=len(paths), records=len(builder.rows), errors=len(failures), elapsed_s=round(time.monotonic()-start, 2))), flush=True)
    return dict(workers=workers, files=len(paths), elapsed_s=time.monotonic()-start, stages=timings, failures=failures, rows=rows)


def benchmark(paths, out):
    out = Path(out)
    out.mkdir(exist_ok=True)
    ordered = sorted(paths, key=lambda p: p.stat().st_size)
    selected = [ordered[i] for i in np.linspace(len(ordered)*.15, len(ordered)*.85, 16, dtype=int)]
    reference = None
    for workers in [1, 4, 8]:
        target = out / ('workers_' + str(workers))
        if target.exists():
            raise FileExistsError(target)
        target.mkdir()
        result = parallel_convert(selected, target, workers)
        fingerprint = {r['source']: r['sha256'] for r in result.pop('rows')}
        if result['failures']:
            raise ValueError(result['failures'])
        if reference is not None and fingerprint != reference:
            raise ValueError('parallel output differs from serial output')
        reference = fingerprint
        print('BENCHMARK', json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--resume-manifest')
    parser.add_argument('--benchmark', action='store_true')
    args = parser.parse_args()
    paths = sorted((DATA/'tekscan_everyday/osfstorage/preprocessed_data').rglob('*.pbz2'))
    if not 1 <= args.workers <= 16:
        parser.error('worker count must be between 1 and 16')
    if args.benchmark:
        benchmark(paths, args.out)
        return
    if Path(args.out).exists():
        raise FileExistsError(args.out)
    builder = Builder(args.out)
    done = set()
    if args.resume_manifest:
        previous = json.loads(Path(args.resume_manifest).read_text())
        for row in previous['rows']:
            path = Path(row['path'])
            if not path.is_absolute():
                path = Path(args.resume_manifest).parent / path
            row['path'] = str(path.resolve())
            if sha256(row['path']) != row['sha256']:
                raise ValueError('resume cache fingerprint mismatch: ' + row['source'])
            builder.rows.append(row)
            done.add(row['source'])
    paths = [p for p in paths if str(p.relative_to(DATA/'tekscan_everyday/osfstorage/preprocessed_data')) not in done]
    builder.save()
    print('RESUME', len(done), 'REMAINING', len(paths), 'WORKERS', args.workers, flush=True)
    result = parallel_convert(paths, args.out, args.workers, builder)
    builder.audit.append(dict(dataset='tekscan_everyday', records=len(builder.rows), workers=args.workers, reused=len(done), elapsed_s=result['elapsed_s']))
    builder.save()
    if result['failures']:
        raise RuntimeError('conversion incomplete; consult manifest errors')
    print('DONE', len(builder.rows), 'elapsed_s', result['elapsed_s'], flush=True)


if __name__ == '__main__':
    main()
