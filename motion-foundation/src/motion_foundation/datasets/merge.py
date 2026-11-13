import argparse
import json
from pathlib import Path
from ..adapters import sha256


def merge(manifests, out):
    target = Path(out)
    if target.exists():
        raise FileExistsError(target)
    rows, provenance, keys = [], [], set()
    for filename in manifests:
        path = Path(filename).resolve()
        payload = json.loads(path.read_text())
        if payload.get('schema') != 'v13-native-corpus-1' or payload.get('errors'):
            raise ValueError('invalid or partial corpus: ' + str(path))
        provenance.append(dict(path=path.name, sha256=sha256(path)))
        for item in payload['rows']:
            row = dict(item)
            if row['split'] == 'quarantine':
                raise ValueError('quarantine is not part of the pretraining pool')
            key = (row['dataset'], row['source'])
            if key in keys:
                raise ValueError('duplicate source; select one frozen version: ' + str(key))
            keys.add(key)
            source = Path(row['path'])
            if not source.is_absolute():
                source = path.parent / source
            import os
            row['path'] = Path(os.path.relpath(source.resolve(), target.parent.resolve())).as_posix()
            rows.append(row)
    mappings = [{} for _ in range(4)]
    for row in rows:
        values = [row['subject'], (row['dataset'], row['source']), row['sha256'], row.get('source_sha256')]
        for value, mapping in zip(values, mappings):
            if not value:
                continue
            if value in mapping and mapping[value] != row['split']:
                raise ValueError('cross-split identity/source/content')
            mapping[value] = row['split']
    payload = dict(schema='v13-native-corpus-1', rows=rows, inputs=provenance, errors=[])
    for split in {r['split'] for r in rows}:
        if split not in ('train', 'validation', 'test'):
            raise ValueError('unsupported split')
    if not rows:
        raise ValueError('empty corpus')
    with target.open('x', encoding='utf-8') as handle:
        json.dump(payload, handle, indent=2)
    return target


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputs', nargs='+', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    print(merge(args.inputs, args.out))


if __name__ == '__main__':
    main()
