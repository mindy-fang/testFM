#!/usr/bin/env python3
"""Pressure/area weighted COP; no force-plate or native Moticon COP input.

CLI: python calculate_cop.py --input pressure.csv --geometry geometry.csv --output cop.csv
API: result = calculate_cop(pressure, xy, area)
Coordinates and COP have the same units and reference frame.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np


def calculate_cop(pressure, xy, area, min_weight=0.0):
    """Return cop (frames, 2), weight_sum (frames,), valid (frames,).

    pressure: (frames, sensors), nonnegative pressure readings.
    xy: (sensors, 2), sensor-center coordinates in one common frame.
    area: (sensors,), positive effective areas or proportional area weights.
    min_weight: require sum(pressure * area) > this threshold, default 0.
    This threshold is NOT necessarily in newtons (depends on input units).

    A missing/nonfinite/negative pressure invalidates its whole frame in this
    sensor group. No clipping, imputation, extrapolation, or filtering occurs.
    All-zero pressure gives weight_sum=0, valid=False, and cop=[nan, nan].
    """
    p = np.asarray(pressure, dtype=np.float64)
    coordinates = np.asarray(xy, dtype=np.float64)
    areas = np.asarray(area, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] == 0:
        raise ValueError('pressure must have shape (frames, sensors), with at least one sensor')
    if coordinates.shape != (p.shape[1], 2) or areas.shape != (p.shape[1],):
        raise ValueError('xy must have shape (sensors, 2); area must have shape (sensors,)')
    if not np.isfinite(coordinates).all():
        raise ValueError('All sensor coordinates must be finite')
    if not np.isfinite(areas).all() or np.any(areas <= 0):
        raise ValueError('All sensor areas must be finite and positive')
    if not np.isfinite(min_weight) or min_weight < 0:
        raise ValueError('min_weight must be finite and nonnegative')
    input_valid = np.isfinite(p).all(axis=1) & (p >= 0).all(axis=1)
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        weighted = p * areas[None, :]
        total = weighted.sum(axis=1)
        numerator = weighted @ coordinates
        valid = input_valid & np.isfinite(total) & (total > min_weight) & np.isfinite(numerator).all(axis=1)
        cop = np.full((len(p), 2), np.nan)
        np.divide(numerator, total[:, None], out=cop, where=valid[:, None])
    valid &= np.isfinite(cop).all(axis=1)
    cop[~valid] = np.nan
    total[~input_valid | ~np.isfinite(total)] = np.nan
    return {'cop': cop, 'weight_sum': total, 'valid': valid}


def _finite_number(value, description):
    try:
        number = float(value)
    except (ValueError, TypeError):
        raise ValueError(description + ' must be numeric')
    if not np.isfinite(number):
        raise ValueError(description + ' must be finite')
    return number


def load_geometry(path, pressure_columns):
    """Load generic mapping CSV or the original Grouvel sensor_pe.json."""
    path = Path(path)
    if path.suffix.lower() == '.json':
        meta = json.loads(path.read_text(encoding='utf-8-sig'))
        rows = []
        for side in ['left', 'right']:
            if side not in meta:
                continue
            sensors = [key for key in meta[side] if re.fullmatch(r'pressure_\d+', key)]
            for key in sorted(sensors, key=lambda key: int(key.split('_')[-1])):
                sensor = meta[side][key]
                number = int(key.split('_')[-1])
                # Exact aliases only; original Moticon COP fields cannot match.
                aliases = [f'{side} pressure {number}[N/cm²]', f'{side}_pressure_{number}']
                present = [col for col in aliases if col in pressure_columns]
                if len(present) != 1:
                    raise ValueError(f'{side}/{key}: expected exactly one pressure column from {aliases}; found {present}. Use a geometry CSV to specify custom names.')
                row = {'group': side, 'pressure_column': present[0], 'x': sensor['x_own'],
                       'y': sensor['y_own'], 'area': sensor['area']}
                if 'x_shared' in sensor and 'y_shared' in sensor:
                    row.update(x_shared=sensor['x_shared'], y_shared=sensor['y_shared'])
                rows.append(row)
        source_info = {'format': 'grouvel_sensor_pe_json', 'original_metadata': meta.get('_meta', {})}
    elif path.suffix.lower() == '.csv':
        with path.open(encoding='utf-8-sig', newline='') as stream:
            reader = csv.DictReader(stream)
            required = {'group', 'pressure_column', 'x', 'y', 'area'}
            if not required.issubset(reader.fieldnames or []):
                raise ValueError('Geometry CSV needs columns: group,pressure_column,x,y,area')
            if len(reader.fieldnames) != len(set(reader.fieldnames)):
                raise ValueError('Geometry CSV contains duplicate header names')
            rows = list(reader)
        source_info = {'format': 'geometry_csv', 'units': 'Same as supplied x/y; area is used exactly as supplied.'}
    else:
        raise ValueError('Geometry must be a .csv file or the original sensor_pe.json')
    if not rows:
        raise ValueError('No sensor geometry rows found')
    groups = {}
    used = set()
    for index, row in enumerate(rows, 1):
        group = str(row.get('group', '')).strip()
        column = row.get('pressure_column')
        if not re.fullmatch(r'[A-Za-z][A-Za-z0-9_]*', group) or group == 'global':
            raise ValueError(f'Geometry row {index}: group must be an ASCII identifier other than global')
        if column not in pressure_columns:
            raise ValueError(f'Geometry row {index}: missing pressure column {column!r}')
        if column in used:
            raise ValueError(f'Pressure column {column!r} is assigned more than once')
        used.add(column)
        sensor = {'pressure_column': column}
        for key in ['x', 'y', 'area']:
            sensor[key] = _finite_number(row.get(key), f'Geometry row {index}, {key}')
        if sensor['area'] <= 0:
            raise ValueError(f'Geometry row {index}: area must be positive')
        has_x = row.get('x_shared') not in (None, '')
        has_y = row.get('y_shared') not in (None, '')
        if has_x != has_y:
            raise ValueError(f'Geometry row {index}: provide both x_shared and y_shared, or neither')
        if has_x:
            sensor['x_shared'] = _finite_number(row['x_shared'], 'x_shared')
            sensor['y_shared'] = _finite_number(row['y_shared'], 'y_shared')
        groups.setdefault(group, []).append(sensor)
    return groups, source_info


def compute_groups(pressure_by_group, geometry, min_weight=0.0, shared_cop=False):
    """Compute each group's local COP, optionally all sensors in a shared frame.

    pressure_by_group[group] must follow geometry[group]'s sensor order.
    Local group coordinates are never combined as if they were shared.
    """
    results = {}
    shared_pressure = []; shared_xy = []; shared_area = []
    frame_count = None
    if set(pressure_by_group) != set(geometry):
        raise ValueError('Pressure groups must match geometry groups exactly')
    for group, sensors in geometry.items():
        p = np.asarray(pressure_by_group[group], dtype=np.float64)
        if p.ndim != 2:
            raise ValueError('Each group pressure array must be two-dimensional')
        if frame_count is not None and len(p) != frame_count:
            raise ValueError('All groups must have the same frame count and timestamps')
        frame_count = len(p)
        xy = np.array([[s['x'], s['y']] for s in sensors])
        area = np.array([s['area'] for s in sensors])
        results[group] = calculate_cop(p, xy, area, min_weight)
        if shared_cop:
            if any('x_shared' not in s or 'y_shared' not in s for s in sensors):
                raise ValueError('--shared-cop requires x_shared,y_shared for every sensor')
            shared_pressure.append(p)
            shared_xy.extend([[s['x_shared'], s['y_shared']] for s in sensors])
            shared_area.extend(area)
    if shared_cop:
        results['global'] = calculate_cop(np.concatenate(shared_pressure, axis=1), shared_xy, shared_area, min_weight)
    return results


def _hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def process_csv(input_path, geometry_path, output_path, min_weight=0.0,
                shared_cop=False, diagnostics=False, replace_cop=False):
    """Append COP to a new CSV; preserve original field strings and row order."""
    input_path = Path(input_path).resolve()
    geometry_path = Path(geometry_path).resolve()
    output_path = Path(output_path).resolve()
    metadata_path = output_path.with_suffix(output_path.suffix+'.meta.json')
    if output_path in [input_path, geometry_path] or metadata_path in [input_path, geometry_path]:
        raise ValueError('Output must differ from all input files')
    if output_path.exists() or metadata_path.exists():
        raise ValueError('Output or metadata already exists; choose a new output name')
    with input_path.open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.reader(stream)
        try:
            header = next(reader)
        except StopIteration:
            raise ValueError('Input CSV is empty')
        if len(header) != len(set(header)):
            raise ValueError('Input CSV contains duplicate header names')
        data = []
        for index, row in enumerate(reader, 2):
            if not row:  # A physically empty line contains no sample.
                continue
            if len(row) != len(header):
                raise ValueError(f'Input CSV record {index}: column count does not match header')
            data.append(row)
    geometry, geometry_info = load_geometry(geometry_path, header)
    matrices = {}; conversion_failures = 0
    for group, sensors in geometry.items():
        indices = [header.index(s['pressure_column']) for s in sensors]
        values = np.empty((len(data), len(indices)))
        for i, row in enumerate(data):
            for j, col in enumerate(indices):
                try:
                    values[i, j] = float(row[col])
                except ValueError:
                    values[i, j] = np.nan
                    conversion_failures += 1
        matrices[group] = values
    result = compute_groups(matrices, geometry, min_weight, shared_cop)
    names = [group+'_cop_'+axis for group in result for axis in ['x', 'y']]
    if diagnostics:
        names += [group+suffix for group in result for suffix in ['_cop_valid', '_load_weight']]
    if len(names) != len(set(names)):
        raise ValueError('Geometry group names create duplicate output fields')
    owned_groups = list(geometry)+['global']
    owned_fields = {group+suffix for group in owned_groups for suffix in ['_cop_x', '_cop_y', '_cop_valid', '_load_weight']}
    collisions = (owned_fields if replace_cop else set(names)) & set(header)
    if collisions and not replace_cop:
        raise ValueError(f'COP output fields already exist: {sorted(collisions)}. Use --replace-cop to recalculate them in a NEW output file.')
    if any(s['pressure_column'] in set(names) | collisions for sensors in geometry.values() for s in sensors):
        raise ValueError('An output field would replace an input pressure field')
    keep = [i for i, col in enumerate(header) if col not in collisions]
    metadata = {'algorithm': 'pressure-area-weighted-centroid-v1',
                'formula': 'COP = sum(pressure_i * area_i * coordinate_i) / sum(pressure_i * area_i)',
                'input_path': str(input_path), 'input_sha256': _hash(input_path),
                'geometry_path': str(geometry_path), 'geometry_sha256': _hash(geometry_path),
                'geometry_info': geometry_info, 'rows': len(data), 'min_weight': min_weight,
                'shared_cop': bool(shared_cop), 'removed_previous_computed_fields': sorted(collisions),
                'non_numeric_or_blank_pressure_cells': conversion_failures,
                'output_coordinate_units': 'Exactly the units/frame of supplied coordinates; no automatic mm conversion.',
                'load_weight_units': 'pressure units * area units; not necessarily N.',
                'invalid_policy': 'Any negative/missing/nonfinite sensor invalidates its group for that frame. Insufficient total load gives blank COP. Shared COP requires all sensors valid.',
                'shared_frame_note': 'Requires compatible common-frame coordinates and comparable cross-group area scales. Original Grouvel shared coordinates are a fixed illustrative layout, not dynamic laboratory COP.',
                'groups': {}}
    for group, r in result.items():
        info = {'valid_rows': int(r['valid'].sum()), 'invalid_rows': int((~r['valid']).sum())}
        if group in geometry:
            p = matrices[group]
            info.update(sensors=len(geometry[group]), pressure_columns=[s['pressure_column'] for s in geometry[group]],
                        nonfinite_rows=int((~np.isfinite(p).all(1)).sum()), negative_rows=int((p < 0).any(1).sum()))
        metadata['groups'][group] = info
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation prevents overwriting inputs or a previous result.
    with output_path.open('x', encoding='utf-8-sig', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow([header[i] for i in keep]+names)
        for i, row in enumerate(data):
            values = []
            for r in result.values():
                values.extend(format(x, '.15g') if np.isfinite(x) else '' for x in r['cop'][i])
            if diagnostics:
                for r in result.values():
                    total = r['weight_sum'][i]
                    values.extend([str(int(r['valid'][i])), format(total, '.15g') if np.isfinite(total) else ''])
            writer.writerow([row[j] for j in keep]+values)
    metadata['output_sha256'] = _hash(output_path)
    with metadata_path.open('x', encoding='utf-8') as stream:
        json.dump(metadata, stream, ensure_ascii=False, indent=2, allow_nan=False)
    return metadata


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--input', required=True, help='Pressure CSV, UTF-8, one row per sample')
    parser.add_argument('--geometry', required=True, help='Geometry CSV or original Grouvel sensor_pe.json')
    parser.add_argument('--output', required=True, help='New output CSV path; input is never overwritten')
    parser.add_argument('--min-weight', type=float, default=0.0, help='Require sum(pressure*area) greater than this value; default 0')
    parser.add_argument('--shared-cop', action='store_true', help='Also output global_cop_x/y using provided shared coordinates')
    parser.add_argument('--diagnostics', action='store_true', help='Append each group cop_valid and load_weight')
    parser.add_argument('--replace-cop', action='store_true', help='Replace existing calculated COP fields in the NEW output copy')
    args = parser.parse_args(argv)
    try:
        result = process_csv(args.input, args.geometry, args.output, args.min_weight,
                             args.shared_cop, args.diagnostics, args.replace_cop)
    except (OSError, ValueError, KeyError, csv.Error) as error:
        parser.exit(2, 'Error: '+str(error)+'\n')
    print(json.dumps({'output': str(Path(args.output).resolve()), 'rows': result['rows'],
                      'groups': result['groups']}, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    sys.exit(main())
