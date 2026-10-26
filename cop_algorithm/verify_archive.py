"""Verify all six stored COP columns in the supplied archive; read only."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import zipfile
import numpy as np
from calculate_cop import compute_groups, load_geometry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', required=True)
    parser.add_argument('--report-dir', default='verification')
    args = parser.parse_args()
    archive_path = Path(args.archive)
    root = Path(__file__).resolve().parent
    report_dir = Path(args.report_dir)
    digest = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    checks = []; total_rows = 0; maximum = 0.0
    with zipfile.ZipFile(archive_path) as archive:
        # Bundled geometry is byte-identical to the source configuration.
        assert archive.read('sensor_pe.json') == (root/'sensor_pe.json').read_bytes()
        names = sorted(n for n in archive.namelist() if n.startswith('P') and n.endswith('.csv'))
        for index, name in enumerate(names, 1):
            reader = csv.reader(io.TextIOWrapper(archive.open(name), encoding='utf-8-sig'))
            header = next(reader); data = list(reader)
            geometry, _ = load_geometry(root/'sensor_pe.json', header)
            pressure = {}
            for group, sensors in geometry.items():
                ids = [header.index(s['pressure_column']) for s in sensors]
                pressure[group] = np.array([[float(row[i]) for i in ids] for row in data], dtype=float)
            result = compute_groups(pressure, geometry, shared_cop=True)
            for group, r in result.items():
                for j, axis in enumerate(['x', 'y']):
                    column = group+'_cop_'+axis
                    col = header.index(column)
                    expected = np.array([float(row[col]) if row[col] else np.nan for row in data])
                    actual = r['cop'][:, j]
                    missing_mismatches = int((np.isnan(expected) != np.isnan(actual)).sum())
                    finite = np.isfinite(expected) & np.isfinite(actual)
                    difference = float(np.max(np.abs(actual[finite]-expected[finite]))) if finite.any() else 0.
                    bad = int((np.abs(actual[finite]-expected[finite]) > 1e-12).sum())
                    assert missing_mismatches == 0 and bad == 0, (name, column, difference)
                    maximum = max(maximum, difference)
                    checks.append({'file': name, 'column': column, 'rows': len(data), 'finite_values': int(finite.sum()),
                                   'max_absolute_error': difference, 'value_mismatches_above_1e_12': bad,
                                   'missing_value_mismatches': missing_mismatches})
            total_rows += len(data)
            if index % 50 == 0: print('Verified CSVs:', index, flush=True)
    assert hashlib.sha256(archive_path.read_bytes()).hexdigest() == digest
    summary = {'archive_sha256': digest, 'csv_files': len(names), 'rows': total_rows, 'COP_columns': 6,
               'max_absolute_difference': maximum, 'tolerance': 1e-12, 'all_values_and_missing_patterns_match': True,
               'source_archive_unchanged': True, 'meaning': 'Arithmetic reproduction only, not force-plate accuracy.'}
    report_dir.mkdir(parents=True, exist_ok=True)
    with (report_dir/'archive_column_checks.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(checks[0])); writer.writeheader(); writer.writerows(checks)
    (report_dir/'archive_reproduction.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__': main()
