"""Physical invariants, invalid-load behavior, and CSV column provenance."""
import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from calculate_cop import calculate_cop, compute_groups, load_geometry, process_csv


class COPTests(unittest.TestCase):
    def test_unequal_area_and_single_loaded_sensor(self):
        r = calculate_cop([[10, 20, 30], [0, 0, 50]], [[0, 0], [100, 0], [0, 100]], [1, 2, 1])
        np.testing.assert_allclose(r['cop'], [[50, 37.5], [0, 100]])
        np.testing.assert_allclose(r['weight_sum'], [80, 50])

    def test_coordinate_and_unit_invariance(self):
        p = np.array([[10., 20., 30.], [20., 12., 19.]])
        xy = np.array([[0., 0.], [100., 0.], [0., 100.]])
        a = np.array([1., 2., 1.])
        base = calculate_cop(p, xy, a)['cop']
        np.testing.assert_allclose(calculate_cop(p*1000, xy, a*7)['cop'], base)
        np.testing.assert_allclose(calculate_cop(p, xy+[7, -5], a)['cop'], base+[7, -5])
        np.testing.assert_allclose(calculate_cop(p, xy/1000, a)['cop'], base/1000)
        order = [2, 0, 1]
        np.testing.assert_allclose(calculate_cop(p[:, order], xy[order], a[order])['cop'], base)

    def test_zero_missing_negative_and_load_threshold(self):
        p = [[0, 0], [np.nan, 5], [3, -1], [np.inf, 4], [2, 2], [3, 2]]
        r = calculate_cop(p, [[0, 0], [10, 10]], [1, 1], min_weight=4)
        self.assertEqual(r['valid'].tolist(), [False, False, False, False, False, True])
        self.assertTrue(np.isnan(r['cop'][:5]).all())
        np.testing.assert_allclose(r['cop'][-1], [4, 4])

    def test_geometry_errors(self):
        for area in [[0, 1], [np.nan, 1], [-1, 2]]:
            with self.assertRaises(ValueError): calculate_cop([[1, 1]], [[0, 0], [1, 1]], area)
        with self.assertRaises(ValueError): calculate_cop([[1, 2]], [[0, 0]], [1, 1])
        with self.assertRaises(ValueError): calculate_cop([[1]], [[np.inf, 0]], [1])
        with self.assertRaises(ValueError): calculate_cop([[1]], [[0, 0]], [1], min_weight=-1)

    def test_shared_frame_and_missing_foot(self):
        geometry = {
            'left': [dict(pressure_column='p1', x=0, y=1, area=1, x_shared=-2, y_shared=1)],
            'right': [dict(pressure_column='p2', x=0, y=1, area=1, x_shared=2, y_shared=1)]}
        r = compute_groups({'left': [[2], [2], [np.nan]], 'right': [[0], [2], [3]]}, geometry, shared_cop=True)
        np.testing.assert_allclose(r['global']['cop'][:2], [[-2, 1], [0, 1]])
        self.assertFalse(r['right']['valid'][0])
        self.assertTrue(r['right']['valid'][2])
        self.assertFalse(r['global']['valid'][2])
        incomplete = {'left': [dict(pressure_column='p1', x=0, y=0, area=1)]}
        with self.assertRaises(ValueError): compute_groups({'left': [[1]]}, incomplete, shared_cop=True)

    def test_csv_uses_names_preserves_native_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            source = folder/'pressure.csv'; geo = folder/'geometry.csv'; out = folder/'result.csv'
            source.write_text('time,p2,p1,left center of pressure X[%]\n0.0100,20,10,0.12345678901234567\n', encoding='utf-8')
            geo.write_text('group,pressure_column,x,y,area\nleft,p1,0,0,1\nleft,p2,100,0,2\n', encoding='utf-8')
            before = source.read_bytes()
            result = process_csv(source, geo, out, diagnostics=True)
            with out.open(encoding='utf-8-sig', newline='') as stream: row = next(csv.DictReader(stream))
            self.assertEqual(row['time'], '0.0100')
            self.assertEqual(row['left center of pressure X[%]'], '0.12345678901234567')
            self.assertEqual(float(row['left_cop_x']), 80)
            self.assertEqual(row['left_cop_valid'], '1')
            self.assertEqual(source.read_bytes(), before)
            self.assertEqual(result['groups']['left']['valid_rows'], 1)
            with self.assertRaises(ValueError): process_csv(source, geo, source)
            with self.assertRaises(ValueError): process_csv(source, geo, out)

    def test_explicit_recompute_removes_stale_shared_output(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory); source = folder/'pressure.csv'; geo = folder/'geometry.csv'
            source.write_text('p,left_cop_x,left_cop_y,global_cop_x,global_cop_y\n1,999,999,999,999\n', encoding='utf-8')
            geo.write_text('group,pressure_column,x,y,area\nleft,p,2,3,1\n', encoding='utf-8')
            out = folder/'out.csv'
            with self.assertRaises(ValueError): process_csv(source, geo, out)
            process_csv(source, geo, out, replace_cop=True)
            with out.open(encoding='utf-8-sig') as stream: row = next(csv.DictReader(stream))
            self.assertNotIn('global_cop_x', row)
            self.assertEqual(float(row['left_cop_x']), 2)

    def test_ambiguous_sensor_mapping_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            geo = Path(directory)/'geometry.csv'
            geo.write_text('group,pressure_column,x,y,area\nleft,p,0,0,1\nright,p,1,0,1\n', encoding='utf-8')
            with self.assertRaises(ValueError): load_geometry(geo, ['p'])
            meta = Path(directory)/'sensor.json'
            meta.write_text(json.dumps({'left': {'pressure_1': {'x_own': 0, 'y_own': 1, 'area': 1}}}), encoding='utf-8')
            with self.assertRaises(ValueError): load_geometry(meta, ['left pressure 1[N/cm²]', 'left_pressure_1'])

    def test_real_example_json_and_csv_formats(self):
        with (ROOT/'examples/grouvel_expected_cop.csv').open(encoding='utf-8-sig') as stream:
            expected_rows = list(csv.DictReader(stream))
        targets = [side+'_cop_'+axis for side in ['left', 'right', 'global'] for axis in ['x', 'y']]
        expected = np.array([[float(r[c]) if r[c] else np.nan for c in targets] for r in expected_rows])
        with tempfile.TemporaryDirectory() as directory:
            for geometry_name in ['sensor_pe.json', 'sensor_coordinates_grouvel.csv']:
                output = Path(directory)/(geometry_name+'.out.csv')
                process_csv(ROOT/'examples/grouvel_pressure.csv', ROOT/geometry_name, output, shared_cop=True)
                with output.open(encoding='utf-8-sig') as stream: rows = list(csv.DictReader(stream))
                actual = np.array([[float(r[c]) if r[c] else np.nan for c in targets] for r in rows])
                np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=0, equal_nan=True)


if __name__ == '__main__': unittest.main(verbosity=2)
