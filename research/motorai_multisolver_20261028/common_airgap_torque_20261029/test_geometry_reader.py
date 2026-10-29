"""Synthetic HDF-like metadata only. No h5py, files, fields, or models are read."""
import unittest
import numpy as np
from audit_dev32_geometry import inspect_case


class Names:
    def asstr(self):
        return np.array(['a1', 'a2'])


class Metadata(dict):
    def __init__(self, values):
        super().__init__(values)
        self.read_keys = []

    def __getitem__(self, key):
        if not (key.startswith('mesh/') or key.startswith('regions/')):
            raise AssertionError('attempted field access')
        self.read_keys.append(key)
        return super().__getitem__(key)


def fixture():
    c = np.sqrt(.5)
    xy = np.array([[100., 0.], [110., 0.], [100*c, 100*c], [110*c, 110*c],
                   [90., 0.], [90., 10.], [95., 0.]])
    tri = np.array([[10, 20, 30], [20, 40, 30], [50, 60, 70]])
    return Metadata({
        'mesh/node_id': np.array([10, 20, 30, 40, 50, 60, 70]),
        'mesh/tri_index': np.array([100, 200, 900]),
        **{f'mesh/node_{k+1}': tri[:, k] for k in range(3)},
        'mesh/reg_code': np.array([1, 1, 2]),
        'regions/reg_code': np.array([1, 2]), 'regions/name': Names(),
        'mesh/moving_reg_codes': np.array([2]),
        'mesh/moving_node_id': np.array([50, 60, 70]),
        'mesh/moving_node_indices': np.array([4, 5, 6]),
        'mesh/node_x_mm': xy[:, 0], 'mesh/node_y_mm': xy[:, 1],
        'mesh/node_x_mm_by_step_moving': np.tile(xy[4:, 0], (45, 1)),
        'mesh/node_y_mm_by_step_moving': np.tile(xy[4:, 1], (45, 1)),
    })


REC = {'case_index': 123, 'source_sha256': 'synthetic', 'stack_length_m': .15}


class GeometryReaderTests(unittest.TestCase):
    def test_metadata_allowlist_and_no_premature_release(self):
        h = fixture()
        ticks = []
        result = inspect_case(h, REC, lambda: ticks.append(1))
        self.assertEqual(len(ticks), 45)
        self.assertEqual(result['a1_native_element_ids'], [100, 200])
        self.assertEqual(result['operator']['symmetry_multiplier'], 8)
        self.assertEqual(result['field_array_reads'], 0)
        self.assertEqual(result['status'], 'GEOMETRY_CHECKED_BINDING_PENDING')
        self.assertEqual(result['dev_bundle_exact_ID_join'], 'REQUIRED_BEFORE_RELEASE')
        self.assertTrue(all(k.startswith(('mesh/', 'regions/')) for k in h.read_keys))

    def test_missing_motion_or_bad_native_mapping(self):
        h = fixture(); del h['mesh/moving_reg_codes']
        with self.assertRaises(KeyError):
            inspect_case(h, REC, lambda: None)
        h = fixture(); h['mesh/moving_node_id'] = np.array([70, 60, 50])
        with self.assertRaisesRegex(ValueError, 'bind'):
            inspect_case(h, REC, lambda: None)

    def test_missing_step_and_duplicate_regions(self):
        h = fixture(); h['mesh/node_x_mm_by_step_moving'] = h['mesh/node_x_mm_by_step_moving'][:44]
        with self.assertRaisesRegex(ValueError, '45'):
            inspect_case(h, REC, lambda: None)
        h = fixture(); h['regions/reg_code'] = np.array([1, 1])
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            inspect_case(h, REC, lambda: None)

    def test_stop_callback_interrupts(self):
        def stop():
            raise InterruptedError('synthetic STOP')
        with self.assertRaises(InterruptedError):
            inspect_case(fixture(), REC, stop)

    def test_stationary_interface_nodes_in_candidate_table(self):
        h = fixture()
        h['mesh/moving_node_id'] = np.array([20, 50, 60, 70])
        h['mesh/moving_node_indices'] = np.array([1, 4, 5, 6])
        h['mesh/node_x_mm_by_step_moving'] = np.tile(h['mesh/node_x_mm'][[1, 4, 5, 6]], (45, 1))
        h['mesh/node_y_mm_by_step_moving'] = np.tile(h['mesh/node_y_mm'][[1, 4, 5, 6]], (45, 1))
        row = inspect_case(h, REC, lambda: None)
        self.assertEqual(row['a1_candidate_moving_nodes'], 1)
        self.assertEqual(row['a1_verified_stationary_candidates'], 1)
        h['mesh/node_x_mm_by_step_moving'][44, 0] += 1e-9
        with self.assertRaises(ValueError):
            inspect_case(h, REC, lambda: None)


if __name__ == '__main__':
    unittest.main()
