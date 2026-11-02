"""Output contracts and threshold boundaries for the selected-filter workflow."""
from pathlib import Path
import tempfile
import unittest

from hal_discovery.mine import (
    FilterParameters, SELECTED_CSV_COLUMNS, _selected_csv_row,
    _threshold_criterion, find_structure_paths, required_atom_names,
)


class SelectedFilterOutputTests(unittest.TestCase):
    def test_threshold_boundaries_and_missing_measurement(self):
        for comparison, expected in [('<', 'fail'), ('>', 'fail'), ('>=', 'pass')]:
            with self.subTest(comparison=comparison):
                criterion = _threshold_criterion(
                    'score', 'potential_cu_2his', .8, comparison, .8, 'Boundary', 'score')
                self.assertEqual(criterion['status'], expected)
                self.assertEqual(criterion['measurement'], 'score')
        missing = _threshold_criterion('distance', 'his_ne2', None, '<', 4., 'Missing', 'distance')
        self.assertEqual(missing['status'], 'not_evaluable')
        self.assertIsNone(missing['measurement'])

    def test_csv_keeps_distinct_donor_and_copper_identities_in_their_columns(self):
        record = {
            'site_id': 'site-1', 'source': {'structure_id': 'example'},
            'atoms': {
                'his1_donor': dict(model_number=2, auth_chain_id='A', auth_seq_id=10,
                                   insertion_code='B', atom_name='ND1'),
                'his2_donor': dict(auth_seq_id=23, insertion_code='C', atom_name='NE2'),
                'copper': dict(auth_chain_id='X', auth_seq_id=901, label_asym_id='CU1'),
            },
            'measurements': {'s1': {'value': .9}, 's2': {'value': .8},
                             'cu_his1': {'value': 2.1}, 'cu_his2': {'value': 2.2}},
            'classifications': {},
        }
        row = _selected_csv_row(record, FilterParameters(('potential_cu_2his',)))
        self.assertEqual(list(row), SELECTED_CSV_COLUMNS)
        self.assertEqual([row[k] for k in ('his1', 'his1_insertion_code', 'donor1')], [10, 'B', 'ND1'])
        self.assertEqual([row[k] for k in ('his2', 'his2_insertion_code', 'donor2')], [23, 'C', 'NE2'])
        self.assertEqual([row[k] for k in ('cu_auth_chain', 'cu_auth_number', 'cu_label_chain')], ['X', 901, 'CU1'])
        self.assertEqual([row[k] for k in ('s1', 's2', 'cu_his1_angstrom', 'cu_his2_angstrom')], [.9, .8, 2.1, 2.2])
        self.assertIsNone(row['donor_angle_degree'])

    def test_structure_discovery_sorts_limits_and_ignores_other_files(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve()
            for name in ('b.cif', 'a.PDB', 'c.mmcif', 'notes.txt'):
                (root / name).touch()
            (root / 'nested.pdb').mkdir()
            self.assertEqual([p.name for p in find_structure_paths(root)], ['a.PDB', 'b.cif', 'c.mmcif'])
            self.assertEqual(find_structure_paths(root, 1), [root / 'a.PDB'])
            self.assertEqual(find_structure_paths(root / 'b.cif'), [root / 'b.cif'])
            with self.assertRaisesRegex(ValueError, 'does not exist'):
                find_structure_paths(root / 'missing')

    def test_required_atoms_follow_selected_measurements(self):
        self.assertEqual(required_atom_names(FilterParameters(('observed_cu_2his',))), ())
        names = required_atom_names(FilterParameters(('his_backbone', 'potential_cu_2his')))
        self.assertEqual(set(names), {'N', 'O', 'CG', 'ND1', 'CE1', 'NE2', 'CD2'})
