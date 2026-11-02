"""Reference recovery and synthetic software controls, not biological mutants."""
import argparse
import contextlib
import copy
from dataclasses import asdict
import io
import json
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
from Bio.PDB import Atom, Chain, Model, PDBIO, Residue, Structure
from Bio.PDB.Atom import DisorderedAtom

from hal_discovery.mine import CopperParameters, FilterParameters, analyze_copper, run_mine
from hal_discovery.pymol_export import export_bundle, validate_record
from hal_discovery.results import sha256
from hal_discovery.structure_io import load_structure

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / 'tests/fixtures/copper'


def add_atom(residue, name, xyz, element, alt=' ', occupancy=1.0):
    atom = Atom.Atom(name, np.array(xyz, dtype=float), 20., occupancy, alt,
                     name.rjust(4), 1, element=element)
    if alt.strip():
        if name not in residue:
            residue.add(DisorderedAtom(name))
        residue[name].disordered_add(atom)
    else:
        residue.add(atom)
    return atom


def add_residue(chain, name, number, insertion=' ', hetero=' '):
    residue = Residue.Residue((hetero, number, insertion), name, ' ')
    chain.add(residue)
    return residue


def synthetic_structure():
    """Two CH units with nonconsecutive author numbers and a separate Cu chain."""
    structure = Structure.Structure('synthetic')
    model = Model.Model(0)
    structure.add(model)
    chain = Chain.Chain('P')
    model.add(chain)
    cys1 = add_residue(chain, 'CYS', 7)
    add_atom(cys1, 'C', (3.67, 0, 0), 'C')
    add_atom(cys1, 'SG', (0, 0, 3), 'S')
    his1 = add_residue(chain, 'HIS', 10)
    add_atom(his1, 'N', (5, 0, 0), 'N')
    add_atom(his1, 'NE2', (2, 0, 0), 'N')
    add_atom(his1, 'ND1', (4, 0, 0), 'N')
    cys2 = add_residue(chain, 'CYS', 17)
    add_atom(cys2, 'C', (4.67, 6, 0), 'C')
    add_atom(cys2, 'SG', (2.05, 0, 3), 'S')
    his2 = add_residue(chain, 'HIS', 20)
    add_atom(his2, 'N', (6, 6, 0), 'N')
    add_atom(his2, 'NE2', (0, 2, 0), 'N')
    add_atom(his2, 'ND1', (0, 4, 0), 'N')
    metals = Chain.Chain('Z')
    model.add(metals)
    copper = add_residue(metals, 'CU', 900, hetero='H_CU')
    add_atom(copper, 'XX', (0, 0, 0), 'CU')  # Deliberately not named CU.
    return structure


class CopperReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.records, cls.issues, cls.facts = analyze_copper(REFERENCE/'8SY3.cif')
        cls.hits = [r for r in cls.records if r['status']=='pass']

    def test_pinned_reference_checksums(self):
        manifest = json.loads((REFERENCE/'provenance.json').read_text())
        for item in manifest['structures']:
            self.assertEqual(sha256(REFERENCE/item['filename']), item['sha256'])

    def test_recovers_both_chains_with_exact_donors_and_distances(self):
        self.assertEqual(self.facts['deposited_copper_atoms'], 2)
        self.assertEqual(len(self.hits), 2)
        expected = {'A': (2.029740, 2.081712, 98.446883), 'B': (2.084698, 2.110239, 102.155802)}
        for r in self.hits:
            atoms = r['atoms']
            chain = atoms['his1_donor']['auth_chain_id']
            self.assertEqual([atoms[k]['auth_seq_id'] for k in ['his1_donor','his2_donor']], [204,230])
            self.assertEqual([atoms[k]['atom_name'] for k in ['his1_donor','his2_donor']], ['NE2','NE2'])
            self.assertEqual(atoms['his2_donor']['auth_chain_id'], chain)
            self.assertEqual(atoms['copper']['occupancy'], .5)
            for name, value in zip(['cu_his1','cu_his2','donor_cu_donor_angle'], expected[chain]):
                self.assertAlmostEqual(r['measurements'][name]['value'], value, places=5)
            self.assertTrue(r['classifications']['burp_like_2xch_evidence'])
            self.assertTrue(r['burp_evidence']['disulfide']['deposited_bond_annotation'])
            self.assertEqual(r['metals']['hypothetical'], [])

    def test_author_and_label_copper_chains_are_kept_separate(self):
        mapping = {(r['atoms']['copper']['auth_chain_id'], r['atoms']['copper']['label_asym_id']) for r in self.hits}
        self.assertEqual(mapping, {('A','C'),('B','D')})
        self.assertTrue(all(r['atoms']['his1_donor']['label_seq_id']=='204' for r in self.hits))

    def test_metal_free_reference_is_not_an_observed_copper_match(self):
        records, diagnostics, facts = analyze_copper(REFERENCE/'8SY2.cif')
        self.assertEqual(records, [])
        self.assertEqual(facts['deposited_copper_atoms'], 0)
        self.assertTrue(any('No deposited copper' in d['reason'] for d in diagnostics))
        self.assertIn('not a biological negative', facts['interpretation'])

    def test_unoccupied_h165_h177_is_not_recovered_as_bound(self):
        for r in self.hits:
            self.assertNotIn(165, [r['atoms'][k]['auth_seq_id'] for k in ['his1_donor','his2_donor']])
            self.assertNotIn(177, [r['atoms'][k]['auth_seq_id'] for k in ['his1_donor','his2_donor']])

    def test_other_contacts_are_reported_without_ligand_assignment(self):
        chain_b = next(r for r in self.hits if r['atoms']['his1_donor']['auth_chain_id']=='B')
        water = [c for c in chain_b['contacts'] if c['atom']['hetero_flag']=='W']
        self.assertTrue(water)
        self.assertAlmostEqual(water[0]['distance_angstrom'], 2.03311, places=4)
        for r in self.hits:
            methionine = [c for c in r['contacts'] if c['atom']['resname']=='MET' and c['atom']['atom_name']=='SD']
            self.assertTrue(methionine)
            self.assertTrue(all('does not assign a ligand' in c['interpretation'] for c in methionine))

    def test_every_distance_independently_matches_the_recorded_coordinates(self):
        for r in self.hits:
            validate_record(r, REFERENCE/'8SY3.cif')
            for measurement in r['measurements'].values():
                if measurement['units']=='angstrom':
                    endpoints = [r['atoms'][key]['coordinates'] for key in measurement['endpoints']]
                    self.assertAlmostEqual(math.dist(*endpoints), measurement['value'], places=6)

    def test_export_contains_original_mmcif_and_observed_copper_center(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'hits.jsonl'
            path.write_text(''.join(json.dumps(r)+'\n' for r in self.hits))
            target=export_bundle(path,self.hits[0]['site_id'],Path(folder)/'bundle')
            self.assertEqual(sha256(target/'structure.cif'),sha256(REFERENCE/'8SY3.cif'))
            report=json.loads((target/'site.json').read_text())
            self.assertEqual(report['points'],{})
            self.assertEqual(report['visualization']['center'],'copper')


class SyntheticCopperTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path=Path(self.temp.name)/'synthetic.pdb'
        self.structure=synthetic_structure()

    def analyze(self, parameters=None):
        writer=PDBIO();writer.set_structure(self.structure);writer.save(str(self.path))
        return analyze_copper(self.path,parameters)

    def passing(self, parameters=None):
        return [r for r in self.analyze(parameters)[0] if r['status']=='pass']

    def test_element_not_atom_name_serial_or_chain_identifies_copper(self):
        hits=self.passing()
        self.assertEqual(len(hits),1)
        self.assertEqual(hits[0]['atoms']['copper']['atom_name'],'XX')
        self.assertEqual(hits[0]['atoms']['copper']['auth_chain_id'],'Z')
        self.assertTrue(hits[0]['classifications']['burp_like_2xch_evidence'])

    def test_donor_can_be_nd1(self):
        his=self.structure[0]['P'][10]
        his['ND1'].coord=np.array([2.,0.,0.]);his['NE2'].coord=np.array([4.,0.,0.])
        self.assertEqual(self.passing()[0]['atoms']['his1_donor']['atom_name'],'ND1')

    def test_altered_geometry_is_a_failed_criterion_not_a_function_claim(self):
        self.structure[0]['P'][10]['NE2'].coord=np.array([3.2,0.,0.])
        records,_,_=self.analyze()
        self.assertTrue(records)
        self.assertFalse(any(r['status']=='pass' for r in records))
        self.assertTrue(any(c['status']=='fail' for r in records for c in r['criteria'] if c['name']=='cu_his1_distance'))

    def test_missing_both_donors_is_not_evaluable(self):
        his=self.structure[0]['P'][10]
        his.detach_child('NE2');his.detach_child('ND1')
        records,diagnostics,_=self.analyze()
        self.assertEqual(records,[])
        self.assertTrue(any(d['status']=='not_evaluable' and 'Missing usable' in d['reason'] for d in diagnostics))

    def test_removed_copper_is_not_a_match(self):
        self.structure[0].detach_child('Z')
        self.assertEqual(self.passing(),[])

    def test_generic_and_strict_presets_treat_missing_staple_differently(self):
        self.structure[0]['P'][7].detach_child('SG')
        generic=self.passing()
        self.assertEqual(len(generic),1)
        self.assertFalse(generic[0]['classifications']['burp_like_2xch_evidence'])
        strict=self.analyze(CopperParameters(preset='burp-like'))[0]
        self.assertTrue(any(r['status']=='not_evaluable' for r in strict))
        self.assertFalse(any(r['status']=='pass' for r in strict))

    def test_non_cysteine_predecessor_does_not_block_generic_match(self):
        self.structure[0]['P'][7].resname='ALA'
        self.assertEqual(len(self.passing()),1)
        self.assertEqual(self.passing()[0]['burp_evidence']['ch1']['status'],'fail')
        self.assertEqual(self.passing(CopperParameters(preset='burp-like')),[])

    def test_missing_peptide_link_does_not_assume_sequence_adjacency(self):
        self.structure[0]['P'][7]['C'].coord=np.array([100.,0.,0.])
        hit=self.passing()[0]
        self.assertEqual(hit['burp_evidence']['ch1']['status'],'not_evaluable')

    def test_renumbering_insertion_codes_and_rigid_transform_preserve_geometry(self):
        before=self.passing()[0]
        for chain in self.structure[0]:
            for r in list(chain):
                chain.detach_child(r.id)
                r.id=(r.id[0],r.id[1]+100,'A' if r.resname=='HIS' else ' ')
                chain.add(r)
        rotation=np.array([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]])
        for atom in self.structure.get_atoms():
            atom.transform(rotation,np.array([10.,20.,30.]))
        after=self.passing()[0]
        self.assertEqual(after['atoms']['his1_donor']['insertion_code'],'A')
        self.assertEqual(before['classifications'],after['classifications'])
        for name,m in before['measurements'].items():
            self.assertAlmostEqual(after['measurements'][name]['value'],m['value'],places=4)

    def test_histidines_from_different_chains_do_not_pair(self):
        model=self.structure[0];his=model['P'][20]
        model['P'].detach_child(his.id);other=Chain.Chain('Q');model.add(other);other.add(his)
        self.assertEqual(self.passing(),[])

    def test_different_models_are_independent(self):
        model=copy.deepcopy(self.structure[0]);model.id=1;model.serial_num=2;self.structure.add(model)
        hits=self.passing()
        self.assertEqual(len(hits),2)
        self.assertEqual(len({r['site_id'] for r in hits}),2)
        for r in hits:
            self.assertEqual(len({a['model_id'] for a in r['atoms'].values()}),1)
            self.assertFalse(r['exportable'])

    def test_multiple_histidines_and_additional_ligands_are_allowed(self):
        chain=self.structure[0]['P']
        his=add_residue(chain,'HIS',30);add_atom(his,'NE2',(-2,0,0),'N')
        water=add_residue(self.structure[0]['Z'],'HOH',901,hetero='W')
        add_atom(water,'O',(0,0,2),'O')
        self.assertEqual(len(self.passing()),3)

    def test_incompatible_altlocs_are_not_combined(self):
        for number,alt in [(10,'A'),(20,'B')]:
            his=self.structure[0]['P'][number]
            position=his['NE2'].coord.copy();his.detach_child('NE2');his.detach_child('ND1')
            add_atom(his,'NE2',position,'N',alt=alt,occupancy=.5)
        self.assertEqual(self.passing(),[])

    def test_compatible_altlocs_are_enumerated_and_exportable(self):
        for number in (10,20):
            his=self.structure[0]['P'][number]
            position=his['NE2'].coord.copy();his.detach_child('NE2');his.detach_child('ND1')
            add_atom(his,'NE2',position,'N',alt='A',occupancy=.5)
            add_atom(his,'NE2',position,'N',alt='B',occupancy=.5)
        hits=self.passing()
        self.assertEqual(len(hits),2)
        for r in hits:
            self.assertEqual(r['atoms']['his1_donor']['altloc'],r['atoms']['his2_donor']['altloc'])
            validate_record(r,self.path)

    def test_zero_occupancy_copper_is_present_but_not_used(self):
        self.structure[0]['Z'][('H_CU',900,' ')]['XX'].occupancy=0
        records,diagnostics,facts=self.analyze()
        self.assertEqual(facts['deposited_copper_atoms'],1)
        self.assertEqual(records,[])
        self.assertTrue(any('occupancy' in d['reason'] for d in diagnostics))

    def test_wrong_element_is_not_used_as_a_histidine_nitrogen(self):
        self.structure[0]['P'][10]['NE2'].element='C'
        records,diagnostics,_=self.analyze()
        self.assertFalse(any(r['status']=='pass' for r in records))
        self.assertTrue(any('name/element conflict' in d['reason'] for d in diagnostics))

    def test_nonfinite_copper_has_a_json_safe_diagnostic(self):
        self.structure[0]['Z'][('H_CU',900,' ')]['XX'].coord=np.array([float('nan'),0.,0.])
        records,diagnostics,_=self.analyze()
        self.assertEqual(records,[])
        issue=next(d for d in diagnostics if 'nonfinite' in d['reason'])
        self.assertIsNone(issue['atom']['coordinates'][0])
        json.dumps(diagnostics,allow_nan=False)

    def test_inclusive_distance_boundary(self):
        self.structure[0]['P'][10]['NE2'].coord=np.array([2.8,0.,0.])
        # PDB parsing uses float32: use the represented boundary explicitly.
        boundary=float(np.float32(2.8))
        self.assertEqual(len(self.passing(CopperParameters(cu_n_max=boundary))),1)
        self.assertEqual(self.passing(CopperParameters(cu_n_max=boundary-0.001)),[])

    def test_parameters_reject_nonfinite_and_reversed_limits(self):
        for values in [dict(cu_n_min=float('nan')),dict(cu_n_min=3,cu_n_max=2),dict(candidate_radius=1)]:
            with self.subTest(values=values),self.assertRaises(ValueError):CopperParameters(**values)

    def test_output_files_keep_failed_candidates_and_valid_empty_csv(self):
        self.analyze()
        out=Path(self.temp.name)/'output'
        args=argparse.Namespace(pdb_dir=str(self.path),outdir=str(out),**asdict(FilterParameters(("observed_cu_2his",))))
        with contextlib.redirect_stdout(io.StringIO()):run_mine(args)
        self.assertEqual(len(pd.read_csv(next(out.glob('selected_sites_*.csv')))),1)
        self.assertGreater(len((next(out.glob('selected_candidates_*.jsonl'))).read_text().splitlines()),1)
        self.assertEqual(len((next(out.glob('selected_sites_*.jsonl'))).read_text().splitlines()),1)
        args.pdb_dir=str(REFERENCE/'8SY2.cif');args.outdir=str(Path(self.temp.name)/'empty')
        with contextlib.redirect_stdout(io.StringIO()):run_mine(args)
        self.assertTrue(pd.read_csv(next(Path(args.outdir).glob('selected_sites_*.csv'))).empty)

    def test_parse_failure_is_reported_and_does_not_look_like_a_success(self):
        self.path.write_text('ATOM      1  CA  ALA A   1      BADCOORDINATES\n')
        args=argparse.Namespace(pdb_dir=str(self.path),outdir=str(Path(self.temp.name)/'bad'),**asdict(FilterParameters(("observed_cu_2his",))))
        with contextlib.redirect_stdout(io.StringIO()),self.assertRaisesRegex(ValueError,'could not be processed'):
            run_mine(args)
        issues=json.loads((next(Path(args.outdir).glob('selected_diagnostics_*.json'))).read_text())
        self.assertIn('Parse/processing error',issues[0]['reason'])


if __name__=='__main__':unittest.main()
