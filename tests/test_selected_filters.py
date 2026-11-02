"""Composable filters: independent scores, same-site composition, and regressions."""
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
from unittest.mock import patch

import numpy as np
from Bio.PDB import MMCIFIO, PDBIO
from Bio.PDB.Atom import DisorderedAtom

from hal_discovery.cli import main
from hal_discovery.mine import FilterParameters, analyze_filters, inward_facing_geometry, run_mine
from hal_discovery.pymol_export import export_bundle, validate_record, VIEW_SCRIPT
from hal_discovery.structure_io import load_structure

REFERENCE = Path(__file__).resolve().parents[1] / 'tests/fixtures/copper'
COMMON = ('his_ne2','potential_cu_2his')


def passing(records):
    return [r for r in records if r['status']=='pass']


def site_key(record):
    a,b = (record['atoms'][role] for role in ('his1_donor','his2_donor'))
    return a['auth_chain_id'],a['auth_seq_id'],b['auth_seq_id'],a['atom_name'],b['atom_name']


class InwardMathTests(unittest.TestCase):
    def setUp(self):
        self.ring1 = [(-1,0,0),(0,-1,0),(1,0,0),(0,1,0),(0,0,0)]
        self.ring2 = [(x+4,y,z) for x,y,z in self.ring1]

    def test_two_inward_scores_are_not_the_mutual_vector_dot(self):
        result = inward_facing_geometry(self.ring1,(1,0,0),self.ring2,(3,0,0))
        self.assertEqual(result['s1'],1)
        self.assertEqual(result['s2'],1)
        self.assertEqual(result['inward_score'],1)
        self.assertEqual(sum(a*b for a,b in zip(result['u1'],result['u2'])),-1)

    def test_away_and_sideways_donors_do_not_have_positive_scores(self):
        for donor,expected in [((-1,0,0),-1),((0,1,0),0)]:
            with self.subTest(donor=donor):
                result=inward_facing_geometry(self.ring1,donor,self.ring2,(3,0,0))
                self.assertEqual(result['s1'],expected)
                self.assertEqual(result['inward_score'],expected)

    def test_pair_reversal_swaps_scores_and_preserves_minimum(self):
        a=inward_facing_geometry(self.ring1,(1,0,0),self.ring2,(3,1,0))
        b=inward_facing_geometry(self.ring2,(3,1,0),self.ring1,(1,0,0))
        self.assertEqual(a['s1'],b['s2'])
        self.assertEqual(a['s2'],b['s1'])
        self.assertEqual(a['inward_score'],b['inward_score'])

    def test_degenerate_and_nonfinite_geometry_is_not_evaluable(self):
        for ring,donor in [(self.ring1,(0,0,0)),(self.ring1,(float('nan'),0,0))]:
            with self.assertRaises(ValueError):inward_facing_geometry(self.ring1,donor,ring,(1,0,0))
        with self.assertRaises(ValueError):inward_facing_geometry(self.ring1[:4],(1,0,0),self.ring2,(3,0,0))

    def test_threshold_validation_and_canonical_filter_order(self):
        for value in (-.1,1.1,float('nan'),float('inf')):
            with self.subTest(value=value),self.assertRaises(ValueError):
                FilterParameters(COMMON,inward_score_min=value)
        self.assertEqual(FilterParameters(tuple(reversed(COMMON))).filters,FilterParameters(COMMON).filters)
        for names in [(),('unknown',)]:
            with self.assertRaises(ValueError):FilterParameters(names)
        with self.assertRaises(ValueError):FilterParameters(('his_ne2',),inward_score_min=.8)


class SelectedReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.records,cls.issues,cls.facts=analyze_filters(REFERENCE/'8SY3.cif',FilterParameters(COMMON))
        cls.hits=passing(cls.records)

    def test_all_four_reference_pairs_match_the_independent_experiment(self):
        expected={('A',204,230):(.973473,.881650),('B',204,230):(.949550,.866887),
                  ('A',165,177):(.936781,.922495),('B',165,177):(.931649,.940531)}
        self.assertEqual(len(self.hits),4)
        for r in self.hits:
            key=site_key(r)
            self.assertEqual(key[3:],('NE2','NE2'))
            s1,s2=expected[key[:3]]
            self.assertAlmostEqual(r['measurements']['s1']['value'],s1,places=5)
            self.assertAlmostEqual(r['measurements']['s2']['value'],s2,places=5)
            self.assertAlmostEqual(r['measurements']['inward_score']['value'],min(s1,s2),places=5)
            self.assertIsNone(r['selection']['parameters']['inward_score_min'])
            self.assertEqual(r['metals']['observed'],[])
            self.assertEqual(r['metals']['hypothetical'],[])
            self.assertEqual(r['metals']['assessment'],'not_assessed_by_selected_filters')

    def test_omitted_and_zero_threshold_keep_the_same_inward_candidates(self):
        zero=passing(analyze_filters(REFERENCE/'8SY3.cif',FilterParameters(COMMON,inward_score_min=0))[0])
        self.assertEqual({site_key(r) for r in zero},{site_key(r) for r in self.hits})

    def test_point_nine_excludes_both_observed_h204_h230_arrangements(self):
        hits=passing(analyze_filters(REFERENCE/'8SY3.cif',FilterParameters(COMMON,inward_score_min=.9))[0])
        self.assertEqual({site_key(r)[:3] for r in hits},{('A',165,177),('B',165,177)})

    def test_threshold_is_inclusive_and_uses_the_weaker_donor(self):
        example=next(r for r in self.hits if site_key(r)[:3]==('A',204,230))
        threshold=example['measurements']['inward_score']['value']
        at=passing(analyze_filters(REFERENCE/'8SY3.cif',FilterParameters(COMMON,inward_score_min=threshold))[0])
        above=passing(analyze_filters(REFERENCE/'8SY3.cif',FilterParameters(COMMON,inward_score_min=threshold+1e-6))[0])
        self.assertIn(site_key(example),{site_key(r) for r in at})
        self.assertNotIn(site_key(example),{site_key(r) for r in above})
        self.assertLess(threshold,example['measurements']['s1']['value'])

    def test_optional_cysteine_filters_keep_reference_hits(self):
        hits=passing(analyze_filters(REFERENCE/'8SY3.cif',FilterParameters(COMMON+('cys_his','cys_disulfide')))[0])
        self.assertEqual({site_key(r) for r in hits},{site_key(r) for r in self.hits})
        self.assertTrue(all(all(status=='pass' for status in r['filter_results'].values()) for r in hits))

    def test_no_legacy_backbone_hx_or_length_filters_are_implicit(self):
        for r in self.hits:
            self.assertEqual(set(r['filter_results']),set(COMMON))
            self.assertFalse(any('backbone' in c['name'] or 'HX' in c['name'] for c in r['criteria']))

    def test_ne2_filter_alone_reports_ne2_pairs_without_unused_donor_assignments(self):
        records,_,_=analyze_filters(REFERENCE/'8SY3.cif',FilterParameters(('his_ne2',)))
        self.assertEqual(len(passing(records)),4)
        self.assertTrue(all(site_key(r)[3:]==('NE2','NE2') for r in records))

    def test_nd1_pairs_are_allowed_when_ne2_distance_filter_is_not_selected(self):
        path=REFERENCE.parents[2]/'demo/demo_dataset/G5H9U4.pdb'
        records,_,_=analyze_filters(path,FilterParameters(('potential_cu_2his',)))
        hit=next(r for r in passing(records) if site_key(r)==('A',76,102,'ND1','ND1'))
        self.assertAlmostEqual(hit['measurements']['s1']['value'],.7017,places=4)
        self.assertEqual(set(hit['filter_results']),{'potential_cu_2his'})

    def test_metal_free_reference_has_potential_but_not_observed_evidence(self):
        hits=passing(analyze_filters(REFERENCE/'8SY2.cif',FilterParameters(COMMON))[0])
        self.assertEqual(len(hits),4)
        self.assertTrue(all(not r['metals']['observed'] and not r['metals']['hypothetical'] for r in hits))
        observed=passing(analyze_filters(REFERENCE/'8SY2.cif',FilterParameters(('observed_cu_2his',)))[0])
        self.assertEqual(observed,[])

    def test_observed_filter_is_independent_of_potential_filter(self):
        observed=passing(analyze_filters(REFERENCE/'8SY3.cif',FilterParameters(('observed_cu_2his',)))[0])
        self.assertEqual({site_key(r)[:3] for r in observed},{('A',204,230),('B',204,230)})
        strong=passing(analyze_filters(REFERENCE/'8SY3.cif',FilterParameters(('observed_cu_2his','potential_cu_2his'),inward_score_min=.9))[0])
        self.assertEqual(strong,[])

    def test_independent_record_verification_rejects_tampered_geometry(self):
        r=self.hits[0]
        validate_record(r,REFERENCE/'8SY3.cif')
        for field in ('point','direction','score'):
            changed=copy.deepcopy(r)
            if field=='point':changed['points']['ring_center1']['coordinates'][0]+=.1
            elif field=='direction':changed['directions']['his1']['vector'][0]+=.1
            else:changed['measurements']['s1']['value']+=.1
            with self.subTest(field=field),self.assertRaises(ValueError):validate_record(changed,REFERENCE/'8SY3.cif')

    def test_generated_script_is_valid_python_and_bundle_contains_no_inferred_metal(self):
        compile(VIEW_SCRIPT.split('python\n',1)[1].rsplit('python end',1)[0],'view.pml','exec')
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'hits.jsonl';path.write_text(json.dumps(self.hits[0])+'\n')
            bundle=export_bundle(path,self.hits[0]['site_id'],Path(folder)/'bundle')
            self.assertTrue((bundle/'structure.cif').exists())
            self.assertIn('not metal positions',(bundle/'README.txt').read_text())


class SelectedMutationTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.path=Path(self.temp.name)/'synthetic.pdb'
        self.structure=load_structure(REFERENCE/'8SY3.cif').structure

    def analyze(self, filters=COMMON, **options):
        writer=PDBIO();writer.set_structure(self.structure);writer.save(str(self.path))
        return analyze_filters(self.path,FilterParameters(filters,**options))

    def test_removing_all_copper_does_not_change_protein_geometry(self):
        before=passing(self.analyze()[0])
        for model in self.structure:
            for chain in model:
                for residue in list(chain):
                    if any(a.element=='CU' for a in residue):chain.detach_child(residue.id)
        after=passing(self.analyze()[0])
        a={site_key(r):r for r in before};b={site_key(r):r for r in after}
        self.assertEqual(set(a),set(b))
        for key in a:
            self.assertEqual(a[key]['measurements'],b[key]['measurements'])
            self.assertEqual(a[key]['filter_results'],b[key]['filter_results'])

    def test_missing_ring_carbon_is_not_evaluable_not_a_measured_failure(self):
        self.structure[0]['A'][204].detach_child('CG')
        records,_,_=self.analyze()
        affected=[r for r in records if site_key(r)[:3]==('A',204,230)]
        self.assertTrue(affected)
        self.assertTrue(all(r['filter_results']['potential_cu_2his']=='not_evaluable' for r in affected))
        self.assertTrue(all(not r['exportable'] for r in affected))

    def test_staple_from_another_site_cannot_satisfy_a_missing_site_staple(self):
        self.structure[0]['A'][203].detach_child('SG')
        records,_,_=self.analyze(COMMON+('cys_disulfide',))
        affected=[r for r in records if site_key(r)[:3]==('A',204,230)]
        self.assertTrue(affected)
        self.assertTrue(all(r['filter_results']['cys_disulfide']=='not_evaluable' for r in affected))
        self.assertNotIn(('A',204,230),{site_key(r)[:3] for r in passing(records)})
        self.assertIn(('A',165,177),{site_key(r)[:3] for r in passing(records)})

    def test_rigid_transform_and_renumbering_preserve_selected_scores(self):
        before={site_key(r):r for r in passing(self.analyze()[0])}
        rotation=np.array([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]])
        for atom in self.structure.get_atoms():atom.transform(rotation,np.array([10.,20.,30.]))
        for model in self.structure:
            for chain in model:
                for residue in list(chain):
                    chain.detach_child(residue.id);residue.id=(residue.id[0],residue.id[1]+1000,'A' if residue.resname=='HIS' else residue.id[2]);chain.add(residue)
        after=passing(self.analyze()[0])
        self.assertEqual(len(after),len(before))
        for r in after:
            key=site_key(r);old=before[(key[0],key[1]-1000,key[2]-1000,*key[3:])]
            self.assertEqual(r['atoms']['his1_donor']['insertion_code'],'A')
            for name in ('s1','s2','inward_score'):
                self.assertAlmostEqual(old['measurements'][name]['value'],r['measurements'][name]['value'],places=5)

    def test_model_and_chain_identity_are_never_mixed(self):
        duplicate=copy.deepcopy(self.structure[0]);duplicate.id=1;duplicate.serial_num=2;self.structure.add(duplicate)
        hits=passing(self.analyze()[0])
        self.assertEqual(len(hits),8)
        self.assertEqual(len({r['site_id'] for r in hits}),8)
        for r in hits:
            self.assertEqual(len({a['model_id'] for a in r['atoms'].values()}),1)
            self.assertEqual(len({r['atoms'][k]['auth_chain_id'] for k in ('his1_donor','his2_donor')}),1)
            self.assertFalse(r['exportable'])

    def test_ring_atoms_from_incompatible_altlocs_cannot_be_combined(self):
        for number,labels in ((204,('A',)),(230,('B',))):
            residue=self.structure[0]['A'][number]
            for name in ('CG','ND1','CE1','NE2','CD2'):
                original=residue[name];residue.detach_child(name)
                disordered=DisorderedAtom(name);residue.add(disordered)
                for label in labels:
                    atom=copy.deepcopy(original);atom.set_altloc(label);atom.set_occupancy(.5)
                    disordered.disordered_add(atom)
        records,diagnostics,_=self.analyze()
        self.assertNotIn(('A',204,230),{site_key(r)[:3] for r in passing(records)})
        self.assertTrue(any(d['status']=='not_evaluable' for d in diagnostics))

    def test_complete_compatible_ring_altlocs_are_enumerated_separately(self):
        for number in (204,230):
            residue=self.structure[0]['A'][number]
            for name in ('CG','ND1','CE1','NE2','CD2'):
                original=residue[name];residue.detach_child(name)
                disordered=DisorderedAtom(name);residue.add(disordered)
                for label in ('A','B'):
                    atom=copy.deepcopy(original);atom.set_altloc(label);atom.set_occupancy(.5)
                    disordered.disordered_add(atom)
        hits=[r for r in passing(self.analyze()[0]) if site_key(r)[:3]==('A',204,230)]
        self.assertEqual(len(hits),2)
        self.assertEqual(len({r['site_id'] for r in hits}),2)
        for r in hits:
            self.assertEqual(len({a['altloc'] for a in r['atoms'].values() if a['altloc']}),1)
            validate_record(r,self.path)

    def test_selected_run_writes_valid_empty_outputs(self):
        writer=PDBIO();writer.set_structure(self.structure);writer.save(str(self.path))
        out=Path(self.temp.name)/'out'
        args=argparse.Namespace(pdb_dir=str(self.path),outdir=str(out),**asdict(FilterParameters(COMMON, inward_score_min=1.)))
        with contextlib.redirect_stdout(io.StringIO()):run_mine(args)
        summary=json.loads(next(out.glob('summary_selected_*.json')).read_text())
        self.assertEqual(summary['passing_records'],0)
        self.assertEqual(next(out.glob('selected_sites_*.jsonl')).read_text(),'')
        self.assertIn('inward_score',next(out.glob('selected_sites_*.csv')).read_text().splitlines()[0])

    def test_cli_uses_mine_and_flat_filters(self):
        out=Path(self.temp.name)/'cli'
        argv=['hal-discovery','mine','--pdb-dir',str(REFERENCE/'8SY3.cif'),'--outdir',str(out),
              '--filters',*COMMON,'--inward-score-min','0.9']
        with patch('sys.argv',argv),contextlib.redirect_stdout(io.StringIO()):main()
        summary=json.loads(next(out.glob('summary_selected_*.json')).read_text())
        self.assertEqual(summary['passing_records'],2)
        with patch('sys.argv',['hal-discovery','copper']),contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):main()


if __name__=='__main__':unittest.main()
