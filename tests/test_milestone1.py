import argparse
import contextlib
import copy
import csv
import io
import json
import math
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

from Bio.PDB import PDBParser, PDBIO
from hal_discovery.mine import main_function, run_mine
from hal_discovery.results import explain_site, PARAMETERS, sha256
from hal_discovery.pymol_export import export_bundle, validate_record, VIEW_SCRIPT

ROOT = Path(__file__).resolve().parents[1]
DEFAULTS = dict(his_ne2_cutoff=4., his_hbond_cutoff=4., asp_glu_cutoff=5., ala_gly_cutoff=4.5,
                require_hx=True, hx_offset=2, min_length=200, acid_exclude_cutoff=5.5,
                exclude_ligand_cutoff=4., ala_gly_presence_cutoff=7., hx_allowed=['ALA','GLY'],
                triad_min_length=0, progress_every=200, max_pdbs=None)


def key(row):
    return tuple(str(row[k]) for k in ('structure','chainA','hisA','chainB','hisB'))


class MilestoneOne(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.directory = Path(cls.temp.name)
        cls.args = argparse.Namespace(**DEFAULTS, pdb_dir=str(ROOT/'demo/demo_dataset'),
                                      outdir=str(cls.directory/'results'), detailed_results=True)
        with contextlib.redirect_stdout(io.StringIO()):
            run_mine(cls.args)
        cls.results = next((cls.directory/'results').glob('site_matches_*.jsonl'))
        cls.records = [json.loads(line) for line in cls.results.read_text().splitlines()]
        cls.example = next(r for r in cls.records if r['source']['structure_id']=='Q9RBY6')

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_full_demo_matches_original_rows_and_classifications(self):
        baseline = json.loads((ROOT/'tests/fixtures/demo_baseline.json').read_text())
        for filename, digest in baseline['input_sha256'].items():
            self.assertEqual(sha256(ROOT/'demo/demo_dataset'/filename), digest)
        actual = {key(r['legacy_row']):r for r in self.records}
        self.assertEqual(set(actual), {key(r) for r in baseline['rows']})
        with next((self.directory/'results').glob('2his_sites_all_*.csv')).open() as handle:
            csv_rows = list(csv.DictReader(handle))
        self.assertEqual(len(csv_rows), 203)
        for expected in baseline['rows']:
            record = actual[key(expected)]
            self.assertEqual(record['classifications'], expected['classifications'])
            for col, value in expected.items():
                if col=='classifications': continue
                measured = record['legacy_row'][col]
                if col.startswith('closest_'):
                    if value=='': self.assertIsNone(measured)
                    else: self.assertAlmostEqual(float(value), measured, places=6)
                else: self.assertEqual(str(measured), value)
        for pattern, classification in [('facial_triad_sites_*.csv','facial_triad'),
                                         ('putative_halogenases_*.csv','putative_halogenase')]:
            with next((self.directory/'results').glob(pattern)).open() as f:
                self.assertEqual({key(r) for r in csv.DictReader(f)},
                                 {key(r['legacy_row']) for r in self.records if r['classifications'][classification]})
        summary = json.loads(next((self.directory/'results').glob('summary_*.json')).read_text())
        self.assertEqual([summary[k] for k in ('total_structures','total_2his_sites',
            'proteins_with_2his_sites','sites_2his_aspglu','sites_putative_halogenase')], [200,203,200,192,10])
        self.assertEqual(json.loads(next((self.directory/'results').glob('processing_errors_*.json')).read_text()), [])

    def test_measurements_independently_recompute_for_every_hit(self):
        for record in self.records:
            for measurement in record['measurements'].values():
                ends = [record['atoms'].get(e,record['points'].get(e))['coordinates'] for e in measurement['endpoints']]
                self.assertAlmostEqual(math.dist(*ends),measurement['value'],places=5)
        validate_record(self.example, ROOT/'demo/demo_dataset/Q9RBY6.pdb')
        self.assertEqual(self.example['legacy_row']['hisA'],116)
        self.assertEqual(self.example['legacy_row']['hisB'],235)
        self.assertEqual(self.example['hx']['auth_seq_id'],118)

    def test_export_checksum_ambiguity_and_no_overwrite(self):
        bundle = export_bundle(self.results,self.example['site_id'],self.directory/'bundle')
        self.assertEqual(sha256(bundle/'structure.pdb'),self.example['source']['sha256'])
        self.assertEqual(set(p.name for p in bundle.iterdir()),
                         {'structure.pdb','site.json','display.json','view.pml','README.txt'})
        with self.assertRaisesRegex(ValueError,'new or empty'):
            export_bundle(self.results,self.example['site_id'],bundle)
        with self.assertRaisesRegex(ValueError,'exactly one'):
            export_bundle(self.results,'unknown',self.directory/'unknown')
        altered = self.directory/'altered.pdb'
        altered.write_bytes((bundle/'structure.pdb').read_bytes()+b'REMARK synthetic\n')
        with self.assertRaisesRegex(ValueError,'SHA-256'):
            validate_record(self.example,altered)
        ambiguous = copy.deepcopy(self.example); ambiguous['exportable']=False
        with self.assertRaisesRegex(ValueError,'ambiguous'):
            validate_record(ambiguous,bundle/'structure.pdb')
        invalid = copy.deepcopy(self.example)
        invalid['measurements']['his_ne2_distance']['value']+=1
        with self.assertRaisesRegex(ValueError,'disagrees'):
            validate_record(invalid,bundle/'structure.pdb')

    def test_missing_data_does_not_become_measured_failure(self):
        record = copy.deepcopy(self.example)
        record['legacy_row']['closest_CYS']=None
        explain_site(record,self.args)
        criterion = next(c for c in record['criteria'] if c['name']=='exclude_CYS')
        self.assertEqual(criterion['status'],'not_evaluable')
        self.assertFalse(record['classifications']['putative_halogenase'])

    def test_insertion_code_and_selected_alternate_location(self):
        source=(ROOT/'demo/demo_dataset/Q9RBY6.pdb').read_text().splitlines(True)
        lines=[]
        for line in source:
            if line.startswith('ATOM') and line[21]=='A' and line[22:26].strip()=='116':
                line=line[:26]+'A'+line[27:]
                if line[12:16].strip()=='NE2':
                    lines.append(line[:16]+'A'+line[17:54]+'  0.30'+line[60:])
                    line=line[:16]+'B'+line[17:54]+'  0.70'+line[60:]
            lines.append(line)
        path=self.directory/'altloc.pdb';path.write_text(''.join(lines))
        details=[];main_function(str(path),4.,4.,2,details)
        record=explain_site(next(r for r in details if r['legacy_row']['hisA']==116),self.args)
        self.assertEqual(record['atoms']['his1_NE2']['altloc'],'B')
        self.assertEqual(record['atoms']['his1_NE2']['insertion_code'],'A')
        validate_record(record,path)

    def test_criterion_boundaries_are_explicit(self):
        record=copy.deepcopy(self.example)
        record['legacy_row']['closest_ASP']=self.args.asp_glu_cutoff
        record['legacy_row']['closest_GLY']=self.args.ala_gly_presence_cutoff
        explain_site(record,self.args)
        criteria={c['name']:c for c in record['criteria']}
        self.assertEqual(criteria['facial_ASP']['status'],'pass')
        self.assertEqual(criteria['presence_GLY']['status'],'fail')

    def test_legacy_cross_chain_hits_keep_exact_chain_identities(self):
        structure=PDBParser(QUIET=True).get_structure('synthetic',str(ROOT/'demo/demo_dataset/Q9RBY6.pdb'))
        chain=copy.deepcopy(structure[0]['A']);chain.id='B';structure[0].add(chain)
        writer=PDBIO();writer.set_structure(structure);path=self.directory/'chains.pdb';writer.save(str(path))
        details=[];main_function(str(path),4.,4.,2,details)
        record=explain_site(next(r for r in details if r['legacy_row']['chainA']=='A'
                                and r['legacy_row']['chainB']=='B'
                                and r['legacy_row']['hisA']==116 and r['legacy_row']['hisB']==235),self.args)
        self.assertEqual(record['atoms']['his1_NE2']['auth_chain_id'],'A')
        self.assertEqual(record['atoms']['his2_NE2']['auth_chain_id'],'B')
        validate_record(record,path)

    def test_synthetic_missing_atom_and_models(self):
        structure = PDBParser(QUIET=True).get_structure('synthetic',str(ROOT/'demo/demo_dataset/Q9RBY6.pdb'))
        structure[0]['A'][116].detach_child('NE2')
        writer=PDBIO();writer.set_structure(structure); path=self.directory/'missing.pdb';writer.save(str(path))
        details,diagnostics=[],[]
        rows=main_function(str(path),4.,4.,2,details,diagnostics)
        self.assertFalse(any(r['hisA']==116 or r['hisB']==116 for r in rows))
        self.assertTrue(any(d['status']=='not_evaluable' and 'NE2' in d['missing_atoms'] for d in diagnostics))
        structure = PDBParser(QUIET=True).get_structure('synthetic',str(ROOT/'demo/demo_dataset/Q9RBY6.pdb'))
        model=copy.deepcopy(structure[0]);model.id=1;model.serial_num=2;structure.add(model)
        writer.set_structure(structure);path=self.directory/'models.pdb';writer.save(str(path))
        details=[];main_function(str(path),4.,4.,2,details)
        self.assertTrue(details)
        self.assertTrue(all(not r['exportable'] for r in details))

    def test_rotation_translation_and_renumbering(self):
        import numpy as np
        structure=PDBParser(QUIET=True).get_structure('synthetic',str(ROOT/'demo/demo_dataset/Q9RBY6.pdb'))
        for atom in structure.get_atoms():
            atom.transform(np.array([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]]),np.array([10.,20.,30.]))
        for chain in structure[0]:
            for residue in list(chain):
                chain.detach_child(residue.id)
                residue.id=(residue.id[0],residue.id[1]+1000,residue.id[2])
                chain.add(residue)
        writer=PDBIO();writer.set_structure(structure);path=self.directory/'transformed.pdb';writer.save(str(path))
        details=[];main_function(str(path),4.,4.,2,details)
        record=explain_site(next(r for r in details if r['legacy_row']['hisA']==1116),self.args)
        self.assertEqual(record['legacy_row']['hisB'],1235)
        self.assertEqual(record['classifications'],self.example['classifications'])
        for name,m in record['measurements'].items():
            self.assertAlmostEqual(m['value'],self.example['measurements'][name]['value'],places=4)

    def test_empty_and_parse_error_reports(self):
        folder=self.directory/'empty_input';folder.mkdir()
        (folder/'malformed.pdb').write_text('ATOM      1  CA  ALA A   1      BADCOORDINATES\n')
        out=self.directory/'empty_results'
        args=argparse.Namespace(**DEFAULTS,pdb_dir=str(folder),outdir=str(out),detailed_results=True)
        with contextlib.redirect_stdout(io.StringIO()): run_mine(args)
        self.assertEqual(next(out.glob('site_matches_*.jsonl')).read_text(),'')
        self.assertTrue(json.loads(next(out.glob('processing_errors_*.json')).read_text()))

    def test_generated_script_with_identity_aware_pymol_standin(self):
        # Exercises selection construction and distance verification, not rendering.
        bundle=export_bundle(self.results,self.example['site_id'],self.directory/'script_bundle')
        structure=PDBParser(QUIET=True).get_structure('test',str(bundle/'structure.pdb'))
        atoms=[]
        for index,a in enumerate(structure.get_atoms(),1):
            r=a.get_parent()
            atoms.append(types.SimpleNamespace(index=index,chain=r.get_parent().id,
                resi=str(r.id[1])+r.id[2].strip(),resn=r.resname,name=a.name,alt=a.get_altloc(),
                hetatm=r.id[0]!=' ',coord=[float(v) for v in a.coord]))
        coordinates={}
        class Command:
            def get_model(self,*a,**kw):return types.SimpleNamespace(atom=atoms)
            def select(self,name,expression):
                if name.startswith('recorded_'):
                    index=int(expression.rsplit(' ',1)[1]);coordinates[name]=atoms[index-1].coord
            def pseudoatom(self,name,**kw):coordinates[name]=kw['pos']
            def get_distance(self,a,b,**kw):return math.dist(coordinates[a],coordinates[b])
            def __getattr__(self,name):return lambda *a,**kw:None
        module=types.ModuleType('pymol');module.cmd=Command()
        original=Path.cwd()
        import os
        try:
            os.chdir(bundle)
            with patch.dict('sys.modules',{'pymol':module}), contextlib.redirect_stdout(io.StringIO()):
                exec(compile(VIEW_SCRIPT.split('python\n',1)[1].rsplit('python end',1)[0],'view.pml','exec'),{})
        finally: os.chdir(original)
        self.assertEqual(len([k for k in coordinates if k.startswith('recorded_')]),len(self.example['atoms']))


if __name__=='__main__':unittest.main()
