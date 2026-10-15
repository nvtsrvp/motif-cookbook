"""Small unit tests for mine.py, using synthetic coordinates rather than proteins.

Tests named ``current_limitation`` document bugs that remain unfixed in this
review. They describe today's behavior, not a recommended scientific policy.
"""
import argparse
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from Bio.PDB import Atom, Chain, Model, Residue, Structure

from hal_discovery.mine import (
    RES_TYPES,
    _get_hx_resname,
    euclidean_distance,
    extract_his_coordinates,
    find_closest_residues,
    find_his_pairs,
    run_mine,
)


def add_residue(model, number, name, atoms, chain_id="A", insertion_code=" "):
    """Build one tiny residue; atoms maps atom names to (x, y, z)."""
    if chain_id not in model:
        model.add(Chain.Chain(chain_id))
    residue = Residue.Residue((" ", number, insertion_code), name, " ")
    for serial, (atom_name, position) in enumerate(atoms.items(), start=1):
        atom = Atom.Atom(
            atom_name, np.array(position, dtype=float), 0.0, 1.0,
            " ", atom_name.rjust(4), serial, element=atom_name[0],
        )
        residue.add(atom)
    model[chain_id].add(residue)
    return residue


def one_model_structure():
    structure = Structure.Structure("synthetic")
    model = Model.Model(0)
    structure.add(model)
    return structure, model


class DistanceTests(unittest.TestCase):
    def test_distance_between_known_points(self):
        self.assertEqual(euclidean_distance((0, 0, 0), (3, 4, 0)), 5)

    def test_distance_to_same_point_is_zero(self):
        self.assertEqual(euclidean_distance((2, -3, 4), (2, -3, 4)), 0)


class HisPairTests(unittest.TestCase):
    def setUp(self):
        self.first = ("A", 10)
        self.second = ("A", 20)
        self.ne2 = {self.first: (0, 0, 0), self.second: (3, 0, 0)}
        self.oxygen = {self.first: (0, 0, 0), self.second: (0, 0, 0)}
        self.nitrogen = {self.first: (0, 1, 0), self.second: (0, 1, 0)}

    def test_pair_passes_when_all_three_distances_are_below_cutoffs(self):
        pairs = find_his_pairs(self.ne2, self.oxygen, self.nitrogen)
        self.assertEqual(pairs, [(self.first, self.second)])

    def test_ne2_distance_equal_to_cutoff_does_not_pass(self):
        self.ne2[self.second] = (4, 0, 0)
        self.assertEqual(find_his_pairs(self.ne2, self.oxygen, self.nitrogen), [])

    def test_either_backbone_distance_equal_to_cutoff_does_not_pass(self):
        for atom in (self.first, self.second):
            with self.subTest(atom=atom):
                nitrogen = dict(self.nitrogen)
                nitrogen[atom] = (0, 4, 0)
                self.assertEqual(find_his_pairs(self.ne2, self.oxygen, nitrogen), [])

    def test_missing_any_required_atom_prevents_pairing(self):
        for atom_group in ("ne2", "oxygen", "nitrogen"):
            with self.subTest(atom_group=atom_group):
                groups = [dict(self.ne2), dict(self.oxygen), dict(self.nitrogen)]
                groups[("ne2", "oxygen", "nitrogen").index(atom_group)].pop(self.second)
                self.assertEqual(find_his_pairs(*groups), [])

    def test_sidechain_and_backbone_cutoffs_are_independent(self):
        self.assertEqual(find_his_pairs(self.ne2, self.oxygen, self.nitrogen,
                                       cutoff_ne2=2, cutoff_hbond=4), [])
        self.assertEqual(find_his_pairs(self.ne2, self.oxygen, self.nitrogen,
                                       cutoff_ne2=4, cutoff_hbond=0.5), [])

    def test_current_limitation_nan_ne2_distance_can_pass(self):
        # NaN >= cutoff is False, so the early rejection is bypassed.
        self.ne2[self.second] = (float("nan"), 0, 0)
        self.assertEqual(find_his_pairs(self.ne2, self.oxygen, self.nitrogen),
                         [(self.first, self.second)])


class CoordinateAndNumberingTests(unittest.TestCase):
    def test_only_histidines_are_extracted_and_chains_stay_distinct(self):
        structure, model = one_model_structure()
        add_residue(model, 10, "HIS", {"NE2": (1, 0, 0)})
        add_residue(model, 10, "HIS", {"NE2": (2, 0, 0)}, chain_id="B")
        add_residue(model, 30, "ALA", {"N": (3, 0, 0)})
        ne2, oxygen, nitrogen = extract_his_coordinates(structure)
        self.assertEqual(set(ne2), {("A", 10), ("B", 10)})
        self.assertEqual(ne2[("A", 10)].tolist(), [1, 0, 0])
        self.assertEqual(ne2[("B", 10)].tolist(), [2, 0, 0])
        self.assertEqual(oxygen, {})
        self.assertEqual(nitrogen, {})

    def test_current_limitation_incomplete_models_can_be_combined(self):
        structure, first_model = one_model_structure()
        add_residue(first_model, 10, "HIS", {"NE2": (0, 0, 0)})
        second_model = Model.Model(1)
        structure.add(second_model)
        add_residue(second_model, 10, "HIS", {"N": (1, 0, 0), "O": (2, 0, 0)})
        ne2, oxygen, nitrogen = extract_his_coordinates(structure)
        # Neither model has a complete His, but the dictionaries suggest one.
        common_keys = set(ne2) & set(oxygen) & set(nitrogen)
        self.assertEqual(common_keys, {("A", 10)})

    def test_current_limitation_insertion_codes_overwrite_same_number(self):
        structure, model = one_model_structure()
        add_residue(model, 10, "HIS", {"NE2": (1, 0, 0)})
        add_residue(model, 10, "HIS", {"NE2": (2, 0, 0)}, insertion_code="A")
        ne2, _, _ = extract_his_coordinates(structure)
        self.assertEqual(len(ne2), 1)
        self.assertEqual(ne2[("A", 10)].tolist(), [2, 0, 0])

    def test_hx_uses_author_number_plus_offset_in_the_same_chain(self):
        structure, model = one_model_structure()
        add_residue(model, 12, "GLY", {})
        add_residue(model, 12, "ALA", {}, chain_id="B")
        self.assertEqual(_get_hx_resname(structure, "A", 10, 2), "GLY")
        self.assertEqual(_get_hx_resname(structure, "B", 10, 2), "ALA")
        self.assertEqual(_get_hx_resname(structure, "A", 10, 3), "NA")


class NeighborhoodTests(unittest.TestCase):
    def test_minima_reset_for_each_site_and_matched_histidines_are_excluded(self):
        structure, model = one_model_structure()
        add_residue(model, 10, "HIS", {"NE2": (0, 0, 0)})
        add_residue(model, 20, "HIS", {"NE2": (2, 0, 0)})
        add_residue(model, 30, "HIS", {"NE2": (10, 0, 0)})
        add_residue(model, 40, "HIS", {"NE2": (12, 0, 0)})
        add_residue(model, 50, "CYS", {"SG": (1, 0, 0)})
        ne2, _, _ = extract_his_coordinates(structure)
        pairs = [(("A", 10), ("A", 20)), (("A", 30), ("A", 40))]

        rows = find_closest_residues(structure, pairs, ne2, 5, "synthetic.pdb", 2)

        self.assertEqual(rows[0]["closest_CYS"], 0)
        self.assertEqual(rows[1]["closest_CYS"], 10)
        self.assertEqual(rows[0]["closest_HIS"], 9)
        self.assertIsNone(rows[0]["closest_ASP"])

    def test_minimum_uses_configured_atoms_and_includes_other_chains(self):
        structure, model = one_model_structure()
        add_residue(model, 30, "CYS", {"CA": (1, 0, 0), "SG": (5, 0, 0)})
        add_residue(model, 40, "CYS", {"SG": (2, 0, 0)}, chain_id="B")
        pair = (("A", 10), ("A", 20))
        ne2 = {pair[0]: (-1, 0, 0), pair[1]: (1, 0, 0)}

        row = find_closest_residues(structure, [pair], ne2, 2, "synthetic.pdb", 2)[0]

        # CYS CA is not a configured key atom; B40 SG supplies the minimum.
        self.assertEqual(row["closest_CYS"], 2)


class FilterOutputTests(unittest.TestCase):
    """Exercise the real pandas filters with one controlled site row."""

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.inputs = self.root / "inputs"
        self.inputs.mkdir()
        (self.inputs / "example.pdb").touch()
        self.args = argparse.Namespace(
            pdb_dir=str(self.inputs), outdir=str(self.root / "outputs"),
            his_ne2_cutoff=4, his_hbond_cutoff=4, hx_offset=2,
            asp_glu_cutoff=5, triad_min_length=0, min_length=200,
            acid_exclude_cutoff=5.5, exclude_ligand_cutoff=4,
            ala_gly_presence_cutoff=7, require_hx=True, hx_allowed=["ALA", "GLY"],
            progress_every=200, max_pdbs=None, detailed_results=False,
        )
        self.row = {f"closest_{residue}": 10.0 for residue in RES_TYPES}
        self.row.update(structure="example", chainA="A", hisA=10,
                        chainB="A", hisB=20, protein_length=201,
                        HXres="ALA", closest_ALA=6.0)

    def run_one_site(self):
        # Parsing is mocked; run_mine still executes its actual filter expressions.
        with patch("hal_discovery.mine.main_function", return_value=[self.row]):
            with contextlib.redirect_stdout(io.StringIO()):
                run_mine(self.args)
        output = Path(self.args.outdir)
        return json.loads(next(output.glob("summary_*.json")).read_text())

    def test_halogenase_passes_when_all_required_rules_pass(self):
        self.assertEqual(self.run_one_site()["sites_putative_halogenase"], 1)

    def test_halogenase_length_equal_to_threshold_fails(self):
        self.row["protein_length"] = 200
        self.assertEqual(self.run_one_site()["sites_putative_halogenase"], 0)

    def test_facial_acid_distance_equal_to_threshold_passes(self):
        self.row["closest_ASP"] = 5.0
        summary = self.run_one_site()
        self.assertEqual(summary["sites_2his_aspglu"], 1)
        self.assertEqual(summary["sites_putative_halogenase"], 0)

    def test_missing_acid_distance_excludes_halogenase_in_legacy_filter(self):
        self.row["closest_ASP"] = None
        self.assertEqual(self.run_one_site()["sites_putative_halogenase"], 0)

    def test_hx_rule_can_be_disabled(self):
        self.row["HXres"] = "SER"
        self.assertEqual(self.run_one_site()["sites_putative_halogenase"], 0)
        self.args.require_hx = False
        self.assertEqual(self.run_one_site()["sites_putative_halogenase"], 1)

    def test_current_limitation_processing_errors_are_hidden_without_details(self):
        with patch("hal_discovery.mine.main_function", side_effect=ValueError("bad coordinates")):
            with contextlib.redirect_stdout(io.StringIO()):
                run_mine(self.args)
        output = Path(self.args.outdir)
        summary = json.loads(next(output.glob("summary_*.json")).read_text())
        self.assertEqual(summary["total_structures"], 1)
        self.assertEqual(summary["total_2his_sites"], 0)
        self.assertEqual(Path(summary["no_pairs_txt"]).read_text(), "")
        self.assertEqual(list(output.glob("processing_errors_*.json")), [])

    def test_current_limitation_missing_input_directory_looks_like_empty_run(self):
        self.args.pdb_dir = str(self.root / "does-not-exist")
        with contextlib.redirect_stdout(io.StringIO()):
            run_mine(self.args)
        output = Path(self.args.outdir)
        summary = json.loads(next(output.glob("summary_*.json")).read_text())
        self.assertEqual(summary["total_structures"], 0)
        # An empty raw CSV has no header and cannot be read as a table.
        with self.assertRaises(pd.errors.EmptyDataError):
            pd.read_csv(summary["raw_sites_csv"])


if __name__ == "__main__":
    unittest.main()
