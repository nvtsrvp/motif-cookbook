"""PDB/mmCIF identity and connectivity for the observed-copper workflow.

The legacy miner deliberately keeps its own parser and conformer policy.
"""
from dataclasses import dataclass, field
from math import dist, isfinite
from pathlib import Path
import warnings

from Bio.PDB import MMCIFParser, PDBParser
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
from Bio.PDB.Polypeptide import is_aa

from .results import atom_record


def clean(value):
    return None if value in (None, ".", "?") else value


def atom_identity(atom):
    residue = atom.get_parent()
    chain = residue.get_parent()
    return (chain.get_parent().serial_num, chain.id, residue.id[1],
            residue.id[2].strip(), residue.resname, atom.name, atom.get_altloc().strip())


def unpacked_atoms(model):
    """Enumerate atom alternate locations; do not mix residue microheterogeneity."""
    for residue in model.get_residues():
        if residue.is_disordered() == 2:
            continue
        yield from residue.get_unpacked_list()


def compatible(*atoms):
    return len({a.get_altloc().strip() for a in atoms if a.get_altloc().strip()}) <= 1


@dataclass
class StructureInput:
    path: Path
    structure: object
    format: str
    metadata: dict = field(default_factory=dict)
    sequence: dict = field(default_factory=dict)
    disulfides: set = field(default_factory=set)
    warnings: list = field(default_factory=list)

    def record(self, atom):
        result = atom_record(atom)
        # Invalid atoms can appear in diagnostics; keep JSON valid rather than
        # serializing NaN as if it were a reproducible coordinate/measurement.
        result["coordinates"] = [v if isfinite(v) else None for v in result["coordinates"]]
        for name in ("occupancy", "b_factor"):
            if result[name] is not None and not isfinite(result[name]):
                result[name] = None
        identity = atom_identity(atom)
        if self.format == "mmcif" and identity not in self.metadata:
            raise ValueError(f"Cannot map parsed atom to deposited mmCIF identifiers: {identity}")
        result.update(self.metadata.get(identity, {}))
        return result

    def predecessor(self, his, conformer):
        """Use label sequence adjacency or a coordinate-supported peptide link.

        No author-number subtraction: renumbering/insertion codes do not define
        sequence adjacency. A gap with no usable sequence/link is not_evaluable.
        """
        chain = his.get_parent()
        first_atom = next(iter(his.get_unpacked_list()), None)
        meta = self.record(first_atom) if first_atom is not None else {}
        label_seq = meta.get("label_seq_id")
        if label_seq is not None:
            expected = int(label_seq) - 1
            previous = None
            for residue in chain:
                if residue.is_disordered() == 2:
                    continue
                atom = next(iter(residue.get_unpacked_list()), None)
                if atom is None:
                    continue
                row = self.record(atom)
                if (row.get("label_asym_id") == meta.get("label_asym_id")
                        and row.get("label_seq_id") == str(expected)):
                    previous = residue
                    break
            name = self.sequence.get((meta.get("label_entity_id"), str(expected)))
            if previous is not None:
                name = previous.resname
            status = "not_evaluable" if name is None else ("pass" if name == "CYS" else "fail")
            return previous, {"status": status, "resname": name,
                              "method": "mmCIF label_seq_id adjacency", "label_seq_id": expected,
                              "coordinates_present": previous is not None}

        residues = [r for r in chain if r.id[0] == " " and is_aa(r, standard=True)]
        try:
            index = residues.index(his)
        except ValueError:
            index = 0
        if index == 0:
            return None, {"status": "not_evaluable", "method": "peptide C-N connectivity",
                          "reason": "No preceding coordinate residue with verified connectivity"}
        previous = residues[index - 1]
        carbons = [a for a in previous.get_unpacked_list() if a.name == "C"]
        nitrogens = [a for a in his.get_unpacked_list() if a.name == "N"]
        connected = any(compatible(c, n, conformer)
                        and (c.occupancy is None or c.occupancy > 0)
                        and (n.occupancy is None or n.occupancy > 0)
                        and 1.0 <= dist(c.coord, n.coord) <= 1.8
                        for c in carbons for n in nitrogens)
        if not connected:
            return None, {"status": "not_evaluable", "method": "peptide C-N connectivity",
                          "reason": "Missing atoms, gap, or no compatible 1.0-1.8 A peptide link"}
        return previous, {"status": "pass" if previous.resname == "CYS" else "fail",
                          "resname": previous.resname, "method": "peptide C-N connectivity",
                          "coordinates_present": True}

    def disulfide_annotation(self, first, second):
        def endpoint(atom):
            r = atom.get_parent()
            return (r.get_parent().id, str(r.id[1]), r.id[2].strip())
        return frozenset((endpoint(first), endpoint(second))) in self.disulfides


def load_structure(path):
    path = Path(path).expanduser().resolve()
    if path.suffix.lower() not in {".pdb", ".cif", ".mmcif"}:
        raise ValueError("Expected a .pdb, .cif or .mmcif structure")
    cif = path.suffix.lower() != ".pdb"
    parser = MMCIFParser(QUIET=False, auth_chains=True, auth_residues=True) if cif else PDBParser(PERMISSIVE=False, QUIET=False)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        structure = parser.get_structure(path.stem, str(path))
    result = StructureInput(path, structure, "mmcif" if cif else "pdb",
                            warnings=[str(w.message) for w in caught])
    if not cif:
        for line in path.read_text().splitlines():
            if line.startswith("SSBOND"):
                if any(line[start:end].strip() not in {"", "1555"} for start, end in ((59, 65), (66, 72))):
                    continue
                a = (line[15:16], line[17:21].strip(), line[21:22].strip())
                b = (line[29:30], line[31:35].strip(), line[35:36].strip())
                result.disulfides.add(frozenset((a, b)))
        return result

    data = MMCIF2Dict(str(path))
    count = len(data.get("_atom_site.id", []))
    def value(name, index, default=None):
        values = data.get(name)
        return default if values is None else clean(values[index])
    for i in range(count):
        key = (int(value("_atom_site.pdbx_PDB_model_num", i, "1")),
               value("_atom_site.auth_asym_id", i), int(value("_atom_site.auth_seq_id", i)),
               value("_atom_site.pdbx_PDB_ins_code", i) or "",
               value("_atom_site.label_comp_id", i), value("_atom_site.label_atom_id", i),
               value("_atom_site.label_alt_id", i) or "")
        if key in result.metadata:
            raise ValueError(f"Ambiguous mmCIF author atom identity: {key}")
        result.metadata[key] = {
            "atom_site_id": value("_atom_site.id", i),
            "label_asym_id": value("_atom_site.label_asym_id", i),
            "label_seq_id": value("_atom_site.label_seq_id", i),
            "label_entity_id": value("_atom_site.label_entity_id", i),
            "label_atom_id": value("_atom_site.label_atom_id", i),
            "label_comp_id": value("_atom_site.label_comp_id", i),
            "auth_atom_id": value("_atom_site.auth_atom_id", i),
            "auth_comp_id": value("_atom_site.auth_comp_id", i),
        }
    for i in range(len(data.get("_entity_poly_seq.num", []))):
        result.sequence[(value("_entity_poly_seq.entity_id", i), value("_entity_poly_seq.num", i))] = value("_entity_poly_seq.mon_id", i)
    for i, kind in enumerate(data.get("_struct_conn.conn_type_id", [])):
        if kind != "disulf":
            continue
        ends = []
        for p in (1, 2):
            # Symmetry-generated partners are not atoms in this coordinate model.
            if value(f"_struct_conn.ptnr{p}_symmetry", i) not in (None, "1_555"):
                break
            ends.append((value(f"_struct_conn.ptnr{p}_auth_asym_id", i),
                         value(f"_struct_conn.ptnr{p}_auth_seq_id", i),
                         value(f"_struct_conn.pdbx_ptnr{p}_PDB_ins_code", i) or ""))
        if len(ends) == 2:
            result.disulfides.add(frozenset(ends))
    return result
