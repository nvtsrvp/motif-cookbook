"""Explainable records captured by the legacy miner, without changing its rules."""
import hashlib
import json
from pathlib import Path

from .mine import euclidean_distance

SCHEMA_VERSION = "1.0"
FILTER_VERSION = "legacy-0.1.0"
PARAMETERS = (
    "his_ne2_cutoff", "his_hbond_cutoff", "asp_glu_cutoff", "ala_gly_cutoff",
    "require_hx", "hx_offset", "min_length", "acid_exclude_cutoff",
    "exclude_ligand_cutoff", "ala_gly_presence_cutoff", "hx_allowed", "triad_min_length",
)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atom_record(atom):
    # Biopython disordered wrappers delegate to the selected conformer.
    residue = atom.get_parent()
    chain = residue.get_parent()
    model = chain.get_parent()
    return {
        "model_id": model.id, "model_number": model.serial_num,
        "auth_chain_id": chain.id, "label_asym_id": None,
        "resname": residue.resname, "hetero_flag": residue.id[0],
        "auth_seq_id": residue.id[1], "insertion_code": residue.id[2].strip(),
        "label_seq_id": None, "atom_name": atom.name,
        "altloc": atom.get_altloc().strip(), "element": atom.element,
        "serial_number": atom.get_serial_number(),
        "coordinates": [float(v) for v in atom.coord],
        "occupancy": atom.occupancy, "b_factor": atom.bfactor,
    }


def capture_site(structure, pair, midpoint, witnesses, row, pdb_path):
    """Retain actual minimum-distance witnesses during evaluation, not after it."""
    from .mine import KEY_ATOMS
    atoms, roles, measurements, warnings = {}, {}, {}, []
    missing_atoms = []
    residues = {}
    ambiguous = len(structure) != 1
    for residue in structure.get_residues():
        key = (residue.get_parent().id, residue.id[1])
        if key in residues:
            ambiguous = True
        residues[key] = residue
        # GLN OE2 is a legacy optional key, not a standard glutamine atom.
        missing = [n for n in KEY_ATOMS.get(residue.resname, [])
                   if n not in residue and not (residue.resname == "GLN" and n == "OE2")]
        if missing:
            missing_atoms.append({"model_id": residue.get_parent().get_parent().id,
                                  "auth_chain_id": key[0], "auth_seq_id": key[1],
                                  "insertion_code": residue.id[2].strip(),
                                  "resname": residue.resname, "atom_names": missing})
    if ambiguous:
        warnings.append("Legacy chain/number keys are ambiguous across models or residue identities; exact export is disabled.")
    if missing_atoms:
        warnings.append("Some legacy key atoms are missing; available-coordinate minima do not establish a complete neighborhood.")

    def retain(atom, role):
        ref = role
        atoms[ref] = atom_record(atom)
        roles[role] = ref
        return ref

    for i, key in enumerate(pair, 1):
        residue = residues[key]
        for name in ("NE2", "N", "O"):
            if name in residue:
                retain(residue[name], f"his{i}_{name}")
            else:
                warnings.append(f"Missing His {key} {name}; legacy keys may have combined incomplete residues.")
                ambiguous = True
    for name, ends in {
        "his_ne2_distance": ["his1_NE2", "his2_NE2"],
        "backbone_o1_n2": ["his1_O", "his2_N"],
        "backbone_n1_o2": ["his1_N", "his2_O"],
    }.items():
        value = None if not all(e in atoms for e in ends) else float(euclidean_distance(
            atoms[ends[0]]["coordinates"], atoms[ends[1]]["coordinates"]))
        measurements[name] = {"endpoints": ends, "value": value, "units": "angstrom", "method": "euclidean"}
    for resname, atom in witnesses.items():
        ref = retain(atom, f"nearest_{resname}")
        measurements[f"closest_{resname}"] = {
            "endpoints": ["midpoint", ref], "value": row[f"closest_{resname}"],
            "units": "angstrom", "method": "minimum_over_legacy_key_atoms",
        }
    checksum = sha256(pdb_path)
    # Parameters are included in the final ID by write_results.
    return {
        "schema_version": SCHEMA_VERSION,
        "source": {"path": str(Path(pdb_path).resolve()), "filename": Path(pdb_path).name,
                   "sha256": checksum, "format": "pdb", "structure_id": row["structure"],
                   "identifier_policy": "PDB author identifiers; mmCIF input is not supported by the legacy miner"},
        "legacy_row": row.copy(), "atoms": atoms, "roles": roles,
        "points": {"midpoint": {"coordinates": [float(v) for v in midpoint],
                    "kind": "computed_geometric_reference", "label": "COMPUTED GEOMETRIC REFERENCE POINT",
                    "method": "arithmetic midpoint of matched His NE2 atoms", "endpoints": ["his1_NE2", "his2_NE2"]}},
        "measurements": measurements,
        "metals": {"observed": [atom_record(a) for a in structure.get_atoms()
                   if a.element.upper() in {"CU", "FE", "ZN", "MN", "CO", "NI", "MG", "CA", "CD", "NA", "K"}],
                   "hypothetical": [], "interpretation": "Deposited atoms only; midpoint is not a metal prediction."},
        "missing_atoms": missing_atoms, "warnings": warnings, "exportable": not ambiguous,
    }


def explain_site(record, args):
    from .mine import KEY_ATOMS
    params = {name: getattr(args, name) for name in PARAMETERS}
    record["filter"] = {"name": "legacy_2his", "version": FILTER_VERSION,
                        "package_version": "0.1.0", "parameters": params, "key_atoms": KEY_ATOMS,
                        "implementation_sha256": {name: sha256(Path(__file__).with_name(name))
                                                  for name in ("mine.py", "results.py", "cli.py")},
                        "altloc_policy": "Biopython selected conformer (legacy behavior)",
                        "neighborhood_scope": "all chains and models; matched chain/number keys excluded",
                        "unused_parameters": ["ala_gly_cutoff"]}
    identity = [record["source"]["sha256"], record["legacy_row"]["chainA"],
                record["legacy_row"]["hisA"], record["legacy_row"]["chainB"],
                record["legacy_row"]["hisB"], FILTER_VERSION, params]
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
    record["site_id"] = f"{record['source']['structure_id']}-{digest}"
    row = record["legacy_row"]
    criteria = []

    def criterion(name, group, value, operator, threshold, measurement=None):
        if value is None or value == "NA":
            status = "not_evaluable"
        else:
            passed = {"<": lambda: value < threshold, "<=": lambda: value <= threshold,
                      ">": lambda: value > threshold, "in": lambda: value in threshold}[operator]()
            status = "pass" if passed else "fail"
        item = {"name": name, "group": group, "value": value, "operator": operator,
                "threshold": threshold, "status": status, "measurement": measurement,
                "explanation": f"{name}: {value!r} {operator} {threshold!r}; {status}"}
        criteria.append(item)
        return status

    for name, limit in [("his_ne2_distance", args.his_ne2_cutoff),
                        ("backbone_o1_n2", args.his_hbond_cutoff),
                        ("backbone_n1_o2", args.his_hbond_cutoff)]:
        criterion(name, "geometry", record["measurements"][name]["value"], "<", limit, name)
    facial = [criterion(f"facial_{r}", "facial_acid_any", row[f"closest_{r}"], "<=",
                        args.asp_glu_cutoff, f"closest_{r}") for r in ("ASP", "GLU")]
    facial_length = criterion("facial_length", "facial_length", row["protein_length"], ">", args.triad_min_length)
    exclusions = []
    for r in ("ASP", "GLU", "ASN", "GLN", "CYS", "MET", "HIS", "TRP", "TYR"):
        cutoff = args.acid_exclude_cutoff if r in ("ASP", "GLU") else args.exclude_ligand_cutoff
        exclusions.append(criterion(f"exclude_{r}", "halogenase_all", row[f"closest_{r}"], ">", cutoff, f"closest_{r}"))
    presence = [criterion(f"presence_{r}", "ala_gly_any", row[f"closest_{r}"], "<",
                          args.ala_gly_presence_cutoff, f"closest_{r}") for r in ("ALA", "GLY")]
    length = criterion("halogenase_length", "halogenase_all", row["protein_length"], ">", args.min_length)
    hx = "pass"
    if args.require_hx:
        hx = criterion("HX_identity", "halogenase_all", row["HXres"], "in", args.hx_allowed)
    # Recover the exact legacy HX target, including its atoms, for display.
    record["hx"] = {"auth_chain_id": row["chainA"], "auth_seq_id": row["hisA"] + args.hx_offset,
                    "insertion_code": "", "resname": row["HXres"],
                    "method": "legacy author residue number + offset, not sequence adjacency"}
    record["criteria"] = criteria
    record["classifications"] = {
        "two_his": True,
        "facial_triad": "pass" in facial and facial_length == "pass",
        "putative_halogenase": all(s == "pass" for s in exclusions + [length, hx]) and "pass" in presence,
    }
    record["criterion_groups"] = {"geometry": "all", "facial_acid_any": "any",
                                  "facial_length": "all", "halogenase_all": "all", "ala_gly_any": "any"}
    if any(c["status"] == "not_evaluable" for c in criteria):
        record["warnings"].append("Unavailable evidence is not_evaluable; legacy classification still treats missing comparisons as false.")
    record["warnings"].append("Legacy minima describe available coordinates, not completeness of the site or demonstrated enzyme function.")
    return record


def write_results(records, args, path):
    with Path(path).open("w") as handle:
        for record in records:
            handle.write(json.dumps(explain_site(record, args), allow_nan=False, sort_keys=True) + "\n")
