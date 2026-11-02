import json
import datetime as dt
from math import dist as euclidean_distance, dist, acos, degrees, isfinite
from dataclasses import asdict, dataclass
from itertools import combinations, product
import hashlib
import math
from pathlib import Path
from typing import Dict, Tuple, List, Any

import pandas as pd
from Bio.PDB import PDBParser
from Bio.PDB.PDBExceptions import PDBException, PDBConstructionException


# ----------------------- HIS pairing (notebook-like) -----------------------

def extract_his_coordinates(structure):
    """
    Extract coordinates for HIS residues:
      - NE2 (side chain)
      - N and O (backbone)
    Keys are (model_id, chain_id, (hetero_flag, resseq, insertion_code)).
    The hetero flag distinguishes ordinary protein residues (`" "`) from water (`"W"`) and other heteroresidues, such as a ligand (`"H_CU"` for a copper component).
    """
    his_ne2 = {}
    his_o = {}
    his_n = {}

    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.get_resname() == "HIS":
                    key = (model.id, chain.id, residue.id)
                    # direct lookup is faster/cleaner than looping atoms
                    if "NE2" in residue:
                        his_ne2[key] = residue["NE2"].get_coord()
                    if "O" in residue:
                        his_o[key] = residue["O"].get_coord()
                    if "N" in residue:
                        his_n[key] = residue["N"].get_coord()

    return his_ne2, his_o, his_n


def find_his_pairs(his_ne2, his_o, his_n, cutoff_ne2=4.0, cutoff_hbond=4.0):
    """
    Find pairs of HIS residues using your original geometric criteria.
    Pairs contain full residue keys and never cross models. Different chains
    in the same model remain eligible, preserving the original chain policy.
    Requires all needed atoms present for both residues.
    """
    keys = sorted(set(his_ne2.keys()) & set(his_o.keys()) & set(his_n.keys()))
    pairs = []

    for idx_i in range(len(keys)):
        for idx_j in range(idx_i + 1, len(keys)):
            i = keys[idx_i]
            j = keys[idx_j]
            if i[0] != j[0]:
                continue

            dist1 = euclidean_distance(his_ne2[i], his_ne2[j])
            if dist1 >= cutoff_ne2:
                continue

            dist2 = euclidean_distance(his_o[i], his_n[j])
            dist3 = euclidean_distance(his_n[i], his_o[j])

            if dist2 < cutoff_hbond and dist3 < cutoff_hbond:
                pairs.append((i, j))

    return pairs


# ----------------------- closest residue distances (notebook-like) -----------------------

KEY_ATOMS = {
    "ASN": ["ND2", "OD1"],
    "GLN": ["NE2", "OE1", "OE2"],
    "CYS": ["SG"],
    "HIS": ["NE2", "ND1"],
    "MET": ["SD"],
    "TYR": ["OH"],
    "TRP": ["NE1"],
    "ASP": ["OD1", "OD2", "CB", "CG"],
    "GLU": ["OE1", "OE2", "CB", "CG", "CD"],
    "ALA": ["CB"],
    "GLY": ["CA"],
    "PHE": ["CD1", "CD2", "CG", "CE1", "CE2", "CZ"],
    "ILE": ["CG1", "CG2", "CD1"],
    "LEU": ["CG", "CD1", "CD2"],
    "VAL": ["CG1", "CG2"],
    "LYS": ["CG", "CD", "CE", "NZ"],
    "PRO": ["CB", "CG", "CD"],
    "SER": ["OG"],
    "THR": ["OG1", "CG2"],
    "ARG": ["CZ", "NE", "NH1", "NH2"],
}
RES_TYPES = list(KEY_ATOMS.keys())


def _get_hx_resname(structure, model_id: int, chain_id: str, resseq: int, hx_offset: int) -> str:
    """
    Notebook-like HX: author number + offset, blank insertion code, in the
    selected model and chain. This deliberately does not use sequence adjacency.
    Returns 'NA' if missing.
    """
    target = (" ", resseq + hx_offset, " ")
    try:
        return structure[model_id][chain_id][target].get_resname()
    except KeyError:
        return "NA"


def find_closest_residues(structure, his_pairs, his_ne2, protein_length, pdb_path, hx_offset: int, details=None):
    """
    For each HIS pair:
      - compute midpoint between NE2 atoms
      - compute closest distances from midpoint to key atoms of candidate residue types
      - determine HX residue (two residues after first HIS) when possible
    Returns a list of row dicts (one per 2-His site).
    """
    structure_id = Path(pdb_path).stem
    rows = []

    for pair in his_pairs:
        (model_id, chainA, residueA), (other_model_id, chainB, residueB) = pair
        if model_id != other_model_id:
            raise ValueError("A His pair cannot span different models")
        model = structure[model_id]
        resiA, resiB = residueA[1], residueB[1]

        midpoint = (
            (his_ne2[pair[0]][0] + his_ne2[pair[1]][0]) / 2,
            (his_ne2[pair[0]][1] + his_ne2[pair[1]][1]) / 2,
            (his_ne2[pair[0]][2] + his_ne2[pair[1]][2]) / 2,
        )

        # IMPORTANT: reset per pair (your earlier bug cause)
        closest = {res: float("inf") for res in RES_TYPES}
        witnesses = {}

        # HX residue (optional use later; we still compute it)
        hx_res = _get_hx_resname(structure, model_id, chainA, resiA, hx_offset=hx_offset)

        for chain in model:
            for residue in chain:
                if (model.id, chain.id, residue.id) in pair:
                    continue

                resname = residue.get_resname()
                if resname not in KEY_ATOMS:
                    continue

                for atom_name in KEY_ATOMS[resname]:
                    if atom_name in residue:
                        d = euclidean_distance(midpoint, residue[atom_name].get_coord())
                        if d < closest[resname]:
                            closest[resname] = d
                            witnesses[resname] = residue[atom_name]

        # convert inf -> blank for writing CSV, but keep numeric for filtering
        row = {
            "structure": structure_id,
            "chainA": chainA,
            "hisA": resiA,
            "chainB": chainB,
            "hisB": resiB,
            # Preserve the legacy residue-count definition, but never count
            # alternative models as extra residues of the selected model.
            "protein_length": protein_length if len(structure) == 1 else len(list(model.get_residues())),
            "HXres": hx_res,
        }
        for res in RES_TYPES:
            row[f"closest_{res}"] = None if closest[res] == float("inf") else float(closest[res])

        # Additive CSV columns retain all existing names and values while making
        # sites with identical author numbers distinguishable.
        row.update(model_id=model_id, model_number=model.serial_num,
                   hisA_insertion_code=residueA[2].strip(), hisB_insertion_code=residueB[2].strip(),
                   hisA_hetero_flag=residueA[0], hisB_hetero_flag=residueB[0])

        rows.append(row)
        if details is not None:
            from .results import capture_site
            details.append(capture_site(structure, pair, midpoint, witnesses, row, pdb_path))

    return rows


def main_function(pdb_path: str, cutoff_ne2: float, cutoff_hbond: float, hx_offset: int, details=None, diagnostics=None) -> List[Dict[str, Any]]:
    """Parse PDB, find His pairs, compute distances and context features."""
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("pdb", pdb_path)
    if diagnostics is not None:
        for residue in structure.get_residues():
            if residue.get_resname() == "HIS":
                missing = [name for name in ("NE2", "N", "O") if name not in residue]
                if missing:
                    diagnostics.append({"source": str(pdb_path), "status": "not_evaluable",
                                        "reason": "missing_required_his_pair_atoms", "missing_atoms": missing,
                                        "model_id": residue.get_parent().get_parent().id,
                                        "chain": residue.get_parent().id, "residue_id": list(residue.id)})

    protein_length = len(list(structure.get_residues()))

    his_ne2, his_o, his_n = extract_his_coordinates(structure)
    his_pairs = find_his_pairs(his_ne2, his_o, his_n, cutoff_ne2=cutoff_ne2, cutoff_hbond=cutoff_hbond)

    if not his_pairs:
        return []

    return find_closest_residues(structure, his_pairs, his_ne2, protein_length, pdb_path, hx_offset=hx_offset, details=details)


# ----------------------- filtering (notebook-like but parameterized) -----------------------

def _coalesce_numeric(s):
    return pd.to_numeric(s, errors="coerce")


def run_mine(args) -> None:
    """
    hal-discovery mine
      Input: folder of PDB files (named <UniProt>.pdb)
      Output: 2-His site table + filtered hit tables + summary
    """
    if getattr(args, "filters", None) is not None:
        return run_selected_filters(args)
    if getattr(args, "inward_score_min", None) is not None:
        raise ValueError("--inward-score-min requires --filters potential_cu_2his")
    pdb_dir = Path(args.pdb_dir).expanduser().resolve()
    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")

    pdb_files = sorted(pdb_dir.glob("*.pdb"))
    if args.max_pdbs is not None:
        pdb_files = pdb_files[: args.max_pdbs]

    print(f"Scanning {len(pdb_files)} PDB files in: {pdb_dir}")

    all_rows: List[Dict[str, Any]] = []
    no_pairs: List[str] = []
    detailed = [] if getattr(args, "detailed_results", False) else None
    failures = []

    for i, pdb_path in enumerate(pdb_files, start=1):
        if i == 1 or i % args.progress_every == 0 or i == len(pdb_files):
            print(f"[{i}/{len(pdb_files)}]")

        try:
            file_details = [] if detailed is not None else None
            rows = main_function(
                str(pdb_path),
                cutoff_ne2=args.his_ne2_cutoff,
                cutoff_hbond=args.his_hbond_cutoff,
                hx_offset=args.hx_offset,
                details=file_details,
                diagnostics=failures if detailed is not None else None,
            )
        except Exception as exc:
            # if a PDB is malformed, skip it
            failures.append({"source": str(pdb_path), "error": str(exc), "status": "parse_or_processing_error"})
            continue

        if detailed is not None:
            detailed.extend(file_details)

        if not rows:
            no_pairs.append(pdb_path.stem)
            continue

        all_rows.extend(rows)

    # Write raw 2-His site table
    df_2his = pd.DataFrame(all_rows)
    out_2his = outdir / f"2his_sites_all_{ts}.csv"
    df_2his.to_csv(out_2his, index=False)

    out_none = outdir / f"no_pairs_{ts}.txt"
    out_none.write_text("\n".join(no_pairs) + ("\n" if no_pairs else ""))

    extra_summary = {}
    if detailed is not None:
        from .results import write_results
        out_details = outdir / f"site_matches_{ts}.jsonl"
        write_results(detailed, args, out_details)
        out_failures = outdir / f"processing_errors_{ts}.json"
        out_failures.write_text(json.dumps(failures, indent=2) + "\n")
        extra_summary = {"detailed_sites_jsonl": str(out_details), "processing_errors_json": str(out_failures)}

    # If empty, still write summary and exit
    if df_2his.empty:
        summary = {
            **extra_summary,
            "total_structures": len(pdb_files),
            "total_2his_sites": 0,
            "proteins_with_2his_sites": 0,
            "sites_2his_aspglu": 0,
            "sites_putative_halogenase": 0,
            "raw_sites_csv": str(out_2his),
            "no_pairs_txt": str(out_none),
        }
        (outdir / f"summary_{ts}.json").write_text(json.dumps(summary, indent=2))
        print("No 2-His sites found.")
        print(summary)
        return

    # Ensure numeric columns
    for res in RES_TYPES:
        col = f"closest_{res}"
        if col in df_2his.columns:
            df_2his[col] = _coalesce_numeric(df_2his[col])

    df_2his["protein_length"] = _coalesce_numeric(df_2his["protein_length"])

    # ---------- Facial triad filter (your notebook logic, fixed parentheses) ----------
    # default cutoff: <5 Å for ASP/GLU, and length > 200
    df_facial = df_2his[
        (
            (df_2his["closest_ASP"] <= args.asp_glu_cutoff)
            | (df_2his["closest_GLU"] <= args.asp_glu_cutoff)
        )
        & (df_2his["protein_length"] > args.triad_min_length)
    ].copy()


    out_facial = outdir / f"facial_triad_sites_{ts}.csv"
    df_facial.to_csv(out_facial, index=False)

    # ---------- Putative halogenase filter (your notebook logic, parameterized) ----------
    # “no 3rd ligand” by requiring acids farther than acid_exclude (default 5.5)
    # exclude other ligands by requiring > exclude_ligand_dist (default 4)
    # require Ala/Gly close (default < 7)
    # optional HX motif requirement (default ON), HX allowed residues default ALA/GLY

    # start with base mask
    mask = (
        (df_2his["closest_ASP"] > args.acid_exclude_cutoff)
        & (df_2his["closest_GLU"] > args.acid_exclude_cutoff)
        & (df_2his["closest_ASN"] > args.exclude_ligand_cutoff)
        & (df_2his["closest_GLN"] > args.exclude_ligand_cutoff)
        & (df_2his["closest_CYS"] > args.exclude_ligand_cutoff)
        & (df_2his["closest_MET"] > args.exclude_ligand_cutoff)
        & (df_2his["closest_HIS"] > args.exclude_ligand_cutoff)
        & (df_2his["closest_TRP"] > args.exclude_ligand_cutoff)
        & (df_2his["closest_TYR"] > args.exclude_ligand_cutoff)
        & (
            (df_2his["closest_ALA"] < args.ala_gly_presence_cutoff)
            | (df_2his["closest_GLY"] < args.ala_gly_presence_cutoff)
        )
        & (df_2his["protein_length"] > args.min_length)
    )

    if args.require_hx:
        mask = mask & (df_2his["HXres"].isin(args.hx_allowed))

    df_hals = df_2his[mask].copy()

    out_hals = outdir / f"putative_halogenases_{ts}.csv"
    df_hals.to_csv(out_hals, index=False)

    out_ids = outdir / f"putative_halogenase_accessions_{ts}.txt"
    df_hals["structure"].drop_duplicates().to_csv(out_ids, index=False, header=False)

    # ---------- printed + saved summary (your preferred wording) ----------
    n_sites_total = len(df_2his)
    n_proteins_total = df_2his["structure"].nunique()
    n_sites_facial = len(df_facial)
    n_sites_hals = len(df_hals)

    summary = {
        **extra_summary,
        "total_structures": len(pdb_files),
        "total_2his_sites": int(n_sites_total),
        "proteins_with_2his_sites": int(n_proteins_total),
        "sites_2his_aspglu": int(n_sites_facial),
        "sites_putative_halogenase": int(n_sites_hals),
        "raw_sites_csv": str(out_2his),
        "facial_triad_csv": str(out_facial),
        "putative_halogenases_csv": str(out_hals),
        "putative_halogenase_ids": str(out_ids),
        "no_pairs_txt": str(out_none),
    }
    (outdir / f"summary_{ts}.json").write_text(json.dumps(summary, indent=2))

    print(
        f"\nThere are {n_sites_total} total 2-His sites identified, in {n_proteins_total} proteins\n"
        f"of which {n_sites_facial} sites are 2-His-1-Asp/Glu,\n"
        f"and {n_sites_hals} are 2-His sites that lack a third metal-coordinating residue and instead contain a nearby Ala/Gly.\n"
    )
    print("Done.")
    print(summary)


# ----------------------- observed Cu–2His geometry (migrated unchanged) -----------------------

import operator

PASS = "pass"
FAIL = "fail"
NOT_EVALUABLE = "not_evaluable"
COMPARISONS = {"<": operator.lt, ">": operator.gt, ">=": operator.ge}

OBSERVED_SCHEMA_VERSION = "1.2"
OBSERVED_FILTER_VERSION = "copper-0.1.0"


def _source_sha256(path):
    # Lazy import keeps results.py's legacy import of euclidean_distance valid.
    from .results import sha256
    return sha256(path)


@dataclass(frozen=True)
class CopperParameters:
    cu_n_min: float = 1.7
    cu_n_max: float = 2.8
    candidate_radius: float = 4.5
    disulfide_min: float = 1.8
    disulfide_max: float = 2.3
    contact_radius: float = 6.0
    preset: str = "cu-2his"

    def __post_init__(self):
        numbers = [self.cu_n_min, self.cu_n_max, self.candidate_radius,
                   self.disulfide_min, self.disulfide_max, self.contact_radius]
        if not all(isfinite(n) and n > 0 for n in numbers):
            raise ValueError("All distance parameters must be finite and positive")
        if self.cu_n_min > self.cu_n_max or self.disulfide_min > self.disulfide_max:
            raise ValueError("Minimum distance must not exceed maximum distance")
        if self.candidate_radius < self.cu_n_max:
            raise ValueError("Candidate radius must cover the Cu-N maximum")
        if self.preset not in {"cu-2his", "burp-like"}:
            raise ValueError("Preset must be cu-2his or burp-like")


def donor_angle(first, copper, second):
    a = [float(x) - float(y) for x, y in zip(first.coord, copper.coord)]
    b = [float(x) - float(y) for x, y in zip(second.coord, copper.coord)]
    norms = sum(x*x for x in a)**0.5 * sum(x*x for x in b)**0.5
    if norms == 0:
        return None
    cosine = sum(x*y for x, y in zip(a, b)) / norms
    return degrees(acos(max(-1.0, min(1.0, cosine))))


def _usable(atom):
    return (all(isfinite(float(v)) for v in atom.coord)
            and (atom.occupancy is None or (isfinite(atom.occupancy) and atom.occupancy > 0)))


def _criterion(name, value, limits=None, status=None, explanation=None, measurement=None):
    if status is None:
        status = "not_evaluable" if value is None else ("pass" if limits[0] <= value <= limits[1] else "fail")
    return {"name": name, "value": value, "operator": "inclusive_range" if limits else "evidence",
            "threshold": list(limits) if limits else None, "status": status,
            "measurement": measurement,
            "explanation": (explanation or f"{name}: {value}") + f"; {status}"}


def _make_record(source, cu, first, second, parameters):
    from .structure_io import atom_identity, compatible, unpacked_atoms
    atoms, roles, measurements, criteria = {}, {}, {}, []
    warnings = list(source.warnings)
    core = (cu, first, second)

    def retain(role, atom):
        atoms[role] = source.record(atom)
        roles[role] = role
        return role

    def measure(name, a, b):
        value = dist(atoms[a]["coordinates"], atoms[b]["coordinates"])
        measurements[name] = {"endpoints": [a, b], "value": value, "units": "angstrom", "method": "euclidean"}
        return value

    retain("copper", cu)
    for i, donor in enumerate((first, second), 1):
        retain(f"his{i}_donor", donor)
        value = measure(f"cu_his{i}", "copper", f"his{i}_donor")
        criteria.append(_criterion(f"cu_his{i}_distance", value,
                                   (parameters.cu_n_min, parameters.cu_n_max),
                                   measurement=f"cu_his{i}",
                                   explanation=f"Cu–His{i} {donor.name}: {value:.4f} A; require {parameters.cu_n_min} <= distance <= {parameters.cu_n_max} A"))
        his = donor.get_parent()
        # Report the other ring nitrogen too; never use CA/residue centers.
        for atom in his.get_unpacked_list():
            if atom.name in ("ND1", "NE2") and atom.element.upper() == "N" and _usable(atom) and compatible(*core, atom):
                role = f"his{i}_{atom.name}_{atom.get_altloc().strip() or 'blank'}"
                retain(role, atom)
                measure(f"cu_{role}", "copper", role)
        missing = [name for name in ("ND1", "NE2") if name not in his]
        if missing:
            warnings.append(f"His{i} missing ring nitrogen(s): {', '.join(missing)}; available donor was measured")
    angle = donor_angle(first, cu, second)
    measurements["donor_cu_donor_angle"] = {
        "endpoints": ["his1_donor", "copper", "his2_donor"], "value": angle,
        "units": "degree", "method": "vector_angle"}
    if angle is None:
        warnings.append("Donor angle not_evaluable: zero-length Cu-donor vector")
    measure("donor_donor_distance", "his1_donor", "his2_donor")
    criteria.extend([
        _criterion("deposited_copper", True, status="pass", explanation="An explicit Cu element atom is present in the source; no position was fitted"),
        _criterion("same_protein_chain_and_model", True, status="pass",
                   explanation="Two distinct HIS residues in the same protein chain and model surround the same deposited Cu atom"),
    ])
    geometry_pass = all(c["status"] == "pass" for c in criteria)

    staple_options = []
    evidence = {}
    for i, donor in enumerate((first, second), 1):
        previous, info = source.predecessor(donor.get_parent(), donor)
        evidence[f"ch{i}"] = info
        criteria.append(_criterion(f"cys_his{i}_adjacency", info.get("resname"), status=info["status"],
                                   explanation=f"CH{i}: {info['status']} using {info['method']}"))
        options = []
        if previous is not None and info["status"] == "pass":
            options = [a for a in previous.get_unpacked_list() if a.name == "SG"
                       and _usable(a) and compatible(*core, a)]
        staple_options.append(options)
    staples = [(a, b) for a, b in product(*staple_options) if compatible(*core, a, b)]
    staple_value, annotated = None, False
    if staples:
        # Same conformer as the donors; prefer occupancy, not an attractive distance.
        staples.sort(key=lambda pair: (-sum(a.occupancy or 0 for a in pair),
                                       tuple(atom_identity(a) for a in pair)))
        sg1, sg2 = staples[0]
        retain("cys1_SG", sg1)
        retain("cys2_SG", sg2)
        staple_value = measure("cys_sg_sg", "cys1_SG", "cys2_SG")
        annotated = source.disulfide_annotation(sg1, sg2)
        if len(staples) > 1:
            warnings.append("Multiple compatible staple conformers; highest occupancy pair selected deterministically")
    staple_criterion = _criterion("disulfide_geometry", staple_value,
                                  (parameters.disulfide_min, parameters.disulfide_max),
                                  measurement="cys_sg_sg" if staples else None,
                                  explanation="SG–SG geometry is provisional evidence of a structural staple, not copper ligation")
    criteria.append(staple_criterion)
    evidence["disulfide"] = {"status": staple_criterion["status"], "sg_sg_angstrom": staple_value,
                             "deposited_bond_annotation": annotated,
                             "interpretation": "Cysteines are structural features; not assigned as copper ligands"}
    burp_pass = geometry_pass and all(evidence[f"ch{i}"]["status"] == "pass" for i in (1, 2)) and staple_criterion["status"] == "pass"
    strict_statuses = [evidence[f"ch{i}"]["status"] for i in (1, 2)] + [staple_criterion["status"]]
    for criterion in criteria:
        criterion["required_for_match"] = (criterion["name"] not in {"cys_his1_adjacency", "cys_his2_adjacency", "disulfide_geometry"}
                                           or parameters.preset == "burp-like")
    if not geometry_pass or (parameters.preset == "burp-like" and "fail" in strict_statuses):
        status = "fail"
    elif parameters.preset == "burp-like" and "not_evaluable" in strict_statuses:
        status = "not_evaluable"
    else:
        status = "pass"
    contacts = []
    for atom in unpacked_atoms(cu.get_parent().get_parent().get_parent()):
        if atom is cu or atom.element.upper() in {"H", "D"} or not _usable(atom) or not compatible(*core, atom):
            continue
        value = dist(atom.coord, cu.coord)
        if value <= parameters.contact_radius:
            contacts.append({"atom": source.record(atom), "distance_angstrom": value,
                             "interpretation": "Nearby atom; proximity alone does not assign a ligand"})
    contacts.sort(key=lambda r: (r["distance_angstrom"], json.dumps(r["atom"], sort_keys=True)))
    identity = [_source_sha256(source.path), OBSERVED_FILTER_VERSION, asdict(parameters), [atom_identity(a) for a in core]]
    site_id = source.path.stem + "-cu-" + hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
    if any(a.occupancy is None for a in core):
        warnings.append("Occupancy is missing for one or more core atoms")
    if any(a.get_altloc().strip() for a in core):
        warnings.append("Alternate locations were enumerated; only compatible labels plus shared blank atoms were combined")
    warnings.extend([
        "Search tolerances are provisional and require scientific calibration",
        "Two donors do not establish a complete coordination-geometry class or enzyme function",
        "Additional ligands are allowed; missing coordinates do not prove absence of other ligands",
    ])
    return {
        "schema_version": OBSERVED_SCHEMA_VERSION, "site_id": site_id, "status": status,
        "source": {"path": str(source.path), "filename": source.path.name, "sha256": _source_sha256(source.path),
                   "format": source.format, "structure_id": source.path.stem,
                   "experimental_resolution_angstrom": source.structure.header.get("resolution"),
                   "identifier_policy": "Author chain/number plus deposited mmCIF label identifiers; model-specific"},
        "filter": {"name": "observed_cu_2his", "version": OBSERVED_FILTER_VERSION, "package_version": "0.1.0",
                   "parameters": asdict(parameters), "tolerances": "provisional, not universal metal-binding cutoffs",
                   "altloc_policy": "enumerate compatible atom labels; blank atoms shared; zero occupancy excluded",
                   "pdb_adjacency_policy": "coordinate order plus compatible peptide C-N distance 1.0-1.8 A",
                   "implementation_sha256": {name: _source_sha256(Path(__file__).with_name(name))
                                             for name in ("mine.py", "structure_io.py")}},
        "atoms": atoms, "roles": roles, "points": {}, "measurements": measurements, "criteria": criteria,
        "visualization": {"center": "copper", "matched_atoms": ["his1_donor", "his2_donor"],
                          "display_measurements": ["cu_his1", "cu_his2", "donor_cu_donor_angle"] + (["cys_sg_sg"] if staples else [])},
        "classifications": {"observed_cu_2his_geometry": geometry_pass,
                            "burp_like_2xch_evidence": burp_pass}, "burp_evidence": evidence,
        "metals": {"observed": [source.record(cu)], "hypothetical": [],
                   "interpretation": "Deposited Cu atom, not a claim of physiological occupancy or catalytic activity"},
        "contacts": contacts, "warnings": warnings, "exportable": len(source.structure) == 1,
    }


def analyze_copper(path, parameters=None, source=None):
    """Return all evaluable local candidates, diagnostics, and file-level facts.

    Failed distance candidates remain in candidates.jsonl; only passing records
    appear in copper_sites.jsonl/CSV. Missing donors are separate not_evaluable
    diagnostics, never invented atom positions.
    """
    from .structure_io import compatible, load_structure, unpacked_atoms
    parameters = parameters or CopperParameters()
    source = source if source is not None else load_structure(path)
    records, diagnostics = [], []
    observed = 0
    for model in source.structure:
        model_atoms = list(unpacked_atoms(model))
        for residue in model.get_residues():
            if residue.is_disordered() == 2:
                diagnostics.append({"status": "not_evaluable", "reason": "Residue microheterogeneity is unsupported",
                                    "model_id": model.id, "chain": residue.get_parent().id, "residue_id": residue.id})
        coppers = []
        for atom in model_atoms:
            element = atom.element.upper()
            component = atom.get_parent().resname.upper()
            if component in {"CU", "CU1", "CU2"} and element != "CU":
                diagnostics.append({"status": "not_evaluable", "reason": "Copper component/element conflict", "atom": source.record(atom)})
            if element == "CU":
                observed += 1
                if _usable(atom):
                    coppers.append(atom)
                else:
                    diagnostics.append({"status": "not_evaluable", "reason": "Cu has nonfinite coordinates or nonpositive occupancy", "atom": source.record(atom)})
        histidines = [r for r in model.get_residues() if r.resname == "HIS" and r.id[0] == " " and r.is_disordered() != 2]
        for cu in coppers:
            groups = []
            for his in histidines:
                ring = [a for a in his.get_unpacked_list() if a.name in ("ND1", "NE2")]
                if any(a.element.upper() != "N" for a in ring):
                    diagnostics.append({"status": "not_evaluable", "reason": "Histidine donor name/element conflict",
                                        "model_id": model.id, "chain": his.get_parent().id, "residue_id": his.id})
                ring = [a for a in ring if a.element.upper() == "N"]
                donors = [a for a in ring if _usable(a) and compatible(cu, a)
                          and dist(cu.coord, a.coord) <= parameters.candidate_radius]
                if not ring or not any(_usable(a) for a in ring):
                    diagnostics.append({"status": "not_evaluable", "reason": "Missing usable histidine ring donor atoms",
                                        "model_id": model.id, "chain": his.get_parent().id, "residue_id": his.id})
                if donors:
                    groups.append((his, donors))
                    if sum(parameters.cu_n_min <= dist(cu.coord, a.coord) <= parameters.cu_n_max for a in donors) > 1:
                        diagnostics.append({"status": "not_evaluable", "reason": "Multiple eligible donor assignments; all compatible assignments enumerated",
                                            "model_id": model.id, "chain": his.get_parent().id, "residue_id": his.id})
            for (his1, donors1), (his2, donors2) in combinations(groups, 2):
                if his1.get_parent().id != his2.get_parent().id:
                    continue
                for first, second in product(donors1, donors2):
                    if compatible(cu, first, second):
                        records.append(_make_record(source, cu, first, second, parameters))
    facts = {"source": str(source.path), "source_sha256": _source_sha256(source.path),
             "deposited_copper_atoms": observed, "candidate_records": len(records),
             "passing_records": sum(r["status"] == "pass" for r in records),
             "interpretation": "No deposited Cu is not a biological negative for copper binding"}
    if not observed:
        diagnostics.append({"status": "not_evaluable", "reason": "No deposited copper atoms; metal-independent search was not performed"})
    return records, diagnostics, facts



# ----------------------- flat filter configuration -----------------------

SITE_FILTERS = ('his_ne2', 'his_backbone', 'potential_cu_2his',
                'observed_cu_2his', 'cys_his', 'cys_disulfide')
RING_ATOMS = ('CG', 'ND1', 'CE1', 'NE2', 'CD2')
SELECTED_SCHEMA_VERSION = '1.3'
SELECTED_FILTER_VERSION = 'site-filters-0.1.0'


@dataclass(frozen=True)
class FilterParameters:
    filters: tuple
    his_ne2_cutoff: float = 4.0
    his_hbond_cutoff: float = 4.0
    inward_score_min: float | None = None
    cu_n_min: float = 1.7
    cu_n_max: float = 2.8
    candidate_radius: float = 4.5
    disulfide_min: float = 1.8
    disulfide_max: float = 2.3
    contact_radius: float = 6.0

    def __post_init__(self):
        names = tuple(sorted(set(self.filters)))
        if not names or any(name not in SITE_FILTERS for name in names):
            raise ValueError(f'Choose at least one known filter: {", ".join(SITE_FILTERS)}')
        object.__setattr__(self, 'filters', names)
        if self.inward_score_min is not None:
            if not isfinite(self.inward_score_min) or not 0 <= self.inward_score_min <= 1:
                raise ValueError('--inward-score-min must be finite and between 0 and 1 inclusive')
            if 'potential_cu_2his' not in names:
                raise ValueError('--inward-score-min requires potential_cu_2his in --filters')
        if not all(isfinite(v) and v > 0 for v in (self.his_ne2_cutoff, self.his_hbond_cutoff)):
            raise ValueError('Histidine distance cutoffs must be finite and positive')
        self.observed_parameters()  # Also validate shared Cu/staple parameters.

    def observed_parameters(self):
        return CopperParameters(cu_n_min=self.cu_n_min, cu_n_max=self.cu_n_max,
                                candidate_radius=self.candidate_radius,
                                disulfide_min=self.disulfide_min, disulfide_max=self.disulfide_max,
                                contact_radius=self.contact_radius, preset='cu-2his')


# ----------------------- ring centers and inward-facing directions -----------------------

def _unit(vector):
    values = [float(v) for v in vector]
    length = math.sqrt(math.fsum(v*v for v in values))
    if not isfinite(length) or length <= 1e-12:
        raise ValueError('Direction is not_evaluable: zero-length or nonfinite vector')
    return [v/length for v in values]


def inward_facing_geometry(ring1, donor1, ring2, donor2):
    """Pure geometry: five ring coordinates per His; no metal input or fitting.

    s1 = u1 dot w; s2 = u2 dot (-w). The mutual u1 dot u2 is NOT
    substituted for either inward score. Degenerate data is not_evaluable.
    """
    if len(ring1) != 5 or len(ring2) != 5:
        raise ValueError('Both ring centers require all five imidazole atoms')
    if not all(isfinite(float(v)) for point in (*ring1, *ring2, donor1, donor2) for v in point):
        raise ValueError('Ring geometry has missing/nonfinite coordinates')
    o1 = [math.fsum(float(p[i]) for p in ring1)/5 for i in range(3)]
    o2 = [math.fsum(float(p[i]) for p in ring2)/5 for i in range(3)]
    u1 = _unit([float(n)-o for n,o in zip(donor1,o1)])
    u2 = _unit([float(n)-o for n,o in zip(donor2,o2)])
    w = _unit([b-a for a,b in zip(o1,o2)])
    s1 = max(-1.0, min(1.0, math.fsum(a*b for a,b in zip(u1,w))))
    s2 = max(-1.0, min(1.0, -math.fsum(a*b for a,b in zip(u2,w))))
    return dict(ring_center1=o1, ring_center2=o2, u1=u1, u2=u2, w=w,
                s1=s1, s2=s2, inward_score=min(s1,s2),
                angle1=degrees(acos(s1)), angle2=degrees(acos(s2)))


def _conformers(residue, names):
    """Enumerate usable, compatible atom maps; missing names stay absent."""
    from .structure_io import compatible
    choices = []
    for name in names:
        element = 'N' if name.startswith('N') else ('O' if name == 'O' else 'C')
        atoms = [a for a in residue.get_unpacked_list()
                 if a.name == name and a.element.upper() == element and _usable(a)]
        if atoms:
            choices.append(atoms)
    for atoms in product(*choices):
        if compatible(*atoms):
            yield {a.name: a for a in atoms}


# ----------------------- Cys–His adjacency and site-associated disulfide -----------------------

def _add_cysteine_evidence(source, record, first, second, core, parameters):
    from .structure_io import compatible, atom_identity
    options, evidence = [], {}
    for i, donor in enumerate((first, second), 1):
        previous, info = source.predecessor(donor.get_parent(), donor)
        evidence[f'ch{i}'] = info
        record['criteria'].append(dict(_criterion(
            f'cys_his{i}_adjacency', info.get('resname'), status=info['status'],
            explanation=f'CH{i}: {info["method"]}'), filter='cys_his'))
        options.append([a for a in previous.get_unpacked_list() if a.name == 'SG'
                        and a.element.upper() == 'S' and _usable(a) and compatible(*core, a)]
                       if previous is not None and info['status'] == 'pass' else [])
    pairs = [(a,b) for a,b in product(*options) if compatible(*core,a,b)]
    value, annotated = None, False
    if pairs:
        pairs.sort(key=lambda pair: (-sum(a.occupancy or 0 for a in pair),
                                     tuple(atom_identity(a) for a in pair)))
        first_sg, second_sg = pairs[0]
        for i, atom in enumerate((first_sg,second_sg),1):
            record['atoms'][f'cys{i}_SG'] = source.record(atom)
        value = dist(first_sg.coord,second_sg.coord)
        record['measurements']['cys_sg_sg'] = dict(endpoints=['cys1_SG','cys2_SG'],value=value,
                                                   units='angstrom',method='euclidean')
        annotated = source.disulfide_annotation(first_sg,second_sg)
    known_not_ch = any(evidence[f'ch{i}']['status']=='fail' for i in (1,2))
    criterion = _criterion('disulfide_geometry',value,(parameters.disulfide_min,parameters.disulfide_max),
                           status='fail' if known_not_ch else None,
                           measurement='cys_sg_sg' if pairs else None,
                           explanation='Structural staple between the two His-associated cysteines; not copper ligation')
    record['criteria'].append(dict(criterion,filter='cys_disulfide'))
    evidence['disulfide'] = dict(status=criterion['status'],sg_sg_angstrom=value,
                                deposited_bond_annotation=annotated)
    record['burp_evidence'] = evidence
    record['classifications']['burp_like_2xch_evidence'] = all(
        evidence[f'ch{i}']['status']=='pass' for i in (1,2)) and criterion['status']=='pass'


# ----------------------- shared candidate records and selected criteria -----------------------

def _selected_record(source, first, second):
    return dict(schema_version=SELECTED_SCHEMA_VERSION, status='not_evaluable',
        source=dict(path=str(source.path),filename=source.path.name,sha256=_source_sha256(source.path),
                    format=source.format,structure_id=source.path.stem,
                    identifier_policy='Full author and mmCIF label identities; one chain and model per site'),
        atoms={'his1_donor':source.record(first),'his2_donor':source.record(second)},
        roles={}, points={}, measurements={}, criteria=[], classifications={}, warnings=list(source.warnings),
        metals=dict(observed=[],hypothetical=[],assessment='not_assessed_by_selected_filters'),
        exportable=len(source.structure)==1)


def _prepare_selected_atoms(source, record, maps, first, observed):
    """Retain measured atoms and recover the observed copper when applicable."""
    from .structure_io import atom_identity, unpacked_atoms
    core = [a for atom_map in maps for a in atom_map.values()]
    if observed:
        cu = next(a for a in unpacked_atoms(source.structure[first.get_parent().get_parent().get_parent().id])
                  if atom_identity(a) == tuple(record['_copper_identity']))
        core.append(cu)
        del record['_copper_identity']
        # The original observed engine's evidence and geometry remain authoritative.
        for c in record['criteria']:
            c['filter'] = ('cys_his' if c['name'].startswith('cys_his') else
                           'cys_disulfide' if c['name']=='disulfide_geometry' else 'observed_cu_2his')
    for i, atom_map in enumerate(maps,1):
        for name, atom in atom_map.items():
            record['atoms'][f'his{i}_{name}'] = source.record(atom)

    return core


def _measure_distance(record: Dict[str, Any], name: str, role_a: str, role_b: str):
    """Store an atom-to-atom distance; return None when an endpoint is missing."""
    atoms = record["atoms"]
    if role_a not in atoms or role_b not in atoms:
        return None

    value = euclidean_distance(atoms[role_a]["coordinates"], atoms[role_b]["coordinates"])
    record["measurements"][name] = {
        "endpoints": [role_a, role_b],
        "value": value,
        "units": "angstrom",
        "method": "euclidean",
    }
    return value


def _threshold_criterion(name, filter_name, value, operator_symbol, threshold,
                         explanation, measurement=None):
    """Compare one measured value to a threshold, preserving missing evidence."""
    if value is None:
        status = NOT_EVALUABLE
    elif COMPARISONS[operator_symbol](value, threshold):
        status = PASS
    else:
        status = FAIL

    return {
        "name": name,
        "filter": filter_name,
        "value": value,
        "operator": operator_symbol,
        "threshold": threshold,
        "status": status,
        "measurement": measurement if value is not None else None,
        "explanation": f"{explanation}; {status}",
    }


def _add_his_distance_criteria(record: Dict[str, Any], parameters: FilterParameters) -> None:
    """Add the selected NE2 and reciprocal backbone distance requirements."""
    distance_filters = {
        "his_ne2": [("his_ne2_distance", "his1_NE2", "his2_NE2", parameters.his_ne2_cutoff)],
        "his_backbone": [
            ("backbone_o1_n2", "his1_O", "his2_N", parameters.his_hbond_cutoff),
            ("backbone_n1_o2", "his1_N", "his2_O", parameters.his_hbond_cutoff),
        ],
    }
    for filter_name, entries in distance_filters.items():
        if filter_name not in parameters.filters:
            continue
        for name, role_a, role_b, cutoff in entries:
            value = _measure_distance(record, name, role_a, role_b)
            record["criteria"].append(_threshold_criterion(
                name, filter_name, value, "<", cutoff,
                explanation=f"{name}: {value}; require distance < {cutoff} A",
                measurement=name,
            ))


def _add_inward_geometry_criteria(record, maps, first, second, parameters):
    """Measure inward geometry and its criteria; return geometry for the view."""
    if 'potential_cu_2his' not in parameters.filters:
        return None
    missing = [f'His{i} {name}' for i,atom_map in enumerate(maps,1) for name in RING_ATOMS if name not in atom_map]
    geometry = None
    if not missing:
        try:
            geometry = inward_facing_geometry(
                [maps[0][name].coord for name in RING_ATOMS], first.coord,
                [maps[1][name].coord for name in RING_ATOMS], second.coord)
        except ValueError as exc:
            record['warnings'].append(str(exc))
    else:
        record['warnings'].append('Missing usable ring atoms: '+', '.join(missing))
    if geometry is not None:
        for i,j in ((1,2),(2,1)):
            record['measurements'][f's{i}'] = dict(endpoints=[f'ring_center{i}',f'his{i}_donor',f'ring_center{j}'],
                value=geometry[f's{i}'],units='dimensionless',method='inward_dot')
            record['measurements'][f'inward_angle{i}'] = dict(endpoints=[f'his{i}_donor',f'ring_center{i}',f'ring_center{j}'],
                value=geometry[f'angle{i}'],units='degree',method='vector_angle')
        record['measurements']['inward_score'] = dict(endpoints=[],dependencies=['s1','s2'],
            value=geometry['inward_score'],units='dimensionless',method='minimum')
    for i in (1, 2):
        value = geometry[f's{i}'] if geometry is not None else None
        record['criteria'].append(_threshold_criterion(
            f'inward_s{i}', 'potential_cu_2his', value, '>', 0,
            explanation=f's{i}={value}; require > 0 (ring-center inward score, not mutual donor dot)',
            measurement=f's{i}',
        ))
    if parameters.inward_score_min is not None:
        value = geometry['inward_score'] if geometry is not None else None
        record['criteria'].append(_threshold_criterion(
            'inward_score_min', 'potential_cu_2his', value, '>=', parameters.inward_score_min,
            explanation=f'min(s1,s2)={value}; require >= {parameters.inward_score_min}',
            measurement='inward_score',
        ))
    record['classifications']['potential_cu_2his_geometry'] = all(
        c['status']=='pass' for c in record['criteria'] if c['filter']=='potential_cu_2his')
    record['warnings'].extend([
        'Inward-facing geometry does not establish copper specificity, affinity, occupancy, or enzyme activity',
        'Protonation/tautomer and AlphaFold side-chain/rotamer uncertainty are not resolved',
        'Steric accessibility is not assessed; no convergence calculation or hypothetical metal was used'])
    return geometry


def _add_selected_cysteine_criteria(source, record, first, second, core, parameters, observed):
    """Keep staple evidence compatible with all atoms used by selected filters."""
    if observed and parameters.filters == ('observed_cu_2his',):
        return
    if observed:
        record['criteria'] = [c for c in record['criteria'] if c['filter'] not in ('cys_his', 'cys_disulfide')]
        record['atoms'].pop('cys1_SG', None)
        record['atoms'].pop('cys2_SG', None)
        record['measurements'].pop('cys_sg_sg', None)
    _add_cysteine_evidence(source, record, first, second, core, parameters)


def _add_selected_visualization(record, first, second, geometry, parameters, observed):
    """Describe computed points/directions separately from filter decisions."""
    if geometry is not None:
        for i in (1,2):
            origin = f'ring_center{i}'
            record['points'][origin] = dict(coordinates=geometry[origin],kind='computed_ring_center',
                method='centroid',endpoints=[f'his{i}_{n}' for n in RING_ATOMS],
                label=f'His{i} COMPUTED RING CENTER')
        record['directions'] = {
            'his1':dict(origin='ring_center1',target='his1_donor',vector=geometry['u1']),
            'his2':dict(origin='ring_center2',target='his2_donor',vector=geometry['u2']),
            'inter_ring':dict(origin='ring_center1',target='ring_center2',vector=geometry['w'])}
    if not observed and 'potential_cu_2his' in parameters.filters:
        record['visualization'] = dict(center='site_center',matched_atoms=['his1_donor','his2_donor'],
            display_measurements=['donor_donor_distance','inward_angle1','inward_angle2'] if geometry else ['donor_donor_distance'],
            potential_site=True,label='POTENTIAL COPPER-BINDING GEOMETRY — NO METAL INFERRED')
        record['exportable'] = record['exportable'] and geometry is not None
    if not observed:
        record['points']['site_center'] = dict(
            coordinates=[(float(a)+float(b))/2 for a,b in zip(first.coord,second.coord)],
            kind='display_reference', method='midpoint', endpoints=['his1_donor','his2_donor'],
            label='COMPUTED DISPLAY REFERENCE — NOT A METAL POSITION')
        record.setdefault('visualization', dict(
            center='site_center', matched_atoms=['his1_donor','his2_donor'],
            display_measurements=['donor_donor_distance']))


def _combined_status(statuses):
    """AND criteria, with a measured failure taking precedence over missing data."""
    if FAIL in statuses:
        return FAIL
    if not statuses or NOT_EVALUABLE in statuses:
        return NOT_EVALUABLE
    return PASS


def _finalize_selected_record(source, record, maps, first, second, parameters, observed):
    """Set selected outcomes, provenance, roles and the deterministic site ID."""
    from .structure_io import atom_identity

    for criterion in record['criteria']:
        criterion['required_for_match'] = criterion['filter'] in parameters.filters
    outcomes = {}
    for name in parameters.filters:
        statuses = [c['status'] for c in record['criteria'] if c['filter'] == name]
        outcomes[name] = _combined_status(statuses)
    record['filter_results'] = outcomes
    record['status'] = _combined_status(list(outcomes.values()))
    record['selection'] = dict(filters=list(parameters.filters), parameters=asdict(parameters),
                               combination='AND at the same site and donor assignment')
    if not observed:
        record['filter'] = dict(
            name='selected_site_filters', version=SELECTED_FILTER_VERSION,
            parameters=asdict(parameters),
            altloc_policy='enumerate compatible labels across all measured atoms; blank shared',
            implementation_sha256={
                'mine.py': _source_sha256(__file__),
                'structure_io.py': _source_sha256(Path(__file__).with_name('structure_io.py'))})
        identity = [record['source']['sha256'], SELECTED_FILTER_VERSION, asdict(parameters),
                    [atom_identity(a) for a in (first,second)],
                    [[atom_identity(a) for a in atom_map.values()] for atom_map in maps]]
        record['site_id'] = source.path.stem+'-site-'+hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()[:16]
    elif parameters.filters != ('observed_cu_2his',):
        identity = [record['site_id'], asdict(parameters),
                    [[atom_identity(a) for a in atom_map.values()] for atom_map in maps]]
        record['site_id'] += '-'+hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()[:12]
    record['schema_version'] = SELECTED_SCHEMA_VERSION
    record['roles'].update({name:name for name in record['atoms']})
    return record


def _apply_selected_criteria(source, record, maps, first, second, parameters, observed=False):
    """Apply the named measurement, evidence, visualization and finalization steps."""
    core = _prepare_selected_atoms(source, record, maps, first, observed)
    _measure_distance(record, 'donor_donor_distance', 'his1_donor', 'his2_donor')
    _add_his_distance_criteria(record, parameters)
    geometry = _add_inward_geometry_criteria(record, maps, first, second, parameters)
    _add_selected_cysteine_criteria(source, record, first, second, core, parameters, observed)
    _add_selected_visualization(record, first, second, geometry, parameters, observed)
    return _finalize_selected_record(source, record, maps, first, second, parameters, observed)


def required_atom_names(parameters):
    """List atoms needed by the selected criteria and donor assignments."""
    required = set() if 'observed_cu_2his' in parameters.filters else {'ND1','NE2'}
    if 'his_ne2' in parameters.filters:
        required.add('NE2')
    if 'potential_cu_2his' in parameters.filters:
        required.update(RING_ATOMS)
    if 'his_backbone' in parameters.filters:
        required.update(('N','O'))
    return tuple(sorted(required))


def _collect_observed_copper_candidates(source, parameters, names):
    """Extend observed-copper records with compatible measured conformers."""
    from .structure_io import compatible, unpacked_atoms, atom_identity
    candidates = []
    path = source.path
    bases, diagnostics, observed_facts = analyze_copper(path,parameters.observed_parameters(),source=source)
    lookup = {atom_identity(a):a for model in source.structure for a in unpacked_atoms(model)}
    for base in bases:
        def resolve(role):
            r=base['atoms'][role]
            return lookup[(r['model_number'],r['auth_chain_id'],r['auth_seq_id'],r['insertion_code'],r['resname'],r['atom_name'],r['altloc'])]
        first,second,cu = (resolve(role) for role in ('his1_donor','his2_donor','copper'))
        for maps in product(_conformers(first.get_parent(),names),_conformers(second.get_parent(),names)):
            maps = (dict(maps[0]),dict(maps[1]))
            maps[0].setdefault(first.name,first)
            maps[1].setdefault(second.name,second)
            if (maps[0].get(first.name) is first and maps[1].get(second.name) is second
                    and compatible(cu,*maps[0].values(),*maps[1].values())):
                candidate = json.loads(json.dumps(base))
                candidate['_copper_identity'] = atom_identity(cu)
                candidates.append((candidate,maps,first,second))
    return candidates, diagnostics, observed_facts


def _collect_his_pair_candidates(source, parameters, names):
    """Collect within-chain His pairs, applying the optional NE2 pre-screen."""
    from .structure_io import compatible
    candidates, diagnostics = [], []
    for model in source.structure:
        for chain in model:
            residues = [r for r in chain if r.resname=='HIS' and r.id[0]==' ' and r.is_disordered()!=2]
            conformers = {r.id:list(_conformers(r,names)) for r in residues}
            for his1,his2 in combinations(residues,2):
                maps1,maps2 = conformers[his1.id],conformers[his2.id]
                if 'his_ne2' in parameters.filters:
                    measured = [dist(a['NE2'].coord,b['NE2'].coord) for a,b in product(maps1,maps2)
                                if 'NE2' in a and 'NE2' in b and compatible(*a.values(),*b.values())]
                    if not measured or min(measured)>=parameters.his_ne2_cutoff:
                        diagnostics.append(dict(status='fail' if measured else 'not_evaluable',filter='his_ne2',
                            model_id=model.id,chain=chain.id,his1=his1.id,his2=his2.id,
                            value=min(measured) if measured else None,threshold=parameters.his_ne2_cutoff,
                            reason='Pair screened by explicit NE2 distance requirement'))
                        continue
                count = 0
                for maps in product(maps1,maps2):
                    if not compatible(*maps[0].values(),*maps[1].values()):
                        continue
                    donor_pairs = product(('NE2','ND1'),repeat=2) if 'potential_cu_2his' in parameters.filters else [
                        ('NE2' if 'NE2' in maps[0] else 'ND1', 'NE2' if 'NE2' in maps[1] else 'ND1')]
                    for name1,name2 in donor_pairs:
                        first,second = maps[0].get(name1),maps[1].get(name2)
                        if first is None or second is None:
                            continue
                        candidates.append((_selected_record(source,first,second),maps,first,second))
                        count+=1
                if not count:
                    diagnostics.append(dict(status='not_evaluable',model_id=model.id,chain=chain.id,
                        his1=his1.id,his2=his2.id,reason='No usable, compatible nitrogen donor assignments'))
    if 'his_ne2' not in parameters.filters:
        diagnostics.append(dict(status='warning',reason='No distance filter selected; orientation alone does not constrain donor separation'))
    facts = {'metal_assessment': 'not_performed; deposited copper is not consulted'}
    return candidates, diagnostics, facts


def analyze_filters(path, parameters):
    """Collect candidates and apply only the explicitly selected criteria."""
    from .structure_io import load_structure
    source = load_structure(path)
    names = required_atom_names(parameters)
    observed = 'observed_cu_2his' in parameters.filters
    collect = _collect_observed_copper_candidates if observed else _collect_his_pair_candidates
    candidates, diagnostics, facts = collect(source, parameters, names)
    records = []
    for base, maps, first, second in candidates:
        records.append(_apply_selected_criteria(
            source, base, maps, first, second, parameters, observed=observed))
    facts = dict(facts)
    facts.update(source=str(source.path), source_sha256=_source_sha256(source.path),
                 candidate_records=len(records), passing_records=sum(r['status'] == PASS for r in records))
    return records, diagnostics, facts


# ----------------------- selected-filter output (legacy CSVs stay unchanged) -----------------------

SELECTED_CSV_COLUMNS = ['site_id','structure','model_number','chain','his1','his1_insertion_code','donor1',
    'his2','his2_insertion_code','donor2','donor_distance_angstrom','s1','s2','inward_score',
    'inward_score_min','cu_auth_chain','cu_auth_number','cu_label_chain','cu_his1_angstrom',
    'cu_his2_angstrom','donor_angle_degree','burp_like_2xch_evidence']


def find_structure_paths(input_path, max_pdbs=None):
    """Find supported files in deterministic order, honoring the input limit."""
    path = Path(input_path).expanduser().resolve()
    if not path.exists():
        raise ValueError(f'Structure input does not exist: {path}')
    if path.is_dir():
        paths = sorted(p for p in path.iterdir()
                       if p.is_file() and p.suffix.lower() in {'.pdb', '.cif', '.mmcif'})
    else:
        paths = [path]
    return paths if max_pdbs is None else paths[:max_pdbs]


def _measured_value(record, name):
    return record['measurements'].get(name, {}).get('value')


def _selected_csv_row(record, parameters):
    """Flatten a passing record with explicit column names."""
    his1 = record['atoms']['his1_donor']
    his2 = record['atoms']['his2_donor']
    copper = record['atoms'].get('copper', {})
    return {
        'site_id': record['site_id'],
        'structure': record['source']['structure_id'],
        'model_number': his1['model_number'],
        'chain': his1['auth_chain_id'],
        'his1': his1['auth_seq_id'],
        'his1_insertion_code': his1['insertion_code'],
        'donor1': his1['atom_name'],
        'his2': his2['auth_seq_id'],
        'his2_insertion_code': his2['insertion_code'],
        'donor2': his2['atom_name'],
        'donor_distance_angstrom': _measured_value(record, 'donor_donor_distance'),
        's1': _measured_value(record, 's1'),
        's2': _measured_value(record, 's2'),
        'inward_score': _measured_value(record, 'inward_score'),
        'inward_score_min': parameters.inward_score_min,
        'cu_auth_chain': copper.get('auth_chain_id'),
        'cu_auth_number': copper.get('auth_seq_id'),
        'cu_label_chain': copper.get('label_asym_id'),
        'cu_his1_angstrom': _measured_value(record, 'cu_his1'),
        'cu_his2_angstrom': _measured_value(record, 'cu_his2'),
        'donor_angle_degree': _measured_value(record, 'donor_cu_donor_angle'),
        'burp_like_2xch_evidence': record['classifications'].get('burp_like_2xch_evidence'),
    }


def _analyze_structure_paths(paths, parameters):
    """Accumulate records and report processing failures per input file."""
    records, diagnostics, facts = [], [], []
    for path in paths:
        try:
            found, issues, summary = analyze_filters(path, parameters)
            records.extend(found)
            facts.append(summary)
            diagnostics.extend(dict(source=str(path), **issue) for issue in issues)
        except (ValueError, KeyError, OSError, PDBException, PDBConstructionException) as exc:
            diagnostics.append(dict(
                source=str(path), status=NOT_EVALUABLE, reason=f'Parse/processing error: {exc}'))
    return records, diagnostics, facts


def _write_selected_outputs(outdir, timestamp, paths, parameters, records, diagnostics, facts):
    """Write the existing CSV, detailed records, diagnostics and summary."""
    passing = [record for record in records if record['status'] == PASS]
    outputs = {}
    # Explicit-filter runs always retain detailed evidence, including failures.
    for name, items in [('selected_sites', passing), ('selected_candidates', records)]:
        target = outdir / f'{name}_{timestamp}.jsonl'
        outputs[name] = str(target)
        target.write_text(''.join(json.dumps(record, sort_keys=True, allow_nan=False) + '\n'
                                  for record in items))

    rows = [_selected_csv_row(record, parameters) for record in passing]
    table = outdir / f'selected_sites_{timestamp}.csv'
    pd.DataFrame(rows, columns=SELECTED_CSV_COLUMNS).to_csv(table, index=False)
    report = outdir / f'selected_diagnostics_{timestamp}.json'
    report.write_text(json.dumps(diagnostics, indent=2, allow_nan=False) + '\n')
    summary = dict(
        parameters=asdict(parameters), total_structures=len(paths), processed_structures=len(facts),
        candidate_records=len(records), passing_records=len(passing), file_summaries=facts,
        outputs=outputs, selected_sites_csv=str(table), diagnostics_json=str(report))
    (outdir / f'summary_selected_{timestamp}.json').write_text(json.dumps(summary, indent=2) + '\n')
    return summary, report


def run_selected_filters(args):
    parameters = FilterParameters(
        filters=tuple(args.filters),
        his_ne2_cutoff=args.his_ne2_cutoff,
        his_hbond_cutoff=args.his_hbond_cutoff,
        inward_score_min=args.inward_score_min,
        cu_n_min=args.cu_n_min,
        cu_n_max=args.cu_n_max,
        candidate_radius=args.candidate_radius,
        disulfide_min=args.disulfide_min,
        disulfide_max=args.disulfide_max,
        contact_radius=args.contact_radius,
    )
    paths = find_structure_paths(args.pdb_dir, getattr(args, 'max_pdbs', None))
    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    timestamp = dt.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    records, diagnostics, facts = _analyze_structure_paths(paths, parameters)
    summary, report = _write_selected_outputs(
        outdir, timestamp, paths, parameters, records, diagnostics, facts)
    print(json.dumps(summary, indent=2))
    if len(facts) != len(paths):
        raise ValueError(f'Some structures could not be processed; see {report}')
