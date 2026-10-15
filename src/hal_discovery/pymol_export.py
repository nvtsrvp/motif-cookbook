"""Portable PyMOL scripts from recorded sites; no PyMOL dependency at export time."""
import json
import math
from pathlib import Path
import shutil

from Bio.PDB import PDBParser
from .results import SCHEMA_VERSION, atom_record, sha256


def validate_record(record, source):
    if record.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported site schema version")
    if not record.get("exportable"):
        raise ValueError("Site has ambiguous legacy identities and cannot be exported exactly")
    if sha256(source) != record["source"]["sha256"]:
        raise ValueError("Structure SHA-256 differs from the filter input")
    structure = PDBParser(QUIET=True).get_structure("source", str(source))
    if len(structure) != 1:
        raise ValueError("Legacy export requires a single model")
    parsed = [atom_record(a) for a in structure.get_atoms()]
    keys = ("model_id", "model_number", "auth_chain_id", "hetero_flag", "auth_seq_id",
            "insertion_code", "resname", "atom_name", "altloc")
    for ref, atom in record["atoms"].items():
        matches = [a for a in parsed if all(a[k] == atom[k] for k in keys)]
        if len(matches) != 1 or math.dist(matches[0]["coordinates"], atom["coordinates"]) > 0.001:
            raise ValueError(f"Recorded atom {ref} does not resolve uniquely to the source coordinates")
    for name, measurement in record["measurements"].items():
        endpoints = [record["atoms"].get(e, record["points"].get(e)) for e in measurement["endpoints"]]
        if any(e is None for e in endpoints) or measurement["value"] is None:
            raise ValueError(f"Measurement {name} lacks an exact endpoint")
        actual = math.dist(*(e["coordinates"] for e in endpoints))
        if abs(actual - measurement["value"]) > 0.001:
            raise ValueError(f"Measurement {name} disagrees with its recorded atoms")


# Runs inside PyMOL. Index selections are constructed only after complete identity
# and coordinate checks, avoiding reliance on PDB serial numbers or guessing chains.
VIEW_SCRIPT = '''# Run from the bundle directory: pymol view.pml
python
import json
import hashlib
import math
from pathlib import Path
from pymol import cmd

record = json.loads(Path("site.json").read_text())
options = json.loads(Path("display.json").read_text())
source = Path("structure.pdb")
if hashlib.sha256(source.read_bytes()).hexdigest() != record["source"]["sha256"]:
    raise ValueError("Structure checksum mismatch")
cmd.delete("protein")
cmd.delete("geometric_reference")
cmd.delete("distance_*")
cmd.load(str(source), "protein")
cmd.hide("everything", "protein")
cmd.show("cartoon", "protein")
cmd.set("cartoon_transparency", 0.25, "protein")
cmd.set("label_size", 14)
cmd.set("label_color", "white")
cmd.bg_color("black")
model = cmd.get_model("protein", state=1)
palette = ["cyan", "salmon", "green", "violet", "yellow", "marine", "orange"]
for i, chain in enumerate(sorted(set(a.chain for a in model.atom))):
    indices = [a.index for a in model.atom if a.chain == chain]
    cmd.select("chain_display_" + str(i), "protein and index " + "+".join(map(str, indices)))
    cmd.color(palette[i % len(palette)], "chain_display_" + str(i))

selections = {}
for ref, atom in record["atoms"].items():
    resi = str(atom["auth_seq_id"]) + atom["insertion_code"]
    candidates = [a for a in model.atom if a.chain == atom["auth_chain_id"]
                  and a.resi == resi and a.resn == atom["resname"]
                  and a.name == atom["atom_name"] and a.alt.strip() == atom["altloc"]
                  and bool(a.hetatm) == (atom["hetero_flag"] != " ")]
    if len(candidates) != 1:
        raise ValueError("Ambiguous or missing PyMOL atom: " + ref)
    actual = candidates[0]
    if math.dist(actual.coord, atom["coordinates"]) > 0.001:
        raise ValueError("PyMOL coordinate mismatch: " + ref)
    selection = "recorded_" + ref
    cmd.select(selection, "protein and index " + str(actual.index))
    selections[ref] = selection
    label = f"{atom['auth_chain_id'] or '(blank chain)'}:{atom['resname']}{resi}/{atom['atom_name']}"
    if atom["altloc"]:
        label += " alt=" + atom["altloc"]
    cmd.label(selection, repr(label))

point = record["points"]["midpoint"]
cmd.pseudoatom("geometric_reference", pos=point["coordinates"], label=point["label"])
cmd.color("magenta", "geometric_reference")
cmd.show("spheres", "geometric_reference")
cmd.set("sphere_scale", 0.18, "geometric_reference")
selections["midpoint"] = "geometric_reference"
matched = [selections[r] for r in ("his1_NE2", "his2_NE2")]
cmd.select("matched_residues", "byres (" + " or ".join(matched) + ")")
cmd.show("sticks", "matched_residues")
cmd.color("yellow", "matched_residues")
cmd.select("local_context", "byres (protein within " + str(options["context_radius"]) + " of geometric_reference)")
cmd.show("sticks", "local_context")
cmd.select("observed_metals", "protein and elem CU+FE+ZN+MN+CO+NI+MG+CA+CD+NA+K")
cmd.show("spheres", "observed_metals")
cmd.set("sphere_scale", 0.35, "observed_metals")
cmd.show("sticks", "local_context and hetatm")
cmd.show("spheres", "local_context and solvent")
cmd.set("sphere_scale", 0.2, "local_context and solvent")

hx = record["hx"]
hx_indices = [a.index for a in model.atom if a.chain == hx["auth_chain_id"]
              and a.resi == str(hx["auth_seq_id"]) and a.resn == hx["resname"] and not a.hetatm]
if hx_indices:
    cmd.select("hx_residue", "protein and index " + "+".join(map(str, hx_indices)))
    cmd.show("sticks", "hx_residue")
    cmd.label("hx_residue and name CA", repr("HX " + hx["auth_chain_id"] + ":" + hx["resname"] + str(hx["auth_seq_id"])))

# Only filter-relevant minima are displayed; all retained measurements remain in site.json.
used = set(c["measurement"] for c in record["criteria"] if c["measurement"])
distant = []
for name in sorted(used):
    measurement = record["measurements"].get(name)
    if measurement is None:
        continue
    ends = [selections[e] for e in measurement["endpoints"]]
    actual = cmd.get_distance(ends[0], ends[1], state=1)
    if abs(actual - measurement["value"]) > 0.001:
        raise ValueError("PyMOL measurement mismatch: " + name)
    obj = "distance_" + name
    cmd.distance(obj, ends[0], ends[1], state=1)
    cmd.show("sticks", "byres (" + " or ".join(ends) + ") and protein")
    if "midpoint" in measurement["endpoints"] and measurement["value"] > options["context_radius"]:
        distant.append(obj)

cmd.orient("protein")
cmd.zoom("protein", 4)
cmd.scene("whole_protein", "store")
evidence = [selections[e] for name in used if name in record["measurements"]
            for e in record["measurements"][name]["endpoints"]]
cmd.zoom("local_context or " + " or ".join(sorted(set(evidence))), 3)
cmd.scene("all_criteria", "store")
for obj in distant:
    cmd.disable(obj)
cmd.hide("sticks", "protein and not (local_context or matched_residues)")
cmd.hide("labels", "protein and not (local_context or matched_residues)")
cmd.zoom("local_context or matched_residues or geometric_reference", 3)
cmd.scene("active_site", "store")
if options["save_session"]:
    cmd.save("site.pse")
if options["png"]:
    cmd.png("closeup.png", width=1600, height=1200, dpi=150, ray=1)
print("Verified site:", record["site_id"])
print("Scenes: active_site, all_criteria (includes distant witnesses), whole_protein")
for c in record["criteria"]:
    print(c["explanation"])
python end
'''


def export_bundle(results, site_id, outdir, source_override=None, context_radius=8.0,
                  save_session=False, png=False):
    if not math.isfinite(context_radius) or context_radius <= 0:
        raise ValueError("Context radius must be finite and positive")
    matches = []
    with Path(results).open() as handle:
        for line in handle:
            if line.strip():
                record = json.loads(line)
                if record.get("site_id") == site_id:
                    matches.append(record)
    if len(matches) != 1:
        raise ValueError("site_id must identify exactly one record")
    record = matches[0]
    source = Path(source_override or record["source"]["path"]).expanduser().resolve()
    validate_record(record, source)
    target = Path(outdir).expanduser().resolve()
    if target.exists() and (not target.is_dir() or any(target.iterdir())):
        raise ValueError("Bundle directory must be new or empty; existing files will not be overwritten")
    target.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target / "structure.pdb")
    (target / "site.json").write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    (target / "display.json").write_text(json.dumps({"context_radius": context_radius,
        "save_session": save_session, "png": png}, indent=2) + "\n")
    (target / "view.pml").write_text(VIEW_SCRIPT)
    (target / "README.txt").write_text(
        "From this directory run: pymol view.pml\n"
        "For headless PyMOL: pymol -cq view.pml\n"
        "Scene recall: scene active_site, recall (or all_criteria / whole_protein).\n"
        "all_criteria shows distant exclusion witnesses omitted from the close-up.\n"
        "Criteria and warnings are in site.json; the script prints criteria after verifying atoms.\n"
        "The magenta midpoint is a COMPUTED GEOMETRIC REFERENCE POINT, not a metal prediction.\n"
        "Deposited metals are shown without claims of enzyme function.\n"
        "Optional session/PNG files are created only when PyMOL runs the script.\n")
    return target
