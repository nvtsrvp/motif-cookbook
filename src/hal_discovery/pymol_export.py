"""Portable PyMOL scripts from recorded sites; no PyMOL dependency at export time."""
import json
import math
from pathlib import Path
import shutil

from .results import SCHEMA_VERSION, sha256
from .structure_io import load_structure, unpacked_atoms


def validate_record(record, source):
    if record.get("schema_version") not in {"1.0", SCHEMA_VERSION, "1.2", "1.3"}:
        raise ValueError("Unsupported site schema version")
    if not record.get("exportable"):
        raise ValueError("Site is ambiguous or uses a multi-model source and cannot be exported exactly; see record warnings")
    if sha256(source) != record["source"]["sha256"]:
        raise ValueError("Structure SHA-256 differs from the filter input")
    loaded = load_structure(source)
    structure = loaded.structure
    if len(structure) != 1:
        raise ValueError("Legacy export requires a single model")
    parsed = [loaded.record(a) for model in structure for a in unpacked_atoms(model)]
    keys = ("model_id", "model_number", "auth_chain_id", "hetero_flag", "auth_seq_id",
            "insertion_code", "resname", "atom_name", "altloc")
    if loaded.format == "mmcif":
        keys += ("label_asym_id", "label_seq_id", "label_entity_id", "label_atom_id", "atom_site_id")
    for ref, atom in record["atoms"].items():
        matches = [a for a in parsed if all(a[k] == atom[k] for k in keys)]
        if len(matches) != 1 or math.dist(matches[0]["coordinates"], atom["coordinates"]) > 0.001:
            raise ValueError(f"Recorded atom {ref} does not resolve uniquely to the source coordinates")
    coordinates = {name:atom['coordinates'] for name,atom in record['atoms'].items()}
    for name, point in record['points'].items():
        endpoints = point.get('endpoints', [])
        if not endpoints or any(e not in coordinates for e in endpoints):
            raise ValueError(f'Computed point {name} lacks source endpoints')
        actual = [math.fsum(coordinates[e][i] for e in endpoints)/len(endpoints) for i in range(3)]
        tolerance = 1e-6 if record['schema_version']=='1.3' else .001
        if math.dist(actual, point['coordinates']) > tolerance:
            raise ValueError(f'Computed point {name} disagrees with its source atoms')
        coordinates[name] = actual
    for name, direction in record.get('directions', {}).items():
        origin, target = (coordinates[direction[k]] for k in ('origin','target'))
        length = math.dist(origin,target)
        if length <= 1e-12:
            raise ValueError(f'Degenerate direction {name}')
        actual = [(b-a)/length for a,b in zip(origin,target)]
        if math.dist(actual,direction['vector']) > 1e-6:
            raise ValueError(f'Direction {name} disagrees with its source atoms')
    for name, measurement in record["measurements"].items():
        if measurement['method'] == 'minimum':
            actual = min(record['measurements'][key]['value'] for key in measurement['dependencies'])
            if abs(actual-measurement['value']) > 1e-6:
                raise ValueError(f'Measurement {name} disagrees with its dependencies')
            continue
        endpoints = [record["atoms"].get(e, record["points"].get(e)) for e in measurement["endpoints"]]
        if any(e is None for e in endpoints) or measurement["value"] is None:
            raise ValueError(f"Measurement {name} lacks an exact endpoint")
        positions = [coordinates[e] for e in measurement['endpoints']]
        if measurement["units"] == "degree":
            a, center, b = positions
            u, v = [x-y for x,y in zip(a, center)], [x-y for x,y in zip(b, center)]
            norm = math.dist(a, center) * math.dist(b, center)
            actual = math.degrees(math.acos(max(-1, min(1, sum(x*y for x,y in zip(u,v))/norm))))
        elif measurement['method'] == 'inward_dot':
            center, donor, other = positions
            u,v = [n-o for n,o in zip(donor,center)], [b-a for a,b in zip(center,other)]
            norm = math.dist(center,donor)*math.dist(center,other)
            if norm <= 1e-12:
                raise ValueError(f'Degenerate inward score {name}')
            actual = max(-1.,min(1.,math.fsum(x*y for x,y in zip(u,v))/norm))
        else:
            actual = math.dist(*positions)
        tolerance = 1e-6 if measurement['units']=='dimensionless' else .001
        if abs(actual - measurement["value"]) > tolerance:
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
source = Path(options.get("structure_filename", "structure.pdb"))
if hashlib.sha256(source.read_bytes()).hexdigest() != record["source"]["sha256"]:
    raise ValueError("Structure checksum mismatch")
cmd.delete("protein")
cmd.delete("geometric_reference")
cmd.delete("distance_*")
cmd.delete("angle_*")
# Use author identifiers consistently with Biopython and the recorded mapping.
cmd.set("cif_use_auth", 1)
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
coordinates = {}
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
    coordinates[ref] = list(actual.coord)
    label = f"{atom['auth_chain_id'] or '(blank chain)'}:{atom['resname']}{resi}/{atom['atom_name']}"
    if atom["altloc"]:
        label += " alt=" + atom["altloc"]
    cmd.label(selection, repr(label))

if "midpoint" in record["points"]:
    point = record["points"]["midpoint"]
    cmd.pseudoatom("geometric_reference", pos=point["coordinates"], label=point["label"])
    cmd.color("magenta", "geometric_reference")
    cmd.show("spheres", "geometric_reference")
    cmd.set("sphere_scale", 0.18, "geometric_reference")
    selections["midpoint"] = "geometric_reference"
    center = "geometric_reference"
else:
    for name, point in record["points"].items():
        ends = point["endpoints"]
        actual = [math.fsum(coordinates[e][i] for e in ends)/len(ends) for i in range(3)]
        if math.dist(actual,point["coordinates"]) > 0.001:
            raise ValueError("Computed point mismatch: " + name)
        obj = "computed_" + name
        cmd.delete(obj)
        label = point["label"]
        if name == "site_center" and record.get("visualization",{}).get("potential_site"):
            label = record["visualization"]["label"]
            label += " | s1=%.4f s2=%.4f min=%.4f" % tuple(record["measurements"][k]["value"] for k in ("s1","s2","inward_score"))
        cmd.pseudoatom(obj,pos=actual,label=label)
        cmd.show("nonbonded",obj)
        selections[name] = obj
        coordinates[name] = actual
    center = selections[record["visualization"]["center"]]
if record.get("directions"):
    from pymol.cgo import CYLINDER, CONE
    for name, direction in record["directions"].items():
        origin, target = (coordinates[direction[k]] for k in ("origin","target"))
        length = math.dist(origin,target)
        unit = [(b-a)/length for a,b in zip(origin,target)]
        if math.dist(unit,direction["vector"]) > 0.001:
            raise ValueError("Direction mismatch: " + name)
        end = [a+2.0*u for a,u in zip(origin,unit)]
        shaft = [a+1.65*u for a,u in zip(origin,unit)]
        color = (1.,0.3,0.3) if name=="his1" else ((0.3,1.,0.3) if name=="his2" else (0.8,0.8,0.8))
        arrow = [CYLINDER,*origin,*shaft,0.045,*color,*color,
                 CONE,*shaft,*end,0.12,0.,*color,*color,1.,1.]
        cmd.load_cgo(arrow,"direction_"+name)
matched_roles = record.get("visualization", {}).get("matched_atoms", ["his1_NE2", "his2_NE2"])
matched = [selections[r] for r in matched_roles]
cmd.select("matched_residues", "byres (" + " or ".join(matched) + ")")
cmd.show("sticks", "matched_residues")
cmd.color("yellow", "matched_residues")
cmd.select("local_context", "byres (protein within " + str(options["context_radius"]) + " of " + center + ")")
cmd.show("sticks", "local_context")
cmd.select("observed_metals", "protein and elem CU+FE+ZN+MN+CO+NI+MG+CA+CD+NA+K")
cmd.show("spheres", "observed_metals")
cmd.set("sphere_scale", 0.35, "observed_metals")
cmd.show("sticks", "local_context and hetatm")
cmd.show("spheres", "local_context and solvent")
cmd.set("sphere_scale", 0.2, "local_context and solvent")
if record.get("visualization",{}).get("potential_site"):
    cmd.hide("everything","observed_metals")

hx = record.get("hx", {})
hx_indices = [a.index for a in model.atom if a.chain == hx["auth_chain_id"]
              and a.resi == str(hx["auth_seq_id"]) and a.resn == hx["resname"] and not a.hetatm] if hx else []
if hx_indices:
    cmd.select("hx_residue", "protein and index " + "+".join(map(str, hx_indices)))
    cmd.show("sticks", "hx_residue")
    cmd.label("hx_residue and name CA", repr("HX " + hx["auth_chain_id"] + ":" + hx["resname"] + str(hx["auth_seq_id"])))

# Only filter-relevant minima are displayed; all retained measurements remain in site.json.
used = set(c["measurement"] for c in record["criteria"] if c["measurement"])
used.update(record.get("visualization", {}).get("display_measurements", []))
distant = []
for name in sorted(used):
    measurement = record["measurements"].get(name)
    if measurement is None:
        continue
    if measurement["units"] == "dimensionless":
        if measurement["method"] == "minimum":
            actual = min(record["measurements"][k]["value"] for k in measurement["dependencies"])
        else:
            o,n,other = [coordinates[e] for e in measurement["endpoints"]]
            u,v = [b-a for a,b in zip(o,n)],[b-a for a,b in zip(o,other)]
            actual = math.fsum(x*y for x,y in zip(u,v))/(math.dist(o,n)*math.dist(o,other))
        if abs(actual-measurement["value"]) > 0.000001:
            raise ValueError("Inward score mismatch: " + name)
        continue
    ends = [selections[e] for e in measurement["endpoints"]]
    is_angle = measurement["units"] == "degree"
    actual = cmd.get_angle(*ends, state=1) if is_angle else cmd.get_distance(*ends, state=1)
    if abs(actual - measurement["value"]) > 0.001:
        raise ValueError("PyMOL measurement mismatch: " + name)
    obj = ("angle_" if is_angle else "distance_") + name
    if is_angle:
        cmd.angle(obj, *ends, state=1)
    else:
        cmd.distance(obj, *ends, state=1)
    cmd.show("sticks", "byres (" + " or ".join(ends) + ") and protein")
    if not is_angle and "midpoint" in measurement["endpoints"] and measurement["value"] > options["context_radius"]:
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
cmd.zoom("local_context or matched_residues or " + center, 3)
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
    structure_filename = "structure.cif" if record["source"]["format"] == "mmcif" else "structure.pdb"
    shutil.copyfile(source, target / structure_filename)
    (target / "site.json").write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    (target / "display.json").write_text(json.dumps({"context_radius": context_radius, "structure_filename": structure_filename,
        "save_session": save_session, "png": png}, indent=2) + "\n")
    (target / "view.pml").write_text(VIEW_SCRIPT)
    (target / "README.txt").write_text(
        "From this directory run: pymol view.pml\n"
        "For headless PyMOL: pymol -cq view.pml\n"
        "Scene recall: scene active_site, recall (or all_criteria / whole_protein).\n"
        "all_criteria shows distant exclusion witnesses omitted from the close-up.\n"
        "Criteria and warnings are in site.json; the script prints criteria after verifying atoms.\n" +
        ("The magenta midpoint is a COMPUTED GEOMETRIC REFERENCE POINT, not a metal prediction.\n"
         if "midpoint" in record["points"] else
         "The view centers on an actual deposited Cu atom; no hypothetical metal is shown.\n"
         if 'copper' in record['atoms'] else
         "Computed ring/display references are not metal positions; no hypothetical copper is shown.\n") +
        "Deposited metals are shown without claims of enzyme function.\n"
        "Optional session/PNG files are created only when PyMOL runs the script.\n")
    return target
