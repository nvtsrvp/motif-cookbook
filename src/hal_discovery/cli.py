import argparse
from pathlib import Path

from .mine import run_mine
from .annotate import run_annotate


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="hal-discovery",
        description="Structural mining of 2-His metal sites from PDBs + optional UniProt annotation filtering."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ---------------- mine ----------------
    p_mine = sub.add_parser("mine", help="Scan a folder of PDBs and identify 2-His sites and putative halogenases.")
    p_mine.add_argument("--pdb-dir", required=True, help="Folder containing PDB files named like <UniProtAccession>.pdb")
    p_mine.add_argument("--outdir", required=True, help="Output folder for all generated results (will be created).")

    # geometry / rules
    p_mine.add_argument("--his-ne2-cutoff", type=float, default=4.0, help="Max NE2–NE2 distance to define 2-His pair.")
    p_mine.add_argument("--his-hbond-cutoff", type=float, default=4.0, help="Max O(i)–N(j) / N(i)–O(j) distances (H-bond-like).")
    p_mine.add_argument("--asp-glu-cutoff", type=float, default=5.0, help="Max distance from 2-His midpoint to Asp/Glu key atoms.")
    p_mine.add_argument("--ala-gly-cutoff", type=float, default=4.5, help="Max distance from 2-His midpoint to Ala/Gly atoms (optional category).")

    # HX rule
    p_mine.add_argument("--require-hx", action="store_true", default=True, help="Require HX motif check (default: on).")
    p_mine.add_argument("--no-require-hx", dest="require_hx", action="store_false", help="Disable HX motif check.")
    p_mine.add_argument("--hx-offset", type=int, default=2, help="Offset from first His residue index used for HX residue (default: +2).")

    # putative halogenase filters (notebook-like defaults)
    p_mine.add_argument("--min-length", type=int, default=200, help="Minimum protein length for downstream filters.")
    p_mine.add_argument("--acid-exclude-cutoff", type=float, default=5.5, help="Require ASP/GLU distances > this for halogenase-like sites.")
    p_mine.add_argument("--exclude-ligand-cutoff", type=float, default=4.0, help="Require other potential ligands > this distance.")
    p_mine.add_argument("--ala-gly-presence-cutoff", type=float, default=7.0, help="Require ALA or GLY within this distance for halogenase-like sites.")
    p_mine.add_argument("--hx-allowed", nargs="*", default=["ALA", "GLY"], help="Allowed HX residue identities if --require-hx is used.")
    p_mine.add_argument("--triad-min-length",type=int, default=0, help="Minimum protein length for counting 2-His-1-Asp/Glu facial triad sites (default: 0).")

    # outputs / performance
    p_mine.add_argument("--progress-every", type=int, default=200, help="Print progress every N PDBs.")
    p_mine.add_argument("--max-pdbs", type=int, default=None, help="Optional: process only first N PDBs (debug).")
    p_mine.add_argument("--detailed-results", action="store_true", help="Also write explainable site records and processing diagnostics; existing CSVs are unchanged.")

    p_export = sub.add_parser("export-pymol", help="Export one recorded legacy filter match as a portable PyMOL bundle.")
    p_export.add_argument("--results", required=True, help="site_matches_*.jsonl from mine --detailed-results")
    p_export.add_argument("--site-id", required=True, help="Exact site_id from the detailed results")
    p_export.add_argument("--outdir", required=True, help="New or empty bundle directory")
    p_export.add_argument("--structure", help="Override source path, with identical SHA-256 required")
    p_export.add_argument("--context-radius", type=float, default=8.0, help="Display neighborhood in Angstrom; does not change filters")
    p_export.add_argument("--save-session", action="store_true", help="Have PyMOL save site.pse when the script is run")
    p_export.add_argument("--png", action="store_true", help="Have PyMOL render closeup.png when the script is run")

    # ---------------- annotate ----------------
    p_ann = sub.add_parser("annotate", help="Merge mined hits with UniProt idmapping.fasta and apply keyword filtering.")
    p_ann.add_argument("--hits-csv", required=True, help="CSV from `mine` (putative_halogenases_*.csv).")
    p_ann.add_argument("--idmapping-fasta", required=True, help="FASTA from UniProt ID mapping tool (idmapping.fasta).")
    p_ann.add_argument("--outdir", required=True, help="Output folder (will be created).")

    p_ann.add_argument(
        "--exclude-keywords",
        nargs="*",
        default=[
            "transcription", "regulator", "AraC", "globin", "AlkB", "glutelin", "TehB",
            "tet", "fragment", "chemotaxis", "helix-turn-helix", "tellurite",
            "adenosyl", "SAM", "ferredoxin",
        ],
        help="Keywords used to exclude obvious false positives (case-insensitive match on UniProt function)."
    )

    args = parser.parse_args()

    if args.command == "mine":
        run_mine(args)
    elif args.command == "annotate":
        run_annotate(args)
    elif args.command == "export-pymol":
        from .pymol_export import export_bundle
        try:
            bundle = export_bundle(args.results, args.site_id, args.outdir, args.structure,
                                   args.context_radius, args.save_session, args.png)
        except (ValueError, OSError) as exc:
            parser.error(str(exc))
        print(f"Wrote {bundle}; run PyMOL from that directory: pymol view.pml")
