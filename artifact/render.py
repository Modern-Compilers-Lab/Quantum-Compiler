"""Turn the summary CSVs into the paper's figures and tables.

    python artifact/render.py --list
    python artifact/render.py tab03
    python artifact/render.py all
    python artifact/render.py all --source committed

By default each item prefers a CSV from generate.py and falls back to the
committed one under dynamic-qlosure/results-summary/. --source generated
reads only generated CSVs; --source committed only the shipped ones.
Output goes to artifact/output/{figures,tables}/.
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import csv_sources
from common import OUTPUT_DIR, ensure_dir
from experiments import LOOP_ITERATIONS, RENDER_DEPENDENCIES

ITEMS = {
    "fig01": ("render.fig01_motivation", "Figure 1  - motivating comparison"),
    "fig08": ("render.fig08_dqueko", "Figure 8  - d-QUEKO routing on IBM Brisbane"),
    "fig09": ("render.fig09_nested", "Figure 9  - nested while loops"),
    "tab03": ("render.tab03_main", "Table 3   - improvements over Sabre"),
    "tab04": ("render.tab04_chiplet", "Table 4   - chiplet QPUs"),
    "tab05": ("render.tab05_surface_code", "Table 5   - surface-code circuits"),
    "tab06": ("render.tab06_time", "Table 6   - mapping time"),
    "fig10": ("render.fig10_ablation", "Figure 10 - ablation study on Brisbane"),
    "tab07": ("render.tab07_ablation", "Table 7   - ablation over the Recon baseline"),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("items", nargs="*", default=["all"],
                    help="one or more of: " + ", ".join(ITEMS) + ", or 'all'")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--source", choices=["auto", "generated", "committed"],
                    default="auto",
                    help="auto (default): prefer CSVs from generate.py, fall back "
                         "to the committed ones. generated: only your own CSVs, "
                         "skipping items you have not generated. committed: only "
                         "the shipped CSVs.")
    ap.add_argument("--csv-root", type=Path, default=csv_sources.GENERATED_ROOT)
    ap.add_argument("--loop-iterations", type=int, default=LOOP_ITERATIONS)
    args = ap.parse_args()

    if args.list:
        for key, (_, desc) in ITEMS.items():
            print(f"  {key}  {desc}   (needs: "
                  f"{', '.join(RENDER_DEPENDENCIES.get(key, []))})")
        return 0

    names = list(ITEMS) if "all" in args.items else args.items
    unknown = [n for n in names if n not in ITEMS]
    if unknown:
        raise SystemExit(f"unknown item(s): {unknown}. Known: {list(ITEMS)}")

    csv_sources.configure(args.source, args.csv_root)
    ensure_dir(OUTPUT_DIR)

    rendered, partial, failures = [], [], []
    for name in names:
        module_name, _ = ITEMS[name]
        try:
            result = importlib.import_module(module_name).run(args)
        except SystemExit as exc:
            print(f"\n  [SKIP] {name}: {exc}")
            failures.append(name)
            continue
        if isinstance(result, dict) and result.get("produced", 1) < result.get("total", 1):
            partial.append((name, result["produced"], result["total"]))
        else:
            rendered.append(name)

    print("\n" + "=" * 78)
    print("  summary")
    print("=" * 78)
    for name in rendered:
        print(f"  {name}  rendered")
    for name, produced, total in partial:
        print(f"  {name}  PARTIAL - {produced}/{total} parts rendered, "
              f"{total - produced} skipped for missing input")
    for name in failures:
        print(f"  {name}  skipped (missing input)")

    # list which CSVs fell back to the committed set
    resolved = csv_sources.used()
    generated = list(dict.fromkeys(rel for rel, o in resolved if o == "generated"))
    committed = list(dict.fromkeys(rel for rel, o in resolved if o == "committed"))
    if resolved:
        print(f"\n  CSV sources: {len(generated)} generated, {len(committed)} committed")
        if committed:
            print("  fell back to committed CSVs (not re-run):")
            for rel in committed:
                print(f"    - {rel}")
    print(f"  output in {OUTPUT_DIR}")
    return 1 if failures or partial else 0


if __name__ == "__main__":
    raise SystemExit(main())
