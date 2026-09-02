"""Run the whole pipeline: every experiment, then every figure and table.

    python artifact/run_all_experiments.py
    python artifact/run_all_experiments.py --jobs 8
    python artifact/run_all_experiments.py --render-only

Equivalent to running generate.py for each experiment followed by render.py all.
Traces are cached, so an interrupted run resumes where it stopped.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from experiments import EXPERIMENTS  # noqa: E402


def run(cmd):
    print(f"\n$ {' '.join(cmd)}", flush=True)
    return subprocess.run(cmd, cwd=HERE.parent).returncode


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--experiments", nargs="+", default=list(EXPERIMENTS),
                    help="restrict to these; default is all")
    ap.add_argument("--render-only", action="store_true",
                    help="skip generation, just rebuild figures and tables")
    ap.add_argument("--source", choices=["auto", "generated", "committed"], default="auto")
    args = ap.parse_args()

    t0 = time.time()
    failed = []

    if not args.render_only:
        for name in args.experiments:
            if name not in EXPERIMENTS:
                raise SystemExit(f"unknown experiment {name!r}. Known: {list(EXPERIMENTS)}")
            rc = run([sys.executable, "-u", "artifact/generate.py", name,
                      "--keep-going", "--jobs", str(args.jobs)])
            if rc != 0:
                failed.append(name)

    rc = run([sys.executable, "-u", "artifact/render.py", "all", "--source", args.source])

    print("\n" + "=" * 78)
    print(f"  finished in {(time.time() - t0) / 60:.0f} min")
    if failed:
        print(f"  experiments with failures: {', '.join(failed)}")
    print(f"  output in {HERE / 'output'}")
    return 1 if failed or rc != 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
