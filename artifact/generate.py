"""Run DynamiQ and Qiskit Sabre, then write the summary CSVs that render.py reads.

    python artifact/generate.py --list
    python artifact/generate.py main
    python artifact/generate.py main --widths 54 --leaf-depths 10 20 --jobs 4
    python artifact/generate.py all --jobs 8

Traces land in dynamic-qlosure/results/{qlosure,sabre}/... and are reused on a
rerun, so an interrupted generate resumes where it stopped. Pass --force to
recompute. CSVs land in artifact/output/csv/ by default.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (OUTPUT_DIR, RESULTS_ROOT, SURFACE_CODE_DIR,
                    banner, ensure_dir, saved)
from experiments import (ABLATION_SUFFIX, EXPERIMENTS, LOOP_ITERATIONS,
                         SELECTION_NOTE, dqueko_jobs, wi_rule_jobs)
from runners import METRIC_KEYS, execute_one, load_backend, score_trace

CSV_ROOT = OUTPUT_DIR / "csv"
METHODS = {"dynamiq": "qlosure", "sabre": "sabre"}


# ── job construction ────────────────────────────────────────────────────────

def build_dqueko_jobs(exp, widths, leaf_depths, force, ablation=None):
    jobs = []
    for backend in exp.backends:
        benches = exp.benches[backend] if isinstance(exp.benches, dict) else exp.benches
        for bench in benches:
            if widths and int(bench.replace("qbt", "")) not in widths:
                continue
            methods = ["dynamiq"] if getattr(exp, "dynamiq_only", False) else ["dynamiq", "sabre"]
            for circuit, config, seed in dqueko_jobs(exp.template, backend, bench,
                                                     exp.seeds, leaf_depths):
                if not circuit.exists():
                    print(f"  [missing] {circuit}")
                    continue
                for method in methods:
                    out = (RESULTS_ROOT / METHODS[method] / exp.template / backend /
                           bench / config / f"SEED_{seed}")
                    jobs.append(dict(method=method, backend=backend, circuit=str(circuit),
                                     seed=seed, out_dir=str(out), force=force,
                                     ablation=ablation))
    return jobs


def build_ablation_jobs(exp, widths, leaf_depths, force):
    """One DynamiQ run per ablation rung, each in its own trace tree."""
    jobs = []
    for cfg in exp.configs:
        for backend in exp.backends:
            for bench in exp.benches:
                if widths and int(bench.replace("qbt", "")) not in widths:
                    continue
                for circuit, config, seed in dqueko_jobs(exp.template, backend, bench,
                                                         exp.seeds, leaf_depths):
                    if not circuit.exists():
                        print(f"  [missing] {circuit}")
                        continue
                    out = (RESULTS_ROOT / f"qlosure_abl{cfg}" / exp.template / backend /
                           bench / config / f"SEED_{seed}")
                    jobs.append(dict(method="dynamiq", backend=backend,
                                     circuit=str(circuit), seed=seed,
                                     out_dir=str(out), force=force, ablation=cfg))
    return jobs


def build_wi_rule_jobs(exp, force):
    jobs = []
    for circuit, config, seed in wi_rule_jobs(exp.bench, exp.leaf_depth, exp.seeds):
        for method in ("dynamiq", "sabre"):
            out = (RESULTS_ROOT / METHODS[method] / "wi_rule" / exp.backend / exp.bench /
                   f"{exp.leaf_depth}Leaf_depth" / config / f"SEED_{seed}")
            jobs.append(dict(method=method, backend=exp.backend, circuit=str(circuit),
                             seed=seed, out_dir=str(out), force=force))
    return jobs


def build_surface_jobs(exp, force):
    jobs = []
    for cfg in exp.configs:
        root = SURFACE_CODE_DIR / cfg["benchmarks"]
        if not root.is_dir():
            print(f"  [missing] {root} - run surface-code/generate_surface_code.py first")
            continue
        backend, _, edges = load_backend(cfg["backend"])
        cap = backend.num_qubits
        for d_dir in sorted(p for p in root.iterdir() if p.is_dir() and p.name.startswith("d")):
            for bench_dir in sorted(p for p in d_dir.iterdir() if p.is_dir()):
                rounds = int(bench_dir.name.split("_")[-1][1:])
                if cfg["rounds"] and rounds not in cfg["rounds"]:
                    continue
                meta = bench_dir / "bench.json"
                if meta.exists() and json.loads(meta.read_text())["total_qubits"] >= cap:
                    continue
                for circuit in sorted(bench_dir.glob("*.qasm")):
                    for seed in exp.seeds:
                        for method in ("dynamiq", "sabre"):
                            out = (RESULTS_ROOT / METHODS[method] / cfg["tag"] / cfg["backend"] /
                                   d_dir.name / bench_dir.name / circuit.stem / f"SEED_{seed}")
                            jobs.append(dict(method=method, backend=cfg["backend"],
                                             circuit=str(circuit), seed=seed,
                                             out_dir=str(out), force=force))
    return jobs


# ── execution ───────────────────────────────────────────────────────────────

def _run_serial(jobs, done, failed, total, t0, offset=0):
    for i, job in enumerate(jobs, 1 + offset):
        ok, msg = execute_one(job)
        done, failed = done + ok, failed + (not ok)
        _progress(i, total, job, ok, msg, t0)
    return done, failed


def run_jobs(jobs, n_jobs):
    """Run every job, halving the worker count and falling back to serial if
    the pool dies (an out-of-memory kill breaks the whole pool)."""
    if not jobs:
        print("  nothing to run")
        return 0, 0
    done = failed = 0
    t0 = time.time()
    total = len(jobs)
    print(f"  {total} mapping runs on {n_jobs} process(es)")

    pending = list(jobs)
    while pending and n_jobs > 1:
        try:
            with ProcessPoolExecutor(max_workers=n_jobs) as pool:
                futures = {pool.submit(execute_one, j): j for j in pending}
                for fut in as_completed(futures):
                    ok, msg = fut.result()
                    done, failed = done + ok, failed + (not ok)
                    _progress(done + failed, total, futures[fut], ok, msg, t0)
            pending = []
        except BrokenProcessPool:
            pending = [j for j in pending
                       if not (Path(j["out_dir"]) / "trace.json").exists()]
            n_jobs = max(1, n_jobs // 2)
            print(f"  [pool died - likely out of memory] retrying "
                  f"{len(pending)} remaining job(s) on {n_jobs} process(es)", flush=True)

    if pending:
        done, failed = _run_serial(pending, done, failed, total, t0, offset=done + failed)

    print(f"  {done} ok, {failed} failed in {time.time() - t0:.0f}s")
    return done, failed


def _progress(i, total, job, ok, msg, t0):
    if not ok:
        print(f"  [{i}/{total}] FAILED {job['method']} {Path(job['circuit']).parent.name}: {msg}")
    elif i % 10 == 0 or i == total:
        rate = (time.time() - t0) / i
        print(f"  [{i}/{total}] {msg}  eta {rate * (total - i):.0f}s")


# ── trace -> CSV ────────────────────────────────────────────────────────────

def collect(results_dir, qubit_props, loop_iterations, key_fn):
    rows = []
    for trace_file in Path(results_dir).rglob("trace.json"):
        key = key_fn(trace_file)
        if key is None:
            continue
        m = score_trace(json.loads(trace_file.read_text()), qubit_props, loop_iterations)
        rows.append({**key, **m})
    return pd.DataFrame(rows)


def leaf_depth_key(trace_file):
    for part in trace_file.parts:
        if "leaf-depth-" in part:
            return {"leaf_depth": int(part.split("leaf-depth-")[-1])}
    return None


def merge_pairwise(df_ours, df_sabre, group):
    def agg(df, suffix):
        if df.empty:
            return pd.DataFrame(columns=[group])
        g = df.groupby(group)[list(METRIC_KEYS)].agg(["mean", "std"]).reset_index()
        g.columns = [group] + [f"{c}_{s}_{suffix}" for c, s in g.columns[1:]]
        return g
    return pd.merge(agg(df_ours, "ours"), agg(df_sabre, "sabre"),
                    on=group, how="outer").sort_values(group)


def summarise_dqueko(exp, csv_root, loop_iterations, widths=None, leaf_depths=None):
    """Aggregate traces into one CSV per (backend, bench), restricted the same
    way as the run."""
    written = []
    for backend in exp.backends:
        benches = exp.benches[backend] if isinstance(exp.benches, dict) else exp.benches
        _, props, _ = load_backend(backend)
        for bench in benches:
            if widths and int(bench.replace("qbt", "")) not in widths:
                continue
            ours = RESULTS_ROOT / "qlosure" / exp.template / backend / bench
            sabre = RESULTS_ROOT / "sabre" / exp.template / backend / bench
            if not ours.exists():
                continue

            def key_fn(trace_file):
                key = leaf_depth_key(trace_file)
                if key is None or (leaf_depths and key["leaf_depth"] not in leaf_depths):
                    return None
                return key

            df = merge_pairwise(collect(ours, props, loop_iterations, key_fn),
                                collect(sabre, props, loop_iterations, key_fn),
                                "leaf_depth")
            if df.empty:
                continue
            tmpl = getattr(exp, "csv_overrides", {}).get(backend, exp.csv_rel)
            out = csv_root / tmpl.format(backend=backend, bench=bench, iters=loop_iterations)
            ensure_dir(out.parent)
            df.to_csv(out, index=False)
            written.append(out)
    return written


def summarise_wi_rule(exp, csv_root):
    _, props, _ = load_backend(exp.backend)
    rows = {}
    for method, sub in (("qroqi", "qlosure"), ("sabre", "sabre")):
        base = (RESULTS_ROOT / sub / "wi_rule" / exp.backend / exp.bench /
                f"{exp.leaf_depth}Leaf_depth")
        if not base.exists():
            continue
        for trace_file in base.rglob("trace.json"):
            folder = trace_file.relative_to(base).parts[0]
            m = score_trace(json.loads(trace_file.read_text()), props, LOOP_ITERATIONS)
            rows.setdefault(folder, {}).setdefault(method, []).append(m)

    out_rows = []
    for folder, per_method in sorted(rows.items()):
        row = {"folder": folder}
        for method in ("sabre", "qroqi"):
            vals = per_method.get(method, [])
            for m in METRIC_KEYS:
                series = [v[m] for v in vals]
                row[f"{method}_mean_{m}"] = np.mean(series) if series else np.nan
                row[f"{method}_std_{m}"] = np.std(series, ddof=1) if len(series) > 1 else 0.0
            row[f"{method}_n"] = len(vals)
        out_rows.append(row)
    out = csv_root / exp.csv_rel
    ensure_dir(out.parent)
    pd.DataFrame(out_rows).to_csv(out, index=False)
    return [out]


def summarise_surface(exp, csv_root):
    if str(SURFACE_CODE_DIR) not in sys.path:
        sys.path.insert(0, str(SURFACE_CODE_DIR))
    from compare_surface_code import collect_results, compute_summary

    written = []
    for cfg in exp.configs:
        _, props, _ = load_backend(cfg["backend"])
        li = cfg["loop_iterations"]
        ours = RESULTS_ROOT / "qlosure" / cfg["tag"] / cfg["backend"]
        sabre = RESULTS_ROOT / "sabre" / cfg["tag"] / cfg["backend"]
        if not ours.exists():
            continue
        d_ours = compute_summary(collect_results(ours, props, li))
        d_sabre = compute_summary(collect_results(sabre, props, li))
        if d_ours.empty:
            continue
        merged = pd.merge(d_ours, d_sabre, on=["distance", "rounds"],
                          suffixes=("_ours", "_sabre"), how="outer")
        out = csv_root / cfg["tag"] / f"{cfg['tag']}_{cfg['backend']}_comparison.csv"
        ensure_dir(out.parent)
        merged.to_csv(out, index=False)
        written.append(out)
    return written


def summarise_ablation(exp, csv_root, loop_iterations, widths=None, leaf_depths=None):
    """One CSV per width, with every rung side by side under its column suffix."""
    written = []
    for backend in exp.backends:
        _, props, _ = load_backend(backend)
        for bench in exp.benches:
            if widths and int(bench.replace("qbt", "")) not in widths:
                continue

            def key_fn(trace_file):
                key = leaf_depth_key(trace_file)
                if key is None or (leaf_depths and key["leaf_depth"] not in leaf_depths):
                    return None
                return key

            merged = None
            for cfg in exp.configs:
                root = RESULTS_ROOT / f"qlosure_abl{cfg}" / exp.template / backend / bench
                if not root.exists():
                    continue
                df = collect(root, props, loop_iterations, key_fn)
                if df.empty:
                    continue
                g = df.groupby("leaf_depth")[list(METRIC_KEYS)].agg(["mean", "std"]).reset_index()
                suffix = ABLATION_SUFFIX[cfg]
                g.columns = ["leaf_depth"] + [f"{c}_{st}_{suffix}" for c, st in g.columns[1:]]
                merged = g if merged is None else pd.merge(merged, g, on="leaf_depth", how="outer")
            if merged is None:
                continue
            out = csv_root / exp.csv_rel.format(backend=backend, bench=bench,
                                                iters=loop_iterations)
            ensure_dir(out.parent)
            merged.sort_values("leaf_depth").to_csv(out, index=False)
            written.append(out)
    return written


def summarise_timing(exp, csv_root, widths=None, leaf_depths=None):
    """Collect per-run mapping times, restricted the same way as the run."""
    rows = []
    for backend in exp.backends:
        for bench in exp.benches:
            if widths and int(bench.replace("qbt", "")) not in widths:
                continue
            base = RESULTS_ROOT / "qlosure" / exp.template / backend / bench
            if not base.exists():
                continue
            for time_file in base.rglob("time.txt"):
                key = leaf_depth_key(time_file)
                if not key or (leaf_depths and key["leaf_depth"] not in leaf_depths):
                    continue
                seed = next((p for p in time_file.parts if p.startswith("SEED_")), "")
                rows.append({"backend": backend, "qubits": int(bench.replace("qbt", "")),
                             "seed": int(seed.split("_")[-1] or 0),
                             "leaf_depth": key["leaf_depth"],
                             "time_s": float(time_file.read_text().strip())})
    out = csv_root / "time" / "mapping_time.csv"
    ensure_dir(out.parent)
    pd.DataFrame(rows).to_csv(out, index=False)
    return [out]


# ── driver ──────────────────────────────────────────────────────────────────

def generate(name, args):
    exp = EXPERIMENTS[name]
    banner(f"generate: {name}", f"kind={exp.kind}  seeds={exp.seeds}")
    if name in ("main", "chiplet"):
        print(f"  note: {SELECTION_NOTE}")

    if exp.kind == "dqueko":
        jobs = build_dqueko_jobs(exp, args.widths, args.leaf_depths, args.force, args.ablation)
    elif exp.kind == "ablation":
        jobs = build_ablation_jobs(exp, args.widths, args.leaf_depths, args.force)
    elif exp.kind == "wi_rule":
        jobs = build_wi_rule_jobs(exp, args.force)
    else:
        jobs = build_surface_jobs(exp, args.force)

    if not args.summarise_only:
        _, failed = run_jobs(jobs, args.jobs)
        if failed and not args.keep_going:
            raise SystemExit(f"  {failed} runs failed; rerun with --keep-going to summarise anyway")

    print("\n  summarising traces -> CSV")
    if exp.kind == "ablation":
        written = summarise_ablation(exp, args.csv_root, args.loop_iterations,
                                     args.widths, args.leaf_depths)
    elif exp.kind == "dqueko" and name == "timing":
        written = summarise_timing(exp, args.csv_root, args.widths, args.leaf_depths)
    elif exp.kind == "dqueko":
        written = summarise_dqueko(exp, args.csv_root, args.loop_iterations,
                                   args.widths, args.leaf_depths)
    elif exp.kind == "wi_rule":
        written = summarise_wi_rule(exp, args.csv_root)
    else:
        written = summarise_surface(exp, args.csv_root)
    for path in written:
        saved(path)
    return written


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("experiments", nargs="*", default=["all"],
                    help="one or more of: " + ", ".join(EXPERIMENTS) + ", or 'all'")
    ap.add_argument("--list", action="store_true", help="list experiments and exit")
    ap.add_argument("--jobs", type=int, default=1, help="parallel worker processes")
    ap.add_argument("--widths", nargs="+", type=int, help="restrict to these circuit widths")
    ap.add_argument("--leaf-depths", nargs="+", type=int, help="restrict to these leaf depths")
    ap.add_argument("--loop-iterations", type=int, default=LOOP_ITERATIONS)
    ap.add_argument("--csv-root", type=Path, default=CSV_ROOT)
    ap.add_argument("--force", action="store_true", help="recompute cached traces")
    ap.add_argument("--ablation", type=int, choices=[1, 2, 3, 4], default=None,
                    help="run DynamiQ at a rung of routing.ABLATION_CONFIGS "
                         "(1=distance only .. 4=full); default is the full method")
    ap.add_argument("--keep-going", action="store_true", help="summarise even if runs failed")
    ap.add_argument("--summarise-only", action="store_true",
                    help="skip mapping; rebuild CSVs from existing traces")
    args = ap.parse_args()

    if args.list:
        for name, exp in EXPERIMENTS.items():
            print(f"  {name:<14s} {exp.kind:<9s} -> {exp.csv_rel or '(custom)'}")
        return 0

    names = list(EXPERIMENTS) if "all" in args.experiments else args.experiments
    unknown = [n for n in names if n not in EXPERIMENTS]
    if unknown:
        raise SystemExit(f"unknown experiment(s): {unknown}. Known: {list(EXPERIMENTS)}")

    ensure_dir(args.csv_root)
    for name in names:
        generate(name, args)
    print(f"\n  CSVs written under {args.csv_root}")
    print("  next: python artifact/render.py all")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
