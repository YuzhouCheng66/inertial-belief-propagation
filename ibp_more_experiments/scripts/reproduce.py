#!/usr/bin/env python3
"""Run the packaged I-BP benchmarks or a fast three-domain smoke test."""
from __future__ import annotations
import argparse
import json
import statistics
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BIN = ROOT / "build" / "ibp_parallel"
DATA = ROOT / "data"
RUNS = ROOT / "runs"

MAIN_CASES = [
    *(f"gauss128_s{s}" for s in (201, 202, 203)),
    *(f"ising128_s{s}" for s in (201, 202, 203)),
    *(f"tn32_s{s}" for s in (201, 202, 203)),
]


def call(case: str, method: int, threads: int, prefix: Path | None = None) -> dict:
    fixture = DATA / f"{case}.bin"
    if not fixture.exists():
        raise FileNotFoundError(f"Missing fixture: {fixture}")
    args = [str(BIN), str(fixture), str(method), str(threads), "5000", "0"]
    args += [str(prefix) if prefix else "-", "-"]
    p = subprocess.run(args, check=True, text=True, capture_output=True)
    result = json.loads(p.stdout.strip().splitlines()[-1])
    if not result.get("converged", False):
        raise RuntimeError(f"{case} method={method} threads={threads} did not converge: {result}")
    return result


def smoke() -> None:
    if not BIN.exists():
        raise FileNotFoundError("Build first with `make`.")
    checks = ["check_0", "check_1", "check_2"]
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for case in checks:
            for method in (0, 1):
                result = call(case, method, 1, tmp / f"{case}_m{method}")
                rows.append(result)
                print(json.dumps(result, sort_keys=True))
    assert len(rows) == 6 and all(r["converged"] for r in rows)
    print("SMOKE TEST PASSED: Gaussian, Ising, and tensor-network BP/I-BP all converged.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="Run the three tiny packaged fixtures and exit.")
    ap.add_argument("--threads", nargs="+", type=int, default=[1, 4])
    ap.add_argument("--repeats", type=int, default=1)
    ns = ap.parse_args()
    if ns.smoke:
        smoke(); return
    if not BIN.exists():
        raise FileNotFoundError("Build first with `make`.")
    RUNS.mkdir(exist_ok=True)
    records = []
    for case in MAIN_CASES:
        for method in (0, 1):
            for threads in ns.threads:
                timings = []
                canonical = None
                for rep in range(ns.repeats):
                    prefix = RUNS / f"{case}_m{method}_t{threads}" if (rep == 0 and threads == 1) else None
                    r = call(case, method, threads, prefix)
                    timings.append(r["seconds"])
                    canonical = r
                canonical = dict(canonical)
                canonical["seconds_median"] = statistics.median(timings)
                canonical["repeats"] = ns.repeats
                records.append(canonical)
                print(case, "I-BP" if method else "BP", f"{threads}T", f"{canonical['seconds_median']:.6g}s")
    out = RUNS / "timings.jsonl"
    out.write_text("\n".join(json.dumps(r, sort_keys=True) for r in records) + "\n")
    print(f"Wrote {out}")
    print("Use the packaged results for the paper figure, or adapt scripts/plot_results.py to runs/*.history.")

if __name__ == "__main__":
    main()
