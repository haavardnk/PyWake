import argparse
import json
import os
import statistics
import subprocess
import sys
import tempfile
import threading

import matrix

SINGLE_THREAD = {
    key: "1"
    for key in (
        "NUMBA_NUM_THREADS",
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    )
}
WORKER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "worker.py")
RSS_UNIT = 1 if sys.platform == "darwin" else 1024


def measure(python, root, case, runs, timeout):
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "timing.json")
        process = subprocess.Popen(
            [python, WORKER, "--root", root, "--case", case, "--out", out, "--runs", str(runs)],
            env={**os.environ, **SINGLE_THREAD},
        )
        timer = threading.Timer(timeout, process.kill)
        timer.start()
        try:
            _, status, usage = os.wait4(process.pid, 0)
        finally:
            timer.cancel()
        process.returncode = os.waitstatus_to_exitcode(status)
        if process.returncode != 0:
            raise RuntimeError(f"{case} failed in {root} with exit code {process.returncode}")
        with open(out) as fid:
            timing = json.load(fid)
    timing["rss"] = usage.ru_maxrss * RSS_UNIT
    return timing


def summary(samples):
    first = [sample["runs"][0]["cpu"] for sample in samples]
    last = [sample["runs"][-1]["cpu"] for sample in samples]
    return {
        "cpu": statistics.median(last),
        "cpu_min": min(last),
        "wall": statistics.median(sample["runs"][-1]["wall"] for sample in samples),
        "first_extra": statistics.median(f - w for f, w in zip(first, last)),
        "build": statistics.median(sample["build"] for sample in samples),
        "rss": statistics.median(sample["rss"] for sample in samples),
        "cpu_samples": last,
    }


def main():
    parser = argparse.ArgumentParser(description="Compare two py_wake checkouts in alternating fresh processes")
    parser.add_argument("--base", required=True)
    parser.add_argument("--head", required=True)
    parser.add_argument("--cases", nargs="*")
    parser.add_argument("--pairs", type=int, default=5)
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=3600)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--out")
    args = parser.parse_args()
    results = {}
    print("| Case | Base s | Head s | Speed-up | Min speed-up | Base GB | Head GB | Memory | Head first-run extra s |")
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for case in matrix.names(args.cases):
        samples = {"base": [], "head": []}
        for pair in range(args.pairs):
            order = ("base", "head") if pair % 2 == 0 else ("head", "base")
            for side in order:
                samples[side].append(measure(args.python, getattr(args, side), case, args.runs, args.timeout))
        base, head = summary(samples["base"]), summary(samples["head"])
        results[case] = {"base": base, "head": head}
        print(
            f"| {case} | {base['cpu']:.2f} | {head['cpu']:.2f} | {base['cpu'] / head['cpu']:.2f}x | "
            f"{base['cpu_min'] / head['cpu_min']:.2f}x | "
            f"{base['rss'] / 1e9:.2f} | {head['rss'] / 1e9:.2f} | {head['rss'] / base['rss']:.2f} | "
            f"{head['first_extra']:.2f} |",
            flush=True,
        )
    if args.out:
        with open(args.out, "w") as fid:
            json.dump({"base": args.base, "head": args.head, "pairs": args.pairs, "results": results}, fid, indent=2)


if __name__ == "__main__":
    main()
