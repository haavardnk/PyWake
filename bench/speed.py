import argparse
import json
import os
import signal
import statistics
import subprocess
import sys
import tempfile
import time

import matrix
from checkout import describe, machine
from memory import footprint, free_percent

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
GB = 1e9


def add_limit_arguments(parser):
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=3600)
    parser.add_argument("--max-memory", type=float, help="kill a run whose memory footprint exceeds this many GB")
    parser.add_argument(
        "--min-free", type=float, default=15, help="kill a run when less system memory is free, in percent"
    )
    parser.add_argument("--python", default=sys.executable)


def limit_breached(memory, started, args):
    if time.monotonic() - started > args.timeout:
        return f"time above {args.timeout:.0f} s"
    if args.max_memory and memory > args.max_memory * GB:
        return f"memory above {args.max_memory:g} GB"
    if free_percent() < args.min_free:
        return f"system memory below {args.min_free:g}% free"
    return None


def measure(root, case, args):
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "timing.json")
        process = subprocess.Popen(
            [args.python, WORKER, "--root", root, "--case", case, "--out", out, "--runs", str(args.runs)],
            env={**os.environ, **SINGLE_THREAD},
        )
        started, failed, peak = time.monotonic(), None, 0
        while (reaped := os.wait4(process.pid, os.WNOHANG))[0] == 0:
            memory = footprint(process.pid)
            peak = max(peak, memory)
            failed = failed or limit_breached(memory, started, args)
            if failed:
                os.kill(process.pid, signal.SIGKILL)
            time.sleep(0.1)
        _, status, usage = reaped
        process.returncode = os.waitstatus_to_exitcode(status)
        rss = usage.ru_maxrss * RSS_UNIT
        if failed:
            return {"failed": failed, "rss": rss, "footprint": peak}
        if process.returncode != 0:
            raise RuntimeError(f"{case} failed in {root} with exit code {process.returncode}")
        with open(out) as fid:
            timing = json.load(fid)
    timing["rss"] = rss
    timing["footprint"] = peak
    return timing


def summary(samples):
    failed = [sample["failed"] for sample in samples if "failed" in sample]
    if failed:
        return {
            "failed": f"{len(failed)}/{len(samples)} {failed[0]}",
            "rss": max(sample["rss"] for sample in samples),
            "footprint": max(sample["footprint"] for sample in samples),
        }
    first = [sample["runs"][0]["cpu"] for sample in samples]
    last = [sample["runs"][-1]["cpu"] for sample in samples]
    return {
        "cpu": statistics.median(last),
        "cpu_min": min(last),
        "wall": statistics.median(sample["runs"][-1]["wall"] for sample in samples),
        "first_extra": statistics.median(f - w for f, w in zip(first, last)) if len(samples[0]["runs"]) > 1 else None,
        "build": statistics.median(sample["build"] for sample in samples),
        "rss": statistics.median(sample["rss"] for sample in samples),
        "footprint": statistics.median(sample["footprint"] for sample in samples),
        "cpu_samples": last,
    }


def seconds(side, key="cpu"):
    if "failed" in side:
        return "killed"
    return "-" if side[key] is None else f"{side[key]:.2f}"


def failed(samples):
    return any("failed" in sample for sample in samples)


def row(case, base, head):
    complete = "failed" not in base and "failed" not in head
    speed_up, min_speed_up, memory = (
        (
            f"{base['cpu'] / head['cpu']:.2f}x",
            f"{base['cpu_min'] / head['cpu_min']:.2f}x",
            f"{head['rss'] / base['rss']:.2f}",
        )
        if complete
        else ("-",) * 3
    )
    cells = [
        case,
        seconds(base),
        seconds(head),
        speed_up,
        min_speed_up,
        f"{base['rss'] / GB:.2f}",
        f"{head['rss'] / GB:.2f}",
    ]
    return "| " + " | ".join([*cells, memory, seconds(head, "first_extra")]) + " |"


def main():
    parser = argparse.ArgumentParser(description="Compare two py_wake checkouts in alternating fresh processes")
    parser.add_argument("--base", required=True)
    parser.add_argument("--head", required=True)
    parser.add_argument("--cases", nargs="*")
    parser.add_argument("--pairs", type=int, default=5)
    parser.add_argument("--out")
    add_limit_arguments(parser)
    args = parser.parse_args()
    report = {
        "base": args.base,
        "head": args.head,
        "pairs": args.pairs,
        "runs": args.runs,
        "python": args.python,
        "checkouts": {"base": describe(os.path.realpath(args.base)), "head": describe(os.path.realpath(args.head))},
        "machine": machine(),
        "results": {},
    }
    print("| Case | Base s | Head s | Speed-up | Min speed-up | Base GB | Head GB | Memory | Head first-run extra s |")
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for case in matrix.names(args.cases):
        samples = {"base": [], "head": []}
        for pair in range(args.pairs):
            order = ("base", "head") if pair % 2 == 0 else ("head", "base")
            for side in order:
                if not failed(samples[side]):
                    samples[side].append(measure(getattr(args, side), case, args))
        base, head = summary(samples["base"]), summary(samples["head"])
        report["results"][case] = {"base": base, "head": head}
        print(row(case, base, head), flush=True)
        if args.out:
            report["machine"]["load_end"] = os.getloadavg()
            with open(args.out, "w") as fid:
                json.dump(report, fid, indent=2)


if __name__ == "__main__":
    main()
