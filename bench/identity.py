import argparse
import json
import os
import sys

import numpy as np
from checkout import describe, use_checkout


def save(args):
    root = use_checkout(args.root)
    import cases

    os.makedirs(args.out, exist_ok=False)
    manifest = {"checkout": describe(root), "cases": {}}
    for name in cases.names(args.cases):
        case = cases.case(name)
        values = cases.outputs(case.run(case.build()))
        np.savez(os.path.join(args.out, f"{name}.npz"), **values)
        manifest["cases"][name] = {key: list(value.shape) for key, value in values.items()}
        print(f"saved {name}", flush=True)
    with open(os.path.join(args.out, "manifest.json"), "w") as fid:
        json.dump(manifest, fid, indent=2)


def difference(expected, actual):
    if expected.shape != actual.shape or expected.dtype != actual.dtype:
        return f"shape/dtype {expected.shape} {expected.dtype} != {actual.shape} {actual.dtype}"
    if expected.tobytes() == actual.tobytes():
        return None
    changed = int(np.sum(~((expected == actual) | (np.isnan(expected) & np.isnan(actual)))))
    return f"{changed} values differ, max abs {np.nanmax(np.abs(expected - actual)):.3e}"


def compare(args):
    with open(os.path.join(args.expected, "manifest.json")) as fid:
        expected_cases = json.load(fid)["cases"]
    with open(os.path.join(args.actual, "manifest.json")) as fid:
        actual_cases = json.load(fid)["cases"]
    failures = 0
    if set(expected_cases) != set(actual_cases):
        print(f"case sets differ: {sorted(set(expected_cases) ^ set(actual_cases))}")
        failures += 1
    for name in sorted(set(expected_cases) & set(actual_cases)):
        with (
            np.load(os.path.join(args.expected, f"{name}.npz"), allow_pickle=False) as expected,
            np.load(os.path.join(args.actual, f"{name}.npz"), allow_pickle=False) as actual,
        ):
            if set(expected.files) != set(actual.files):
                print(f"{name}: keys {sorted(expected.files)} != {sorted(actual.files)}")
                failures += 1
                continue
            problems = {key: difference(expected[key], actual[key]) for key in expected.files}
        problems = {key: problem for key, problem in problems.items() if problem}
        failures += bool(problems)
        print(f"{name}: " + ("identical" if not problems else "; ".join(f"{k} {v}" for k, v in problems.items())))
    print("IDENTICAL" if not failures else f"DIFFERENT ({failures} cases)")
    return int(failures > 0)


def main():
    parser = argparse.ArgumentParser(
        description="Save simulation outputs from a py_wake checkout and compare them bytewise"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    save_parser = commands.add_parser("save")
    save_parser.add_argument("--root", required=True)
    save_parser.add_argument("--out", required=True)
    save_parser.add_argument("--cases", nargs="*")
    compare_parser = commands.add_parser("compare")
    compare_parser.add_argument("expected")
    compare_parser.add_argument("actual")
    args = parser.parse_args()
    if args.command == "save":
        save(args)
        return 0
    return compare(args)


if __name__ == "__main__":
    sys.exit(main())
