import argparse
import json
import time

from checkout import use_checkout


def main():
    parser = argparse.ArgumentParser(description="Time one benchmark case in this process")
    parser.add_argument("--root", required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--runs", type=int, default=2)
    args = parser.parse_args()
    start = time.perf_counter()
    use_checkout(args.root)
    import cases

    imported = time.perf_counter()
    case = cases.case(args.case)
    wfm = case.build()
    built = time.perf_counter()
    runs = []
    for _ in range(args.runs):
        cpu, wall = time.process_time(), time.perf_counter()
        case.run(wfm)
        runs.append({"cpu": time.process_time() - cpu, "wall": time.perf_counter() - wall})
    with open(args.out, "w") as fid:
        json.dump({"import": imported - start, "build": built - imported, "runs": runs}, fid)


if __name__ == "__main__":
    main()
