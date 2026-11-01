import argparse
import json
import os
import subprocess

import matrix
from checkout import machine
from speed import GB, add_limit_arguments, failed, measure, seconds, summary


def git(repo, *args):
    return subprocess.run(["git", "-C", repo, *args], capture_output=True, text=True, check=True).stdout.strip()


def changes_library(repo, commit):
    return bool(git(repo, "diff", "--name-only", f"{commit}^", commit, "--", "py_wake", ":!py_wake/tests"))


def commits(repo, refs):
    selected = []
    for ref in refs:
        if ".." in ref:
            selected += [
                commit
                for commit in git(repo, "rev-list", "--reverse", "--first-parent", ref).split()
                if changes_library(repo, commit)
            ]
        else:
            selected.append(git(repo, "rev-parse", ref))
    return selected


def checkout(repo, commit, workdir):
    path = os.path.join(workdir, commit[:10])
    if not os.path.isdir(path):
        git(repo, "-c", "core.hooksPath=/dev/null", "worktree", "add", "--detach", "-q", path, commit)
    return {"commit": commit, "subject": git(repo, "log", "-1", "--format=%s", commit), "root": path}


def main():
    parser = argparse.ArgumentParser(description="Time a sequence of commits in alternating fresh processes")
    parser.add_argument("--repo", required=True)
    parser.add_argument("--commits", nargs="+", required=True, help="refs and A..B ranges of library commits")
    parser.add_argument("--workdir", required=True, help="directory for detached worktrees, reused between calls")
    parser.add_argument("--cases", nargs="*")
    parser.add_argument("--pairs", type=int, default=3)
    parser.add_argument("--out")
    add_limit_arguments(parser)
    args = parser.parse_args()
    os.makedirs(args.workdir, exist_ok=True)
    steps = [checkout(args.repo, commit, os.path.realpath(args.workdir)) for commit in commits(args.repo, args.commits)]
    report = {
        "pairs": args.pairs,
        "runs": args.runs,
        "python": args.python,
        "machine": machine(),
        "steps": steps,
        "results": {},
    }
    for case in matrix.names(args.cases):
        samples = [[] for _ in steps]
        for pair in range(args.pairs):
            order = range(len(steps)) if pair % 2 == 0 else reversed(range(len(steps)))
            for index in order:
                if not failed(samples[index]):
                    samples[index].append(measure(steps[index]["root"], case, args))
        summaries = [summary(step_samples) for step_samples in samples]
        report["results"][case] = summaries
        first = summaries[0]
        print(f"\n{case}\n\n| Commit | Subject | s | Min speed-up | GB |\n| --- | --- | ---: | ---: | ---: |")
        for step, result in zip(steps, summaries):
            speed_up = (
                "-" if "failed" in result or "failed" in first else f"{first['cpu_min'] / result['cpu_min']:.2f}x"
            )
            print(
                f"| {step['commit'][:8]} | {step['subject']} | {seconds(result)} | {speed_up} | {result['rss'] / GB:.2f} |"
            )
        if args.out:
            report["machine"]["load_end"] = os.getloadavg()
            with open(args.out, "w") as fid:
                json.dump(report, fid, indent=2)


if __name__ == "__main__":
    main()
