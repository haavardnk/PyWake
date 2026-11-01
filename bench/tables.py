import argparse

from results import GB, MODELS, comparisons, read


def rounded(value, two_decimals_below, one_decimal_below):
    if round(value, 2) < two_decimals_below:
        return f"{value:.2f}"
    if round(value, 1) < one_decimal_below:
        return f"{value:.1f}"
    return f"{value:.0f}"


def seconds(value):
    return f"{rounded(value, 1, 100)} s"


def gigabytes(value):
    return f"{rounded(value / GB, 10, 100)} GB"


def factor(value):
    return f"{rounded(value, 0, 10)}x"


def markdown_row(cells):
    return "|" + "|".join(f" {cell} " if cell else " " for cell in cells) + "|"


def time_cell(side):
    return "killed" if "failed" in side else seconds(side["cpu"])


def memory_cell(value, side):
    return (">" if "failed" in side else "") + gigabytes(value)


def compare(args):
    print("| Farm | Model | Time before | Time after | Speed-up | Memory before | Memory after | Less memory |")
    print("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    previous = None
    for (size, model), sides in comparisons(args.inputs, args.footprint).items():
        if args.farms and size not in args.farms:
            continue
        base, head = sides["base"], sides["head"]
        complete = "failed" not in base and "failed" not in head
        cells = [
            f"{size} WTGs" if size != previous else "",
            MODELS[model],
            time_cell(base),
            time_cell(head),
            factor(base["cpu"] / head["cpu"]) if complete else "-",
            memory_cell(base["memory"], base),
            memory_cell(head["memory"], head),
            factor(base["memory"] / head["memory"]) if complete else "-",
        ]
        previous = size
        print(markdown_row(cells))


def waterfall(args):
    report = read(args.input)
    cases = sorted(report["results"], key=lambda case: list(MODELS).index(case.split("-")[0]))
    names = [MODELS[case.split("-")[0]] for case in cases]
    for title, cell in (("Time (speed-up over master)", time_step), ("Peak memory (less than master)", memory_step)):
        print(f"\n{title}\n")
        print("| Commit | Change | " + " | ".join(names) + " |")
        print("| --- | --- | " + " | ".join("---:" for _ in names) + " |")
        summaries = [report["results"][case] for case in cases]
        for index, step in enumerate(report["steps"]):
            cells = [cell(summary[0], summary[index], index > 0) for summary in summaries]
            change = "master" if index == 0 else step["subject"]
            print(markdown_row([step["commit"][:8], change, *cells]))


def time_step(first, step, with_factor):
    if "failed" in first or "failed" in step or not with_factor:
        return time_cell(step)
    return f"{seconds(step['cpu'])} ({factor(first['cpu'] / step['cpu'])})"


def memory_step(first, step, with_factor):
    if "failed" in first or "failed" in step or not with_factor:
        return memory_cell(step["rss"], step)
    return f"{gigabytes(step['rss'])} ({factor(first['rss'] / step['rss'])})"


def main():
    parser = argparse.ArgumentParser(description="Print speed.py and waterfall.py results as before and after tables")
    commands = parser.add_subparsers(dest="command", required=True)
    compare_parser = commands.add_parser("compare", help="one table from speed.py reports, later files win")
    compare_parser.add_argument("inputs", nargs="+")
    compare_parser.add_argument(
        "--footprint", nargs="*", default=[], help="reports whose memory is the footprint, applied last"
    )
    compare_parser.add_argument("--farms", nargs="*", type=int, help="farm sizes to keep")
    waterfall_parser = commands.add_parser("waterfall", help="per-commit tables from a waterfall.py report")
    waterfall_parser.add_argument("input")
    args = parser.parse_args()
    {"compare": compare, "waterfall": waterfall}[args.command](args)


if __name__ == "__main__":
    main()
