import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
from results import GB, MODELS, comparisons, read

SIDES = {"base": ("before", "o-"), "head": ("after", "s-")}
FARM_TICKS = [250, 500, 1000, 2000]


def scaling(args):
    rows = comparisons(args.inputs, args.footprint)
    models = [model for model in MODELS if any(row_model == model for _, row_model in rows)]
    figure, axes = plt.subplots(2, len(models), figsize=(2.6 * len(models), 5.6), sharex=True, squeeze=False)
    for column, model in enumerate(models):
        points = [(size, sides) for (size, row_model), sides in rows.items() if row_model == model]
        for side, (label, style) in SIDES.items():
            done = [(size, sides[side]) for size, sides in points if "failed" not in sides[side]]
            sizes = [size for size, _ in done]
            line = axes[0, column].plot(sizes, [run["cpu"] for _, run in done], style, label=label)[0]
            axes[1, column].plot(sizes, [run["memory"] / GB for _, run in done], style, color=line.get_color())
            footprints = [(size, run) for size, run in done if run["is_footprint"]]
            axes[1, column].plot(
                [size for size, _ in footprints],
                [run["memory"] / GB for _, run in footprints],
                style[0],
                color=line.get_color(),
                markerfacecolor="white",
            )
        axes[0, column].set_title(MODELS[model])
        axes[1, column].set_xlabel("wind turbines")
        for axis in axes[:, column]:
            axis.set_xscale("log")
            axis.set_yscale("log")
            axis.grid(True, which="both", alpha=0.3)
        axes[1, column].set_xticks(FARM_TICKS, [str(size) for size in FARM_TICKS])
        axes[1, column].xaxis.set_minor_formatter(NullFormatter())
    axes[0, 0].set_ylabel("CPU time (s)")
    axes[1, 0].set_ylabel("peak memory (GB, hollow: footprint)")
    axes[0, 0].legend()
    figure.tight_layout()
    figure.savefig(args.out, dpi=120)


def waterfall(args):
    report = read(args.input)
    labels = ["master"] + [step["subject"] for step in report["steps"][1:]]
    positions = range(len(labels))
    figure, (speed, memory) = plt.subplots(1, 2, figsize=(13, 0.3 * len(labels) + 1.5), sharey=True)
    for case in sorted(report["results"], key=lambda case: list(MODELS).index(case.split("-")[0])):
        summaries = report["results"][case]
        if "failed" in summaries[0]:
            continue
        complete = [(index, summary) for index, summary in enumerate(summaries) if "failed" not in summary]
        first = summaries[0]
        name = MODELS[case.split("-")[0]]
        indices = [index for index, _ in complete]
        speed.plot([first["cpu"] / summary["cpu"] for _, summary in complete], indices, "o-", ms=3, label=name)
        memory.plot([summary["rss"] / GB for _, summary in complete], indices, "o-", ms=3, label=name)
    speed.set_yticks(positions, labels, fontsize=7)
    speed.invert_yaxis()
    speed.set_xlabel("speed-up over master")
    memory.set_xlabel("peak memory (GB)")
    memory.set_xlim(left=0)
    for axis in (speed, memory):
        axis.grid(True, alpha=0.3)
    speed.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(args.out, dpi=120)


def main():
    parser = argparse.ArgumentParser(description="Plot speed.py and waterfall.py results")
    commands = parser.add_subparsers(dest="command", required=True)
    scaling_parser = commands.add_parser("scaling", help="time and memory against farm size, later files win")
    scaling_parser.add_argument("inputs", nargs="+")
    scaling_parser.add_argument(
        "--footprint", nargs="*", default=[], help="reports whose memory is the footprint, applied last"
    )
    scaling_parser.add_argument("--out", required=True)
    waterfall_parser = commands.add_parser("waterfall", help="speed-up and memory after each commit")
    waterfall_parser.add_argument("input")
    waterfall_parser.add_argument("--out", required=True)
    args = parser.parse_args()
    {"scaling": scaling, "waterfall": waterfall}[args.command](args)


if __name__ == "__main__":
    main()
