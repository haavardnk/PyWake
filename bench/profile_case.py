import argparse
import cProfile
import pstats

from checkout import use_checkout


def main():
    parser = argparse.ArgumentParser(description="Profile one benchmark case after a warm-up run")
    parser.add_argument("--root", required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--sort", default="tottime")
    parser.add_argument("--top", type=int, default=30)
    parser.add_argument("--out")
    args = parser.parse_args()
    use_checkout(args.root)
    import cases

    case = cases.case(args.case)
    wfm = case.build()
    case.run(wfm)
    profiler = cProfile.Profile()
    profiler.runcall(case.run, wfm)
    if args.out:
        profiler.dump_stats(args.out)
    pstats.Stats(profiler).strip_dirs().sort_stats(args.sort).print_stats(args.top)


if __name__ == "__main__":
    main()
