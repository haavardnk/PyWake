# Benchmarks

Fork tooling for comparing two py_wake checkouts. It is not part of any upstream merge request.

Every script takes `--root`, a py_wake checkout, and imports `py_wake` from there. One environment
(numpy, numba, xarray, scipy) can therefore time and compare any two commits, for example a
`git worktree` of `master` against a branch.

```sh
python bench/identity.py save --root ../pywake-master --out ref-master
python bench/identity.py save --root . --out head
python bench/identity.py compare ref-master head

python bench/speed.py --base ../pywake-master --head . --pairs 5 --out speed.json
python bench/speed.py --base ../pywake-master --head . --pairs 1 --runs 1 --max-memory 26 --cases turbopark-scale2000
python bench/waterfall.py --repo . --commits master master..performance --workdir ../waterfall --cases turbopark-grid400
python bench/profile_case.py --root . --case turbopark-grid400
```

- `identity.py` stores `WS_eff`, `TI_eff`, `CT`, `Power` and total AEP per case and compares them
  bytewise (`tobytes()`), so the comparison also catches last-bit differences.
- `speed.py` runs each case in fresh, single-threaded processes, alternating base and head. It
  reports the median CPU time of the last run, the ratio of the fastest runs (robust when the machine
  is loaded, since load only adds time), the peak RSS of the process and the extra CPU time of the
  first run (numba compilation). Each run starts after `gc.collect()`: py_wake results sit in
  reference cycles, so without it the previous run's arrays can survive into the next run and
  inflate its peak memory by an amount that depends on garbage-collector timing. A run is killed and
  reported as `killed` when it exceeds `--timeout` or its memory footprint exceeds `--max-memory` GB,
  or when less than `--min-free` percent of system memory is free (default 15, `0` disables it); its
  row still shows the RSS reached and the side is not run again for that case. On macOS the footprint
  is the kernel's physical footprint and the free share is `kern.memorystatus_level`: under memory
  pressure macOS compresses pages, which leaves them out of RSS (master `turbopark-scale2000` reached
  a 26 GB footprint at 12.6 GB RSS), so an RSS limit can let a run exhaust the machine. Elsewhere both
  come from psutil (RSS and available memory). The JSON records the peak footprint next to the RSS. With `--runs 1` the timed run includes numba
  compilation. The JSON output records both checkouts and the machine.
- `waterfall.py` times a sequence of commits the same way, alternating their order between pairs,
  and reports the fastest-run ratio against the first commit. Ranges expand to the first-parent
  commits that change library code under `py_wake/`, outside the tests; each commit is checked out
  once as a detached worktree under `--workdir` (remove them with `git worktree remove`).
- Cases are `<model>-<farm>`. Models: `turbopark` (`Nygaard_2022`: TurboGaussian, Mirror, Gaussian
  overlap, SquaredSum, deficit scaled with the downstream ambient speed), `nojlocal` (`NOJLocalDeficit`
  with area overlap, `ct2a_mom1d`, `a=[0, 0.04]`, no effective TI, LinearSum), `fugamultilut` (two LUTs),
  `noj`, `bastankhah`, `niayifar` (Gaussian with CrespoHernandez added turbulence), `fuga`. Farms: `hr1`
  (Horns Rev 1, 360 wd x 23 ws), `grid400` (400 WTGs of two types, 72 wd x 22 ws), `aligned100`
  (100 WTGs on an exact grid, 24 wd including the grid axes, so turbines sit exactly side by side)
  and, outside the default matrix, `scale<N>` (N WTGs laid out like `grid400`, same 72 wd x 22 ws;
  `scale400` is `grid400`) and `scale<N>c<k>` (the same farm run with `wd_chunks=k`).
