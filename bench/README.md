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
python bench/profile_case.py --root . --case ogtp-grid400
```

- `identity.py` stores `WS_eff`, `TI_eff`, `CT`, `Power` and total AEP per case and compares them
  bytewise (`tobytes()`), so the comparison also catches last-bit differences.
- `speed.py` runs each case in fresh, single-threaded processes, alternating base and head. It
  reports the median CPU time of the last run, the peak RSS of the process and the extra CPU time of
  the first run (numba compilation).
- Cases are `<model>-<farm>`. Models: `ogtp` (TurboGaussian, Mirror, Gaussian overlap, SquaredSum,
  deficit scaled with the downstream ambient speed), `park2` (NOJLocal with area overlap,
  LinearSum), `fugamultilut` (two LUTs), `turbopark`, `noj`, `bastankhah`, `fuga`. Farms: `hr1`
  (Horns Rev 1, 360 wd x 23 ws), `grid400` (400 WTGs of two types, 72 wd x 22 ws), `aligned100`
  (100 WTGs on an exact grid, 24 wd including the grid axes, so turbines sit exactly side by side)
  and `grid1000` (1000 WTGs, 12 wd, not in the default matrix).
