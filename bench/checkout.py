import os
import platform
import subprocess
import sys


def use_checkout(root):
    root = os.path.realpath(root)
    sys.path.insert(0, root)
    import py_wake

    loaded = os.path.realpath(py_wake.__file__)
    if not loaded.startswith(root + os.sep):
        raise RuntimeError(f"py_wake loaded from {loaded}, expected a module under {root}")
    return root


def describe(root):
    import numba
    import numpy
    import scipy
    import xarray

    commit = subprocess.run(
        ["git", "-C", root, "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    dirty = (
        subprocess.run(
            ["git", "-C", root, "status", "--porcelain", "--untracked-files=no"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        != ""
    )
    return {
        "root": root,
        "commit": commit,
        "dirty": dirty,
        "python": sys.version.split()[0],
        "numpy": numpy.__version__,
        "numba": numba.__version__,
        "scipy": scipy.__version__,
        "xarray": xarray.__version__,
        "platform": sys.platform,
    }


def cpu_name():
    if sys.platform == "darwin":
        return subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True, check=True
        ).stdout.strip()
    with open("/proc/cpuinfo") as fid:
        return next(
            (line.split(":", 1)[1].strip() for line in fid if line.startswith("model name")), platform.machine()
        )


def machine():
    import psutil

    return {
        "cpu": cpu_name(),
        "cores": os.cpu_count(),
        "memory_gb": psutil.virtual_memory().total / 1e9,
        "platform": platform.platform(),
        "load": os.getloadavg(),
    }
