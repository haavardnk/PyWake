import re
from dataclasses import dataclass, field

import numpy as np
from matrix import MODEL_NAMES, names
from py_wake import NOJ
from py_wake.deficit_models import fuga
from py_wake.deficit_models.fuga import FugaMultiLUTDeficit
from py_wake.deficit_models.gaussian import IEA37SimpleBastankhahGaussian
from py_wake.deficit_models.noj import NOJLocalDeficit
from py_wake.deficit_models.utils import ct2a_mom1d
from py_wake.examples.data.hornsrev1 import Hornsrev1Site, HornsrevV80, wt_x, wt_y
from py_wake.literature.gaussian_models import Niayifar_PorteAgel_2016
from py_wake.literature.turbopark import Nygaard_2022
from py_wake.rotor_avg_models.area_overlap_model import AreaOverlapAvgModel
from py_wake.superposition_models import LinearSum
from py_wake.tests.test_files import tfp
from py_wake.wind_farm_models.engineering_models import PropagateDownwind
from py_wake.wind_turbines import WindTurbines
from py_wake.wind_turbines.generic_wind_turbines import GenericWindTurbine

FUGA_2MW = tfp + "fuga/2MW/Z0=0.03000000Zi=00401Zeta0=0.00E+00.nc"
FUGA_LUTS = [
    tfp + "fuga/LUTs_Zeta0=0.00_16_32_D80_zhub70_zi400_z0=0.00001000_z29.6-207.9_UL_nx512_ny128_dx20.0_dy5.0.nc",
    tfp + "fuga/LUTs_Zeta0=0.00_16_32_D120_zhub90_zi400_z0=0.00001000_z29.6-207.9_UL_nx512_ny128_dx30.0_dy7.5.nc",
]


def noj_local(site, wt):
    deficit = NOJLocalDeficit(
        ct2a=ct2a_mom1d, a=[0, 0.04], use_effective_ws=True, use_effective_ti=False, rotorAvgModel=AreaOverlapAvgModel()
    )
    return PropagateDownwind(site, wt, wake_deficitModel=deficit, superpositionModel=LinearSum())


def fuga_multilut(site, wt):
    return PropagateDownwind(
        site, wt, wake_deficitModel=FugaMultiLUTDeficit(LUT_path=FUGA_LUTS), superpositionModel=LinearSum()
    )


def fuga_2mw(site, wt):
    return fuga.Fuga(FUGA_2MW, site, wt)


MODELS = {
    "turbopark": Nygaard_2022,
    "nojlocal": noj_local,
    "fugamultilut": fuga_multilut,
    "noj": NOJ,
    "bastankhah": IEA37SimpleBastankhahGaussian,
    "niayifar": Niayifar_PorteAgel_2016,
    "fuga": fuga_2mw,
}
assert tuple(MODELS) == MODEL_NAMES
__all__ = ["case", "names", "outputs"]


@dataclass(frozen=True)
class Layout:
    x: np.ndarray
    y: np.ndarray
    types: np.ndarray
    wd: np.ndarray
    ws: np.ndarray
    wd_chunks: int | None = None


def two_types():
    return WindTurbines.from_WindTurbine_lst(
        [GenericWindTurbine("G80", 80, 70, power_norm=2000), GenericWindTurbine("G120", 120, 90, power_norm=4500)]
    )


def hornsrev1():
    return Layout(np.asarray(wt_x), np.asarray(wt_y), np.zeros(len(wt_x), dtype=int), np.arange(360), np.arange(3, 26))


def grid(n_wtg, wd_step, wd_chunks=None, jitter=50.0):
    rng = np.random.default_rng(n_wtg)
    side = int(np.ceil(np.sqrt(n_wtg)))
    ix, iy = np.divmod(np.arange(n_wtg), side)
    x = ix * 700.0 + rng.uniform(-jitter, jitter, n_wtg)
    y = iy * 700.0 + rng.uniform(-jitter, jitter, n_wtg)
    return Layout(x, y, (ix + iy) % 2, np.arange(0, 360, wd_step), np.arange(4, 26), wd_chunks)


@dataclass(frozen=True)
class Case:
    model: str
    farm: str
    site: object = field(repr=False)
    wind_turbines: object = field(repr=False)
    layout: Layout = field(repr=False)

    @property
    def name(self):
        return f"{self.model}-{self.farm}"

    def build(self):
        return MODELS[self.model](self.site, self.wind_turbines)

    def run(self, wfm):
        layout = self.layout
        return wfm(
            layout.x, layout.y, type=layout.types, wd=layout.wd, ws=layout.ws, wd_chunks=layout.wd_chunks, n_cpu=1
        )


FARMS = {
    "hr1": lambda: (HornsrevV80(), hornsrev1()),
    "grid400": lambda: (two_types(), grid(400, 5)),
    "aligned100": lambda: (two_types(), grid(100, 15, jitter=0.0)),
}
SCALE = re.compile(r"scale([1-9]\d*)(?:c([1-9]\d*))?")


def farm(name):
    if name in FARMS:
        return FARMS[name]()
    match = SCALE.fullmatch(name)
    if match is None:
        raise KeyError(f"unknown farm {name}")
    n_wtg, wd_chunks = match.groups()
    return two_types(), grid(int(n_wtg), 5, wd_chunks=int(wd_chunks) if wd_chunks else None)


def case(name):
    model, farm_name = name.split("-")
    wind_turbines, layout = farm(farm_name)
    return Case(model, farm_name, Hornsrev1Site(), wind_turbines, layout)


def outputs(sim_res):
    return {
        "WS_eff": sim_res.WS_eff.values,
        "TI_eff": sim_res.TI_eff.values,
        "CT": sim_res.CT.values,
        "Power": sim_res.Power.values,
        "aep": np.asarray(sim_res.aep().sum().values),
    }
