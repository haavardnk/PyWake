import functools
import numba
import numpy
from numpy import newaxis as na
from py_wake import np
from py_wake.turbulence_models.turbulence_model import TurbulenceModel
from py_wake.superposition_models import SqrMaxSum
from py_wake.utils.gradients import cabs
from py_wake.deficit_models.deficit_model import WakeRadiusTopHat
from py_wake.rotor_avg_models.area_overlap_model import AreaOverlapAvgModel
from py_wake.deficit_models.utils import ct2a_madsen


def _added_turbulence(ti, dw_ratio, cw, wake_radius, dw, TI_add):
    I, J, L, K = TI_add.shape
    radius_step = wake_radius.shape[3] > 1
    for i in range(I):
        for j in range(J):
            for l in range(L):
                ratio, cw_ijl, downstream = dw_ratio[i, j, l, 0], cw[i, j, l, 0], 1. if dw[i, j, l, 0] > 0 else 0.
                for k in range(K):
                    in_wake = ti[i, l, k] * ratio * (1. if cw_ijl < wake_radius[i, j, l, k * radius_step] else 0.)
                    TI_add[i, j, l, k] = in_wake * downstream


@functools.lru_cache
def _added_turbulence_kernel():
    ilk_in = numba.types.Array(numba.float64, 3, 'C', readonly=True)
    ijlk_in = numba.types.Array(numba.float64, 4, 'A', readonly=True)
    return numba.njit(numba.void(ilk_in, ijlk_in, ijlk_in, ijlk_in, ijlk_in, numba.float64[:, :, :, ::1]),
                      error_model='numpy')(_added_turbulence)


def _compiled_added_turbulence(ti_ijlk, dw_ratio_ijlk, cw_ijlk, wake_radius_ijlk, dw_ijlk):
    arrays = ti_ijlk, dw_ratio_ijlk, cw_ijlk, wake_radius_ijlk, dw_ijlk
    if any(type(x) is not numpy.ndarray or x.dtype != numpy.float64 or x.ndim != 4 or not x.flags.c_contiguous
           for x in arrays) or ti_ijlk.shape[1] != 1 or any(x.shape[3] != 1 for x in (dw_ratio_ijlk, cw_ijlk, dw_ijlk)):
        return None
    # NaN payloads depend on operand order, which LLVM may commute, unless every NaN is generated in the loop
    if numpy.isnan(ti_ijlk).any() or numpy.isnan(dw_ratio_ijlk).any():
        return None
    shape = numpy.broadcast_shapes(*[x.shape for x in arrays])
    I, J, L, K = shape
    ti_ilk = numpy.ascontiguousarray(numpy.broadcast_to(ti_ijlk[:, 0], (I, L, K)))
    TI_add_ijlk = numpy.empty(shape)
    _added_turbulence_kernel()(ti_ilk, *[numpy.broadcast_to(x, (I, J, L, x.shape[3])) for x in arrays[1:]], TI_add_ijlk)
    return TI_add_ijlk


class CrespoHernandez(TurbulenceModel, WakeRadiusTopHat):
    """
    Implemented according to:
    A. Crespo and J. Hernández
    Turbulence characteristics in wind-turbine wakes
    J. of Wind Eng. and Industrial Aero. 61 (1996) 71-85

    """

    def __init__(self, ct2a=ct2a_madsen, c=[0.73, 0.8325, -0.0325, -0.32],
                 addedTurbulenceSuperpositionModel=SqrMaxSum(),
                 rotorAvgModel=AreaOverlapAvgModel(), groundModel=None):
        TurbulenceModel.__init__(self, addedTurbulenceSuperpositionModel, rotorAvgModel=rotorAvgModel,
                                 groundModel=groundModel)
        self.c = c
        self.ct2a = ct2a

    def calc_added_turbulence(self, dw_ijlk, cw_ijlk, D_src_il, ct_ilk, TI_ilk, D_dst_ijl, wake_radius_ijlk, **_):
        """ Calculate the added turbulence intensity at locations specified by
        downstream distances (dw_jl) and crosswind distances (cw_jl)
        caused by the wake of a turbine (diameter: D_src_l, thrust coefficient: Ct_lk).

        Returns
        -------
        TI_add_ijlk: array:float
            Added turbulence intensity weighted by wake-turbine overlap [-]
        """
        # induction factor
        a_ilk = self.ct2a(ct_ilk)
        a_ilk = np.maximum(a_ilk, 1e-10)  # avoid error in gradient of a_ilk**0.8325 > 0.8325*a_ilk**-.1675
        # added turbulence (Eq. 21)
        dw_ijlk_gt0 = np.maximum(dw_ijlk, 1e-10)  # avoid divide by zero and sqrt of negative number
        ti_ijlk = self.c[0] * a_ilk[:, na, :, :]**self.c[1] * TI_ilk[:, na, :, :]**self.c[2]
        dw_ratio_ijlk = (dw_ijlk_gt0 / D_src_il[:, na, :, na])**self.c[3]
        TI_add_ijlk = _compiled_added_turbulence(ti_ijlk, dw_ratio_ijlk, cw_ijlk, wake_radius_ijlk, dw_ijlk)
        if TI_add_ijlk is not None:
            return TI_add_ijlk
        TI_add_ijlk = ti_ijlk * dw_ratio_ijlk

        return TI_add_ijlk * (cw_ijlk < wake_radius_ijlk) * (dw_ijlk > 0)  # ensure zero upstream
