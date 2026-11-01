import matplotlib.pyplot as plt
import pytest

from py_wake import np
from py_wake.deficit_models.deficit_model import WakeRadiusTopHat, BlockageDeficitModel
from py_wake.deficit_models.gaussian import ZongGaussian, BastankhahGaussianDeficit
from py_wake.deficit_models.noj import NOJ, NOJDeficit
from py_wake.examples.data.dtu10mw._dtu10mw import ct_curve
from py_wake.examples.data.hornsrev1 import Hornsrev1Site, V80
from py_wake.examples.data.iea37._iea37 import IEA37_WindTurbines
from py_wake.rotor_avg_models.area_overlap_model import AreaOverlapAvgModel
from py_wake.rotor_avg_models.gaussian_overlap_model import GaussianOverlapAvgModel
from py_wake.rotor_avg_models.rotor_avg_model import EqGridRotorAvg
from py_wake.site._site import UniformSite
from py_wake.tests import npt
from py_wake.turbulence_models.crespo import CrespoHernandez
from py_wake.turbulence_models.turbulence_model import TurbulenceModel
from py_wake.utils.model_utils import get_models
from py_wake.wind_farm_models.engineering_models import PropagateDownwind, All2AllIterative
from py_wake.wind_turbines._wind_turbines import WindTurbines
from py_wake.wind_turbines.power_ct_functions import PowerCtFunction
from py_wake.superposition_models import WeightedSum, SquaredSum
import warnings
from py_wake.ground_models.ground_models import Mirror
from numpy import newaxis as na
from py_wake.deficit_models.utils import ct2a_mom1d
from py_wake.deficit_models.gaussian import NiayifarGaussianDeficit, CarbajofuertesGaussianDeficit, \
    ZongGaussianDeficit, TurboGaussianDeficit, IEA37SimpleBastankhahGaussianDeficit
from py_wake.flow_map import XYGrid
from py_wake.turbulence_models.stf import STF2017TurbulenceModel
from py_wake.utils.gradients import autograd, cs, fd, cabs
from py_wake.rotor_avg_models import area_overlap_model, gaussian_overlap_model
from autograd.numpy.numpy_boxes import ArrayBox
from py_wake.turbulence_models.gcl_turb import GCLTurbulence
from py_wake.utils.grid_interpolator import GridInterpolator
from py_wake.utils.numpy_utils import Numpy32
import functools


def test_overlapping_area_factor_shapes():
    site = Hornsrev1Site()
    windTurbines = IEA37_WindTurbines()
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', '.* model is not representative of the setup')
        wfm = ZongGaussian(site, windTurbines, rotorAvgModel=EqGridRotorAvg(9),
                           turbulenceModel=CrespoHernandez())
    wfm([0, 1000], [0, 0])


def test_area_overlapping():
    wts = WindTurbines(names=['V80'] * 2, diameters=[80] * 2,
                       hub_heights=[70] * 2,
                       # only define for ct
                       powerCtFunctions=[PowerCtFunction(['ws'], lambda ws, run_only: np.interp(ws, ct_curve[:, 0], ct_curve[:, 1]), 'w'),
                                         PowerCtFunction(['ws'], lambda ws, run_only: ws * 0, 'w')])

    site = UniformSite([1], 0.1)
    wfm = NOJ(site, wts)

    y_lst = np.linspace(-250, 250, 50)
    y = np.r_[0, y_lst]
    x = np.r_[0, y_lst * 0 + 400]
    t = np.r_[0, y_lst * 0 + 1]
    WS_eff = wfm(x, y, type=t, ws=10, wd=270).WS_eff.values[1:, 0, 0]
    plt.plot(y_lst, WS_eff, '.-')
    # print(list(np.round(WS_eff[12:23], 2)))
    ref = [10.0, 9.99, 9.88, 9.71, 9.51, 9.29, 9.06, 8.83, 8.64, 8.58, 8.58]
    npt.assert_array_almost_equal(ref, WS_eff[12:23], decimal=2)
    wfm = PropagateDownwind(site, wts, NOJDeficit(rotorAvgModel=None))
    WS_eff = wfm(x, y, type=t, ws=10, wd=270).WS_eff.values[1:, 0, 0]
    # print(list(np.round(WS_eff[12:23], 2)))
    plt.plot(y_lst, WS_eff)
    ref = [10.0, 10.0, 10.0, 10.0, 10.0, 8.58, 8.58, 8.58, 8.58, 8.58, 8.58]
    plt.plot(y_lst[12:23], ref)
    if 0:
        plt.show()
    npt.assert_array_almost_equal(ref, WS_eff[12:23], decimal=2)
    plt.close('all')


@pytest.mark.parametrize('turbulenceModel', get_models(TurbulenceModel))
def test_AreaOverlapAvgModel_turbulence(turbulenceModel):
    if turbulenceModel is None:
        return
    wfm = PropagateDownwind(UniformSite(), V80(), NOJDeficit(),
                            turbulenceModel=turbulenceModel(rotorAvgModel=AreaOverlapAvgModel()))
    if isinstance(wfm.turbulenceModel, WakeRadiusTopHat):
        wfm([0, 1000], [0, 0])
    else:
        with pytest.raises(AssertionError, match='AreaOverlapAvgModel uses the wake_radius'):
            wfm([0, 1000], [0, 0])


@pytest.mark.parametrize('turbulenceModel', get_models(TurbulenceModel))
def test_GaussianOverlapAvgModel_turbulence(turbulenceModel):
    if turbulenceModel is None:
        return
    wfm = PropagateDownwind(UniformSite(), V80(), BastankhahGaussianDeficit(),
                            turbulenceModel=turbulenceModel(rotorAvgModel=GaussianOverlapAvgModel()))
    with pytest.raises(AttributeError, match=f"'{turbulenceModel.__name__}' has no attribute 'sigma_ijlk'"):
        wfm([0, 1000], [0, 0])


@pytest.mark.parametrize('blockageDeficitModel', get_models(BlockageDeficitModel))
def test_GaussianOverlapAvgModel_blockage(blockageDeficitModel):
    if blockageDeficitModel is None:
        return
    wfm = All2AllIterative(UniformSite(), V80(), BastankhahGaussianDeficit(),
                           blockage_deficitModel=blockageDeficitModel(rotorAvgModel=GaussianOverlapAvgModel()),
                           turbulenceModel=CrespoHernandez())
    with pytest.raises(AttributeError, match=f"'{blockageDeficitModel.__name__}' has no attribute 'sigma_ijlk'"):
        wfm([0, 1000], [0, 0], yaw=0)


def test_GaussianOverlapAvgModel_WeightedSum():
    with pytest.raises(AssertionError, match=r"WeightedSum and CumulativeWakeSum only work with NodeRotorAvgModel-based rotor average models"):
        wfm = PropagateDownwind(UniformSite(), V80(), BastankhahGaussianDeficit(rotorAvgModel=GaussianOverlapAvgModel()),
                                WeightedSum())
    wfm = PropagateDownwind(UniformSite(), V80(), BastankhahGaussianDeficit(groundModel=Mirror()),
                            WeightedSum())
    with pytest.raises(NotImplementedError, match=r"calc_deficit_convection \(WeightedSum\) cannot be used in combination with GroundModels"):
        wfm([0, 1000], [0, 0])


def test_area_overlapping_deprecated_way():
    wts = WindTurbines(names=['V80'] * 2, diameters=[80] * 2,
                       hub_heights=[70] * 2,
                       # only define for ct
                       powerCtFunctions=[PowerCtFunction(['ws'], lambda ws, run_only: np.interp(ws, ct_curve[:, 0], ct_curve[:, 1]), 'w'),
                                         PowerCtFunction(['ws'], lambda ws, run_only: ws * 0, 'w')])

    site = UniformSite([1], 0.1)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=DeprecationWarning)
        wfm = PropagateDownwind(site, wts, rotorAvgModel=AreaOverlapAvgModel(),
                                wake_deficitModel=NOJDeficit(rotorAvgModel=None),
                                superpositionModel=SquaredSum())

    y_lst = np.linspace(-250, 250, 50)
    y = np.r_[0, y_lst]
    x = np.r_[0, y_lst * 0 + 400]
    t = np.r_[0, y_lst * 0 + 1]
    WS_eff = wfm(x, y, type=t, ws=10, wd=270).WS_eff.values[1:, 0, 0]
    plt.plot(y_lst, WS_eff, '.-')
    # print(list(np.round(WS_eff[12:23], 2)))
    ref = [10.0, 9.99, 9.88, 9.71, 9.51, 9.29, 9.06, 8.83, 8.64, 8.58, 8.58]
    npt.assert_array_almost_equal(ref, WS_eff[12:23], decimal=2)
    wfm = PropagateDownwind(site, wts, NOJDeficit(rotorAvgModel=None))
    WS_eff = wfm(x, y, type=t, ws=10, wd=270).WS_eff.values[1:, 0, 0]
    # print(list(np.round(WS_eff[12:23], 2)))
    plt.plot(y_lst, WS_eff)
    ref = [10.0, 10.0, 10.0, 10.0, 10.0, 8.58, 8.58, 8.58, 8.58, 8.58, 8.58]
    plt.plot(y_lst[12:23], ref)
    if 0:
        plt.show()
    npt.assert_array_almost_equal(ref, WS_eff[12:23], decimal=2)
    plt.close('all')


def stock_calc_deficit_centre(self, sigma_ijlk, cw_ijlk, **kwargs):
    return self.calc_deficit(cw_ijlk=cw_ijlk * 0, **kwargs)


class ShiftedBastankhahGaussianDeficit(BastankhahGaussianDeficit):
    def _calc_deficit(self, D_src_il, dw_ijlk, ct_ilk, **kwargs):
        return BastankhahGaussianDeficit._calc_deficit(self, D_src_il, dw_ijlk + 100, ct_ilk, **kwargs)


overlap_deficit_models = [
    lambda: BastankhahGaussianDeficit(rotorAvgModel=GaussianOverlapAvgModel()),
    lambda: NiayifarGaussianDeficit(rotorAvgModel=GaussianOverlapAvgModel()),
    lambda: CarbajofuertesGaussianDeficit(rotorAvgModel=GaussianOverlapAvgModel()),
    lambda: ZongGaussianDeficit(rotorAvgModel=GaussianOverlapAvgModel()),
    lambda: TurboGaussianDeficit(rotorAvgModel=GaussianOverlapAvgModel(), groundModel=None),
    lambda: TurboGaussianDeficit(rotorAvgModel=GaussianOverlapAvgModel(), groundModel=Mirror()),
    lambda: IEA37SimpleBastankhahGaussianDeficit(rotorAvgModel=GaussianOverlapAvgModel()),
    lambda: ShiftedBastankhahGaussianDeficit(rotorAvgModel=GaussianOverlapAvgModel()),
]


@pytest.mark.parametrize('wfm_cls', [PropagateDownwind, All2AllIterative])
@pytest.mark.parametrize('deficitModel', overlap_deficit_models)
def test_GaussianOverlapAvgModel_deficit_centre(wfm_cls, deficitModel, monkeypatch):
    x, y, h = [0, 400, 800, 200, 600], [0, 50, -30, 300, 0], [50, 70, 60, 80, 70]

    def get_sim_res():
        wfm = wfm_cls(UniformSite(), V80(), deficitModel(), turbulenceModel=STF2017TurbulenceModel())
        sim_res = wfm(x, y, h=h, wd=[260, 270, 280], ws=[6, 10])
        return sim_res.WS_eff.values, sim_res.flow_map(XYGrid(resolution=20), wd=270, ws=10).WS_eff.values

    sim_res = get_sim_res()
    monkeypatch.setattr(BastankhahGaussianDeficit, '_calc_deficit_centre', stock_calc_deficit_centre)
    for actual, expected in zip(sim_res, get_sim_res()):
        npt.assert_array_equal(actual, expected)


@pytest.mark.parametrize('gradient_method', [autograd, cs, fd])
@pytest.mark.parametrize('groundModel', [None, Mirror()])
def test_GaussianOverlapAvgModel_deficit_centre_gradients(gradient_method, groundModel, monkeypatch):
    x, y, h = np.array([0., 400, 800]), np.array([0., 50, -30]), np.array([50., 70, 60])
    wfm = PropagateDownwind(UniformSite(), V80(),
                            TurboGaussianDeficit(rotorAvgModel=GaussianOverlapAvgModel(), groundModel=groundModel))

    def get_gradients():
        return wfm.aep_gradients(gradient_method, ['x', 'y', 'h'], x=x, y=y, h=h, wd=[260, 270, 280], ws=[6, 10])

    gradients = get_gradients()
    monkeypatch.setattr(BastankhahGaussianDeficit, '_calc_deficit_centre', stock_calc_deficit_centre)
    if gradient_method is autograd:
        # one shared sigma node sums its adjoints in another order, i.e. last-bit differences
        npt.assert_array_almost_equal(gradients, get_gradients(), 15)
    else:
        npt.assert_array_equal(gradients, get_gradients())


@pytest.mark.parametrize('cw', [0., np.inf, np.nan, 1e300 + 1e300j, 1 + 1e-20j, np.inf * 1j])
def test_calc_deficit_centre_non_finite(cw):
    kwargs = dict(D_src_il=np.array([[80., 0., 1e-161, 240.]]),
                  dw_ijlk=np.array([0., 400., -50.])[na, :, na, na],
                  ct_ilk=np.array([.8, 0., .5, 1.2])[na, :, na],
                  TI_ilk=np.array([.06, 0., .1, .06])[na, :, na],
                  WS_ref_ijlk=np.full((1, 1, 4, 1), 10.))
    cw_ijlk = np.broadcast_to(np.array([1., cw, 0.])[na, :, na, na], (1, 3, 4, 1))
    deficitModel = TurboGaussianDeficit(A=.04, ct2a=ct2a_mom1d, ctlim=.96)
    with np.errstate(all='ignore'):
        expected = deficitModel.calc_deficit(cw_ijlk=cw_ijlk * 0, **kwargs)
        actual = deficitModel._calc_deficit_centre(deficitModel.sigma_ijlk(**kwargs), cw_ijlk=cw_ijlk, **kwargs)
    assert np.isnan(expected).any() and np.isfinite(expected).any()
    assert actual.dtype == expected.dtype
    assert np.broadcast_to(actual, expected.shape).tobytes() == expected.tobytes()


@pytest.mark.parametrize('dcw', [0, 1e-20j])
def test_calc_deficit_centre_skips_unit_exponential(dcw, monkeypatch):
    kwargs = dict(D_src_il=np.array([[80., 120.]]), dw_ijlk=np.array([200., 400.])[na, :, na, na],
                  ct_ilk=np.array([.8, .5])[na, :, na], TI_ilk=np.full((1, 2, 1), .06),
                  WS_ref_ijlk=np.full((1, 1, 2, 1), 10.))
    cw_ijlk = np.array([[30., -50.]])[..., na, na] + dcw
    deficitModel = TurboGaussianDeficit()
    expected = deficitModel.calc_deficit(cw_ijlk=cw_ijlk * 0, **kwargs)
    if not dcw:
        monkeypatch.setattr(np, 'exp', None)
    actual = deficitModel._calc_deficit_centre(deficitModel.sigma_ijlk(**kwargs), cw_ijlk=cw_ijlk, **kwargs)
    assert actual.dtype == expected.dtype
    assert actual.tobytes() == expected.tobytes()


def stock_overlap_call(self, func, cw_ijlk, D_dst_ijl, **kwargs):
    deficit_model = func.__self__
    if not hasattr(deficit_model, 'sigma_ijlk'):
        raise AttributeError(
            f"'{deficit_model.__class__.__name__}' has no attribute 'sigma_ijlk', which is needed by the GaussianOverlapAvgModel")
    sigma_ijlk = deficit_model.sigma_ijlk(**kwargs)
    if hasattr(deficit_model, '_calc_deficit_centre') and func == deficit_model.calc_deficit:
        res_ijlk = deficit_model._calc_deficit_centre(sigma_ijlk, cw_ijlk=cw_ijlk, D_dst_ijl=D_dst_ijl, **kwargs)
    else:
        res_ijlk = func(cw_ijlk=cw_ijlk * 0, D_dst_ijl=D_dst_ijl, **kwargs)
    r_sigma_ijlk = (D_dst_ijl / 2)[..., na] / sigma_ijlk
    cw_sigma_ijlk = cabs(cw_ijlk / sigma_ijlk)
    shape = np.broadcast_shapes(r_sigma_ijlk.shape, cw_sigma_ijlk.shape)
    xp = np.stack([np.broadcast_to(r_sigma_ijlk, shape), np.broadcast_to(cw_sigma_ijlk, shape)]).reshape(2, -1).T
    overlap_factor_ijlk = self.overlap_interpolator(xp).reshape(shape)
    return res_ijlk * overlap_factor_ijlk


def compiled_and_stock_overlap(monkeypatch, func):
    build, calls = gaussian_overlap_model._overlap_factor_kernel, {'fast': 0, 'nan': 0}

    def spy_kernel():
        kernel = build()

        def spy(*args):
            nan = kernel(*args)
            calls['nan' if nan else 'fast'] += 1
            return nan
        return spy
    monkeypatch.setattr(gaussian_overlap_model, '_overlap_factor_kernel', spy_kernel)
    res = func()
    monkeypatch.setattr(GaussianOverlapAvgModel, '__call__', stock_overlap_call)
    return res, func(), calls


class FixedSigmaDeficit:
    def __init__(self, sigma_ijlk):
        self._sigma_ijlk = sigma_ijlk

    def sigma_ijlk(self):
        return self._sigma_ijlk

    def calc_deficit(self, cw_ijlk, D_dst_ijl):
        return 1.


@functools.cache
def overlap_table():
    return GaussianOverlapAvgModel().overlap_interpolator


def overlap_call(D_dst_ijl, cw_ijlk, sigma_ijlk, interpolator=None):
    model = GaussianOverlapAvgModel()
    model._overlap_interpolator = interpolator or overlap_table()
    return lambda: model(FixedSigmaDeficit(sigma_ijlk).calc_deficit, cw_ijlk=cw_ijlk, D_dst_ijl=D_dst_ijl)


class SubArray(np.ndarray):
    pass


def with_values(x, values):
    x = x.copy()
    for index, value in values.items():
        x[index] = value
    return x


overlap_rng = np.random.default_rng(0)
overlap_J, overlap_L, overlap_K = 23, 4, 5
mirror_D_ijl = overlap_rng.uniform(40, 250, (1, overlap_J, overlap_L))
mirror_cw_ijlk = overlap_rng.normal(0, 800, (2, overlap_J, overlap_L, 1))
mirror_sigma_ijlk = np.exp(overlap_rng.uniform(np.log(2), np.log(3000), (1, overlap_J, overlap_L, overlap_K)))
mirror = mirror_D_ijl, mirror_cw_ijlk, mirror_sigma_ijlk
rows_D_ijl = overlap_rng.uniform(40, 250, (3, overlap_J, overlap_L))
rows_cw_ijlk = overlap_rng.normal(0, 800, (3, overlap_J, overlap_L, overlap_K))
rows_sigma_ijlk = np.exp(overlap_rng.uniform(np.log(2), np.log(3000), (3, overlap_J, overlap_L, overlap_K)))
infinite = (with_values(mirror_D_ijl, {(0, 1, 1): 0.}),
            with_values(mirror_cw_ijlk, {(0, 2, 0, 0): np.inf, (1, 2, 0, 0): -np.inf, (0, 3, 0, 0): -0.}),
            with_values(mirror_sigma_ijlk, {(0, 0, 0, 0): np.inf, (0, 0, 0, 1): -np.inf, (0, 0, 0, 2): 0.,
                                            (0, 0, 0, 3): 5e-324}))
r_edges, cw_edges = np.array([0., .01, 19.99, 20., 20.01, 1e300]), np.array([-0., 0., 9.99, 10., 10.01, -1e300])


@pytest.mark.parametrize('D_dst_ijl,cw_ijlk,sigma_ijlk,fast,nan', [
    pytest.param(*mirror, 1, 0, id='mirror'),
    pytest.param(mirror_D_ijl, mirror_cw_ijlk[:1], mirror_sigma_ijlk, 1, 0, id='one_row'),
    pytest.param(mirror_D_ijl, mirror_cw_ijlk, mirror_sigma_ijlk[..., :1], 1, 0, id='one_k'),
    pytest.param(mirror_D_ijl, mirror_cw_ijlk[:1], mirror_sigma_ijlk[..., :1], 0, 0, id='one_row_one_k'),
    pytest.param(mirror_D_ijl, rows_cw_ijlk, mirror_sigma_ijlk, 1, 0, id='cw_rows'),
    pytest.param(mirror_D_ijl[:, :1, :1], mirror_cw_ijlk, mirror_sigma_ijlk[..., :2], 1, 0, id='broadcast_D'),
    pytest.param(np.asfortranarray(mirror_D_ijl), np.asfortranarray(mirror_cw_ijlk), np.asfortranarray(mirror_sigma_ijlk),
                 1, 0, id='fortran'),
    pytest.param(mirror_D_ijl, np.resize(mirror_cw_ijlk, (2, overlap_J, 2 * overlap_L, 1))[:, ::-1, ::2],
                 np.resize(mirror_sigma_ijlk, (1, 2 * overlap_J, overlap_L, 2 * overlap_K))[:, ::2, :, ::2], 1, 0,
                 id='strided'),
    pytest.param(np.broadcast_to(mirror_D_ijl[:, :1], mirror_D_ijl.shape), mirror_cw_ijlk,
                 np.broadcast_to(mirror_sigma_ijlk, mirror_sigma_ijlk.shape), 1, 0, id='zero_stride'),
    pytest.param(mirror_D_ijl[:, :0], mirror_cw_ijlk[:, :0], mirror_sigma_ijlk[:, :0], 1, 0, id='empty_j'),
    pytest.param(mirror_D_ijl, mirror_cw_ijlk[:0], mirror_sigma_ijlk, 1, 0, id='empty_i'),
    pytest.param((2 * r_edges)[na, :, na], np.array([cw_edges, -cw_edges])[:, na, :, na], np.ones((1, 1, 1, 1)), 1, 0,
                 id='table_edges'),
    pytest.param(*infinite, 1, 0, id='infinite'),
    pytest.param(mirror_D_ijl, mirror_cw_ijlk, with_values(mirror_sigma_ijlk, {(0, 3, 1, 2): np.nan}), 0, 1,
                 id='nan_sigma'),
    pytest.param(with_values(mirror_D_ijl, {(0, 4, 2): -np.nan}), mirror_cw_ijlk, mirror_sigma_ijlk, 0, 1,
                 id='negative_nan_D'),
    pytest.param(with_values(mirror_D_ijl, {(0, 5, 0): 0.}), mirror_cw_ijlk,
                 with_values(mirror_sigma_ijlk, {(0, 5, 0, 1): 0.}), 0, 1, id='zero_over_zero'),
    pytest.param(rows_D_ijl, rows_cw_ijlk, rows_sigma_ijlk, 0, 0, id='all2all'),
    pytest.param(mirror_D_ijl, rows_cw_ijlk[:2], rows_sigma_ijlk[:2], 0, 0, id='sigma_rows'),
    pytest.param(rows_D_ijl[:2], mirror_cw_ijlk, mirror_sigma_ijlk, 0, 0, id='D_rows'),
    pytest.param(rows_D_ijl[:0], mirror_cw_ijlk[:0], mirror_sigma_ijlk, 0, 0, id='empty_D_rows'),
    pytest.param(mirror_D_ijl, mirror_cw_ijlk + 1e-20j, mirror_sigma_ijlk, 0, 0, id='complex_cw'),
    pytest.param(mirror_D_ijl, mirror_cw_ijlk, mirror_sigma_ijlk.astype(np.float32), 0, 0, id='float32_sigma'),
    pytest.param(np.round(mirror_D_ijl).astype(int), mirror_cw_ijlk, mirror_sigma_ijlk, 0, 0, id='int_D'),
    pytest.param(mirror_D_ijl[0], mirror_cw_ijlk, mirror_sigma_ijlk, 0, 0, id='two_d_D'),
    pytest.param(mirror_D_ijl, mirror_cw_ijlk[na], mirror_sigma_ijlk, 0, 0, id='five_d_cw'),
    pytest.param(mirror_D_ijl, mirror_cw_ijlk, mirror_sigma_ijlk[0], 0, 0, id='three_d_sigma'),
    pytest.param(mirror_D_ijl, mirror_cw_ijlk.view(SubArray), mirror_sigma_ijlk, 0, 0, id='subclass_cw')])
@pytest.mark.filterwarnings('ignore::RuntimeWarning')
def test_overlap_kernel(D_dst_ijl, cw_ijlk, sigma_ijlk, fast, nan, monkeypatch):
    res, ref, calls = compiled_and_stock_overlap(monkeypatch, overlap_call(D_dst_ijl, cw_ijlk, sigma_ijlk))
    assert calls == {'fast': fast, 'nan': nan}
    assert type(res) is type(ref)
    assert res.dtype == ref.dtype
    assert res.shape == ref.shape
    assert res.tobytes() == ref.tobytes()


def test_overlap_kernel_numpy32(monkeypatch):
    with Numpy32():
        res, ref, calls = compiled_and_stock_overlap(monkeypatch, overlap_call(*mirror))
    assert calls == {'fast': 0, 'nan': 0}
    assert res.dtype == ref.dtype
    assert res.tobytes() == ref.tobytes()


small_r, small_cw = np.arange(0, 7.01, .5), np.arange(0, 60.1, 2.5)
small_V = np.random.default_rng(1).normal(size=(len(small_r), len(small_cw)))


@pytest.mark.parametrize('interpolator,fast', [
    pytest.param(GridInterpolator([small_r - .3, small_cw + .2], small_V, bounds='limit'), 1, id='other_grid'),
    pytest.param(GridInterpolator([small_r, small_cw], np.asfortranarray(small_V), bounds='limit'), 1, id='fortran_V'),
    pytest.param(GridInterpolator([small_r, small_cw], small_V, method=['linear', 'linear'], bounds='limit'), 1,
                 id='linear_list'),
    pytest.param(GridInterpolator([small_r, small_cw], small_V, method='nearest', bounds='limit'), 0, id='nearest'),
    pytest.param(GridInterpolator([small_r, small_cw], small_V, method=['linear', 'nearest'], bounds='limit'), 0,
                 id='linear_nearest'),
    pytest.param(GridInterpolator([small_r, small_cw], small_V, bounds='check'), 0, id='check_bounds'),
    pytest.param(GridInterpolator([small_r, small_cw ** 1.1], small_V, bounds='limit'), 0, id='irregular'),
    pytest.param(GridInterpolator([small_r, small_cw], small_V.astype(np.float32), bounds='limit'), 0, id='float32_V')])
def test_overlap_kernel_interpolators(interpolator, fast, monkeypatch):
    rng = np.random.default_rng(2)
    D_dst_ijl = rng.uniform(40, 250, (1, overlap_J, overlap_L))
    cw_ijlk = rng.uniform(-1000, 1000, (2, overlap_J, overlap_L, 1))
    sigma_ijlk = rng.uniform(20, 1000, (1, overlap_J, overlap_L, overlap_K))
    res, ref, calls = compiled_and_stock_overlap(monkeypatch, overlap_call(D_dst_ijl, cw_ijlk, sigma_ijlk, interpolator))
    assert calls == {'fast': fast, 'nan': 0}
    assert res.tobytes() == ref.tobytes()


def test_overlap_kernel_negative_zero_table(monkeypatch):
    interpolator = GridInterpolator([small_r, small_cw], np.full(small_V.shape, -0.), bounds='limit')
    res, ref, calls = compiled_and_stock_overlap(monkeypatch, overlap_call(*mirror, interpolator))
    assert calls == {'fast': 1, 'nan': 0}
    assert not np.signbit(ref).any()
    assert res.tobytes() == ref.tobytes()


def test_overlap_kernel_clamps_upper_corner(monkeypatch):
    interpolator = GridInterpolator([small_r, small_cw], with_values(small_V, {(slice(None), 0): np.inf}), bounds='limit')
    cw_ijlk = np.full(mirror_cw_ijlk.shape, 1e5)
    res, ref, calls = compiled_and_stock_overlap(monkeypatch, overlap_call(mirror_D_ijl, cw_ijlk, mirror_sigma_ijlk,
                                                                            interpolator))
    assert calls == {'fast': 1, 'nan': 0}
    assert np.isfinite(ref).all()
    assert res.tobytes() == ref.tobytes()


@pytest.mark.parametrize('wfm_cls', [PropagateDownwind, All2AllIterative])
@pytest.mark.parametrize('deficitModel', overlap_deficit_models)
def test_overlap_kernel_wfm(wfm_cls, deficitModel, monkeypatch):
    x, y, h = [0, 400, 800, 200, 600], [0, 50, -30, 300, 0], [50, 70, 60, 80, 70]

    def run():
        wfm = wfm_cls(UniformSite(), V80(), deficitModel(), turbulenceModel=STF2017TurbulenceModel())
        sim_res = wfm(x, y, h=h, wd=[260, 270, 280], ws=[6, 10])
        return sim_res.WS_eff.values, sim_res.TI_eff.values, sim_res.flow_map(XYGrid(resolution=20), wd=270,
                                                                              ws=10).WS_eff.values
    res, ref, calls = compiled_and_stock_overlap(monkeypatch, run)
    ws_independent_sigma = isinstance(deficitModel(), IEA37SimpleBastankhahGaussianDeficit)
    assert calls == {'fast': calls['fast'], 'nan': 0} and (calls['fast'] > 0) != ws_independent_sigma
    for actual, expected in zip(res, ref):
        assert actual.tobytes() == expected.tobytes()


@pytest.mark.parametrize('gradient_method', [autograd, cs, fd])
def test_overlap_kernel_gradients(gradient_method, monkeypatch):
    x, y, h = np.array([0., 400, 800]), np.array([0., 50, -30]), np.array([50., 70, 60])
    wfm = PropagateDownwind(UniformSite(), V80(), TurboGaussianDeficit(rotorAvgModel=GaussianOverlapAvgModel()))

    def run():
        return wfm.aep_gradients(gradient_method, ['x', 'y', 'h'], x=x, y=y, h=h, wd=[260, 270, 280], ws=[6, 10])
    res, ref, calls = compiled_and_stock_overlap(monkeypatch, run)
    assert (calls['fast'] > 0) == (gradient_method is fd)
    assert res.tobytes() == ref.tobytes()


def _stock_cal_overlapping_area_factor(self, R1, R2, d):
    shape = tuple(np.max([R1.shape, R2.shape, d.shape], 0))
    R1, R2, d = [np.broadcast_to(a, shape) for a in [R1, R2, d]]

    Rmax = np.where(R1 < R2, R2, R1)
    Rmin = np.where(R1 < R2, R1, R2)

    index_fullwake = (d <= (Rmax - Rmin))
    dtype = (float, np.complex128)[bool(np.any([np.iscomplexobj(x) for x in [R1, R2, d]]))]
    A_ol_f = np.where(index_fullwake, 1, 0).astype(dtype)

    mask = (d > (Rmax - Rmin)) & (d < (Rmin + Rmax))
    if any([isinstance(x, ArrayBox) for x in [R1, R2, d]]):
        p_wake_mask = mask
        mask = slice(None)
    else:
        p_wake_mask = None

    eps = 2 * np.finfo(float).eps

    def arccos_lim(x):
        return np.arccos((np.clip(x, -1.0 + eps, +1.0 - eps)))

    alpha = arccos_lim((Rmax[mask]**2.0 + d[mask]**2 - Rmin[mask]**2) /
                       (2.0 * np.maximum(Rmax[mask] * d[mask], eps)))

    beta = arccos_lim((Rmin[mask]**2.0 + d[mask]**2 - Rmax[mask]**2) /
                      (2.0 * np.maximum(Rmin[mask] * d[mask], eps)))

    A_triangle = np.sin(alpha) * R1[mask] * d[mask]

    p_wake_f = np.maximum((alpha * Rmax[mask]**2 + beta * Rmin[mask]**2 -
                           A_triangle) / (R2[mask]**2 * np.pi), 0)

    if p_wake_mask is None:
        A_ol_f[mask] = p_wake_f
    else:
        A_ol_f = np.where(p_wake_mask, p_wake_f, A_ol_f)
    return A_ol_f


def compiled_and_stock_area_overlap(monkeypatch, func):
    build, calls = area_overlap_model._classify_overlap_kernel, []

    def spy_build():
        kernel = build()

        def spy(*args):
            calls.append(kernel(*args))
            return calls[-1]
        return spy
    monkeypatch.setattr(area_overlap_model, '_classify_overlap_kernel', spy_build)
    res = func()
    monkeypatch.setattr(AreaOverlapAvgModel, '_cal_overlapping_area_factor', _stock_cal_overlapping_area_factor)
    return res, func(), calls


area_rng = np.random.default_rng(1)
area_R1 = area_rng.uniform(20, 120, (1, 37, 6, 5))
area_R2 = area_rng.uniform(20, 80, (1, 37, 6, 1))
area_d = np.abs(area_rng.normal(0, 150, (1, 37, 6, 1)))
area_values = [0., -0., 5e-324, 10., 20., 30., 40., 50., 70., 80., 1e300, np.inf, np.nan, -np.nan]
area_edges = [x.reshape(1, -1, 1, 1).copy() for x in np.meshgrid(area_values, area_values, area_values, indexing='ij')]
area_rows = (area_rng.uniform(20, 120, (3, 37, 6, 5)), area_rng.uniform(20, 80, (1, 37, 6, 1)),
             np.abs(area_rng.normal(0, 150, (3, 37, 6, 1))))


@pytest.mark.parametrize('R1,R2,d,fast', [
    pytest.param(area_R1, area_R2, area_d, 1, id='niayifar'),
    pytest.param(area_R1[..., :1].copy(), area_R2, area_d, 1, id='park2'),
    pytest.param(*area_rows, 1, id='all2all'),
    pytest.param(*area_edges, 1, id='edges'),
    pytest.param(np.reshape(area_values, (1, 14, 1, 1)), np.reshape(area_values, (1, 1, 14, 1)),
                 np.reshape(area_values, (1, 1, 1, 14)), 1, id='edges_broadcast'),
    pytest.param(np.empty((37, 6, 5, 1)).transpose(3, 0, 1, 2), area_R2, area_d, 1, id='relaxed_strides'),
    pytest.param(area_R2, area_R1, area_d, 1, id='swapped'),
    pytest.param(area_R1[:, :0], area_R2[:, :0], area_d[:, :0], 1, id='empty'),
    pytest.param(np.asfortranarray(area_R1), area_R2, area_d, 0, id='fortran_R1'),
    pytest.param(area_R1[:, ::2], area_R2[:, ::2], area_d[:, ::2], 0, id='strided'),
    pytest.param(area_R1, area_R2.view(SubArray), area_d, 0, id='subclass'),
    pytest.param(area_R1, area_R2, area_d + 0j, 0, id='complex'),
    pytest.param(area_R1.astype(np.float32), area_R2, area_d, 0, id='float32'),
    pytest.param(area_R1[0], area_R2[0], area_d[0], 0, id='three_d'),
    pytest.param(area_R1[..., na], area_R2[..., na], area_d[..., na], 0, id='five_d')])
@pytest.mark.filterwarnings('ignore::RuntimeWarning')
@pytest.mark.filterwarnings('ignore::numpy.exceptions.ComplexWarning')
def test_area_overlap_kernel(R1, R2, d, fast, monkeypatch):
    def run():
        return AreaOverlapAvgModel()._cal_overlapping_area_factor(R1, R2, d)
    res, ref, calls = compiled_and_stock_area_overlap(monkeypatch, run)
    assert len(calls) == fast
    assert type(res) is type(ref)
    assert res.dtype == ref.dtype
    assert res.shape == ref.shape
    assert res.strides == ref.strides
    assert res.tobytes() == ref.tobytes()


def test_area_overlap_kernel_numpy32(monkeypatch):
    def run():
        with Numpy32():
            return AreaOverlapAvgModel()._cal_overlapping_area_factor(area_R1, area_R2, area_d)
    res, ref, calls = compiled_and_stock_area_overlap(monkeypatch, run)
    assert calls == []
    assert res.dtype == ref.dtype
    assert res.tobytes() == ref.tobytes()


def test_area_overlap_kernel_edges_cover_all_cases():
    with np.errstate(all='ignore'):
        ref = _stock_cal_overlapping_area_factor(None, *area_edges)
    assert {0., 1.} < set(ref.ravel().tolist())
    assert ((ref > 0) & (ref < 1)).any()
    assert np.isnan(ref).any()


@pytest.mark.parametrize('wfm_cls', [PropagateDownwind, All2AllIterative])
@pytest.mark.parametrize('model', [
    pytest.param(lambda wfm_cls, site, wt: wfm_cls(site, wt, NOJDeficit(), superpositionModel=SquaredSum()), id='NOJ'),
    pytest.param(lambda wfm_cls, site, wt: wfm_cls(site, wt, NOJDeficit(groundModel=Mirror()),
                                                   superpositionModel=SquaredSum()), id='NOJ_mirror'),
    pytest.param(lambda wfm_cls, site, wt: wfm_cls(site, wt, NiayifarGaussianDeficit(), superpositionModel=SquaredSum(),
                                                   turbulenceModel=CrespoHernandez()), id='niayifar'),
    pytest.param(lambda wfm_cls, site, wt: wfm_cls(site, wt, BastankhahGaussianDeficit(),
                                                   turbulenceModel=GCLTurbulence()), id='gcl')])
def test_area_overlap_kernel_wfm(wfm_cls, model, monkeypatch):
    site = Hornsrev1Site()
    x, y = site.initial_position.T
    wfm = model(wfm_cls, site, V80())

    def run():
        sim_res = wfm(x, y, wd=np.arange(0, 360, 15), ws=[6, 10])
        flow_map = sim_res.flow_map(XYGrid(resolution=30), wd=[270, 285], ws=10)
        return [sim_res[k].values for k in ['WS_eff', 'TI_eff', 'CT', 'Power']] + [flow_map.WS_eff.values,
                                                                                  flow_map.TI_eff.values]
    res, ref, calls = compiled_and_stock_area_overlap(monkeypatch, run)
    assert calls
    for actual, expected in zip(res, ref):
        assert actual.tobytes() == expected.tobytes()


@pytest.mark.parametrize('gradient_method', [autograd, cs, fd])
def test_area_overlap_kernel_gradients(gradient_method, monkeypatch):
    x, y = np.array([0., 400, 800]), np.array([0., 50, -30])
    wfm = PropagateDownwind(UniformSite(), V80(), NOJDeficit(), superpositionModel=SquaredSum(),
                            turbulenceModel=CrespoHernandez())

    def run():
        return wfm.aep_gradients(gradient_method, ['x', 'y'], x=x, y=y, wd=[260, 270, 280], ws=[6, 10])
    res, ref, calls = compiled_and_stock_area_overlap(monkeypatch, run)
    assert bool(calls) == (gradient_method is fd)
    assert res.tobytes() == ref.tobytes()
