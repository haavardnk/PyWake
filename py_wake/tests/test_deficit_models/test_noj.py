import pytest
from py_wake import np
from py_wake import NOJ, NOJLocal
from py_wake.site._site import UniformSite
from py_wake.tests import npt
from py_wake.flow_map import HorizontalGrid, XYGrid
from py_wake.wind_turbines import WindTurbines
from py_wake.wind_farm_models.engineering_models import All2AllIterative, PropagateDownwind
from py_wake.deficit_models.noj import NOJDeficit, NOJLocalDeficit
from py_wake.superposition_models import LinearSum, SquaredSum, WeightedSum
from py_wake.turbulence_models.stf import STF2017TurbulenceModel
from py_wake.wind_turbines.power_ct_functions import PowerCtFunction
from py_wake.deficit_models.utils import ct2a_mom1d
from numpy import newaxis as na
from warnings import catch_warnings, filterwarnings
from py_wake.deficit_models import noj
from py_wake.examples.data.hornsrev1 import Hornsrev1Site, V80
from py_wake.ground_models import Mirror
from py_wake.rotor_avg_models import GridRotorAvg
from py_wake.utils.numpy_utils import Numpy32


def NibeA0():
    # Two turbines, 0: Nibe-A, 1:Ct=0
    return WindTurbines(names=['Nibe-A', 'None'], diameters=[40] * 2,
                        hub_heights=[50] * 2,
                        # only define for ct
                        powerCtFunctions=[PowerCtFunction(['ws'], lambda ws, run_only: ws * 0 + 8 / 9, 'w'),
                                          PowerCtFunction(['ws'], lambda ws, run_only: ws * 0, 'w')])


def test_NOJ_Nibe_result():
    # Replicate result from: Jensen, Niels Otto. "A note on wind generator interaction." (1983).

    site = UniformSite([1], 0.1)
    x_i = [0, 0, 0]
    y_i = [0, -40, -100]
    h_i = [50, 50, 50]
    wfm = All2AllIterative(site, NibeA0(), wake_deficitModel=NOJDeficit(ct2a=ct2a_mom1d),
                           superpositionModel=LinearSum())
    WS_eff = wfm(x_i, y_i, h_i, type=[0, 1, 1], wd=0.0, ws=8.1).WS_eff
    npt.assert_array_almost_equal(WS_eff.squeeze(), [8.1, 4.35, 5.7])


def test_NOJ_Nibe_result_wake_map():
    # Replicate result from: Jensen, Niels Otto. "A note on wind generator interaction." (1983).
    def ct_func(_):
        return 8 / 9

    def power_func(*_):
        return 0
    windTurbines = NibeA0()
    site = UniformSite([1], 0.1)
    wake_model = NOJ(site, windTurbines, ct2a=ct2a_mom1d)
    sim_res = wake_model(x=[0], y=[0], wd=[0], ws=[8.1])
    WS_eff_xy = sim_res.flow_map(HorizontalGrid(x=[0], y=[-40, -100], h=50)).WS_eff_xylk.mean(['wd', 'ws'])
    npt.assert_array_almost_equal(WS_eff_xy[:, 0], [4.35, 5.7])


@pytest.mark.parametrize('wdir,x,y', [(0, [0, 0, 0], [0, -40, -100]),
                                      (90, [0, -40, -100], [0, 0, 0]),
                                      (180, [0, 0, 0], [0, 40, 100]),
                                      (270, [0, 40, 100], [0, 0, 0])])
def test_NOJ_two_turbines_in_row(wdir, x, y):
    # Two turbines in a row, North-South
    # Replicate result from: Jensen, Niels Otto. "A note on wind generator interaction." (1983).

    windTurbines = NibeA0()
    site = UniformSite([1], 0.1)
    wfm = NOJ(site, windTurbines, ct2a=ct2a_mom1d)
    wfm.verbose = False
    h_i = [50, 50, 50]
    WS_eff = wfm(x, y, h_i, wd=wdir, ws=8.1).WS_eff
    ws_wt3 = 8.1 - np.hypot(8.1 * 2 / 3 * (20 / 26)**2, 8.1 * 2 / 3 * (20 / 30)**2)
    npt.assert_array_almost_equal(WS_eff.squeeze(), [8.1, 4.35, ws_wt3])


def test_NOJ_6_turbines_in_row():
    n_wt = 6
    x = [0] * n_wt
    y = - np.arange(n_wt) * 40 * 2

    site = UniformSite([1], 0.1)
    wfm = NOJ(site, NibeA0(), ct2a=ct2a_mom1d)
    wfm.verbose = False
    WS_eff = wfm(x, y, 50, wd=0.0, ws=11.0).WS_eff.squeeze()
    npt.assert_array_almost_equal(
        WS_eff[1:], 11 - np.sqrt(np.cumsum(((11 * 2 / 3 * 20**2)**2) / (20 + 8 * np.arange(1, 6))**4)))


def test_NOJLocal_6_turbines_in_row():
    n_wt = 6
    x = [0] * n_wt
    y = - np.arange(n_wt) * 40 * 2

    site = UniformSite([1], 0.1)
    wfm = NOJLocal(site, NibeA0(), turbulenceModel=STF2017TurbulenceModel(), ct2a=ct2a_mom1d)
    wfm.verbose = False
    WS_eff = wfm(x, y, 50, wd=0.0, ws=11.0).WS_eff
    npt.assert_array_almost_equal(WS_eff.squeeze()[1:], [5.62453869, 5.25806829, 5.64808912, 6.07792364, 6.44549094])


def test_NOJConvection():
    site = UniformSite([1], 0.1)
    with pytest.raises(AssertionError, match='WeightedSum and CumulativeWakeSum only work with NodeRotorAvgModel-based rotor average models'):
        wfm = NOJ(site, NibeA0(), superpositionModel=WeightedSum())
        wfm([0, 500], [0, 0])


def test_NOJConvection2():
    site = UniformSite([1], 0.1)
    with pytest.raises(NotImplementedError, match='calc_deficit_convection not implemented for NOJ'):
        wfm = NOJ(site, NibeA0(), rotorAvgModel=None, superpositionModel=WeightedSum())
        wfm([0, 500], [0, 0])


def test_NOJLocal_ti_dependence():
    def get(ti):
        site = UniformSite([1], ti)
        wfm = PropagateDownwind(site, NibeA0(), NOJLocalDeficit(use_effective_ti=False))
        return wfm([0, 40], [0, 0], wd=270, ws=[10])
    sim_res1, sim_res2 = [get(ti) for ti in [0.1, 0.2]]
    npt.assert_array_almost_equal(sim_res1.TI.squeeze(), [.1])
    npt.assert_array_almost_equal(sim_res2.TI.squeeze(), [.2])
    npt.assert_array_almost_equal(sim_res1.WS_eff.isel(wt=1), [[4.43458331]])
    npt.assert_array_almost_equal(sim_res2.WS_eff.isel(wt=1), [[5.13995521]])


def test_NOJLocal_ti_eff_dependence():
    site = UniformSite([1], 0.1)
    wfm = NOJLocal(site, NibeA0(), ct2a=ct2a_mom1d)
    sim_res = wfm([0, 40], [0, 0], wd=270, ws=[10, 10], TI=[[[.1, .2]]])
    npt.assert_array_almost_equal(sim_res.TI.squeeze(), [.1, .2])
    npt.assert_array_almost_equal(sim_res.TI_eff.squeeze(), [[.1, .2],
                                                             [0.43738364, 0.47043007]])
    npt.assert_array_almost_equal(sim_res.WS_eff.isel(wt=1), [[4.32651153, 5.045581]])


def _stock_noj_layout_terms(self, D_src_il, wake_radius_ijl, dw_ijlk, cw_ijlk, **kwargs):
    WS_ref_ilk = kwargs[self.WS_key]
    R_src_il = D_src_il / 2
    with catch_warnings():
        filterwarnings('ignore', r'invalid value encountered in true_divide')
        filterwarnings('ignore', r'invalid value encountered in divide')
        filterwarnings('ignore', r'divide by zero encountered in divide')

        term_denominator_ijlk = np.where(dw_ijlk > 0, ((wake_radius_ijl / R_src_il[:, na, :])**2)[..., na], 1)

    in_wake_ijlk = wake_radius_ijl[..., na] > cw_ijlk

    with catch_warnings():
        filterwarnings('ignore', r'invalid value encountered in true_divide')
        filterwarnings('ignore', r'invalid value encountered in divide')
        self.layout_factor_ijlk = WS_ref_ilk[:, na] * (in_wake_ijlk / term_denominator_ijlk)


def _stock_noj_calc_deficit(self, ct_ilk, **kwargs):
    if not self.deficit_initalized:
        self._calc_layout_terms(ct_ilk=ct_ilk, **kwargs)
    term_numerator_ilk = 2. * self.ct2a(ct_ilk)
    return term_numerator_ilk[:, na] * self.layout_factor_ijlk


def _stock_noj_local_calc_deficit(self, ct_ilk, D_src_il, wake_radius_ijlk, dw_ijlk, cw_ijlk, **kwargs):
    WS_ref_ilk = kwargs[self.WS_key]
    R_src_il = D_src_il / 2
    with catch_warnings():
        filterwarnings('ignore', r'invalid value encountered in true_divide')
        term_denominator_ijlk = np.where(dw_ijlk > 0, ((wake_radius_ijlk / R_src_il[:, na, :, na])**2), 1)

    in_wake_ijlk = wake_radius_ijlk > cw_ijlk

    with catch_warnings():
        filterwarnings(
            'ignore', r'invalid value encountered in true_divide')
        layout_factor_ijlk = WS_ref_ilk[:, na] * (in_wake_ijlk / term_denominator_ijlk)

    term_numerator_ilk = 2. * self.ct2a(ct_ilk)
    return term_numerator_ilk[:, na] * layout_factor_ijlk


def _spy_kernel(monkeypatch):
    build, calls = noj._noj_deficit_kernel, []

    def spy_build():
        kernel = build()

        def spy(*args):
            calls.append(args[-1].shape)
            return kernel(*args)
        return spy
    monkeypatch.setattr(noj, '_noj_deficit_kernel', spy_build)
    return calls


def _noj_inputs(I, J, L, K, ratio_shape=None, nan=None):
    rng = np.random.default_rng(0)
    inputs = rng.uniform(0, 1.2, (I, L, K)), rng.uniform(3, 25, (I, L, K)), rng.uniform(0, 1, ratio_shape or (I, J, L, 1))
    if nan is not None:
        inputs[nan].flat[7] = np.nan
    return inputs


def _special_inputs(I):
    numerator_ilk, WS_ilk, ratio_ijlk = _noj_inputs(I, 40, 24, 22)
    numerator_ilk[:, 0, :6] = [0., -0., 1e-300, 1e300, np.inf, -np.inf]
    WS_ilk[:, 1, :5] = [0., -0., np.inf, 1e300, 5e-324]
    ratio_ijlk[:, :6, 0, 0] = [np.inf, -0., 1e-300, 1e300, 0., 1.]
    ratio_ijlk[:, :4, 1, 0] = [np.inf, 1e300, 1e-300, -0.]
    return numerator_ilk, WS_ilk, ratio_ijlk


class SubArray(np.ndarray):
    pass


special = _special_inputs(1)
finite = _noj_inputs(1, 40, 24, 22)


def test_noj_special_inputs():
    numerator_ilk, WS_ilk, ratio_ijlk = special
    with np.errstate(all='ignore'):
        ref = numerator_ilk[:, na] * (WS_ilk[:, na] * ratio_ijlk)
    assert np.isnan(ref).any()
    assert np.isposinf(ref).any() and np.isneginf(ref).any()
    assert (np.signbit(ref) & (ref == 0)).any()
    assert ((ref != 0) & (np.abs(ref) < np.finfo(float).tiny)).any()


@pytest.mark.parametrize('inputs,fast', [
    pytest.param(special, 1, id='special'),
    pytest.param(_special_inputs(3), 1, id='all2all'),
    pytest.param((special[0][[0, 0, 0]], special[1], special[2]), 1, id='broadcast_i'),
    pytest.param((special[0], special[1][:, :1], special[2]), 1, id='broadcast_l'),
    pytest.param((special[0], special[1], np.asfortranarray(special[2])), 0, id='fortran_ratio'),
    pytest.param((special[0], special[1], _noj_inputs(1, 80, 24, 22)[2][:, ::2]), 0, id='strided_ratio'),
    pytest.param((np.asfortranarray(special[0]), special[1], special[2]), 0, id='fortran_numerator'),
    pytest.param((special[0], special[1][..., ::-1], special[2]), 0, id='reversed_WS'),
    pytest.param(_noj_inputs(1, 32, 32, 16), 1, id='threshold'),
    pytest.param(_noj_inputs(1, 31, 32, 16), 0, id='below_threshold'),
    pytest.param(_noj_inputs(1, 256, 32, 2), 1, id='k2'),
    pytest.param(_noj_inputs(1, 1000, 32, 1), 0, id='k1'),
    pytest.param(_noj_inputs(1, 40, 24, 22, ratio_shape=(1, 40, 24, 22)), 0, id='ratio_ijlk'),
    pytest.param(_noj_inputs(1, 40, 24, 22, nan=0), 0, id='nan_numerator'),
    pytest.param(_noj_inputs(1, 40, 24, 22, nan=1), 0, id='nan_WS'),
    pytest.param(_noj_inputs(1, 40, 24, 22, nan=2), 0, id='nan_ratio'),
    pytest.param((special[0].view(SubArray), special[1], special[2]), 0, id='subclass'),
    pytest.param((special[0], special[1] + 0j, special[2]), 0, id='complex'),
    pytest.param((finite[0], finite[1], finite[2].astype(np.float32)), 0, id='float32'),
    pytest.param((special[0][:, :, na], special[1][:, :, na], special[2][..., na]), 0, id='five_d'),
    pytest.param((special[0][:, na], special[1], special[2]), 0, id='numerator_4d'),
    pytest.param((special[0], special[1], special[2][0]), 0, id='ratio_3d')])
@pytest.mark.filterwarnings('ignore::RuntimeWarning')
@pytest.mark.filterwarnings('ignore::numpy.exceptions.ComplexWarning')
def test_noj_deficit(inputs, fast, monkeypatch):
    calls = _spy_kernel(monkeypatch)
    numerator_ilk, WS_ilk, ratio_ijlk = inputs
    deficit = noj._noj_deficit(numerator_ilk, WS_ilk, ratio_ijlk)
    ref = numerator_ilk[:, na] * (WS_ilk[:, na] * ratio_ijlk)
    assert len(calls) == fast
    assert type(deficit) is type(ref)
    assert deficit.dtype == ref.dtype
    assert deficit.shape == ref.shape
    assert deficit.strides == ref.strides
    assert deficit.tobytes() == ref.tobytes()


def test_noj_deficit_numpy32(monkeypatch):
    calls = _spy_kernel(monkeypatch)
    with Numpy32():
        numerator_ilk, WS_ilk, ratio_ijlk = [np.asarray(x) for x in finite]
        deficit = noj._noj_deficit(numerator_ilk, WS_ilk, ratio_ijlk)
        ref = numerator_ilk[:, na] * (WS_ilk[:, na] * ratio_ijlk)
    assert calls == []
    assert deficit.tobytes() == ref.tobytes()


def _stock_noj(monkeypatch):
    monkeypatch.setattr(NOJDeficit, '_calc_layout_terms', _stock_noj_layout_terms)
    monkeypatch.setattr(NOJDeficit, 'calc_deficit', _stock_noj_calc_deficit)
    monkeypatch.setattr(NOJLocalDeficit, 'calc_deficit', _stock_noj_local_calc_deficit)


@pytest.mark.parametrize('wfm_cls', [PropagateDownwind, All2AllIterative])
@pytest.mark.parametrize('deficit,superposition,turbulence,kernel', [
    pytest.param(NOJDeficit(), SquaredSum(), None, True, id='NOJ'),
    pytest.param(NOJDeficit(groundModel=Mirror()), SquaredSum(), None, True, id='NOJ_mirror'),
    pytest.param(NOJDeficit(rotorAvgModel=GridRotorAvg()), SquaredSum(), None, False, id='NOJ_grid_rotor_avg'),
    pytest.param(NOJLocalDeficit(ct2a=ct2a_mom1d, a=[0, .04], use_effective_ti=False), LinearSum(), None, True,
                 id='NOJLocal'),
    pytest.param(NOJLocalDeficit(), LinearSum(), STF2017TurbulenceModel(), False, id='NOJLocal_ti_eff')])
def test_noj_deficit_wfm(wfm_cls, deficit, superposition, turbulence, kernel, monkeypatch):
    site = Hornsrev1Site()
    x, y = site.initial_position.T
    wfm = wfm_cls(site, V80(), deficit, superpositionModel=superposition, turbulenceModel=turbulence)

    def run():
        sim_res = wfm(x, y, wd=np.arange(0, 360, 10), ws=np.arange(4, 12))
        return sim_res, sim_res.flow_map(XYGrid(resolution=40), wd=[270, 280], ws=[8, 10])
    calls = _spy_kernel(monkeypatch)
    sim_res, flow_map = run()
    assert bool(calls) == kernel
    _stock_noj(monkeypatch)
    ref, ref_flow_map = run()
    for k in ['WS_eff', 'TI_eff', 'CT', 'Power']:
        assert sim_res[k].values.tobytes() == ref[k].values.tobytes()
    assert flow_map.WS_eff.values.tobytes() == ref_flow_map.WS_eff.values.tobytes()


def test_noj_layout_terms_reused():
    deficit = NOJDeficit()
    deficit.layout_factor_ijlk = np.full((1, 2, 3, 1), 2.)
    deficit.deficit_initalized = True
    ct_ilk = np.full((1, 3, 4), .8)
    npt.assert_array_equal(deficit.calc_deficit(ct_ilk=ct_ilk), (2. * deficit.ct2a(ct_ilk))[:, na] * deficit.layout_factor_ijlk)
