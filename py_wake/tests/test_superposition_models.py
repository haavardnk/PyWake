import pytest
from py_wake import np
from py_wake import NOJ
from py_wake.site._site import UniformSite
from py_wake.superposition_models import LinearSum, SquaredSum, MaxSum, SqrMaxSum, WeightedSum, CumulativeWakeSum, \
    SuperpositionModel
from py_wake.tests import npt
from py_wake.wind_farm_models import engineering_models
from py_wake.wind_farm_models.engineering_models import PropagateDownwind, PropagateUpDownIterative, \
    All2AllIterative
from py_wake.deficit_models.noj import NOJDeficit
from py_wake.turbulence_models import TurbulenceModel
from py_wake.flow_map import HorizontalGrid, Points
from py_wake.tests.test_deficit_models.test_noj import NibeA0
from py_wake.examples.data.hornsrev1 import Hornsrev1Site, V80, wt9_x, wt9_y
from py_wake.deficit_models.deficit_model import BlockageDeficitModel, WakeDeficitModel
from py_wake.deficit_models import NoWakeDeficit
from py_wake.examples.data.iea37._iea37 import IEA37Site, IEA37_WindTurbines
from py_wake.deficit_models.gaussian import BastankhahGaussianDeficit, NiayifarGaussianDeficit
from py_wake.deficit_models.utils import ct2a_mom1d
from py_wake.deficit_models.selfsimilarity import SelfSimilarityDeficit
from py_wake.utils.model_utils import get_models
from py_wake.tests.test_wind_farm_models.test_enginering_wind_farm_model import OperatableV80
from py_wake.turbulence_models.crespo import CrespoHernandez
from py_wake.turbulence_models.gcl_turb import GCLTurbulence
from py_wake.turbulence_models.stf import STF2017TurbulenceModel
from py_wake.rotor_avg_models.rotor_avg_model import CGIRotorAvg
from py_wake.utils.gradients import autograd, cs, fd
from py_wake.utils.numpy_utils import Numpy32

d02 = 8.1 - 5.7
d12 = 8.1 - 4.90473373


@pytest.mark.parametrize('superpositionModel,res', [(LinearSum(), 8.1 - (d02 + d12)),
                                                    (SquaredSum(), 8.1 - np.hypot(d02, d12)),
                                                    (MaxSum(), 8.1 - d12)])
def test_superposition_models(superpositionModel, res):
    site = UniformSite([1], 0.1)
    wake_model = NOJ(site, NibeA0(), ct2a=ct2a_mom1d, superpositionModel=superpositionModel)
    x_i = [0, 0, 0]
    y_i = [0, -40, -100]
    h_i = [50, 50, 50]
    WS_eff = wake_model(x_i, y_i, h_i, type=[0, 0, 1], wd=0.0, ws=8.1).WS_eff
    npt.assert_array_almost_equal(WS_eff.squeeze()[-1], res)


@pytest.mark.parametrize('superpositionModel,res', [(LinearSum(), 0.1 + (.6 + .2)),
                                                    (SquaredSum(), .1 + np.hypot(.6, .2)),
                                                    (MaxSum(), .1 + .6),
                                                    (SqrMaxSum(), np.hypot(.1, .6))])
def test_superposition_models_TI(superpositionModel, res):
    site = UniformSite([1], 0.1)

    class MyTurbulenceModel(TurbulenceModel):

        def calc_added_turbulence(self, dw_ijlk, **_):
            return 1.2 - dw_ijlk / 100

    wake_model = PropagateDownwind(
        site,
        NibeA0(),
        NoWakeDeficit(),
        turbulenceModel=MyTurbulenceModel(superpositionModel))
    x_i = [0, 0, 0]
    y_i = [0, -40, -100]
    h_i = [50, 50, 50]
    TI_eff = wake_model(x_i, y_i, h_i, type=[0, 0, 1], wd=0.0, ws=8.1).TI_eff
    npt.assert_array_almost_equal(TI_eff[-1, 0, 0], res)


@pytest.mark.parametrize('superpositionModel,sum_func', [(LinearSum(), np.sum),
                                                         (SquaredSum(), lambda x: np.hypot(*x)),
                                                         (MaxSum(), np.max)])
def test_superposition_model_indices(superpositionModel, sum_func):
    class WTSite(UniformSite):
        def local_wind(self, x=None, y=None, h=None,
                       wd=None, ws=None, time=None, wd_bin_size=None, ws_bins=None, **_):
            lw = UniformSite.local_wind(self, x=x, y=y, h=h, wd=wd, ws=ws,
                                        wd_bin_size=wd_bin_size, ws_bins=ws_bins)
            lw['WS_ilk'] = lw.WS_ilk + np.arange(len(x))[:, np.newaxis, np.newaxis]
            return lw

    site = WTSite([1], 0.1)

    x_i = [0, 0, 0]
    y_i = [0, -40, -100]
    h_i = [50, 50, 50]

    # WS_ilk different at each wt position
    WS_ilk = site.local_wind(x_i, y_i, h_i, wd=0, ws=8.1).WS_ilk
    npt.assert_array_equal(WS_ilk, np.reshape([8.1, 9.1, 10.1], (3, 1, 1)))

    for wake_model in [PropagateDownwind(site, NibeA0(), wake_deficitModel=NOJDeficit(ct2a=ct2a_mom1d), superpositionModel=superpositionModel),
                       All2AllIterative(site, NibeA0(), wake_deficitModel=NOJDeficit(ct2a=ct2a_mom1d), superpositionModel=superpositionModel)]:

        # No wake (ct = 0), i.e. WS_eff == WS
        WS_eff = wake_model(x_i, y_i, h_i, type=[1, 1, 1], wd=0.0, ws=8.1).WS_eff
        npt.assert_array_equal(WS_eff, WS_ilk)

        ref = WS_ilk - np.reshape([0, 3.75, sum_func([2.4, 3.58974359])], (3, 1, 1))

        # full wake (CT=8/9)
        WS_eff = wake_model(x_i, y_i, h_i, wd=0.0, ws=8.1).WS_eff
        npt.assert_array_almost_equal(WS_eff, ref)

        sim_res = wake_model(x_i, y_i, h_i, type=[0, 0, 0], wd=0.0, ws=8.1)
        WS_eff_ilk = sim_res.flow_map(HorizontalGrid(x=[0], y=np.array(y_i) + 1e-10, h=50)).WS_eff.squeeze()

        npt.assert_array_almost_equal(WS_eff_ilk, ref.squeeze())


def test_diff_wake_blockage_superposition():
    site = UniformSite([1], 0.1)

    class MyWakeDeficit(WakeDeficitModel):
        def calc_deficit(self, dw_ijlk, **_):
            return (dw_ijlk > 0) * 2

    class MyBlockageDeficit(BlockageDeficitModel):

        def __init__(self, superpositionModel=None):
            BlockageDeficitModel.__init__(self, upstream_only=True, superpositionModel=superpositionModel)

        def calc_deficit(self, dw_ijlk, **_):
            return (dw_ijlk < 0) * .3

    wfm = All2AllIterative(site, V80(), wake_deficitModel=MyWakeDeficit(), superpositionModel=SquaredSum(),
                           blockage_deficitModel=MyBlockageDeficit(superpositionModel=LinearSum()))
    x = np.arange(5) * 160
    y = x * 0
    sim_res = wfm(x, y, ws=10, wd=270)
    npt.assert_array_almost_equal(sim_res.WS_eff.squeeze(), [10 - (4 - i) * .3 - np.sqrt(i * 2**2) for i in range(5)])


@pytest.mark.parametrize(
    'superpositionModel,ref',
    [(WeightedSum(), 782.7222806),
     (CumulativeWakeSum(), 797.4772703),
     ])
def test_complex_superposition_blockage(superpositionModel, ref):
    site = IEA37Site(16)
    x, y = site.initial_position.T
    windTurbines = IEA37_WindTurbines()

    wfm = All2AllIterative(site, windTurbines, wake_deficitModel=BastankhahGaussianDeficit(),
                           blockage_deficitModel=SelfSimilarityDeficit(),
                           superpositionModel=superpositionModel)
    sim_res = wfm(x, y, ws=[8., 9., 10.], wd=[270., 280.])
    npt.assert_array_almost_equal(np.sum(sim_res.WS_eff), ref, 7)


def test_weighted_sum_gradients():
    wfm = All2AllIterative(Hornsrev1Site(), V80(), wake_deficitModel=NiayifarGaussianDeficit(),
                           superpositionModel=WeightedSum(), turbulenceModel=CrespoHernandez())
    dx, dy = wfm.aep_gradients(x=[0., 500., 1000.], y=[0., 20., -10.], wd=[270.], ws=[8.], TI=[.06], time=True)
    assert np.isfinite([dx, dy]).all()


@pytest.mark.parametrize('superpositionModel', get_models(SuperpositionModel, exclude_None=True))
def test_superpositionModels(superpositionModel):

    kwargs = dict(site=UniformSite(), windTurbines=OperatableV80(), wake_deficitModel=NiayifarGaussianDeficit(),
                  superpositionModel=superpositionModel(), turbulenceModel=CrespoHernandez())
    wfm_a2a = All2AllIterative(**kwargs)
    wfm_pdw = PropagateDownwind(**kwargs)
    operating = [1] * 8 + [0]
    sim_res_a2a = wfm_a2a(wt9_x, wt9_y, operating=operating)
    sim_res_pdw = wfm_pdw(wt9_x, wt9_y, operating=operating)
    npt.assert_array_almost_equal(sim_res_a2a.WS_eff.sel(wt=8), sim_res_pdw.WS_eff.sel(wt=8))
    fm_a2a = sim_res_a2a.flow_map(Points(wt9_x[-1:], wt9_y[-1:], [70]), wd=sim_res_a2a.wd, ws=sim_res_a2a.ws)
    fm_pdw = sim_res_a2a.flow_map(Points(wt9_x[-1:], wt9_y[-1:], [70]), wd=sim_res_pdw.wd, ws=sim_res_pdw.ws)
    npt.assert_array_almost_equal(sim_res_a2a.WS_eff.sel(wt=8).squeeze(), fm_a2a.WS_eff)
    npt.assert_array_almost_equal(sim_res_a2a.WS_eff.sel(wt=8).squeeze(), fm_pdw.WS_eff)


@pytest.mark.parametrize('superpositionModel', get_models(SuperpositionModel, exclude_None=True))
def test_superpositionModels_rotor_avg_model(superpositionModel):

    kwargs = dict(site=UniformSite(), windTurbines=OperatableV80(), wake_deficitModel=NiayifarGaussianDeficit(rotorAvgModel=CGIRotorAvg(7)),
                  superpositionModel=superpositionModel(), turbulenceModel=CrespoHernandez())
    wfm_a2a = All2AllIterative(**kwargs)
    wfm_pdw = PropagateDownwind(**kwargs)
    wfm_rc = PropagateDownwind(**{**kwargs, 'wake_deficitModel': NiayifarGaussianDeficit()})

    operating = [1] * 8 + [0]
    sim_res_a2a = wfm_a2a(wt9_x, wt9_y, operating=operating)
    sim_res_pdw = wfm_pdw(wt9_x, wt9_y, operating=operating)
    npt.assert_array_almost_equal(sim_res_a2a.WS_eff.sel(wt=8), sim_res_pdw.WS_eff.sel(wt=8))
    sim_res_rc = wfm_rc(wt9_x, wt9_y, operating=operating)
    npt.assert_array_less(sim_res_rc.WS_eff.sel(wd=270)[3:], sim_res_pdw.WS_eff.sel(wd=270)[3:])


def test_squaredSum_speedups():
    wfm = All2AllIterative(site=UniformSite(), windTurbines=V80(), wake_deficitModel=NOJDeficit(),
                           blockage_deficitModel=SelfSimilarityDeficit(exclude_wake=False), superpositionModel=SquaredSum())
    with pytest.raises(AssertionError, match='SquaredSum only works for deficit - not speedups'):
        wfm([0, 200, 400], [0, 0, 0], wd=270)


def _accumulating_and_stock(wfm, monkeypatch, *args, **kwargs):
    accumulate, calls = engineering_models._accumulate, []
    monkeypatch.setattr(engineering_models, '_accumulate', lambda *args: calls.append(accumulate(*args)))
    sim_res = wfm(*args, **kwargs)
    monkeypatch.setattr(engineering_models, '_accumulates', lambda values: False)
    return sim_res, wfm(*args, **kwargs), len(calls)


@pytest.mark.parametrize('wfm_cls,turbulenceModel,blockage', [
    (PropagateDownwind, None, False),
    (PropagateDownwind, STF2017TurbulenceModel(), False),
    (PropagateDownwind, CrespoHernandez(), False),
    (PropagateUpDownIterative, STF2017TurbulenceModel(), True),
    (All2AllIterative, CrespoHernandez(), True)])
@pytest.mark.parametrize('superpositionModel', [LinearSum(), SquaredSum()])
@pytest.mark.parametrize('wd,ws', [(np.arange(0, 360, 30), [6, 8, 10]), (np.arange(0, 360, 30), 8), (270, [6, 8]), (270, 8)])
def test_accumulating_superposition(wfm_cls, turbulenceModel, blockage, superpositionModel, wd, ws, monkeypatch):
    kwargs = {'turbulenceModel': turbulenceModel}
    if blockage:
        kwargs['blockage_deficitModel'] = SelfSimilarityDeficit(superpositionModel=LinearSum(), use_effective_ws=True)
    site = Hornsrev1Site()
    x, y = site.initial_position.T
    wfm = wfm_cls(site, V80(), BastankhahGaussianDeficit(use_effective_ws=True), superpositionModel=superpositionModel,
                  **kwargs)
    sim_res, ref, calls = _accumulating_and_stock(wfm, monkeypatch, x, y, wd=wd, ws=ws)
    assert (calls > 0) == (np.size(wd) * np.size(ws) > 1)
    for k in ['WS_eff', 'TI_eff', 'CT', 'Power']:
        assert sim_res[k].values.tobytes() == ref[k].values.tobytes()


def test_accumulating_superposition_numpy32(monkeypatch):
    site = Hornsrev1Site()
    x, y = site.initial_position.T
    with Numpy32():
        sim_res, ref, calls = _accumulating_and_stock(PropagateDownwind(site, V80(), NOJDeficit()), monkeypatch,
                                                      x, y, wd=np.arange(0, 360, 30), ws=[6, 8, 10])
    assert calls == 0
    assert sim_res.WS_eff.values.tobytes() == ref.WS_eff.values.tobytes()


@pytest.mark.parametrize('wake_deficitModel,turbulenceModel', [
    (NiayifarGaussianDeficit(), GCLTurbulence(LinearSum(), rotorAvgModel=None)),
    (NiayifarGaussianDeficit(), GCLTurbulence(SqrMaxSum(), rotorAvgModel=None)),
    (BastankhahGaussianDeficit(), CrespoHernandez())])
@pytest.mark.filterwarnings('ignore:Output seems independent of input')
def test_accumulating_superposition_gradients(wake_deficitModel, turbulenceModel):
    site = Hornsrev1Site()
    x, y = site.initial_position[:9].T
    wfm = PropagateDownwind(site, V80(), wake_deficitModel, turbulenceModel=turbulenceModel)
    dAEP_fd, dAEP_cs, dAEP_autograd = [wfm.aep_gradients(method, ['TI'], x=x, y=y, wd=270, ws=[8, 10], TI=.08)
                                       for method in [fd, cs, autograd]]
    npt.assert_array_almost_equal(dAEP_autograd, dAEP_cs, 10)
    npt.assert_allclose(dAEP_cs, dAEP_fd, rtol=1e-4)


def test_accumulating_superposition_subclass():
    class MaxLinearSum(LinearSum):
        def __call__(self, value_jxxx):
            return np.max(value_jxxx, 0)

    kwargs = dict(site=UniformSite(), windTurbines=V80(), wake_deficitModel=BastankhahGaussianDeficit())
    sim_res = PropagateDownwind(superpositionModel=MaxLinearSum(), **kwargs)(wt9_x, wt9_y)
    ref = PropagateDownwind(superpositionModel=MaxSum(), **kwargs)(wt9_x, wt9_y)
    npt.assert_array_equal(sim_res.WS_eff, ref.WS_eff)


def test_accumulating_squaredSum_speedups():
    class SpeedupDeficit(NOJDeficit):
        def calc_deficit(self, ct_ilk, **kwargs):
            return -NOJDeficit.calc_deficit(self, ct_ilk=ct_ilk, **kwargs)

    wfm = PropagateDownwind(UniformSite(), V80(), SpeedupDeficit(), superpositionModel=SquaredSum())
    with pytest.raises(AssertionError, match='SquaredSum only works for deficit - not speedups'):
        wfm([0, 200, 400], [0, 0, 0], wd=[270, 280])
