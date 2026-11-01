from numpy import nan
from py_wake.site._site import UniformSite
from py_wake.examples.data.hornsrev1 import V80
from py_wake.ground_models import Mirror, MultiMirror
from py_wake.deficit_models.noj import NOJ, NOJDeficit
import matplotlib.pyplot as plt
from py_wake.flow_map import YZGrid
from py_wake import np
from py_wake.tests import npt
from py_wake.wind_turbines import WindTurbines
from py_wake.superposition_models import LinearSum, SquaredSum
from py_wake.wind_farm_models.engineering_models import PropagateDownwind, All2AllIterative
import pytest
from py_wake.deficit_models.gaussian import ZongGaussianDeficit
from py_wake.turbulence_models.stf import STF2017TurbulenceModel
from py_wake.deficit_models.no_wake import NoWakeDeficit
from py_wake.flow_map import XYGrid, XZGrid
from py_wake.wind_turbines import WindTurbine
from py_wake.wind_turbines.power_ct_functions import PowerCtTabular
from py_wake.deficit_models import SelfSimilarityDeficit2020
import warnings
from py_wake.tests.test_wind_farm_models.test_enginering_wind_farm_model import OperatableV80
from py_wake.deficit_models.gaussian import BastankhahGaussianDeficit, TurboGaussianDeficit
from py_wake.rotor_avg_models import CGIRotorAvg
from py_wake.rotor_avg_models.gaussian_overlap_model import GaussianOverlapAvgModel
from numpy import newaxis as na
from py_wake.ground_models import ground_models
from py_wake.utils.gradients import autograd, cs, fd


@pytest.mark.parametrize('wfm_cls', [PropagateDownwind,
                                     All2AllIterative])
@pytest.mark.parametrize('superpositionModel', [LinearSum(), SquaredSum()])
def test_Mirror_NOJ(wfm_cls, superpositionModel):
    # Compare points in flow map with ws of WT at same position
    site = UniformSite([1], ti=0.1)
    V80_D0 = V80()
    V80_D0._diameters = [0]
    wt = WindTurbines.from_WindTurbine_lst([V80(), V80_D0])
    wfm_ref = wfm_cls(site, wt, wake_deficitModel=NOJDeficit(k=.5), superpositionModel=superpositionModel)
    fm = wfm_ref([0, 0], [0, 0], h=[50, -50], wd=0).flow_map(YZGrid(x=0, y=np.arange(-70, 0, 20), z=10))
    ref = fm.WS_eff.squeeze()

    wfm = wfm_cls(site, wt, wake_deficitModel=NOJDeficit(k=.5, groundModel=Mirror()),
                  superpositionModel=superpositionModel,)
    sim_res = wfm([0], [0], h=[50], wd=0)
    fm_res = sim_res.flow_map(YZGrid(x=0, y=np.arange(-70, 0, 20), z=10)).WS_eff.squeeze()
    with warnings.catch_warnings():
        # D=0 gives divide by zero warning
        warnings.filterwarnings('ignore', r'divide by zero encountered in true_divide')
        res = np.array([wfm([0, 0], [0, y], [50, 10], type=[0, 1], wd=0).WS_eff.sel(wt=1).item()
                        for y in [-70, -50, -30, -10]])  # ref.y])

    if 0:
        sim_res.flow_map(YZGrid(x=0, y=np.arange(-100, 10, 1))).plot_wake_map()
        plt.plot(ref.y, ref.y * 0 + ref.h, '.')
        plt.plot(ref.y, ref * 10, label='ref, WS*10')
        plt.plot(ref.y, res * 10, label='Res, WS*10')
        plt.plot(fm_res.y, fm_res * 10, label='Res flowmap, WS*10')

        plt.legend()
        plt.show()
    plt.close('all')
    npt.assert_array_equal(res, ref)
    npt.assert_array_equal(fm_res, ref)


def test_Mirror_All2AllIterative():
    # Compare points in flow map with ws of WT at same position
    site = UniformSite([1], ti=0.1)
    V80_D0 = V80()
    V80_D0._diameters = [0]
    wt = WindTurbines.from_WindTurbine_lst([V80(), V80_D0])
    wfm = All2AllIterative(site, wt, NOJDeficit(k=.5, groundModel=Mirror()))
    sim_res = wfm([0], [0], h=[50], wd=0)
    fm_ref = sim_res.flow_map(YZGrid(x=0, y=np.arange(-70, 0, 20), z=10))
    ref = fm_ref.WS_eff_xylk[:, 0, 0, 0].values
    with warnings.catch_warnings():
        # D=0 gives divide by zero warning
        warnings.filterwarnings('ignore', r'divide by zero encountered in true_divide')
        res = np.array([wfm([0, 0], [0, y], [50, 10], type=[0, 1], wd=0).WS_eff.sel(wt=1).item() for y in fm_ref.X[0]])

    if 0:
        fm_res = sim_res.flow_map(YZGrid(x=0, y=np.arange(-100, 10, 1)))
        fm_res.plot_wake_map()
        plt.plot(fm_ref.X[0], fm_ref.Y[0], '.')
        plt.plot(fm_ref.X[0], ref * 10, label='ref, WS*10')
        plt.plot(fm_ref.X[0], res * 10, label='Res, WS*10')

        plt.legend()
        plt.show()
    plt.close('all')
    npt.assert_array_equal(res, ref)


@pytest.mark.parametrize('wfm_cls', [PropagateDownwind, All2AllIterative])
def test_Mirror(wfm_cls):
    # Compare points in flow map with ws of WT at same position. All2Alliterative failing with NOJ and WT.diameter=0
    # and therefore this cannot be tested above
    site = UniformSite([1], ti=0.1)
    wt = V80()
    wfm = wfm_cls(site, wt, ZongGaussianDeficit(a=[0, 1], groundModel=Mirror()),
                  turbulenceModel=STF2017TurbulenceModel())
    sim_res = wfm([0], [0], h=[50], wd=0,)
    fm_ref = sim_res.flow_map(YZGrid(x=0, y=np.arange(-70, 0, 20), z=10))
    ref = fm_ref.WS_eff_xylk[:, 0, 0, 0].values

    res = np.array([wfm([0, 0], [0, y], [50, 10], wd=0).WS_eff.sel(wt=1).item() for y in fm_ref.X[0]])

    if 0:
        fm_res = sim_res.flow_map(YZGrid(x=0, y=np.arange(-100, 10, 1)))
        fm_res.plot_wake_map()
        plt.plot(fm_ref.X[0], fm_ref.Y[0], '.')
        plt.plot(fm_ref.X[0], ref * 10, label='ref, WS*10')
        plt.plot(fm_ref.X[0], res * 10, label='Res, WS*10')

        plt.legend()
        plt.show()
    plt.close('all')
    npt.assert_array_equal(res, ref)


@pytest.mark.parametrize('wake,superpositionModel,ref_min_ws_h', [
    (1, LinearSum, [nan, nan, nan, nan, nan, 47.2, 45.2, 41.1, 31.7, 2.0]),
    (1, SquaredSum, [nan, nan, nan, nan, nan, 49.9, 49.9, 49.6, 49.0, 47.5]),
    (0, LinearSum, [2., 21.2, 39.5, 47., 49.7, nan, nan, nan, nan, nan]),
    (0, LinearSum, [2.0, 21.2, 39.5, 47.0, 49.7, nan, nan, nan, nan, nan])])
def test_Mirror_superposition(wake, superpositionModel, ref_min_ws_h):
    site = UniformSite([1], ti=0.1)
    wt = OperatableV80()
    if wake:
        wm = ZongGaussianDeficit(a=[0, 1], groundModel=Mirror(superpositionModel()))
        bm = None
    else:
        wm = NoWakeDeficit()
        bm = SelfSimilarityDeficit2020(upstream_only=1, groundModel=Mirror(superpositionModel()))
    wfm = All2AllIterative(site, wt, wm, blockage_deficitModel=bm, turbulenceModel=STF2017TurbulenceModel())
    sim_res = wfm([0], [0], h=[50], wd=270)
    fm_ws = sim_res.flow_map(XZGrid(y=0, x=np.linspace(-100, 100, 10), z=np.linspace(0, 100)))
    h_ref = fm_ws.h[5]
    sim_res_ws = np.array([wfm([0, 0], [0, y], [50, h_ref], wd=0, operating=[
                          1, 0]).WS_eff.sel(wt=1).item() for y in -fm_ws.x])

    if 0:
        fm_res = sim_res.flow_map(XZGrid(y=0, x=np.arange(-100, 300, 1), z=np.linspace(-100, 100)))

        fm_res.plot_wake_map()
        plt.plot(fm_res.x, fm_res.min_WS_eff())
        plt.plot(fm_ws.x, fm_ws.min_WS_eff())
        plt.plot(fm_ws.x, fm_ws.x * 0 + h_ref, '.')
        plt.plot(fm_ws.x, fm_ws.WS_eff.interp(h=h_ref).squeeze() * 10, label='flowmap, WS*10')
        plt.plot(fm_ws.x, sim_res_ws * 10, label='sim_res, WS*10')

        plt.legend()
        plt.show()
    plt.close('all')

    npt.assert_array_equal(fm_ws.WS_eff.interp(h=h_ref).squeeze(), sim_res_ws)
    # print(np.round(fm_ws.min_WS_eff().values, 1).tolist())
    npt.assert_array_almost_equal(fm_ws.min_WS_eff(), ref_min_ws_h, 1)


def test_Mirror_superposition_far_downstream():
    site = UniformSite([1], ti=0.1)
    wt = OperatableV80()

    def get_ws(wm):
        wfm = All2AllIterative(site, wt, wm, turbulenceModel=STF2017TurbulenceModel())
        sim_res = wfm([0], [0], h=[50], wd=270)
        return sim_res.flow_map(XZGrid(y=0, x=np.linspace(-100, 5000, 100), z=50)).WS_eff.squeeze()

    wm_dict = {'Zong': ZongGaussianDeficit(),
               'Zong,Mirror(LinearSum)': ZongGaussianDeficit(groundModel=Mirror()),
               'Zong,Mirror(SquaredSum)': ZongGaussianDeficit(groundModel=Mirror(SquaredSum()))}
    fm_ws_dict = {l: get_ws(wm) for l, wm in wm_dict.items()}

    # print('ref={' + ",\n".join([f'"{l}": {np.round(fm_ws.values[::10], 2).tolist()}'
    #                            for l, fm_ws in fm_ws_dict.items()]) + '}')
    ref = {"Zong": [12.0, 9.33, 11.01, 11.48, 11.68, 11.78, 11.84, 11.88, 11.91, 11.92],
           "Zong,Mirror(LinearSum)": [12.0, 9.25, 10.76, 11.23, 11.48, 11.62, 11.72, 11.78, 11.82, 11.86],
           "Zong,Mirror(SquaredSum)": [12.0, 9.33, 10.98, 11.42, 11.62, 11.73, 11.8, 11.84, 11.88, 11.9]}
    if 0:
        for l, fm_ws in fm_ws_dict.items():
            c = plt.plot(fm_ws.x, fm_ws, label=l)[0].get_color()
            plt.plot(fm_ws.x[::10], ref[l], 'x', color=c)
        plt.legend()
        plt.show()

    for k in fm_ws_dict:
        npt.assert_array_almost_equal(fm_ws_dict[k][::10], ref[k], 2)


@pytest.mark.parametrize('wfm_cls', [PropagateDownwind, All2AllIterative])
@pytest.mark.parametrize('groundModel,superpositionModel', [(Mirror(), LinearSum()),
                                                            (Mirror(), SquaredSum()), ])
def test_Mirror_flow_map(wfm_cls, groundModel, superpositionModel):
    site = UniformSite([1], ti=0.1)
    wt = V80()
    wfm = NOJ(site, wt, k=.5, superpositionModel=superpositionModel)

    fm_ref = wfm([0, 0 + 1e-20], [0, 0 + 1e-20], wd=0, h=[50, -50]
                 ).flow_map(YZGrid(x=0, y=np.arange(-100, 100, 1) + .1, z=np.arange(1, 100)))
    fm_ref.plot_wake_map()
    plt.title("Underground WT added manually")

    plt.figure()
    wfm = wfm_cls(site, wt, NOJDeficit(k=.5, groundModel=groundModel),
                  superpositionModel=superpositionModel)
    fm_res = wfm([0], [0], wd=0, h=[50]).flow_map(YZGrid(x=0, y=np.arange(-100, 100, 1) + .1, z=np.arange(1, 100)))
    fm_res.plot_wake_map()
    plt.title("With Mirror GroundModel")

    if 0:
        plt.show()
    plt.close('all')
    npt.assert_array_equal(fm_ref.WS_eff, fm_res.WS_eff)


def test_Mirror_flow_map_multiple_wd():
    site = UniformSite([1], ti=0.1)
    wt = V80()
    wfm = NOJ(site, wt, k=.5, superpositionModel=LinearSum())

    sim_res_ref = wfm([0, 0 + 1e-20], [0, 0 + 1e-20], wd=[0, 5], h=[50, -50]
                      )
    fm_ref = sim_res_ref.flow_map(YZGrid(x=0, y=np.arange(-100, 100, 1) + .1, z=np.arange(1, 100)), wd=sim_res_ref.wd)
    fm_ref.plot_wake_map()
    plt.title("Underground WT added manually")

    plt.figure()
    wfm = All2AllIterative(site, wt, NOJDeficit(k=.5, groundModel=Mirror()),
                           superpositionModel=LinearSum())
    sim_res = wfm([0], [0], wd=[0, 5], h=[50])
    fm_res = sim_res.flow_map(YZGrid(x=0, y=np.arange(-100, 100, 1) + .1, z=np.arange(1, 100)), wd=sim_res.wd)
    fm_res.plot_wake_map()
    plt.title("With Mirror GroundModel")

    if 0:
        plt.show()
    plt.close('all')
    npt.assert_array_equal(fm_ref.WS_eff, fm_res.WS_eff)


def test_MultiMirror():

    # create dummy sites and turbine
    class Dummy(WindTurbine):
        def __init__(self, name='dummy', ct=1.0, D=1., zh=1.5):
            WindTurbine.__init__(self, name=name, diameter=D,
                                 hub_height=zh, powerCtFunction=PowerCtTabular([-100, 100], [0, 0], 'kW', [ct, ct]))

    class BastankhahSite(UniformSite):
        def __init__(self, ws=1., ti=0.1):
            UniformSite.__init__(self, ti=ti, ws=ws)

    ct = 1.
    D = 1.
    hub_height = 0.6 * D
    mirror_height = 1.2 * D

    x = np.linspace(-2. * D, 2. * D, 401)
    y = np.linspace(0., 1.2 * D, 201)
    hgrid = XYGrid(x=x, y=y)
    vgrid = XZGrid(0.0, x=x, z=y)

    bmodels = {
        "MultiMirror(n_reps=0)": SelfSimilarityDeficit2020(groundModel=MultiMirror(mirror_height, n_reps=0)),
        "MultiMirror(n_reps=4)": SelfSimilarityDeficit2020(groundModel=MultiMirror(mirror_height, n_reps=4)),
        "MultiMirror(n_reps=0) Verification": SelfSimilarityDeficit2020(),
    }
    wfms = []
    for name, bmodel in bmodels.items():
        wfms.append(All2AllIterative(BastankhahSite(), Dummy(ct=ct, D=D, zh=hub_height),
                                     wake_deficitModel=NoWakeDeficit(),
                                     blockage_deficitModel=bmodel))

    fms = []
    for i in range(len(wfms) - 1):
        fms.append(wfms[i](x=[0], y=[0], wd=270., ws=1.).flow_map(vgrid))
    wt_y = [hub_height,
            -hub_height,
            mirror_height + (mirror_height - hub_height),
            mirror_height + (mirror_height - hub_height) + 2 * hub_height]
    fms.append(wfms[-1](x=[0, 0, 0, 0], y=wt_y,
                        wd=270., ws=1.).flow_map(hgrid))
    # check verification
    npt.assert_array_almost_equal(fms[0].WS_eff, fms[-1].WS_eff, 10)

    # check whether several repetitions are also working
    npt.assert_almost_equal(np.sum(fms[1].WS_eff), 80601.)


class DuplicatingMirror(Mirror):
    def _update_kwargs(self, **kwargs):
        new_kwargs = Mirror._update_kwargs(self, **kwargs)
        if kwargs['IJLK'][0] == 1:
            new_kwargs.update({k: np.concatenate([v, v], 0) for k, v in kwargs.items()
                               if '_ijlk' in k and k not in ['dh_ijlk', 'cw_ijlk']})
        return new_kwargs


mirror_deficit_models = [
    lambda groundModel: NOJDeficit(k=.1, groundModel=groundModel),
    lambda groundModel: ZongGaussianDeficit(groundModel=groundModel),
    lambda groundModel: BastankhahGaussianDeficit(rotorAvgModel=CGIRotorAvg(7), groundModel=groundModel),
    lambda groundModel: TurboGaussianDeficit(rotorAvgModel=GaussianOverlapAvgModel(), groundModel=groundModel),
]


@pytest.mark.parametrize('wfm_cls', [PropagateDownwind, All2AllIterative])
@pytest.mark.parametrize('deficitModel', mirror_deficit_models)
def test_Mirror_shared_inputs(wfm_cls, deficitModel):
    site = UniformSite([1], ti=0.1)
    x, y, h = [0, 400, 800, 200], [0, 50, -30, 300], [50, 70, 60, 80]

    def get_sim_res(groundModel):
        wfm = wfm_cls(site, V80(), deficitModel(groundModel), turbulenceModel=STF2017TurbulenceModel())
        return wfm(x, y, h=h, wd=[260, 270, 280], ws=[6, 10])

    sim_res_ref, sim_res = get_sim_res(DuplicatingMirror()), get_sim_res(Mirror())
    npt.assert_array_equal(sim_res.WS_eff, sim_res_ref.WS_eff)

    grid = XZGrid(y=0, x=np.linspace(-100, 1000, 12), z=np.linspace(1, 150, 6))
    npt.assert_array_equal(sim_res.flow_map(grid, wd=270, ws=10).WS_eff,
                           sim_res_ref.flow_map(grid, wd=270, ws=10).WS_eff)


@pytest.mark.parametrize('deficitModel', mirror_deficit_models[1:])
def test_Mirror_shared_inputs_gradients(deficitModel):
    site = UniformSite([1], ti=0.1)
    x, y, h = np.array([0., 400, 800]), np.array([0., 50, -30]), np.array([50., 70, 60])

    def get_gradients(groundModel):
        wfm = PropagateDownwind(site, V80(), deficitModel(groundModel), turbulenceModel=STF2017TurbulenceModel())
        return wfm.aep_gradients(autograd, ['x', 'y', 'h'], x=x, y=y, h=h, wd=[260, 270, 280], ws=[6, 10])

    npt.assert_array_almost_equal(get_gradients(Mirror()), get_gradients(DuplicatingMirror()), 10)


def _stock_mirror_call(self, func, h_ilk, dh_ijlk, IJLK, **kwargs):
    new_kwargs = self._update_kwargs(h_ilk=h_ilk, dh_ijlk=dh_ijlk, IJLK=IJLK, **kwargs)
    above_ground = ((new_kwargs['h_ilk'][:, na, :] + new_kwargs['dh_ijlk']) > 0)
    values_pijlk = func(**new_kwargs)
    deficit_mijlk = np.reshape(values_pijlk * above_ground, (2,) + IJLK)
    superpositionModel = self.superpositionModel or getattr(func, '__self__', func).superpositionModel
    return superpositionModel(deficit_mijlk)


def _spy_above_ground_only(monkeypatch):
    above_ground_only, calls = ground_models._above_ground_only, {'skip': 0, 'multiply': 0}

    def spy(values_pijlk, above_ground_pijlk):
        res = above_ground_only(values_pijlk, above_ground_pijlk)
        calls['skip' if res is values_pijlk else 'multiply'] += 1
        return res
    monkeypatch.setattr(ground_models, '_above_ground_only', spy)
    return calls


class SubArray(np.ndarray):
    pass


mirror_rng = np.random.default_rng(3)
mirror_values = np.r_[mirror_rng.uniform(0, 2, 24), -0., 0., np.inf, -np.inf, np.nan, -np.nan, 5e-324, 1e308]
mirror_values = mirror_values.reshape(2, 4, 4, 1) * np.ones(3)
mirror_above = np.ones((2, 4, 4, 1), dtype=bool)
mirror_partly = mirror_above.copy()
mirror_partly[1, 2, 0] = False


@pytest.mark.parametrize('values,above_ground,skip', [
    pytest.param(mirror_values, mirror_above, True, id='above'),
    pytest.param(mirror_values, mirror_above[..., 0][..., na] * np.ones(3, dtype=bool), True, id='full_mask'),
    pytest.param(mirror_values, np.ones((1, 1, 1, 1), dtype=bool), True, id='broadcast_mask'),
    pytest.param(mirror_values[:, :0], mirror_above[:, :0], True, id='empty'),
    pytest.param(mirror_values.astype(np.float32), mirror_above, True, id='float32'),
    pytest.param(mirror_values, mirror_partly, False, id='partly_below'),
    pytest.param(mirror_values, ~mirror_above, False, id='below'),
    pytest.param(np.ascontiguousarray(mirror_values[..., :1]), mirror_above * np.ones(3, dtype=bool), False, id='mask_broadcasts_values'),
    pytest.param(np.asfortranarray(mirror_values), mirror_above, False, id='fortran'),
    pytest.param(mirror_values.astype('>f8'), mirror_above, False, id='big_endian'),
    pytest.param(np.broadcast_to(mirror_values, mirror_values.shape), mirror_above, False, id='readonly'),
    pytest.param(mirror_values[0], mirror_above, False, id='mask_more_dims'),
    pytest.param(mirror_values[:, :, ::2], mirror_above[:, :, ::2], False, id='strided'),
    pytest.param(mirror_values + 0j, mirror_above, False, id='complex'),
    pytest.param(np.ones((2, 4, 4, 3), dtype=int), mirror_above, False, id='int'),
    pytest.param(mirror_values.view(SubArray), mirror_above, False, id='subclass_values'),
    pytest.param(mirror_values, mirror_above.view(SubArray), False, id='subclass_mask'),
    pytest.param(mirror_values, mirror_above.astype(float), False, id='float_mask'),
    pytest.param(mirror_values, mirror_above.astype(float) * 2, False, id='scaling_mask')])
def test_above_ground_only(values, above_ground, skip):
    res = ground_models._above_ground_only(values, above_ground)
    ref = values * above_ground
    assert (res is values) == skip
    assert res.dtype == ref.dtype
    assert res.shape == ref.shape
    assert res.strides == ref.strides or res.size == 0
    assert res.tobytes() == ref.tobytes()


@pytest.mark.parametrize('wfm_cls', [PropagateDownwind, All2AllIterative])
@pytest.mark.parametrize('deficitModel', mirror_deficit_models)
def test_Mirror_above_ground_only(wfm_cls, deficitModel, monkeypatch):
    site = UniformSite([1], ti=0.1)
    x, y, h = [0, 400, 800, 200], [0, 50, -30, 300], [50, 70, 60, 80]
    wfm = wfm_cls(site, V80(), deficitModel(Mirror()), turbulenceModel=STF2017TurbulenceModel())
    grid = XZGrid(y=0, x=np.linspace(-100, 1000, 12), z=np.linspace(0, 150, 6))

    def run():
        sim_res = wfm(x, y, h=h, wd=[260, 270, 280], ws=[6, 10])
        return sim_res, sim_res.flow_map(grid, wd=270, ws=10)
    calls = _spy_above_ground_only(monkeypatch)
    sim_res, flow_map = run()
    assert calls['skip'] > 0 and calls['multiply'] > 0
    monkeypatch.setattr(Mirror, '__call__', _stock_mirror_call)
    ref, ref_flow_map = run()
    for k in ['WS_eff', 'TI_eff', 'CT', 'Power']:
        assert sim_res[k].values.tobytes() == ref[k].values.tobytes()
    assert flow_map.WS_eff.values.tobytes() == ref_flow_map.WS_eff.values.tobytes()


@pytest.mark.parametrize('gradient_method,skip', [(autograd, False), (cs, False), (fd, True)])
def test_Mirror_above_ground_only_gradients(gradient_method, skip, monkeypatch):
    site = UniformSite([1], ti=0.1)
    x, y, h = np.array([0., 400, 800]), np.array([0., 50, -30]), np.array([50., 70, 60])
    wfm = PropagateDownwind(site, V80(), mirror_deficit_models[3](Mirror()), turbulenceModel=STF2017TurbulenceModel())

    def get_gradients():
        return wfm.aep_gradients(gradient_method, ['x', 'y', 'h'], x=x, y=y, h=h, wd=[260, 270, 280], ws=[6, 10])
    calls = _spy_above_ground_only(monkeypatch)
    gradients = get_gradients()
    assert (calls['skip'] > 0) == skip
    monkeypatch.setattr(Mirror, '__call__', _stock_mirror_call)
    npt.assert_array_equal(gradients, get_gradients())
