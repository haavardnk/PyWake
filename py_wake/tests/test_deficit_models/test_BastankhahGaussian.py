from numpy import newaxis as na
import pytest
from py_wake import np
from py_wake.deficit_models import gaussian
from py_wake.deficit_models.gaussian import BastankhahGaussianDeficit, CarbajofuertesGaussianDeficit, \
    NiayifarGaussianDeficit, TurboGaussianDeficit, ZongGaussianDeficit
from py_wake.deficit_models.utils import ct2a_mom1d
from py_wake.examples.data.hornsrev1 import Hornsrev1Site, V80
from py_wake.flow_map import XYGrid
from py_wake.ground_models.ground_models import Mirror
from py_wake.rotor_avg_models.gaussian_overlap_model import GaussianOverlapAvgModel
from py_wake.superposition_models import LinearSum, SquaredSum
from py_wake.turbulence_models.crespo import CrespoHernandez
from py_wake.utils.gradients import autograd, cs, fd
from py_wake.utils.numpy_utils import Numpy32
from py_wake.wind_farm_models.engineering_models import PropagateDownwind, All2AllIterative
from py_wake.examples.data.iea37 import iea37_path
from py_wake.examples.data.iea37.iea37_reader import read_iea37_windfarm
from py_wake.tests import npt
import matplotlib.pyplot as plt
from py_wake.literature import IEA37CaseStudy1


def test_IEA37SimpleBastankhahGaussian_all_ex():
    """check AEP matches IEA37 value in file"""
    for n_wt in [9, 16, 36, 64]:
        x, y, aep_ref = read_iea37_windfarm(iea37_path + 'iea37-ex%d.yaml' % n_wt)
        if 0:
            plt.plot(x, y, '2k')
            for i, (x_, y_) in enumerate(zip(x, y)):
                plt.annotate(i, (x_, y_))
            plt.axis('equal')
            plt.show()
        wake_model = IEA37CaseStudy1(n_wt)

        aep_ilk = wake_model(x, y, wd=np.arange(0, 360, 22.5), ws=[9.8]).aep_ilk(normalize_probabilities=True)
        aep_MW_l = aep_ilk.sum((0, 2)) * 1000
        # test that the result is equal to results provided for IEA task 37
        npt.assert_almost_equal(aep_ref[0], aep_MW_l.sum(), 5)
        npt.assert_array_almost_equal(aep_ref[1], aep_MW_l, 5)


def _stock_calc_deficit_centre(self, sigma_ijlk, cw_ijlk, **kwargs):
    return self.calc_deficit(cw_ijlk=cw_ijlk * 0, **kwargs)


def _stock_deficit_centre(sigma_ijlk, cw_ijlk, D_src_il, ct_ilk, WS_ref_ijlk, **_):
    sigma_sqr_ijlk = sigma_ijlk**2
    deficit_centre_ijlk = WS_ref_ijlk * np.minimum(1, 2. * ct2a_mom1d(ct_ilk[:, na] * D_src_il[:, na, :, na]**2 /
                                                                    (8. * sigma_sqr_ijlk)))
    return deficit_centre_ijlk, deficit_centre_ijlk * np.exp(-1 / (2 * sigma_sqr_ijlk) * (cw_ijlk * 0)**2)


def _spy_kernel(monkeypatch):
    build, calls = gaussian._deficit_centre_kernel, {'fast': 0, 'tiny': 0}

    def spy_kernel():
        kernel = build()

        def spy(*args):
            tiny = kernel(*args)
            calls['tiny' if tiny else 'fast'] += 1
            return tiny
        return spy
    monkeypatch.setattr(gaussian, '_deficit_centre_kernel', spy_kernel)
    return calls


def _centre_inputs(I, J, L, K, cw_rows=1):
    rng = np.random.default_rng([I, J, L, K, cw_rows])
    sigma_ijlk = rng.uniform(1., 300., (I, J, L, K))
    sigma_ijlk[0, 0, :2, :5] = np.array([[5e153, 2e-154, 1e-3, 5., -.5], [1e155, 1e200, np.inf, -40., 1e-3]])[:L, :K]
    D_src_il = rng.choice([80., 120., 0.], (I, L))
    D_src_il[0, :2] = [1e150, 1e155][:L]
    ct_ilk = rng.choice([0., 1e-300, .1, .8, 1., 1.2, -.1], (I, L, K))
    ct_ilk[0, :2] = .8
    return dict(sigma_ijlk=sigma_ijlk, cw_ijlk=rng.uniform(-500., 500., (cw_rows * I, J, L, 1)), D_src_il=D_src_il,
                ct_ilk=ct_ilk, WS_ref_ijlk=rng.choice([10., 0., 25., np.inf], (I, 1, L, K)), dw_ijlk=np.zeros((I, J, L, 1)))


def _with(kwargs, **values):
    return {**kwargs, **values}


def _set(x, index, value):
    x = x.copy()
    x[index] = value
    return x


class SubArray(np.ndarray):
    pass


base = _centre_inputs(1, 37, 3, 5)
base_sigma = base['sigma_ijlk']


@pytest.mark.parametrize('kwargs,fast,tiny', [
    pytest.param(base, 1, 0, id='niayifar'),
    pytest.param(_centre_inputs(1, 37, 3, 5, cw_rows=2), 1, 0, id='mirror'),
    pytest.param(_centre_inputs(3, 37, 2, 5), 1, 0, id='all2all'),
    pytest.param(_centre_inputs(2, 37, 1, 1), 1, 0, id='one_case'),
    pytest.param(_with(base, sigma_ijlk=base_sigma[:, :0], cw_ijlk=base['cw_ijlk'][:, :0]), 1, 0, id='empty'),
    pytest.param(_with(base, WS_ref_ijlk=np.arange(15).reshape(1, 1, 3, 5)), 1, 0, id='int_WS'),
    pytest.param(_with(base, WS_ref_ijlk=base['WS_ref_ijlk'].astype(np.float32)), 1, 0, id='float32_WS'),
    pytest.param(_with(base, WS_ref_ijlk=base['WS_ref_ijlk'].astype('>f8')), 1, 0, id='big_endian_WS'),
    pytest.param(_with(base, WS_ref_ijlk=np.asfortranarray(np.broadcast_to(base['WS_ref_ijlk'], base_sigma.shape))), 1, 0,
                 id='fortran_WS'),
    pytest.param(_with(base, ct_ilk=np.asfortranarray(base['ct_ilk'])), 1, 0, id='fortran_ct'),
    pytest.param(_with(base, sigma_ijlk=_set(base_sigma, (0, 5, 1, 2), 1e-160)), 0, 1, id='subnormal_sigma_sqr'),
    pytest.param(_with(base, sigma_ijlk=_set(base_sigma, (0, 36, 2, 4), 0.)), 0, 1, id='zero_sigma'),
    pytest.param(_with(base, sigma_ijlk=_set(base_sigma, (0, 5, 1, 2), np.nan)), 0, 1, id='nan_sigma'),
    pytest.param(_with(base, sigma_ijlk=_set(base_sigma, (0, 5, 1, 2), -np.nan)), 0, 1, id='negative_nan_sigma'),
    pytest.param(_with(base, ct_ilk=_set(base['ct_ilk'], (0, 2, 4), np.nan)), 0, 0, id='nan_ct'),
    pytest.param(_with(base, D_src_il=_set(base['D_src_il'], (0, 2), np.nan)), 0, 0, id='nan_D'),
    pytest.param(_with(base, WS_ref_ijlk=_set(base['WS_ref_ijlk'], (0, 0, 2, 4), np.nan)), 0, 0, id='nan_WS'),
    pytest.param(_with(base, cw_ijlk=_set(base['cw_ijlk'], (0, 36, 2, 0), np.inf)), 0, 0, id='inf_cw'),
    pytest.param(_with(base, cw_ijlk=_set(base['cw_ijlk'], (0, 36, 2, 0), np.nan)), 0, 0, id='nan_cw'),
    pytest.param(_with(base, cw_ijlk=base['cw_ijlk'] + 0j), 0, 0, id='complex_cw'),
    pytest.param(_with(base, sigma_ijlk=base_sigma + 0j), 0, 0, id='complex_sigma'),
    pytest.param(_with(base, ct_ilk=base['ct_ilk'] + 0j), 0, 0, id='complex_ct'),
    pytest.param(_with(base, WS_ref_ijlk=base['WS_ref_ijlk'] + 0j), 0, 0, id='complex_WS'),
    pytest.param(_with(base, sigma_ijlk=base_sigma.astype(np.float32)), 0, 0, id='float32_sigma'),
    pytest.param(_with(base, D_src_il=base['D_src_il'].astype(np.float32)), 0, 0, id='float32_D'),
    pytest.param(_with(base, sigma_ijlk=base_sigma.view(SubArray)), 0, 0, id='subclass_sigma'),
    pytest.param(_with(base, sigma_ijlk=base_sigma.astype('>f8')), 0, 0, id='big_endian_sigma'),
    pytest.param(_with(base, sigma_ijlk=np.asfortranarray(base_sigma)), 0, 0, id='fortran_sigma'),
    pytest.param(_with(base, sigma_ijlk=np.repeat(base_sigma, 2, -1)[..., ::2]), 0, 0, id='strided_sigma'),
    pytest.param(_with(base, sigma_ijlk=np.ascontiguousarray(base_sigma[..., :1])), 0, 0, id='sigma_without_k'),
    pytest.param(_with(base, sigma_ijlk=base_sigma[na]), 0, 0, id='five_d_sigma')])
@pytest.mark.filterwarnings('ignore::RuntimeWarning')
def test_deficit_centre_kernel(kwargs, fast, tiny, monkeypatch):
    calls = _spy_kernel(monkeypatch)
    deficitModel = NiayifarGaussianDeficit(ct2a=ct2a_mom1d)
    actual = deficitModel._calc_deficit_centre(**kwargs)
    assert calls == {'fast': fast, 'tiny': tiny}
    deficit_centre, expected = _stock_deficit_centre(**kwargs)
    assert actual.dtype == expected.dtype
    assert np.broadcast_to(actual, expected.shape).tobytes() == expected.tobytes()
    if fast:
        assert (actual.shape, actual.strides) == (deficit_centre.shape, deficit_centre.strides)


def test_deficit_centre_kernel_inputs_cover_edges():
    with np.errstate(all='ignore'):
        numerator = base['ct_ilk'][:, na] * base['D_src_il'][:, na, :, na]**2
        ratio = numerator / (8. * base_sigma**2)
        deficit_centre, _ = _stock_deficit_centre(**base)
    assert (ratio > 1).any() and (ratio < 1).any() and np.isnan(ratio).any()
    assert (np.isfinite(base_sigma**2) & np.isinf(8. * base_sigma**2) & (numerator > 1e290)).any()
    assert np.isnan(deficit_centre).any() and np.isinf(deficit_centre).any() and (deficit_centre == 0).any()


@pytest.mark.filterwarnings('ignore::RuntimeWarning')
def test_deficit_centre_kernel_numpy32(monkeypatch):
    calls = _spy_kernel(monkeypatch)
    with Numpy32():
        actual = NiayifarGaussianDeficit(ct2a=ct2a_mom1d)._calc_deficit_centre(**base)
        expected, _ = _stock_deficit_centre(**base)
    assert calls == {'fast': 0, 'tiny': 0}
    assert actual.dtype == expected.dtype
    assert actual.tobytes() == expected.tobytes()


@pytest.mark.parametrize('deficitModel,fast', [
    (NiayifarGaussianDeficit(ct2a=ct2a_mom1d), 1),
    (BastankhahGaussianDeficit(ct2a=ct2a_mom1d), 1),
    (CarbajofuertesGaussianDeficit(ct2a=ct2a_mom1d), 1),
    (TurboGaussianDeficit(ct2a=ct2a_mom1d), 1),
    (NiayifarGaussianDeficit(), 0),
    (ZongGaussianDeficit(ct2a=ct2a_mom1d), 0)])
def test_deficit_centre_kernel_models(deficitModel, fast, monkeypatch):
    rng = np.random.default_rng(2)
    kwargs = dict(D_src_il=np.array([[80., 120., 100.]]), dw_ijlk=rng.uniform(1., 2000., (1, 37, 3, 1)),
                  ct_ilk=rng.uniform(0., 1., (1, 3, 5)), TI_ilk=np.full((1, 3, 1), .06),
                  TI_eff_ilk=rng.uniform(.06, .2, (1, 3, 5)), WS_ref_ijlk=rng.uniform(4., 25., (1, 1, 3, 5)))
    cw_ijlk = rng.uniform(-500., 500., (1, 37, 3, 1))
    calls = _spy_kernel(monkeypatch)
    actual = deficitModel._calc_deficit_centre(deficitModel.sigma_ijlk(**kwargs), cw_ijlk=cw_ijlk, **kwargs)
    assert calls == {'fast': fast, 'tiny': 0}
    expected = deficitModel.calc_deficit(cw_ijlk=cw_ijlk * 0, **kwargs)
    assert actual.tobytes() == expected.tobytes()


def _overlap_wfm(wfm_cls, deficit):
    if deficit == 'niayifar':
        deficitModel = NiayifarGaussianDeficit(ct2a=ct2a_mom1d, rotorAvgModel=GaussianOverlapAvgModel())
        return wfm_cls(Hornsrev1Site(), V80(), deficitModel, superpositionModel=LinearSum(),
                       turbulenceModel=CrespoHernandez(ct2a=ct2a_mom1d, c=[0.73, 0.8325, 0.0325, -0.32]))
    deficitModel = TurboGaussianDeficit(ct2a=ct2a_mom1d, ctlim=.96, groundModel=Mirror(superpositionModel=SquaredSum()),
                                        rotorAvgModel=GaussianOverlapAvgModel())
    deficitModel.WS_key = 'WS_jlk'
    return wfm_cls(Hornsrev1Site(), V80(), deficitModel, superpositionModel=SquaredSum())


@pytest.mark.parametrize('wfm_cls', [PropagateDownwind, All2AllIterative])
@pytest.mark.parametrize('deficit', ['niayifar', 'turbopark'])
def test_deficit_centre_kernel_wfm(wfm_cls, deficit, monkeypatch):
    wfm = _overlap_wfm(wfm_cls, deficit)
    x, y = wfm.site.initial_position.T

    def run():
        sim_res = wfm(x, y, wd=np.arange(0, 360, 30), ws=[6, 8])
        return sim_res, sim_res.flow_map(XYGrid(resolution=20), wd=270, ws=8)
    calls = _spy_kernel(monkeypatch)
    sim_res, flow_map = run()
    assert calls['fast'] > 0
    monkeypatch.setattr(BastankhahGaussianDeficit, '_calc_deficit_centre', _stock_calc_deficit_centre)
    ref, ref_flow_map = run()
    for k in ['WS_eff', 'TI_eff', 'CT', 'Power']:
        assert sim_res[k].values.tobytes() == ref[k].values.tobytes()
    assert flow_map.WS_eff.values.tobytes() == ref_flow_map.WS_eff.values.tobytes()


@pytest.mark.parametrize('gradient_method,fast', [(autograd, False), (cs, False), (fd, True)])
def test_deficit_centre_kernel_gradients(gradient_method, fast, monkeypatch):
    x, y = np.array([0., 400, 800]), np.array([0., 50, -30])
    wfm = _overlap_wfm(PropagateDownwind, 'niayifar')

    def get_gradients():
        return wfm.aep_gradients(gradient_method, ['x', 'y'], x=x, y=y, wd=[260, 270, 280], ws=[6, 10])
    calls = _spy_kernel(monkeypatch)
    gradients = get_gradients()
    assert (calls['fast'] > 0) == fast
    monkeypatch.setattr(BastankhahGaussianDeficit, '_calc_deficit_centre', _stock_calc_deficit_centre)
    if gradient_method is autograd:
        npt.assert_array_almost_equal(gradients, get_gradients(), 15)
    else:
        npt.assert_array_equal(gradients, get_gradients())
