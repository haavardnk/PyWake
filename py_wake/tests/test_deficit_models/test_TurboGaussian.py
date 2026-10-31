from numpy import newaxis as na
import pytest
from py_wake import np
from py_wake.deficit_models import gaussian
from py_wake.deficit_models.gaussian import TurboGaussianDeficit
from py_wake.examples.data.hornsrev1 import Hornsrev1Site, V80
from py_wake.flow_map import XYGrid
from py_wake.rotor_avg_models.gaussian_overlap_model import GaussianOverlapAvgModel
from py_wake.turbulence_models import CrespoHernandez
from py_wake.utils.gradients import cabs
from py_wake.utils.numpy_utils import Numpy32
from py_wake.wind_farm_models.engineering_models import PropagateDownwind, All2AllIterative


def _stock_sigma_ijlk(self, D_src_il, dw_ijlk, ct_ilk, **kwargs):
    TI_ref_ilk = kwargs[self.TI_key]
    c1, c2 = self.cTI
    alpha_ilk = c1 * TI_ref_ilk
    ct_ilk = np.maximum(ct_ilk, 1e-20)
    beta_ilk = c2 * TI_ref_ilk / np.sqrt(ct_ilk)

    fac_ilk = self.A * TI_ref_ilk * D_src_il[..., na] / beta_ilk
    term1_ijlk = np.sqrt((alpha_ilk[:, na] + beta_ilk[:, na] * dw_ijlk / D_src_il[:, na, :, na])**2 + 1)
    term2_ilk = np.sqrt(1 + alpha_ilk**2)
    term3_ijlk = (term1_ijlk + 1) * alpha_ilk[:, na]
    term4_ijlk = (term2_ilk[:, na] + 1) * (alpha_ilk[:, na] +
                                           beta_ilk[:, na] * cabs(dw_ijlk) / D_src_il[:, na, :, na])

    expansion_ijlk = fac_ilk[:, na] * (term1_ijlk - term2_ilk[:, na] - np.log(term3_ijlk / term4_ijlk))

    return expansion_ijlk + self.epsilon_ilk(ct_ilk)[:, na] * D_src_il[:, na, :, na]


def _compiled_and_stock(monkeypatch, func):
    build, calls = gaussian._turbo_kernels, {'fast': 0, 'nan': 0}

    def spy_kernels():
        expansion_terms, sigma = build()

        def spy_sigma(*args):
            nan = sigma(*args)
            calls['nan' if nan else 'fast'] += 1
            return nan
        return expansion_terms, spy_sigma
    monkeypatch.setattr(gaussian, '_turbo_kernels', spy_kernels)
    res = func()
    monkeypatch.setattr(TurboGaussianDeficit, 'sigma_ijlk', _stock_sigma_ijlk)
    return res, func(), calls


def _inputs(I, J, L, K, dw_shape, ct_shape, TI_shape, dw_values, ct_values, TI_values):
    rng = np.random.default_rng(0)
    return dict(D_src_il=np.resize([80., 120., 1e-3], (I, L)),
                dw_ijlk=np.resize(np.r_[dw_values, rng.uniform(-500, 2000, 64)], dw_shape),
                ct_ilk=np.resize(ct_values, ct_shape),
                TI_ilk=np.resize(TI_values, TI_shape))


finite = dict(dw_values=[-1e3, -1., -0., 0., 1e-300, 1., 80., 1e4, 1e100], ct_values=[0, 1e-30, .1, .8, 1., 1.2],
              TI_values=[.06, 1e-3, .1])
negative_TI = {**finite, 'TI_values': [-.06, -1e-3]}
nan = dict(dw_values=[-np.inf, -1., 0., 80., 1e300, np.inf, np.nan], ct_values=[0, .8, np.nan, 1.2],
           TI_values=[.06, 0., -np.nan, np.nan])
mirror = _inputs(1, 76, 3, 6, (1, 76, 3, 1), (1, 3, 6), (1, 3, 1), **finite)


class SubArray(np.ndarray):
    pass


@pytest.mark.parametrize('kwargs,fast,nan_calls', [
    pytest.param(mirror, 1, 0, id='mirror_shared'),
    pytest.param(_inputs(3, 76, 2, 6, (3, 76, 2, 6), (3, 2, 6), (3, 2, 6), **finite), 1, 0, id='all2all'),
    pytest.param(_inputs(2, 76, 1, 1, (2, 76, 1, 1), (2, 1, 1), (2, 1, 1), **finite), 1, 0, id='one_case'),
    pytest.param({**mirror, 'dw_ijlk': np.asfortranarray(np.broadcast_to(mirror['dw_ijlk'], (1, 76, 3, 6)))},
                 1, 0, id='fortran_dw'),
    pytest.param({**mirror, 'dw_ijlk': np.resize(mirror['dw_ijlk'], (1, 152, 3, 2))[:, ::2, :, ::2]},
                 1, 0, id='strided_dw'),
    pytest.param({**mirror, 'dw_ijlk': np.broadcast_to(mirror['dw_ijlk'], (1, 76, 3, 6))}, 1, 0, id='zero_stride_dw'),
    pytest.param({**mirror, 'dw_ijlk': mirror['dw_ijlk'][:, :0]}, 1, 0, id='empty'),
    pytest.param(_inputs(3, 76, 2, 6, (3, 76, 2, 6), (3, 2, 6), (3, 2, 6), **negative_TI), 1, 0, id='negative_TI'),
    pytest.param(dict(D_src_il=np.array([[1.0929646618417081e+292]]), dw_ijlk=np.array([[[[2.8740664042081332e+287]]]]),
                      ct_ilk=np.array([[[1.118158735479755]]]), TI_ilk=np.array([[[4.498365519782593e+20]]])),
                 1, 0, id='inf'),
    pytest.param(_inputs(1, 76, 3, 6, (1, 76, 3, 1), (1, 3, 6), (1, 3, 1), **nan), 0, 1, id='nan'),
    pytest.param(_inputs(3, 76, 2, 6, (3, 76, 2, 6), (3, 2, 6), (3, 2, 6), **nan), 0, 1, id='nan_all2all'),
    pytest.param({**mirror, 'TI_ilk': np.full((1, 3, 1), -np.nan), 'ct_ilk': np.full((1, 3, 6), np.nan)},
                 0, 1, id='negative_nan_TI'),
    pytest.param(_inputs(1, 76, 3, 6, (2, 1, 76, 3, 6), (1, 3, 6), (1, 3, 1), **finite), 0, 0, id='five_d'),
    pytest.param({**mirror, 'dw_ijlk': mirror['dw_ijlk'].view(SubArray)}, 0, 0, id='subclass_dw'),
    pytest.param({**mirror, 'dw_ijlk': mirror['dw_ijlk'] + 0j}, 0, 0, id='complex_dw'),
    pytest.param({**mirror, 'ct_ilk': mirror['ct_ilk'] + 1e-20j}, 0, 0, id='complex_ct'),
    pytest.param({**mirror, 'TI_ilk': np.full((1, 3, 1), .06, dtype=np.float32)}, 0, 0, id='float32_TI')])
@pytest.mark.filterwarnings('ignore::RuntimeWarning')
def test_sigma_ijlk_kernel(kwargs, fast, nan_calls, monkeypatch):
    sigma_ijlk, ref, calls = _compiled_and_stock(monkeypatch, lambda: TurboGaussianDeficit().sigma_ijlk(**kwargs))
    assert calls == {'fast': fast, 'nan': nan_calls}
    assert sigma_ijlk.dtype == ref.dtype
    assert sigma_ijlk.shape == ref.shape
    assert sigma_ijlk.tobytes() == ref.tobytes()


@pytest.mark.filterwarnings('ignore::RuntimeWarning')
def test_sigma_ijlk_kernel_numpy32(monkeypatch):
    with Numpy32():
        sigma_ijlk, ref, calls = _compiled_and_stock(monkeypatch, lambda: TurboGaussianDeficit().sigma_ijlk(**mirror))
    assert calls == {'fast': 0, 'nan': 0}
    assert sigma_ijlk.tobytes() == ref.tobytes()


@pytest.mark.parametrize('wfm_cls', [PropagateDownwind, All2AllIterative])
@pytest.mark.parametrize('use_effective_ti', [False, True])
def test_sigma_ijlk_kernel_wfm(wfm_cls, use_effective_ti, monkeypatch):
    site = Hornsrev1Site()
    x, y = site.initial_position.T
    wfm = wfm_cls(site, V80(), TurboGaussianDeficit(rotorAvgModel=GaussianOverlapAvgModel(),
                                                    use_effective_ti=use_effective_ti),
                  turbulenceModel=CrespoHernandez())

    def run():
        sim_res = wfm(x, y, wd=np.arange(0, 360, 30), ws=[6, 8])
        return sim_res, sim_res.flow_map(XYGrid(resolution=20), wd=270, ws=8)
    (sim_res, flow_map), (ref, ref_flow_map), calls = _compiled_and_stock(monkeypatch, run)
    assert calls['fast'] > 0
    for k in ['WS_eff', 'TI_eff', 'CT', 'Power']:
        assert sim_res[k].values.tobytes() == ref[k].values.tobytes()
    assert flow_map.WS_eff.values.tobytes() == ref_flow_map.WS_eff.values.tobytes()
