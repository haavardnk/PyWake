import pytest
from numpy import newaxis as na

from py_wake import np
from py_wake.deficit_models.gaussian import NiayifarGaussianDeficit
from py_wake.deficit_models.utils import ct2a_mom1d
from py_wake.examples.data.hornsrev1 import Hornsrev1Site, V80
from py_wake.flow_map import XYGrid
from py_wake.ground_models.ground_models import Mirror
from py_wake.rotor_avg_models.gaussian_overlap_model import GaussianOverlapAvgModel
from py_wake.superposition_models import LinearSum
from py_wake.tests import npt
from py_wake.turbulence_models import crespo
from py_wake.turbulence_models.crespo import CrespoHernandez
from py_wake.utils.gradients import autograd, cs, fd
from py_wake.utils.numpy_utils import Numpy32
from py_wake.wind_farm_models.engineering_models import PropagateDownwind, All2AllIterative


def _stock_calc_added_turbulence(self, dw_ijlk, cw_ijlk, D_src_il, ct_ilk, TI_ilk, D_dst_ijl, wake_radius_ijlk, **_):
    a_ilk = self.ct2a(ct_ilk)
    a_ilk = np.maximum(a_ilk, 1e-10)
    dw_ijlk_gt0 = np.maximum(dw_ijlk, 1e-10)
    TI_add_ijlk = self.c[0] * a_ilk[:, na, :, :]**self.c[1] * TI_ilk[:, na, :, :]**self.c[2] * \
        (dw_ijlk_gt0 / D_src_il[:, na, :, na])**self.c[3]
    return TI_add_ijlk * (cw_ijlk < wake_radius_ijlk) * (dw_ijlk > 0)


def _spy_kernel(monkeypatch):
    build, outputs = crespo._added_turbulence_kernel, []

    def spy_kernel():
        kernel = build()

        def spy(*args):
            outputs.append(args[-1])
            return kernel(*args)
        return spy
    monkeypatch.setattr(crespo, '_added_turbulence_kernel', spy_kernel)
    return outputs


def _inputs(I, J, L, K, radius_k=None):
    rng = np.random.default_rng([I, J, L, K])
    dw_ijlk = rng.uniform(-500., 3000., (I, J, L, 1))
    dw_ijlk.reshape(-1)[:9] = [-0., 0., 5e-324, -5e-324, 1e-10, 1e308, np.inf, -np.inf, 1e-300][:dw_ijlk.size]
    cw_ijlk = rng.uniform(-400., 400., (I, J, L, 1))
    cw_ijlk.reshape(-1)[:4] = [np.nan, np.inf, -0., 100.][:cw_ijlk.size]
    wake_radius_ijlk = rng.uniform(0., 300., (I, J, L, K if radius_k is None else radius_k))
    wake_radius_ijlk.reshape(-1)[:4] = [np.nan, np.inf, -0., 100.][:wake_radius_ijlk.size]
    D_src_il = rng.choice([80., 120.], (I, L))
    D_src_il[0, -1] = np.inf if L > 1 else 80.
    TI_ilk = rng.uniform(.02, .2, (I, L, K))
    TI_ilk[0, 0, :2] = [0., 1e308][:K]
    return dict(dw_ijlk=dw_ijlk, cw_ijlk=cw_ijlk, D_src_il=D_src_il, ct_ilk=rng.uniform(0., .99, (I, L, K)), TI_ilk=TI_ilk,
                D_dst_ijl=np.full((I, J, L), 80.), wake_radius_ijlk=wake_radius_ijlk)


def _with(kwargs, **values):
    return {**kwargs, **values}


def _set(x, index, value):
    x = x.copy()
    x[index] = value
    return x


class SubArray(np.ndarray):
    pass


base = _inputs(1, 37, 3, 5)


with np.errstate(all='ignore'):
    kernel_cases = [
        pytest.param(base, True, id='niayifar'),
        pytest.param(_inputs(3, 37, 2, 5), True, id='all2all'),
        pytest.param(_inputs(2, 7, 1, 1), True, id='one_case'),
        pytest.param(_inputs(1, 37, 3, 5, radius_k=1), True, id='radius_without_k'),
        pytest.param(_with(base, cw_ijlk=base['cw_ijlk'] * 0.), True, id='area_overlap_cw'),
        pytest.param(_with(base, cw_ijlk=base['wake_radius_ijlk'][..., :1].copy()), True, id='cw_on_radius'),
        pytest.param(_with(base, **{k: base[k][:, :0] for k in ['dw_ijlk', 'cw_ijlk', 'wake_radius_ijlk', 'D_dst_ijl']}), True,
                     id='empty'),
        pytest.param(_with(base, TI_ilk=base['TI_ilk'][..., :1].copy()), True, id='TI_without_k'),
        pytest.param(_with(base, wake_radius_ijlk=np.broadcast_to(base['wake_radius_ijlk'], (1, 37, 3, 5))), True,
                     id='readonly_radius'),
        pytest.param(_with(base, TI_ilk=_set(base['TI_ilk'], (0, 2, 4), np.nan)), False, id='nan_TI'),
        pytest.param(_with(base, TI_ilk=_set(base['TI_ilk'], (0, 2, 4), -.1)), False, id='negative_TI'),
        pytest.param(_with(base, ct_ilk=_set(base['ct_ilk'], (0, 2, 4), np.nan)), False, id='nan_ct'),
        pytest.param(_with(base, ct_ilk=_set(base['ct_ilk'], (0, 2, 4), 1.2)), True, id='ct_above_one'),
        pytest.param(_with(base, dw_ijlk=_set(base['dw_ijlk'], (0, 20, 1, 0), -np.nan)), False, id='nan_dw'),
        pytest.param(_with(base, D_src_il=_set(base['D_src_il'], (0, 1), np.nan)), False, id='nan_D'),
        pytest.param(_with(base, TI_ilk=_set(base['TI_ilk'], (0, 2, 4), np.nan),
                           dw_ijlk=_set(base['dw_ijlk'], (0, 20, 2, 0), -np.nan)), False, id='nan_TI_and_dw'),
        pytest.param(_with(base, dw_ijlk=base['dw_ijlk'] + 0j), False, id='complex_dw'),
        pytest.param(_with(base, cw_ijlk=base['cw_ijlk'] + 0j), False, id='complex_cw'),
        pytest.param(_with(base, TI_ilk=base['TI_ilk'] + 0j), False, id='complex_TI'),
        pytest.param(_with(base, dw_ijlk=base['dw_ijlk'].astype(np.float32)), False, id='float32_dw'),
        pytest.param(_with(base, cw_ijlk=base['cw_ijlk'].astype(np.float32)), False, id='float32_cw'),
        pytest.param(_with(base, wake_radius_ijlk=base['wake_radius_ijlk'].astype(np.float32)), False, id='float32_radius'),
        pytest.param(_with(base, TI_ilk=base['TI_ilk'].astype(np.float32)), True, id='float32_TI'),
        pytest.param(_with(base, wake_radius_ijlk=base['wake_radius_ijlk'].astype('>f8')), False, id='big_endian_radius'),
        pytest.param(_with(base, cw_ijlk=base['cw_ijlk'].view(SubArray)), False, id='subclass_cw'),
        pytest.param(_with(base, wake_radius_ijlk=base['wake_radius_ijlk'].view(SubArray)), False, id='subclass_radius'),
        pytest.param(_with(base, wake_radius_ijlk=np.asfortranarray(base['wake_radius_ijlk'])), False, id='fortran_radius'),
        pytest.param(_with(base, cw_ijlk=np.asfortranarray(base['cw_ijlk'])), False, id='fortran_cw'),
        pytest.param(_with(base, dw_ijlk=np.asfortranarray(base['dw_ijlk'])), False, id='fortran_dw'),
        pytest.param(_with(base, TI_ilk=np.asfortranarray(base['TI_ilk']), ct_ilk=np.asfortranarray(base['ct_ilk'])), False,
                     id='fortran_TI_ct'),
        pytest.param(_with(base, dw_ijlk=base['dw_ijlk'] * np.ones(5)), False, id='dw_with_k'),
        pytest.param(_with(base, cw_ijlk=base['cw_ijlk'] * np.ones(5)), False, id='cw_with_k'),
        pytest.param(_with(base, wake_radius_ijlk=base['wake_radius_ijlk'][na]), False, id='five_d_radius'),
        pytest.param(_with(base, cw_ijlk=base['cw_ijlk'][..., 0]), False, id='three_d_cw')]


@pytest.mark.parametrize('kwargs,fast', kernel_cases)
@pytest.mark.filterwarnings('ignore::RuntimeWarning')
def test_added_turbulence_kernel(kwargs, fast, monkeypatch):
    outputs = _spy_kernel(monkeypatch)
    model = CrespoHernandez(ct2a=ct2a_mom1d, c=[0.73, 0.8325, 0.0325, -0.32])
    try:
        expected = _stock_calc_added_turbulence(model, **kwargs)
    except (ValueError, IndexError) as e:
        with pytest.raises(type(e)):
            model.calc_added_turbulence(**kwargs)
        assert outputs == []
        return
    actual = model.calc_added_turbulence(**kwargs)
    assert len(outputs) == fast
    assert not fast or actual is outputs[0]
    assert type(actual) is type(expected)
    assert (actual.dtype, actual.shape, actual.strides) == (expected.dtype, expected.shape, expected.strides)
    assert actual.tobytes() == expected.tobytes()


def test_added_turbulence_kernel_inputs_cover_edges():
    model = CrespoHernandez(ct2a=ct2a_mom1d, c=[0.73, 0.8325, 0.0325, -0.32])
    with np.errstate(all='ignore'):
        TI_add = _stock_calc_added_turbulence(model, **base)
    assert np.isnan(TI_add).any() and np.isinf(TI_add).any() and (TI_add == 0).any() and (TI_add > 0).any()


overflow = _with(base, dw_ijlk=_set(base['dw_ijlk'], (0, 30, 1, 0), 1e-6), cw_ijlk=_set(base['cw_ijlk'], (0, 30, 1, 0), 1e3))


@pytest.mark.parametrize('c0,kwargs', [(-0.73, base), (1e308, overflow)])
@pytest.mark.filterwarnings('ignore::RuntimeWarning')
def test_added_turbulence_kernel_coefficients(c0, kwargs, monkeypatch):
    outputs = _spy_kernel(monkeypatch)
    model = CrespoHernandez(ct2a=ct2a_mom1d, c=[c0, 0.8325, 0.0325, -0.32])
    actual = model.calc_added_turbulence(**kwargs)
    expected = _stock_calc_added_turbulence(model, **kwargs)
    assert len(outputs) == 1 and actual is outputs[0]
    if c0 < 0:
        assert (np.signbit(expected) & (expected == 0)).any()
    else:
        assert np.isnan(expected[0, 30, 1]).any()
    assert actual.tobytes() == expected.tobytes()


@pytest.mark.parametrize('kwargs,fast', [(base, False), (_with(base, D_src_il=np.full((1, 3), 80.)), True),
                                         (_with(base, TI_ilk=base['TI_ilk'].astype(np.float32)), False)])
@pytest.mark.filterwarnings('ignore::RuntimeWarning')
def test_added_turbulence_kernel_numpy32(kwargs, fast, monkeypatch):
    outputs = _spy_kernel(monkeypatch)
    model = CrespoHernandez()
    with Numpy32():
        actual = model.calc_added_turbulence(**kwargs)
        expected = _stock_calc_added_turbulence(model, **kwargs)
    assert len(outputs) == fast
    assert (actual.dtype, actual.strides) == (expected.dtype, expected.strides)
    assert actual.tobytes() == expected.tobytes()


@pytest.mark.parametrize('c', [[np.array([.73, .5, .9])[na, :, na, na], 0.8325, 0.0325, -0.32],
                               [0.73, np.array([.8, .9, 1.])[na, :, na, na], 0.0325, -0.32],
                               [0.73, 0.8325, 0.0325, np.array([-.3, -.4, -.2])[na, na, :, na]]])
@pytest.mark.filterwarnings('ignore::RuntimeWarning')
def test_added_turbulence_kernel_array_coefficients(c, monkeypatch):
    kwargs = _inputs(1, 3, 3, 5)
    outputs = _spy_kernel(monkeypatch)
    model = CrespoHernandez(ct2a=ct2a_mom1d, c=c)
    actual = model.calc_added_turbulence(**kwargs)
    expected = _stock_calc_added_turbulence(model, **kwargs)
    assert len(outputs) == (np.ndim(c[3]) == 4)
    assert (actual.dtype, actual.shape, actual.strides) == (expected.dtype, expected.shape, expected.strides)
    assert actual.tobytes() == expected.tobytes()


def _wfm(wfm_cls, turbulenceModel):
    deficitModel = NiayifarGaussianDeficit(ct2a=ct2a_mom1d, rotorAvgModel=GaussianOverlapAvgModel())
    return wfm_cls(Hornsrev1Site(), V80(), deficitModel, superpositionModel=LinearSum(), turbulenceModel=turbulenceModel)


@pytest.mark.parametrize('wfm_cls', [PropagateDownwind, All2AllIterative])
@pytest.mark.parametrize('turbulenceModel', [CrespoHernandez(ct2a=ct2a_mom1d, c=[0.73, 0.8325, 0.0325, -0.32]),
                                             CrespoHernandez(rotorAvgModel=None),
                                             CrespoHernandez(groundModel=Mirror())])
def test_added_turbulence_kernel_wfm(wfm_cls, turbulenceModel, monkeypatch):
    wfm = _wfm(wfm_cls, turbulenceModel)
    x, y = wfm.site.initial_position.T

    def run():
        sim_res = wfm(x, y, wd=np.arange(0, 360, 30), ws=[6, 8])
        return sim_res, sim_res.flow_map(XYGrid(resolution=20), wd=270, ws=8)
    outputs = _spy_kernel(monkeypatch)
    sim_res, flow_map = run()
    assert len(outputs) > 0
    monkeypatch.setattr(CrespoHernandez, 'calc_added_turbulence', _stock_calc_added_turbulence)
    ref, ref_flow_map = run()
    for k in ['WS_eff', 'TI_eff', 'CT', 'Power']:
        assert sim_res[k].values.tobytes() == ref[k].values.tobytes()
    for k in ['WS_eff', 'TI_eff']:
        assert flow_map[k].values.tobytes() == ref_flow_map[k].values.tobytes()


@pytest.mark.parametrize('gradient_method,fast', [(autograd, False), (cs, False), (fd, True)])
def test_added_turbulence_kernel_gradients(gradient_method, fast, monkeypatch):
    x, y = np.array([0., 400, 800]), np.array([0., 50, -30])
    wfm = _wfm(PropagateDownwind, CrespoHernandez(ct2a=ct2a_mom1d, c=[0.73, 0.8325, 0.0325, -0.32]))

    def get_gradients():
        return wfm.aep_gradients(gradient_method, ['x', 'y'], x=x, y=y, wd=[260, 270, 280], ws=[6, 10])
    outputs = _spy_kernel(monkeypatch)
    gradients = get_gradients()
    assert (len(outputs) > 0) == fast
    monkeypatch.setattr(CrespoHernandez, 'calc_added_turbulence', _stock_calc_added_turbulence)
    npt.assert_array_equal(gradients, get_gradients())
