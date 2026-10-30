
import pickle

import matplotlib.pyplot as plt
import pytest
import xarray as xr

from py_wake import np
from py_wake.tests import npt
from py_wake.tests.check_speed import timeit
from py_wake.utils.numpy_utils import Numpy32
from py_wake.utils.grid_interpolator import (
    EqDistRegGrid2DInterpolator,
    GridInterpolator,
)


def test_grid_interpolator_not_match():
    with pytest.raises(ValueError, match="Lengths of x does not match shape of V"):
        GridInterpolator([[5, 10, 15], [200, 300]], np.arange(12).reshape(3, 4))


def test_grid_interpolator_wrong_method():
    with pytest.raises(AssertionError, match='method must be "linear" or "nearest"'):
        GridInterpolator([[10, 20, 30]], [1, 2, 3], method="cubic")(None)


def test_grid_interpolator_outside_area_bounds_check():
    with pytest.raises(ValueError, match='Point 0, dimension 0 with value 9.900000 is outside range 10.000000-30.000000'):
        GridInterpolator([[10, 20, 30]], [1, 2, 3])(9.9)

    with pytest.raises(ValueError, match='Point 0, dimension 0 with value 30.100000 is outside range 10.000000-30.000000'):
        GridInterpolator([[10, 20, 30]], [1, 2, 3])(30.1)

    with pytest.raises(ValueError, match='Point 1, dimension 0 with value 30.100000 is outside range 10.000000-30.000000'):
        GridInterpolator([[10, 20, 30]], [1, 2, 3])([[20], [30.1]])

    with pytest.raises(ValueError, match='Point 0, dimension 1 with value 300.100000 is outside range 200.000000-300.000000'):
        GridInterpolator([[5, 10, 15], [200, 300]], np.arange(6).reshape(3, 2))([(5, 300.1)])

    with pytest.raises(ValueError, match='Point 1, dimension 0 with value 4.800000 is outside range 5.000000-15.000000'):
        GridInterpolator([[5, 10, 15], [200, 300]], np.arange(6).reshape(3, 2))([(5, 199.9), (4.8, 200), (5, 301)])


def test_grid_interpolator_outside_area_bounds_limit():
    gi = GridInterpolator([[10, 20, 30]], [1, 2, 3])
    npt.assert_array_equal(gi([[10]]), gi([[9.9]], bounds='limit'))
    npt.assert_array_equal(gi([[30]]), gi([[30.1]], bounds='limit'))
    npt.assert_array_equal(gi([[20], [30]]), gi([[20], [30.1]], bounds='limit'))
    gi = GridInterpolator([[5, 10, 15], [200, 300]], np.arange(6).reshape(3, 2))
    npt.assert_array_equal(gi([(5, 300)]), gi([(5, 300.1)], bounds='limit'))
    npt.assert_array_equal(gi([(5, 200), (5, 200), (5, 300)]), gi([(5, 199.9), (4.8, 200), (5, 301)], bounds='limit'))


def test_grid_interpolator_2d():
    x = [5, 10, 15]
    y = [200, 300, 400, 500]
    v = np.arange(12).reshape(3, 4)
    eq = GridInterpolator([x, y], v)
    xp = [[i, 200] for i in np.arange(5, 16)]
    npt.assert_array_almost_equal(eq(xp), np.linspace(0, 8, 11))
    npt.assert_array_almost_equal(eq(xp, 'nearest'), np.round(np.linspace(0, 8, 11) / 4) * 4)

    X, Y = np.meshgrid(x, y)
    xp, yp = np.linspace(5, 15, 11), np.linspace(200, 500, 13)
    Xp, Yp = np.meshgrid(xp, yp)

    co = np.array([Xp, Yp]).T.reshape(-1, 2)
    for method in ['nearest', 'linear']:
        Z = eq(co, method).reshape(Xp.T.shape)

        plt.figure()
        c = plt.contourf(Xp, Yp, Z.T, np.arange(12))
        plt.plot(X, Y, 'xw',)
        plt.colorbar(c)
    if 0:
        plt.show()
    plt.close('all')


@pytest.mark.parametrize('method,ref', [
    ('linear', [15.5, 40.5, 42.5, 47.5, 48]),
    (['linear', 'linear', 'linear'], [15.5, 40.5, 42.5, 47.5, 48]),
    ('nearest', [26, 47, 49, 54, 54]),
    (['nearest', 'nearest', 'nearest'], [26, 47, 49, 54, 54]),
    ('linear', [15.5, 40.5, 42.5, 47.5, 48]),
    (['linear', 'linear', 'nearest'], [16., 41., 43., 48., 48.]),
    (['nearest', 'linear', 'nearest'], [24., 49., 51., 56., 56.]),

])
def test_grid_interpolator_3d_methods(method, ref):

    x = [5, 10, 15]
    y = [200, 300, 400, 500]
    z = [1000, 1002, 1004, 1006, 1008]
    v = np.arange(3 * 4 * 5).reshape(3, 4, 5)
    eq = GridInterpolator([x, y, z], v)
    # print(v)
    xp = [[8, 260, 1001],
          [13, 340, 1003],
          [13, 340, 1007],
          [13, 440, 1007],
          [13, 440, 1008],
          ]
    npt.assert_array_almost_equal(eq(xp, method=method), ref, 10)


def test_grid_interpolator_2d_plus_1d():
    eq = GridInterpolator([[5, 10, 15], [200, 300, 400, 500]], np.arange(12).repeat(2).reshape(3, 4, 2))
    x = [[i, 200] for i in np.arange(5, 16)]
    npt.assert_array_almost_equal(eq(x).sum(1), np.linspace(0, 8, 11) * 2)
    npt.assert_array_equal(eq(x).shape, (11, 2))


def test_grid_interpolator_2d_plus_2d():
    eq = GridInterpolator([[5, 10, 15], [200, 300, 400, 500]], np.arange(12).repeat(10).reshape(3, 4, 2, 5))
    x = [[i, 200] for i in np.arange(5, 16)]
    npt.assert_array_almost_equal(eq(x).sum((1, 2)), np.linspace(0, 8, 11) * 10)
    npt.assert_array_equal(eq(x).shape, (11, 2, 5))


def test_grid_interpolator_irregular():

    x = [6, 10, 14]
    y = [200, 300, 400, 500]
    z = [80, 90, 100]
    v = np.arange(36).reshape(3, 4, 3) / 3

    eq = GridInterpolator([x, y, z], v)

    xp = [[i, 200, 80] for i in np.linspace(6, 14, 9)]
    npt.assert_array_almost_equal(eq(xp), np.linspace(0, 8, 9))

    x = [6, 12, 14]
    y = [200, 300, 400, 500]
    z = [80, 90, 110]
    v[1, 0, 0] = 6
    eq = GridInterpolator([x, y, z], v)

    xp = [[i, 200, 80] for i in np.linspace(6, 14, 9)]
    npt.assert_array_almost_equal(eq(xp), np.linspace(0, 8, 9))

    xp = [[i, 200, 80] for i in np.linspace(6, 14, 10)]
    npt.assert_array_almost_equal(eq(xp), np.linspace(0, 8, 10))
    npt.assert_array_almost_equal(eq(xp, 'nearest'), [0., 0., 0., 0., 6., 6., 6., 6., 8., 8.])


def test_grid_interpolator_init_twice():
    x = [np.array([0.0, 2.99999, 3.0, 3.5, 4.0, 4.5]),
         np.array([0.84, 0.85, 0.86, 0.87, 0.88, ])
         ]
    V = np.reshape(np.array(range(30)), (6, 5))
    GridInterpolator(x, V)
    GridInterpolator(x, V)


def test_EqDistRegGrid2DInterpolator_non_eqdist():
    x = [6, 12, 18]
    y = [200, 300, 400, 500]
    v = np.arange(12).reshape(3, 4)
    with pytest.raises(AssertionError, match="x is not equidistant"):
        EqDistRegGrid2DInterpolator([6, 12, 15], y, v)
    with pytest.raises(AssertionError, match="y is not equidistant"):
        EqDistRegGrid2DInterpolator(x, [200, 300, 400, 501], v)


def test_EqDistRegGrid2DInterpolator_bound():
    x = [6, 12, 18]
    y = [200, 300, 400, 500]
    v = np.arange(12).reshape(3, 4)
    interp = EqDistRegGrid2DInterpolator(x, y, v)
    npt.assert_array_equal(interp(np.array([6., 18]), np.array([200, 500])), [0, 11])


def compare_speed():
    x = np.arange(0, 100)
    y = np.arange(10, 100)
    z = np.arange(30, 33)
    wd = np.arange(360)
    ws = np.arange(3, 26)
    V = np.random.rand(len(x), len(y), len(z), len(wd), len(ws))
    for x in [x, np.r_[x[:-1], 100]]:
        da = xr.DataArray(V, coords=[('x', x), ('y', y), ('z', z), ('wd', wd), ('ws', ws)])
        coords = xr.Dataset(coords={'wd': np.arange(360), 'ws': [9, 10], 'i': np.arange(16),
                                    'x': ('i', np.linspace(0, 9, 16)),
                                    'y': ('i', np.linspace(10, 20, 16)),
                                    'z': ('i', np.linspace(30, 32, 16))})

        x1, t1 = timeit(da.sel_interp_all, verbose=True)(coords)

        I, L, K = len(coords.x), len(coords.wd), len(coords.ws)

        def interp():
            c = np.array([coords.x.data.repeat(L * K), coords.y.data.repeat(L * K), coords.z.data.repeat(L * K),
                          np.tile(coords.wd.data.repeat(K), I), np.tile(coords.ws.data, I * L)]).T
            return GridInterpolator([x, y, z, wd, ws], da.values)(c)
        x2, t2 = timeit(interp, verbose=True, line_profile=0, profile_funcs=[GridInterpolator.__call__])()
        npt.assert_array_almost_equal(x1, x2.reshape(16, len(coords.wd), len(coords.ws)))
        # print(np.mean(t1) / np.mean(t2))


def stock_interpolation(interpolator, xp, monkeypatch, **kwargs):
    with monkeypatch.context() as m:
        m.setattr(GridInterpolator, '_compiled_limit', lambda *args: None)
        return interpolator(xp, **kwargs)


def random_axis(rng, n, irregular):
    if irregular:
        return np.cumsum(rng.uniform(.1, 3, n)) - 2
    return np.arange(n) * rng.choice([.01, .5, 20.]) + rng.uniform(-5, 5)


def random_table(rng, shape):
    order = rng.permutation(len(shape))
    V = rng.choice([1., -1., 1e16, 3.3, -0., 1e-300], size=tuple(np.array(shape)[order]))
    return (V * rng.uniform(.5, 2, V.shape)).transpose(np.argsort(order))


def random_points(rng, x, n_points):
    columns = []
    for xi in x:
        span = xi[-1] - xi[0]
        column = rng.uniform(xi[0] - span * .2, xi[-1] + span * .2, n_points)
        column = np.where(rng.random(n_points) < .2, rng.choice(xi, n_points), column)
        column = np.where(rng.random(n_points) < .1, rng.choice([np.inf, -np.inf, -0.], n_points), column)
        columns.append(column)
    return np.array(columns).T


@pytest.mark.parametrize('seed', range(4))
def test_grid_interpolator_compiled_limit_matches_stock(seed, monkeypatch):
    rng = np.random.default_rng(seed)
    for _ in range(200):
        n_dims = rng.integers(1, 5)
        x = [random_axis(rng, rng.integers(2, 9), rng.random() < .3) for _ in range(n_dims)]
        method = list(rng.choice(['linear', 'nearest'], n_dims))
        interpolator = GridInterpolator(x, random_table(rng, [len(xi) for xi in x]), method=method, bounds='limit')
        xp = random_points(rng, x, rng.choice([1, 3, 50]))
        assert interpolator._compiled_limit(xp, [m == 'linear' for m in method]) is not None
        assert interpolator(xp).tobytes() == stock_interpolation(interpolator, xp, monkeypatch).tobytes()


@pytest.mark.parametrize('x,V,xp', [
    ([np.array([0., 1, 1, 2])], np.array([0., 1, 2, 3]), np.array([[1.], [.5], [1.5]])),
    ([np.array([0., 1, 1, 2])], np.array([0., 1, 2, 3]), np.array([[1.]])),
    ([np.array([0., 1, 3])] * 7, np.random.default_rng(0).random((3,) * 7), np.full((1, 7), 1.3)),
    ([np.array([0., 1, 3])] * 7, np.random.default_rng(0).random((3,) * 7), np.full((2, 7), 1.3)),
])
def test_grid_interpolator_compiled_limit_cases(x, V, xp, monkeypatch):
    interpolator = GridInterpolator(x, V, bounds='limit')
    assert interpolator._compiled_limit(xp, [True] * len(x)) is not None
    assert interpolator(xp).tobytes() == stock_interpolation(interpolator, xp, monkeypatch).tobytes()


def test_grid_interpolator_compiled_limit_negative_zero(monkeypatch):
    interpolator = GridInterpolator([np.arange(3.)] * 4, -np.zeros((3, 3, 3, 3)), bounds='limit')
    for xp in [np.full((1, 4), .5), np.full((3, 4), .5)]:
        assert interpolator(xp).tobytes() == stock_interpolation(interpolator, xp, monkeypatch).tobytes()


def test_grid_interpolator_compiled_limit_nan(monkeypatch):
    interpolator = GridInterpolator([np.arange(3.), np.array([0., 1, 3])], np.arange(9.).reshape(3, 3), bounds='limit')
    xp = np.array([[np.nan, 1.], [1., np.nan], [.5, 2.]])
    res = interpolator(xp)
    assert np.isnan(res[:2]).all()
    assert res[2:].tobytes() == stock_interpolation(interpolator, xp[2:], monkeypatch).tobytes()


axis3, axis4 = np.array([0., 1, 2]), np.array([100., 200, 300, 400])


@pytest.mark.parametrize('x,V,xp,kwargs', [
    ([axis3, axis4], np.arange(12.).reshape(3, 4), np.array([[1., 250.]]) * 1j, {}),
    ([axis3, axis4], np.arange(12.).reshape(3, 4), np.array([[1., 250.]], dtype=np.float32), {}),
    ([axis3, axis4], np.arange(12.).reshape(3, 4), np.array([[1., 250.]]), {'bounds': 'check'}),
    ([axis3, axis4], np.arange(12.).reshape(3, 4), np.array([[1., 250.]]), {'deg': True}),
    ([axis3, axis4], np.arange(12).reshape(3, 4), np.array([[1., 250.]]), {}),
    ([axis3, axis4], np.arange(12.).reshape(3, 4)[:, ::-1], np.array([[1., 250.]]), {}),
    ([axis3, axis4], np.arange(24.).reshape(3, 4, 2), np.array([[1., 250.]]), {}),
    ([axis3, [100, 200, 300, 400]], np.arange(12.).reshape(3, 4), np.array([[1., 250.]]), {}),
    ([axis3, np.array([0.])], np.arange(3.).reshape(3, 1), np.array([[1., 0.]]), {}),
    ([axis3, np.array([0., 3, 1, 4])], np.arange(12.).reshape(3, 4), np.array([[1., 2.5]]), {}),
    ([axis3, np.array([0., np.nan, 1, 4])], np.arange(12.).reshape(3, 4), np.array([[1., 2.5]]), {}),
    ([np.arange(2.)] * 8, np.arange(256.).reshape((2,) * 8), np.full((1, 8), .3), {}),
    ([axis3, axis4], np.arange(12.).reshape(3, 4), np.array([[[1., 250.]]]), {}),
])
def test_grid_interpolator_compiled_limit_fallback(x, V, xp, kwargs, monkeypatch):
    interpolator = GridInterpolator(x, V, bounds='limit')
    calls = []
    compiled_limit = GridInterpolator._compiled_limit
    monkeypatch.setattr(GridInterpolator, '_compiled_limit',
                        lambda self, *args: calls.append(compiled_limit(self, *args)) or calls[-1])
    res = interpolator(xp, **kwargs)
    assert len(calls) <= 1 and all(c is None for c in calls)
    npt.assert_array_equal(res, stock_interpolation(interpolator, xp, monkeypatch, **kwargs))


def test_grid_interpolator_compiled_limit_numpy32():
    interpolator = GridInterpolator([axis3, axis4], np.arange(12.).reshape(3, 4), bounds='limit')
    xp = np.array([[1., 250.]])
    assert interpolator._compiled_limit(xp, [True, True]) is not None
    with Numpy32():
        assert interpolator._compiled_limit(xp, [True, True]) is None


def test_grid_interpolator_compiled_limit_pickle(monkeypatch):
    V = np.arange(24.).reshape(4, 3, 2).transpose(1, 2, 0)
    interpolator = GridInterpolator([axis3, np.array([0., 1]), axis4], V, bounds='limit')
    xp = np.array([[.5, .3, 220.], [2., 1., 400.]])
    res = interpolator(xp)
    assert '_compiled_layout' in interpolator.__dict__
    state = interpolator.__getstate__()
    assert '_compiled_layout' not in state
    unpickled = pickle.loads(pickle.dumps(interpolator))
    assert unpickled(xp).tobytes() == res.tobytes() == stock_interpolation(unpickled, xp, monkeypatch).tobytes()
    assert np.shares_memory(unpickled._compiled_layout.V, unpickled.V)


if __name__ == '__main__':
    compare_speed()
