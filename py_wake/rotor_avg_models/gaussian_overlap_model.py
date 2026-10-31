import functools
import numba
import numpy
import os
from py_wake.rotor_avg_models.rotor_avg_model import RotorAvgModel
from py_wake import np
from numpy import newaxis as na
import xarray as xr
from tqdm import tqdm
from py_wake.utils.gradients import trapz
from py_wake.utils.grid_interpolator import GridInterpolator, _grid_position
from scipy.interpolate import RectBivariateSpline
from py_wake.utils import gradients


@numba.njit(error_model='numpy', inline='always')
def _axis_corners(value, axes, start, x0, dx, n, stride):
    position = _grid_position(value, axes, start, x0, dx, n, False, True)
    lower = 0 if numpy.isnan(position) else int(position)
    weight = position - lower
    return 1 - weight, weight, lower * stride, min(lower + 1, n - 1) * stride


def _overlap_factor_loop(D_jl, cw_ijlk, sigma_jlk, V, strides, axes, x0, dx, n, overlap_ijlk):
    I, J, L, K = overlap_ijlk.shape
    for j in range(J):
        D_l, cw_ilk, sigma_lk, overlap_ilk = D_jl[j], cw_ijlk[:, j], sigma_jlk[j], overlap_ijlk[:, j]
        for l in range(L):
            radius = D_l[l] / 2
            for k in range(K):
                sigma = sigma_lk[l, k]
                r0, r1, r_index0, r_index1 = _axis_corners(radius / sigma, axes, 0, x0[0], dx[0], n[0], strides[0])
                for i in range(I):
                    c0, c1, c_index0, c_index1 = _axis_corners(abs(cw_ilk[i, l, k] / sigma), axes, n[0], x0[1], dx[1],
                                                               n[1], strides[1])
                    overlap = 0. + r0 * c0 * V[r_index0 + c_index0] + r0 * c1 * V[r_index0 + c_index1] + \
                        r1 * c0 * V[r_index1 + c_index0] + r1 * c1 * V[r_index1 + c_index1]
                    if overlap != overlap:
                        return True
                    overlap_ilk[i, l, k] = overlap
    return False


@functools.lru_cache
def _overlap_factor_kernel():
    def readonly(ndim, layout='A'):
        return numba.types.Array(numba.float64, ndim, layout, readonly=True)
    ints, floats = numba.types.UniTuple(numba.int64, 2), numba.types.UniTuple(numba.float64, 2)
    signature = numba.boolean(readonly(2), readonly(4), readonly(3), readonly(1, 'C'), ints, readonly(1, 'C'), floats,
                              floats, ints, numba.float64[:, :, :, ::1])
    return numba.njit(signature, error_model='numpy')(_overlap_factor_loop)


class GaussianOverlapAvgModel(RotorAvgModel):
    def __init__(self, filename=os.path.dirname(__file__) + f'/gaussian_overlap_.02_.02_128_512.nc'):
        self.filename = filename

    @property
    def overlap_interpolator(self):
        if not hasattr(self, '_overlap_interpolator'):
            table = xr.load_dataarray(self.filename, engine='h5netcdf')
            R_sigma = np.arange(0, 20.001, 0.01)
            CW_sigma = np.arange(0, 10.01, 0.01)
            # behavior changed in xarray=24.11.0 such that
            # dat = table.interp(R_sigma=R_sigma, CW_sigma=CW_sigma, method='cubic')
            # performs a two-step interpolation which is slower and less accurate than
            # the previous one-step interpolation
            dat = RectBivariateSpline(table.R_sigma.values, table.CW_sigma.values, table.values)(R_sigma, CW_sigma)
            self._overlap_interpolator = GridInterpolator([R_sigma, CW_sigma], dat, bounds='limit')
        return self._overlap_interpolator

    def _calc_layout_terms(self, func, cw_ijlk, **kwargs):
        func(cw_ijlk=cw_ijlk * 0, **kwargs)

    def __call__(self, func, cw_ijlk, D_dst_ijl, **kwargs):
        deficit_model = func.__self__
        if not hasattr(deficit_model, 'sigma_ijlk'):
            raise AttributeError(
                f"'{deficit_model.__class__.__name__}' has no attribute 'sigma_ijlk', which is needed by the GaussianOverlapAvgModel")
        sigma_ijlk = deficit_model.sigma_ijlk(**kwargs)
        if hasattr(deficit_model, '_calc_deficit_centre') and func == deficit_model.calc_deficit:
            res_ijlk = deficit_model._calc_deficit_centre(sigma_ijlk, cw_ijlk=cw_ijlk, D_dst_ijl=D_dst_ijl, **kwargs)
        else:
            res_ijlk = func(cw_ijlk=cw_ijlk * 0, D_dst_ijl=D_dst_ijl, **kwargs)
        overlap_factor_ijlk = self._compiled_overlap_factor(D_dst_ijl, cw_ijlk, sigma_ijlk)
        if overlap_factor_ijlk is None:
            r_sigma_ijlk = (D_dst_ijl / 2)[..., na] / sigma_ijlk
            cw_sigma_ijlk = gradients.cabs(cw_ijlk / sigma_ijlk)
            shape = np.broadcast_shapes(r_sigma_ijlk.shape, cw_sigma_ijlk.shape)
            xp = np.stack([np.broadcast_to(r_sigma_ijlk, shape), np.broadcast_to(cw_sigma_ijlk, shape)]).reshape(2, -1).T
            overlap_factor_ijlk = self.overlap_interpolator(xp).reshape(shape)
        return res_ijlk * overlap_factor_ijlk

    def _compiled_overlap_factor(self, D_dst_ijl, cw_ijlk, sigma_ijlk):
        layout = self.overlap_interpolator._linear_limit_layout()
        arrays = D_dst_ijl, cw_ijlk, sigma_ijlk
        if np.backend is not numpy or layout is None or \
                any(type(x) is not numpy.ndarray or x.dtype != numpy.float64 for x in arrays) or \
                [x.ndim for x in arrays] != [3, 4, 4] or len(D_dst_ijl) != 1 or len(sigma_ijlk) != 1:
            return None
        shape = numpy.broadcast_shapes(D_dst_ijl.shape + (1,), cw_ijlk.shape, sigma_ijlk.shape)
        if shape[0] == shape[3] == 1:
            return None
        D_jl, sigma_jlk = numpy.broadcast_to(D_dst_ijl[0], shape[1:3]), numpy.broadcast_to(sigma_ijlk[0], shape[1:])
        cw_ijlk, overlap_ijlk = numpy.broadcast_to(cw_ijlk, shape), numpy.empty(shape)
        # NaN sign bits depend on operand order, which LLVM may commute
        if _overlap_factor_kernel()(D_jl, cw_ijlk, sigma_jlk, layout.V, layout.strides, layout.axes, layout.x0,
                                    layout.dx, layout.n, overlap_ijlk):
            return None
        return overlap_ijlk


def make_lookup_table(n_theta, n_r, dr=.5, dcw=.5):  # pragma: no cover
    # Downstream rotor radius normalized with characteristic wake width, sigma
    R_sigma = np.arange(0, 20 + dr, dr)
    # Crosswind distance normalized with characteristic wake width, sigma
    cw_sigma = np.arange(0, 10 + dcw, dcw)[:, na, na]
    theta = np.linspace(0, 2 * np.pi, n_theta)[na, :, na]  # Azimuthal discretization of downstream rotor
    r_lst = np.linspace(0, 1, n_r)[na, na, :]

    def cw_table(R_sigma):
        if R_sigma > 0:
            r = r_lst * R_sigma  # Radial discretization of downstream rotor
            dat = np.exp(- 1 / 2 * (r ** 2 + cw_sigma**2 - 2 * r * cw_sigma * np.cos(theta))) * r
            dtheta = np.diff(theta.flatten()[:2])
            dr = np.diff(r.flatten()[:2])
            return trapz(trapz(dat, dx=dtheta, axis=1), dx=dr, axis=1)
        else:
            return np.exp(-(cw_sigma[:, 0, 0]**2) / 2)  # use point value
    dat = np.array([cw_table(R) for R in tqdm(R_sigma)])
    m = R_sigma > 0
    dat[m] = dat[m] / (np.pi * R_sigma[m, na]**2)
    return xr.DataArray(dat, dims=('R_sigma', 'CW_sigma'), coords={'R_sigma': R_sigma, 'CW_sigma': cw_sigma[:, 0, 0]})


def find_n_r(n_theta, max_err):  # pragma: no cover
    res = make_lookup_table(n_theta, 2)
    for n in 2**np.arange(2, 12):
        ref = make_lookup_table(n_theta, n)
        print(n, np.abs(ref - res).max())
        if np.abs(ref - res).max() < max_err:
            return n
        res = ref
    raise ValueError("Error limit not reached")


def find_n_theta(n_r, max_err):  # pragma: no cover
    res = make_lookup_table(2, n_r)
    for n in 2**np.arange(2, 12):
        ref = make_lookup_table(n, n_r)
        print(n, np.abs(ref - res).max())
        if np.abs(ref - res).max() < max_err:
            return n
        res = ref
    raise ValueError("Error limit not reached")


def compare2orsted(table):  # pragma: no cover
    import scipy.io
    mat = scipy.io.loadmat(r'C:\tmp\TurbOPark/gauss_lookup_table')
    dist, radius_down, overlap = mat['overlap_lookup_table'][0][0]
    ref = xr.DataArray(overlap, dims=('CW_sigma', 'R_sigma'), coords={'R_sigma': radius_down[0], 'CW_sigma': dist[0]})
    # ref.to_netcdf(os.path.dirname(__file__) + f'/gaussian_overlap_orsted.nc')

    interp = GridInterpolator([table.R_sigma.values, table.CW_sigma.values], table.values)

    X, Y = np.meshgrid(ref.R_sigma, ref.CW_sigma)
    err = ref - interp(np.array([X.flatten(), Y.flatten()]).T).reshape(X.shape)
    err = ref - table.interp(R_sigma=ref.R_sigma, CW_sigma=ref.CW_sigma, method='cubic')
    print(np.abs(err).max())
    (err).plot()
    import matplotlib.pyplot as plt
    plt.show()


if __name__ == '__main__':  # pragma: no cover
    print(find_n_r(256, 1e-6))
    n_r = 128
    n_r = 512
    print(find_n_theta(512, 1e-6))
    n_theta = 64
    n_theta = 128

    #
    table = make_lookup_table(n_theta, n_r, .02, .02)
    table.to_netcdf(os.path.dirname(__file__) + f'/gaussian_overlap_.02_.02_{n_theta}_{n_r}.nc')
    #
    # interp = GridInterpolator([table.R_sigma.values, table.CW_sigma.values], table.values)
    # da = da.sel(radius_down=radius_down[0, :], dist=dist[0, :])
    # X, Y = np.meshgrid(da.radius_down, da.dist)
    # err = da - interp(np.array([X.flatten(), Y.flatten()]).T).reshape(X.shape)
    # print(np.abs(err).max())
    # (err).plot()
    #
    # table.plot()
    table = xr.load_dataarray(os.path.dirname(__file__) + f'/gaussian_overlap_.02_.02_{n_theta}_{n_r}.nc')

    compare2orsted(table)
    print(table)
