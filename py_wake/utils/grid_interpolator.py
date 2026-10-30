from autograd.numpy.numpy_boxes import ArrayBox
import functools
import numba
import numpy
from numpy import atleast_1d
from numpy import newaxis as na
from typing import NamedTuple

from py_wake import np
from py_wake.utils import gradients


_NUMPY_PAIRWISE_BLOCKSIZE = 128


@numba.njit(error_model='numpy')
def _numpy_pairwise_sum(terms):
    r = terms[:8].copy()
    i = 8
    while i < len(terms) - len(terms) % 8:
        for j in range(8):
            r[j] = r[j] + terms[i + j]
        i += 8
    res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]))
    while i < len(terms):
        res = res + terms[i]
        i += 1
    return 0. + res


@numba.njit(error_model='numpy')
def _search_right(axis, start, n, value):
    lo, hi = 0, n
    while lo < hi:
        mid = (lo + hi) >> 1
        if value < axis[start + mid]:
            hi = mid
        else:
            lo = mid + 1
    return lo


@numba.njit(error_model='numpy')
def _grid_position(value, axes, start, x0, dx, n, irregular, linear):
    if irregular:
        i0 = min(max(_search_right(axes, start, n, value) - 1, 0), n - 2)
        xpi = i0 + (value - axes[start + i0]) / (axes[start + i0 + 1] - axes[start + i0])
    else:
        xpi = (value - x0) / dx
    if xpi <= 0:
        xpi = 0.
    if xpi >= n - 1:
        xpi = n - 1.
    if linear:
        return xpi
    return numpy.floor(xpi + .5)


def _corner_bits(linear):
    bits, bit = [], sum(linear)
    for is_linear in linear:
        bit -= is_linear
        bits.append(bit if is_linear else -1)
    return tuple(bits)


class _CompiledLayout(NamedTuple):
    V: numpy.ndarray
    strides: tuple
    axes: numpy.ndarray
    axes_start: tuple
    x0: tuple
    dx: tuple
    n: tuple


@functools.lru_cache
def _limit_kernel(corner_bit, irregular):
    n_dims = len(corner_bit)
    n_corners = 2**sum(bit >= 0 for bit in corner_bit)
    pairwise_sum = n_corners >= 8

    @numba.njit(error_model='numpy')
    def interpolate(xp_T, V, strides, axes, axes_start, x0, dx, n, res):
        n_points = xp_T.shape[1]
        pairwise = pairwise_sum and n_points == 1
        terms = numpy.empty(n_corners)
        weight = numpy.empty((n_dims, 2))
        offset = numpy.empty((n_dims, 2), dtype=numpy.int64)
        for p in range(n_points):
            for d in range(n_dims):
                xpi = _grid_position(xp_T[d, p], axes, axes_start[d], x0[d], dx[d], n[d], irregular, corner_bit[d] >= 0)
                lower = 0 if numpy.isnan(xpi) else int(xpi)
                xpif = xpi - lower
                weight[d, 0] = 1 - xpif
                weight[d, 1] = xpif
                offset[d, 0] = lower * strides[d]
                offset[d, 1] = min(lower + 1, n[d] - 1) * strides[d]
            total = 0.
            for corner in range(n_corners):
                w = 1.
                index = 0
                for d in range(n_dims):
                    side = (corner >> corner_bit[d]) & 1 if corner_bit[d] >= 0 else 0
                    w = w * weight[d, side]
                    index += offset[d, side]
                if pairwise:
                    terms[corner] = w * V[index]
                else:
                    total = total + w * V[index]
            res[p] = _numpy_pairwise_sum(terms) if pairwise else total
    return interpolate


class GridInterpolator(object):
    # Faster than scipy.interpolate.interpolate.RegularGridInterpolator
    def __init__(self, x, V, method='linear', bounds='check'):
        """
        Parameters
        ----------
        x : list of array_like
            Interpolation coordinates
        V : array_like
            Interpolation data (more dimensions than coordinates is possible)
        method : {'linear' or 'nearest} or [{'linear' or 'nearest}, ...]
            default interpolation method
        bounds : {'check', 'limit', 'ignore'}
            Specifies how bounds is handled:\n
            - 'check': bounds check is performed. An error is raised if interpolation point outside area
            - 'limit': interpolation points are forced inside the area
            - 'ignore': Faster option with no check. Use this option if data is guaranteed to be inside the area
        """
        self.x = self.grid = x.copy()

        self.bounds = bounds
        self.method = method
        self.n = np.array([len(x) for x in x], dtype=int)
        self.x0 = np.array([x[0] for x in x])
        dx = np.array([x[min(1, len(x) - 1)] - x[0] for x in x])
        self.dx = np.where(dx == 0, 1, dx)
        self.irregular_axes = np.array([np.allclose(np.diff(x), dx) is False for dx, x in zip(self.dx, x)])
        self.irregular_axes_indexes = np.where(self.irregular_axes)[0]
        for i in self.irregular_axes_indexes:
            self.x[i] = np.r_[self.x[i], self.x[i][-1] + 1]
        self.V = self.values = np.asarray(V)
        if not np.all(np.asarray(self.V.shape[:len(self.n)]) == self.n):
            raise ValueError("Lengths of x does not match shape of V")
        ui = np.array([[0], [1]])
        for _ in range(len(x) - 1):
            ui = np.array([(np.r_[ui, 0], np.r_[ui, 1]) for ui in ui])
            ui = ui.reshape((ui.shape[0] * ui.shape[1], ui.shape[2]))
        ui[:, dx == 0] = 0
        self.ui = ui.astype(int)

    def __call__(self, xp, method=None, bounds=None, deg=False):
        """Interpolate points

        Parameters
        ----------
        xp : array_like
            Interpolation points, shape=(n_points, interpolation_dimensions)
        method : {'linear' or 'nearest} or [{'linear' or 'nearest}, ...] or None
            Overrides self.method if not None
        bounds : {'check', 'limit', 'ignore'} or None
            Overrides self.bounds if not None
        """
        if len(np.atleast_1d(xp)) == 0:
            return np.array([])
        if method is None:
            method = self.method
        method = np.atleast_1d(method)
        assert all([m in ['linear', 'nearest'] for m in method]), 'method must be "linear" or "nearest"'
        assert len(method) in [1, len(self.x)]
        linear = [method[min(len(method) - 1, i)] == 'linear' for i in range(len(self.x))]
        if (bounds or self.bounds) == 'limit' and not deg:
            res = self._compiled_limit(xp, linear)
            if res is not None:
                return res
        chunks = np.maximum(2**np.sum(linear) * np.prod(np.shape(xp)) * 8 / 1024**2 // 100, 1)
        if chunks > 1:
            return np.concatenate([self.__call__(xp_, method, bounds, deg) for xp_ in np.array_split(xp, chunks)])
        bounds = bounds or self.bounds
        assert bounds in ['check', 'limit', 'ignore'], 'bounds must be "check", "limit" or "ignore"'
        xp = np.atleast_2d(xp)
        xp_shape = xp.shape
        assert xp_shape[-1] == len(self.x), xp_shape
        xp = np.reshape(xp, (-1, xp_shape[-1]))
        if len(self.irregular_axes_indexes):
            xpi0 = np.array([np.clip(np.searchsorted(x, xp, side='right') - 1, 0, n - 2, dtype=int)
                             for x, xp, n in zip(self.x, xp.T, self.n)], dtype=int)
            xp0 = np.array([np.asarray(x)[xpi0] for x, xpi0 in zip(self.x, xpi0)])
            xp1 = np.array([np.asarray(x)[xpi0 + 1] for x, xpi0 in zip(self.x, xpi0)])
            xp_dx = xp1 - xp0
            xpi = xpi0.T + (xp - xp0.T) / xp_dx.T
        else:
            xpi = (xp - self.x0) / self.dx

        if bounds == 'check' and (np.any(xpi < 0) or np.any(xpi + 1 > self.n[na])):
            if -xpi.min() > (xpi + 1 - self.n[na]).max():
                point, dimension = np.unravel_index(xpi.argmin(), np.atleast_2d(xpi).shape)
            else:
                point, dimension = np.unravel_index(((xpi + 1 - self.n[na])).argmax(), np.atleast_2d(xpi).shape)
            raise ValueError("Point %d, dimension %d with value %f is outside range %f-%f" %
                             (point, dimension, np.atleast_2d(xp)[point, dimension], self.x[dimension][0], self.x[dimension][-1]))
        if bounds == 'limit':
            xpi = np.minimum(np.maximum(xpi, 0), self.n - 1)

        if 'nearest' in method:
            # round x.5 up (np.round rounds x.5 towards nearest even number)
            xpi = np.where(np.asarray(linear), xpi, gradients.floor(xpi + .5))

        xpif, xpi0 = gradients.modf(xpi)

        int_box_axes = [(np.asarray([0], dtype=int), np.asarray([0, 1], dtype=int))[l] for l in linear]
        ui = np.moveaxis(np.asarray(np.meshgrid(*int_box_axes, indexing='ij'), dtype=int), 0, -1)

        ui = ui.reshape((-1, len(self.x)))
        indexes = (ui.T[:, :, na] + xpi0.T[:, na])

        indexes = np.minimum(indexes, (self.n - 1)[:, na, na], dtype=int)
        v = np.moveaxis(self.V[tuple(indexes)], [0, 1], [-2, -1])
        if deg:
            v = (v + 180) % 360 - 180  # -180..180 > 0-360

        for i, x in enumerate(zip((1 - xpif).T, xpif.T)):
            if linear[i]:
                if i == 0:
                    w = np.array(x)
                else:
                    w = w[..., na, :] * np.expand_dims(np.array(x), tuple(range(i)))
            else:
                if i == 0:
                    w = x[0]
                else:
                    w = w * x[0]  # np.expand_dims(x[0], tuple(range(i)))
        w = np.reshape(w, (-1, xpif.shape[0]))

        # w = np.prod(np.array([xpif1, xpif])[ui, :, range(len(self.x))], 1) # slower
        # w = np.product(np.take_along_axis(xpif10[:, :, na], self.ui[na, na], 0).squeeze(), 2)  # even slower

        res = np.moveaxis((w * v).sum(-2), -1, 0)
        if deg:
            res = gradients.mod(res, 360)
        return np.reshape(res, xp_shape[:-1] + self.V.shape[len(self.x):])

    def __getstate__(self):
        return {k: v for k, v in self.__dict__.items() if k != '_compiled_layout'}

    def _compiled_limit(self, xp, linear):
        if np.backend is not numpy or type(xp) is not numpy.ndarray or xp.dtype != numpy.float64 or \
                not 1 <= xp.ndim <= 2 or xp.shape[-1] != len(self.x) or 2**sum(linear) > _NUMPY_PAIRWISE_BLOCKSIZE or \
                self._compiled_layout is None:
            return None
        layout = self._compiled_layout
        xp_T = numpy.require(numpy.atleast_2d(xp).T, requirements=['C', 'W'])
        res = numpy.empty(xp_T.shape[1])
        interpolate = _limit_kernel(_corner_bits(linear), len(self.irregular_axes_indexes) > 0)
        interpolate(xp_T, layout.V, layout.strides, layout.axes, layout.axes_start, layout.x0, layout.dx, layout.n, res)
        return res

    @functools.cached_property
    def _compiled_layout(self):
        V = self.V
        axes = [x[:n] for x, n in zip(self.x, self.n)]
        if not (V.dtype == numpy.float64 and V.ndim == len(axes) and all(self.n >= 2) and
                all(type(x) is numpy.ndarray and x.dtype == numpy.float64 for x in axes)):
            return None
        if len(self.irregular_axes_indexes) and not all(numpy.all(numpy.diff(x) >= 0) for x in axes):
            return None
        V_memory_order = V.transpose(numpy.argsort(V.strides)[::-1])
        if not V_memory_order.flags.c_contiguous:
            return None
        return _CompiledLayout(V=V_memory_order.reshape(-1), strides=tuple(s // V.itemsize for s in V.strides),
                               axes=numpy.concatenate(axes), axes_start=tuple([0] + numpy.cumsum(self.n).tolist()[:-1]),
                               x0=tuple(self.x0.tolist()), dx=tuple(self.dx.tolist()), n=tuple(self.n.tolist()))


class EqDistRegGrid2DInterpolator():
    def __init__(self, x, y, Z):
        self.x = x
        self.y = y
        self.Z = Z
        self.dx, self.dy = [xy[1] - xy[0] for xy in [x, y]]
        assert all(np.diff(x) == self.dx), "x is not equidistant"
        assert all(np.diff(y) == self.dy), "y is not equidistant"
        self.x0 = x[0]
        self.y0 = y[0]
        xi_valid = np.where(np.any(~np.isnan(self.Z), 1))[0]
        yi_valid = np.where(np.any(~np.isnan(self.Z), 0))[0]
        self.xi_valid_min, self.xi_valid_max = xi_valid[0], xi_valid[-1]
        self.yi_valid_min, self.yi_valid_max = yi_valid[0], yi_valid[-1]

    def __call__(self, x, y, mode='valid'):
        xp, yp = x, y
        xif, xi0 = gradients.modf((xp - self.x0) / self.dx)
        yif, yi0 = gradients.modf((yp - self.y0) / self.dy)

        if mode == 'extrapolate':
            xif[xi0 < self.xi_valid_min] = 0
            xif[xi0 > self.xi_valid_max - 2] = 1
            yif[yi0 < self.yi_valid_min] = 0
            yif[yi0 > self.yi_valid_max - 2] = 1
            xi0 = np.minimum(np.maximum(xi0, self.xi_valid_min), self.xi_valid_max - 2)
            yi0 = np.minimum(np.maximum(yi0, self.yi_valid_min), self.yi_valid_max - 2)
        xi1 = xi0 + (xif > 0)
        yi1 = yi0 + (yif > 0)
        if isinstance(xp, ArrayBox) or isinstance(yp, ArrayBox):
            valid = slice(None)
        else:
            valid = (xif >= 0) & (yif >= 0) & (xi1 < len(self.x)) & (yi1 < len(self.y))
        xi0, xi1, xif, yi0, yi1, yif = [v[valid] for v in [xi0, xi1, xif, yi0, yi1, yif]]
        z00 = self.Z[xi0, yi0]
        z10 = self.Z[xi1, yi0]
        z01 = self.Z[xi0, yi1]
        z11 = self.Z[xi1, yi1]
        z0 = z00 + (z10 - z00) * xif
        z1 = z01 + (z11 - z01) * xif
        if isinstance(xp, ArrayBox) or isinstance(yp, ArrayBox):
            z = z0 + (z1 - z0) * yif
        else:
            dtype = float
            if np.iscomplexobj(xp):
                dtype = xp.dtype
            if np.iscomplexobj(yp):
                dtype = yp.dtype

            z = np.full(xp.shape, np.nan, dtype=dtype)
            z[valid] = z0 + (z1 - z0) * yif
        return z
