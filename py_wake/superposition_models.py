from py_wake import np
from abc import ABC, abstractmethod
import numba
import numpy
from numpy import newaxis as na
from py_wake.utils.gradients import cabs


class SuperpositionModel(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def __call__(self, value_jxxx):
        """Calculate the sum of jxxx

        This method must be overridden by subclass

        Parameters
        ----------

        deficit_jxxx : array_like
            deficit caused by source turbines(j) on xxx (xxx optionally includes
            destination turbine/site, wind directions, wind speeds

        Returns
        -------
        sum_xxx : array_like
            sum for xxx (see above)
        """

    def superpose_deficit(self, deficit_jxxx, **kwargs):
        return self(deficit_jxxx, **kwargs)


class AddedTurbulenceSuperpositionModel():
    def calc_effective_TI(self, TI_xxx, add_turb_jxxx):
        """Calculate effective turbulence intensity

        Parameters
        ----------
        TI_xxx : array_like
            Local turbulence intensity. xxx optionally includes destination turbine/site, wind directions, wind speeds
        add_turb_jxxx : array_like
            added turbulence caused by source turbines(j) on xxx (see above)

        Returns
        -------
        TI_eff_xxx : array_like
            Effective turbulence intensity xxx (see TI_xxx)
        """
        return TI_xxx + self(add_turb_jxxx)

    def accumulated_effective_TI(self, TI_xxx, sum_xxx):
        return TI_xxx + self.accumulated(sum_xxx)


@numba.njit(error_model='numpy')
def _add_squares_kernel(sum_n, value_jn):
    negative = False
    for j in range(value_jn.shape[0]):
        for n in range(value_jn.shape[1]):
            value = value_jn[j, n]
            negative |= value < 0
            sum_n[n] = sum_n[n] + value * value
    return negative


def _is_c_float64_array(x):
    return type(x) is numpy.ndarray and x.dtype == numpy.float64 and x.flags.c_contiguous


def _add_squares(sum_xxx, value_jxxx):
    negative = _add_squares_kernel(sum_xxx.reshape(-1), value_jxxx.reshape(len(value_jxxx), sum_xxx.size))
    assert not negative, "SquaredSum only works for deficit - not speedups"


class SquaredSum(SuperpositionModel, AddedTurbulenceSuperpositionModel):
    def __call__(self, value_jxxx):
        # np.sum(value_jxxx, 0) is sequential only when j is outermost in memory and each row holds several values
        if np.backend is numpy and _is_c_float64_array(value_jxxx) and numpy.prod(value_jxxx.shape[1:]) > 1:
            sum_xxx = numpy.zeros(value_jxxx.shape[1:])
            _add_squares(sum_xxx, value_jxxx)
            return numpy.sqrt(sum_xxx, out=sum_xxx)
        assert not np.any(value_jxxx < 0), "SquaredSum only works for deficit - not speedups"
        return np.sqrt(np.sum(np.power(value_jxxx, 2), 0))

    def accumulator(self, shape):
        return np.zeros(shape)

    def accumulate(self, sum_xxx, value_xxx):
        if np.backend is numpy and _is_c_float64_array(sum_xxx) and _is_c_float64_array(value_xxx) and \
                value_xxx.shape == sum_xxx.shape and not numpy.may_share_memory(sum_xxx, value_xxx):
            _add_squares(sum_xxx, value_xxx[na])
            return
        assert not np.any(value_xxx < 0), "SquaredSum only works for deficit - not speedups"
        sum_xxx += np.power(value_xxx, 2)

    def accumulated(self, sum_xxx):
        return np.sqrt(sum_xxx)


class LinearSum(SuperpositionModel, AddedTurbulenceSuperpositionModel):
    def __call__(self, value_jxxx):
        return np.sum(value_jxxx, 0)

    def accumulator(self, shape):
        return np.zeros(shape)

    def accumulate(self, sum_xxx, value_xxx):
        sum_xxx += value_xxx

    def accumulated(self, sum_xxx):
        return sum_xxx


class MaxSum(SuperpositionModel, AddedTurbulenceSuperpositionModel):
    def __call__(self, value_jxxx):
        return np.max(value_jxxx, 0)


class SqrMaxSum(AddedTurbulenceSuperpositionModel):
    def calc_effective_TI(self, TI_xxx, add_turb_jxxx):
        return np.sqrt(TI_xxx**2 + np.max(add_turb_jxxx, 0)**2)

    def accumulator(self, shape):
        return np.full(shape, -np.inf)

    def accumulate(self, max_xxx, add_turb_xxx):
        np.maximum(max_xxx, add_turb_xxx, out=max_xxx)

    def accumulated_effective_TI(self, TI_xxx, max_xxx):
        return np.sqrt(TI_xxx**2 + max_xxx**2)


class WeightedSum(SuperpositionModel):
    """
    Implemented according to the paper by:
    Haohua Zong and Fernando Porté-Agel
    A momentum-conserving wake superposition method for wind farm power prediction
    J. Fluid Mech. (2020), vol. 889, A8; doi:10.1017/jfm.2020.77
    """

    def __init__(self, delta=0.01, max_err=1e-3, max_iter=5, weight_limit=10 ** 10):
        # minimum deficit (as fraction of free-stream) to invoke weighted summation
        self.delta = delta
        # convergence limit for computing global convection velocity
        self.max_err = max_err
        # maximum number of iterations used in computing weights
        self.max_iter = max_iter
        # Limit weights to this value. A limit of 1.0 means not exceeding linear superposition
        self.weight_limit = weight_limit

    def __call__(self, deficit_jxxx, WS_xxx,
                 convection_velocity_jxxx,
                 sigma_sqr_jxxx, cw_jxxx, hcw_jxxx, dh_jxxx, deficit_centre_jxxx):

        Ws = WS_xxx + np.zeros(deficit_centre_jxxx.shape[1:])

        usc = deficit_centre_jxxx
        uc = convection_velocity_jxxx
        sigma_sqr = sigma_sqr_jxxx
        cw = cw_jxxx
        hcw = hcw_jxxx * np.ones_like(usc)
        dh = dh_jxxx * np.ones_like(usc)
        Us = np.zeros_like(Ws)
        # Determine non-centreline deficit ratio
        # Local deficit
        us = usc * np.exp(-1 / (2 * (sigma_sqr + 1e-30)) * cw**2)
        us = np.where(usc == 0, 0.0, us)

        # Set lower deficit limit below which deficits are linearly added
        us_lim = max(self.delta, 1e-30) * Ws[na]
        # Get indices
        Il = us >= us_lim

        ucn = np.ones_like(uc)
        # Only start weighting computation if at least two deficits need to be combined
        # Computations are performed where velocities exceed the specified limit (us_lim).
        # The more complex indexing leads to more complex code.
        if Il.any():
            # Get indices where deficits need to be combined
            Ilx = Il.any(axis=0)
            # Total cross-wind integrated deficit
            us_int = np.where(Il, usc * 2 * np.pi * sigma_sqr, 0.0)
            # initialize combined quanatities
            Uc = Ws.copy()
            Uc_star = Ws.copy()
            Us = np.zeros_like(Ws)
            Us_int = np.zeros_like(Ws)

            # Initialize
            count = 0
            Uc_star = 10 * Uc
            if np.iscomplexobj(Ws) or np.iscomplexobj(uc):
                Uc_star = Uc_star.astype(np.complex128)
            sum1, sum2, ucn = np.zeros_like(Us), np.zeros_like(Us), np.ones_like(uc)
            n_wt = us.shape[0]
            # Iterate until combined convection velocity converges
            while (np.max(cabs((Uc[Ilx] - Uc_star[Ilx]) / Uc_star[Ilx])) > self.max_err) and (count < self.max_iter):
                # Initialize combined convection velocity
                if count == 0:
                    # Take maximum across all turbines to initilize global convection velocity
                    Uc = np.max(np.where(Il, uc, 0), 0)
                else:
                    Uc = Uc_star
                # Initialize and avoid division by zero
                Inz = Uc != 0
                # Limit weights to self.weight_limit
                safe_uc = np.where(Inz, Uc, 1.0)
                ucn = np.where(
                    Inz[na],
                    np.minimum(uc / safe_uc[na], self.weight_limit),
                    1.0,
                )

                # Combined local deficit
                tmpUS = np.where(Il, ucn * us, 0.0)
                Us = np.sum(tmpUS, axis=0)
                # Combined deficit integrated to infinity in cross-wind direction
                tmpUSint = np.where(Il, ucn * us_int, 0.0)
                Us_int = np.sum(tmpUSint, axis=0)

                # First sum of momentum deficit
                # sum_i^N (uc_i u_i)^2
                tmp1 = np.where(Il, ucn**2 * us_int * usc / 2, 0.0)
                sum1 = np.sum(tmp1, axis=0)
                # Second sum which represents the cross terms
                # 2 sum i>j (uc_i u_i) (uc_j u_j)
                if n_wt > 1:
                    # Initialize
                    sum2 = np.zeros_like(Us)
                    for j in range(n_wt - 1):
                        # Only cross with larger indices
                        k = np.arange(j + 1, n_wt, dtype=int)
                        # Find indices where deficits need to be combined
                        Ilxx = Il[j][na] & Il[k]
                        if Ilxx.any():
                            # To keep the shape, arrays are repeated and a dummy initilized
                            # Instead of a dummy one could use a loop, but this seemed faster
                            sigma_j = np.repeat(sigma_sqr[j][na], len(k), axis=0)
                            sigma_k = sigma_sqr[k]
                            sigma_sum = np.where(Ilxx, sigma_j + sigma_k, 1.0)
                            w2w_hcw = cabs(hcw[j][na] - hcw[k])
                            w2w_dh = cabs(dh[j][na] - dh[k])
                            cross_sigma_jk = 2 * np.exp(-(w2w_hcw**2 + w2w_dh**2) /
                                                        (2 * sigma_sum)) * np.pi * sigma_j * sigma_k / sigma_sum
                            pair_term = 2 * np.repeat((ucn[j] * usc[j])[na], len(k), axis=0) * (ucn[k] * usc[k])
                            tmp2 = np.where(Ilxx, pair_term * cross_sigma_jk, 0.0)
                            sum2 = sum2 + np.sum(tmp2, axis=0)

                # Avoid division by zero
                safe_Us_int = np.where(Us_int == 0, 1.0, Us_int)
                # Update combined convection velocity
                next_Uc_star = Ws - (sum1 + sum2) / safe_Us_int
                Uc_star = np.where(Ilx, next_Uc_star, Uc_star)

                count += 1
        # Replace non-finite values where convection variables are undefined (i.e. upstream of wt)
        ucn = np.where(np.isfinite(ucn), ucn, 1.0)
        # Reset weights to linear sum where WeightedSum is not applied
        ucn = np.where(Il, ucn, 1.0)
        return np.sum(deficit_jxxx * ucn, axis=0)


class CumulativeWakeSum(SuperpositionModel):
    """
    Implemention of the cumulative wake model:
    Majid Bastankhah, Bridget L. Welch, Luis A. Martínez-Tossas, Jennifer King and Paul Fleming
    Analytical solution for the cumulative wake of wind turbines in wind farms
    J. Fluid Mech. (2021), vol. 911, A53, doi:10.1017/jfm.2020.1037
    """

    def __init__(self, alpha=2.):
        # somewhat empirical factor to scale results to fit LES predictions
        self.alpha = alpha  # alpha in 8.3 (1 in eq 6.4 and 2 in eq 4.9)

    def superpose_deficit(self, deficit_jxxx, **kwargs):
        return self(**kwargs)

    def __call__(self, WS0_xxx, WS_eff_xxx, ct_xxx, D_xx, sigma_sqr_jxxx, cw_jxxx, hcw_jxxx, dh_jxxx):

        U0 = WS0_xxx
        WS_eff = WS_eff_xxx
        ct = ct_xxx
        D = D_xx
        downwind = (sigma_sqr_jxxx > 1e-10)
        sigma_sqr = np.where(downwind, sigma_sqr_jxxx, 1)
        cw = cw_jxxx * np.ones_like(sigma_sqr)
        hcw = hcw_jxxx * np.ones_like(sigma_sqr)
        dh = dh_jxxx * np.ones_like(sigma_sqr)

        n_wt = sigma_sqr.shape[0]
        lamCsum = np.zeros(sigma_sqr.shape[1:])

        C_lst = []
        for n in range(n_wt):
            if n > 0:
                sigma_sqr_tot = sigma_sqr[n:n + 1] + sigma_sqr[:n]
                # eq 4.9
                lam = self.alpha * sigma_sqr[:n] / sigma_sqr_tot * np.exp(-(hcw[n][na, ...] - hcw[:n])**2 / (
                    2. * sigma_sqr_tot)) * np.exp(-(dh[n][na, ...] - dh[:n])**2 / (2. * sigma_sqr_tot))
                # lambda sum term in 4.10
                lamCsum = np.sum(lam * np.array(C_lst)[:n], axis=0)

            # sqrt term in eq 4.10
            sqrt_term = (U0[n] - lamCsum)**2 - (ct[n] * D[n, ..., na]**2 * WS_eff[n]**2) / (8. * sigma_sqr[n])
            sqrt_term = np.maximum(sqrt_term, 0)
            Clim = (WS_eff[n] * (1. - np.sqrt(np.maximum(0, 1. - ct[n]))))
            C = (U0[n] - lamCsum) - np.sqrt(sqrt_term)  # Eq 4.10
            C = np.where((sqrt_term > 0.) & (C <= Clim), C, Clim)
            C *= downwind[n]
            C_lst.append(C)
        C = np.array(C_lst)

        exponent = -1 / (2 * sigma_sqr) * cw**2

        return np.sum(C * np.exp(exponent), axis=0)
