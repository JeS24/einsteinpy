import warnings

import numpy as np

from einsteinpy.coordinates.utils import spherical_to_cartesian_fast
from einsteinpy.geodesic.utils import (
    _calc_state,
    _christoffels_M,
    _f_geod_M,
    _rk4_step,
    _step_size,
    check,
)
from einsteinpy.integrators import RK45


class Nulllike:
    """
    Class for defining Null-like Geodesics

    """

    # - ????? Too many parameters -
    # Consolidate alpha, beta, rcam & inclination
    # Consolidate end_lambda, step_size, return_cartesian as ODEKwargs
    def __init__(
        self,
        alpha,
        beta,
        rcam=1e4,
        inclination=np.pi / 2,
        a=0.0,
        # end_lambda=10.0,
        step_size=3e-2,
        max_steps=1e5,  # - ????? Other params - end_lambda
        return_cartesian=True,
    ):
        """
        Parameters
        ----------
        # - Docstring ?????
        end_lambda : float
            Affine Parameter, Lambda, where iterations will stop
        step_size : float, optional
            Size of each geodesic integration step
            Defaults to ``3e-2``
        return_cartesian : bool, optional
            Whether to return calculated values in Cartesian Coordinates
            Defaults to ``True``

        """
        self.alpha = alpha
        self.beta = beta
        self.rcam = rcam
        self.inclination = inclination
        self.a = a
        # self.step_size = step_size
        # self.max_steps = max_steps

        # self._sch_rad = 2
        self._state = self._calculate_state()
        self._trajectory = self.calculate_trajectory(
            step_size=step_size, max_steps=max_steps, return_cartesian=return_cartesian
        )[1]

    def __repr__(self):
        return f"NullGeodesic Object:\n\
            Initial Conditions = (\n\
                alpha: {self.alpha},\n\
                beta: {self.beta},\n\
                rcam: {self.rcam},\n\
                inclination: {self.inclination},\n\
                a: {self.a}\n\
            ),\n\
            Initial State = ({self.state}),\n\
            Trajectory = ({self.trajectory})"

    def __str__(self):
        return f"NullGeodesic Object:\n\
            Initial Conditions = (\n\
                alpha: {self.alpha},\n\
                beta: {self.beta},\n\
                rcam: {self.rcam},\n\
                inclination: {self.inclination},\n\
                a: {self.a}\n\
            ),\n\
            Initial State = ({self.state}),\n\
            Trajectory = ({self.trajectory})"

    @property
    def state(self):
        """
        Returns the Initial State Vector of the Geodesic

        """
        return self._state

    @property
    def trajectory(self):
        """
        Returns the "Trajectory" of the Geodesic

        """
        return self._trajectory

    def _calculate_state(self):
        """
        Prepares and returns the Initial State Vector of the massless test particle

        Source: RAPTOR (?????)

        Returns
        -------
        state : ~numpy.ndarray
            Initial State Vector of the massless test particle
            Length-8 Array

        """
        a = self.a
        r = self.rcam
        alpha, beta = self.alpha, self.beta
        theta = self.inclination
        E = 1.0

        return _calc_state(alpha, beta, r, theta, a, E)

    # Move to Kerr - ?????
    # Not until units + coordinate switching is figured out
    def _ch_sym_M(self, x_vec):
        """
        Returns Christoffel Symbols for Kerr Metric \
        in Boyer-Lindquist Coordinates, in M-Units

        Parameters
        ----------
        x_vec : array_like
            Position 4-Vector

        Returns
        -------
        ~numpy.ndarray
            Christoffel Symbols for Kerr Metric \
            in Boyer-Lindquist Coordinates \
            in M-Units
            Numpy array of shape (4,4,4)

        """
        r, th = x_vec[1], x_vec[2]

        return _christoffels_M(self.a, r, th)

    # Move to Kerr - ?????
    # Not until units + coordinate switching is figured out
    def _f_vec_M(self, lambda_, vec):
        """
        Returns f_vec for Kerr Metric \
        in Boyer-Lindquist Coordinates \
        in M-Units (G = c = M = 1)

        To be used for solving Geodesics ODE

        Source: RAPTOR (?????)

        Parameters
        ----------
        lambda_ : float
            Parameterizes current integration step
            Used by ODE Solver

        vec : array_like
            Length-8 Vector, containing 4-Position & 4-Velocity

        Returns
        -------
        ~numpy.ndarray
            f_vec for Kerr Metric in Boyer-Lindquist Coordinates
            Numpy array of shape (8)

        """
        chl = self._ch_sym_M(vec[:4])
        vals = np.zeros(shape=vec.shape, dtype=vec.dtype)

        return _f_geod_M(chl, vals, vec)

    def calculate_trajectory(  # - ????? Name
        self, step_size=3e-2, max_steps=1e5, return_cartesian=True
    ):
        """
        Calculate trajectory in spacetime, according to Geodesic Equations

        Parameters
        ----------
        return_cartesian : bool, optional
            Whether to return calculated values in Cartesian Coordinates
            Defaults to ``True``

        Returns
        -------
        ~numpy.ndarray
            N-element numpy array containing Lambda, where the geodesic equations were evaluated
        ~numpy.ndarray
            (n,8) shape numpy array containing [x0, x1, x2, x3, v0, v1, v2, v3] for each Lambda

        """
        a = self.a
        y = self.state
        r = y[1]
        steps = 0
        stability_factor = 10

        rdot, thdot, pdot = y[5], y[6], y[7]

        # Termination conditions
        cutoff_outer = r * 1.01
        cutoff_inner = (1.0 + np.sqrt(1.0 - a ** 2)) * 1.1

        vecs = list()
        vecs.append(y)
        lambdas = list()
        lambdas.append(step_size)

        data = open("data.txt", "w")  # To dump y - ?????
        while r > cutoff_inner and r < cutoff_outer and steps < max_steps:
            print(y)
            rdot, thdot, pdot = y[5], y[6], y[7]
            # print(rdot, thdot, pdot)
            # Getting size of next step
            dlambda_next = _step_size(step_size, y[:4], y[4:])
            y = _rk4_step(a, dlambda_next, y)

            # DUMPING DATA TO FILE
            out = repr(y)
            out = out.replace("array(", "").replace("\n", "").replace(")", "").replace("      ", "")
            out += ", "
            data.write(out)

            if r < cutoff_inner or r > cutoff_outer:
                warnings.warn("Photon has reached cutoff bounds. ", RuntimeWarning)
                break
            vecs.append(y)

            # print(rdot, y[5] / rdot)
            # print(thdot, y[6] / thdot)
            print(pdot, y[7] / pdot)
            # Extra stability conditions
            if rdot != 0. and np.abs(y[5] / rdot) > stability_factor:
                y[5] = rdot
            if thdot != 0. and np.abs(y[6] / thdot) > stability_factor:
                y[6] = thdot
            if pdot != 0. and np.abs(y[7] / pdot) > stability_factor:
                y[7] = pdot

            print(dlambda_next)
            lambdas.append(dlambda_next)

            r = y[1]
            steps += 1
            # print('STEP: ', steps)
            # print()

        data.close()
        vecs, lambdas = np.array(vecs), np.array(lambdas)

        if return_cartesian:
            cart_vecs = list()
            for v in vecs:
                # Using sph_to_cart from Symmetry considerations
                # (As considered in RAPTOR & other papers)
                vals = spherical_to_cartesian_fast(
                    v[0], v[1], v[2], v[3], v[5], v[6], v[7], velocities_provided=True,
                )
                cart_vecs.append(np.hstack((vals[:4], v[4], vals[4:])))

            return lambdas, np.array(cart_vecs)

        return lambdas, vecs

    # def calculate_trajectory(
    #     self,
    #     end_lambda=10.0,
    #     OdeMethodKwargs={"stepsize": 3e-2},
    #     return_cartesian=True,
    # ):
    #     """
    #     Calculate trajectory in spacetime, according to Geodesic Equations

    #     Parameters
    #     ----------
    #     end_lambda : float, optional
    #         Affine Parameter, Lambda, where iterations will stop
    #         Equivalent to Proper Time for Timelike Geodesics
    #         Defaults to ``10.0``
    #     OdeMethodKwargs : dict, optional
    #         Kwargs to be supplied to the ODESolver
    #         Dictionary with key 'stepsize' along with a float value is expected
    #         Defaults to ``{'stepsize': 3e-2}``
    #     return_cartesian : bool, optional
    #         Whether to return calculated values in Cartesian Coordinates
    #         Defaults to ``True``

    #     Returns
    #     -------
    #     ~numpy.ndarray
    #         N-element numpy array containing Lambda, where the geodesic equations were evaluated
    #     ~numpy.ndarray
    #         (n,8) shape numpy array containing [x0, x1, x2, x3, v0, v1, v2, v3] for each Lambda

    #     """
    #     ODE = RK45(
    #         fun=self._f_vec_M,
    #         t0=0.0,
    #         y0=self.state,
    #         t_bound=end_lambda,
    #         **OdeMethodKwargs,
    #     )

    #     a = self.a
    #     r = self.rcam

    #     # Termination conditions
    #     cutoff_outer = r * 1.01
    #     cutoff_inner = (1.0 + np.sqrt(1.0 - a ** 2)) * 1.1

    #     vecs = list()
    #     lambdas = list()

    #     while ODE.t < end_lambda:
    #         vecs.append(ODE.y)
    #         lambdas.append(ODE.t)

    #         r_curr = ODE.y[1]

    #         # Checking termination conditions
    #         if r_curr < cutoff_inner or r_curr > cutoff_outer:
    #             message = "Inner" if r_curr < cutoff_inner else "Outer"
    #             warnings.warn(f"Light Ray has reached {message} cut-off bound.", RuntimeWarning)
    #             break

    #         ODE.step()

    #     vecs, lambdas = np.array(vecs), np.array(lambdas)

    #     if return_cartesian:
    #         cart_vecs = list()
    #         for v in vecs:
    #             vals = bl_to_cartesian_fast(
    #                 v[0],
    #                 v[1],
    #                 v[2],
    #                 v[3],
    #                 a,
    #                 v[5],
    #                 v[6],
    #                 v[7],
    #                 velocities_provided=True,
    #             )
    #             cart_vecs.append(np.hstack((vals[:4], v[4], vals[4:])))

    #         return lambdas, np.array(cart_vecs)

    #     return lambdas, vecs


class NullBundle:
    """
    Class for generating a photon sheet and performing
    geodesic integration for Radiative Transfer applications

    """
