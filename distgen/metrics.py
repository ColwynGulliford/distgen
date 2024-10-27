#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  9 21:18:56 2022

@author: colwyngulliford
"""

import numpy as np

from matplotlib import pyplot as plt

from .tools import linspace
from .tools import trapz

from .physical_constants import unit_registry

from .dist import Uniform
from .dist import SuperGaussian


# --------------------------------------------------------------
# Comparing distributions / shaping metrics
# --------------------------------------------------------------


def mean_and_sigma(x, rho):
    x0 = np.trapezoid(rho * x, x)
    x2 = np.trapezoid(rho * (x - x0) ** 2, x)

    return x0, np.sqrt(x2)


# Distribution comparison functions:


def resample_pq(xp, P, xq, Q, plot=False):
    # Get the new grid:
    xmin = min([xp.min(), xq.min()])
    xmax = max([xp.max(), xq.max()])

    dxp, dxq = np.mean(np.diff(xp)), np.mean(np.diff(xq))
    dx = 0.5 * (dxp + dxq)

    x = linspace(xmin, xmax, int(np.floor((xmax - xmin) / dx)))

    # Interpolate to grid
    Pi = np.interp(x, xp, P, left=0, right=0)
    Qi = np.interp(x, xq, Q, left=0, right=0)

    # Renormalize:
    Pi, Qi = Pi / trapz(Pi, x), Qi / trapz(Qi, x)

    if plot:
        plt.plot(x, Pi, color="tab:blue")
        plt.plot(x, Qi, color="tab:orange")

    return (x, Pi, Qi)


def kullback_liebler_div(xp, P, xq, Q, adjusted=False, as_float=True):
    # Check that input P, Q are PDFs (up to normalization):
    if np.sum(P) == 0.0:
        raise ValueError("PDF array P sums to zero!")

    if np.sum(Q) == 0.0:
        raise ValueError("PDF array Q sums to zero!")

    if len(P[P < 0]) > 0:
        raise ValueError("P array has negative values, and is not a true PDF.")

    if len(Q[Q < 0]) > 0:
        raise ValueError("Q array has negative values, and is not a true PDF.")

    xi, P, Q = resample_pq(xp, P, xq, Q)  # Interpolates to same grid, and renormalizes

    if adjusted:
        q0 = Q == 0
        P0 = P[q0]
        Q[q0] = P0 * np.exp(-P0 / P.max() ** 2)

    p_and_q_nonzero = (P > 0) & (Q > 0)

    P0 = P[p_and_q_nonzero]
    Q0 = Q[p_and_q_nonzero]
    x0 = xi[p_and_q_nonzero]

    KLdiv = np.trapezoid(P0 * (np.log(P0 / Q0)), x0)

    if as_float:
        return KLdiv.magnitude
    else:
        return KLdiv


def res2(xp, P, xq, Q, as_float=True, normalize=False):
    xi, P, Q = resample_pq(xp, P, xq, Q)  # Interpolates to same grid, and renormalizes

    if normalize:
        N = np.trapezoid(Q**2, xi)
    else:
        N = 1

    residuals2 = np.trapezoid((P - Q) ** 2, xi) / N

    if as_float:
        return residuals2.magnitude
    else:
        return residuals2


def get_1d_profile(particle_group, var, bins=None):
    if not bins:
        n = len(particle_group)
        bins = int(n / 100)

    w = particle_group["weight"]

    hist, bin_edges = np.histogram(particle_group[var], bins=bins, weights=w)
    hist_x = bin_edges[:-1] + np.diff(bin_edges) / 2

    return hist_x, hist


"""
def get_1d_profile(particle_group, var, bins=None):

    if not bins:
        n = len(particle_group)
        bins = int(n/100)

    w = particle_group['weight']

    hist, bin_edges = np.histogram(particle_group[var], bins=bins, weights=w)
    hist_x = bin_edges[:-1] + np.diff(bin_edges) / 2

    return hist_x, hist
"""


def get_current_profile(particle_group, bins=None):
    t, rho = get_1d_profile(particle_group, "t", bins=bins)
    return t, particle_group.charge * rho / np.trapezoid(rho, t)


# --------------------------------------------------------------------
# Non-uniformity Measures
# --------------------------------------------------------------------
def rms_equivalent_1d_nonuniformity(
    particle_group, var, bins=None, method="res2", **kwargs
):
    hist_x, hist = get_1d_profile(particle_group, var, bins=bins)

    mean_x, sigma_x = particle_group[f"mean_{var}"], particle_group[f"sigma_{var}"]

    x_units = particle_group.units(var).unitSymbol

    params = {
        f"avg_{var}": mean_x * unit_registry(x_units),
        f"sigma_{var}": sigma_x * unit_registry(x_units),
    }

    # Get RMS equivalent uniform beam and compute non-uniformity
    if method == "kl_div":
        if "p" in kwargs:
            p = kwargs["p"]
        else:
            p = 12

        params["p"] = p

        sg = SuperGaussian(var, **params)
        x = sg.get_x_pts(n=len(hist))
        px = sg.pdf(x)

        return kullback_liebler_div(
            hist_x, hist, x.magnitude, px.magnitude, as_float=True
        )

    elif method == "res2":
        u = Uniform("t", **params)
        x = u.get_x_pts(n=len(hist))
        px = u.pdf(x)

        return (
            res2(
                hist_x, hist, x.magnitude, px.magnitude, as_float=True, normalize=False
            )
            / max(px).magnitude
        )

    else:
        raise ValueError(f"Unsupported method type: {method}")


def rms_equivalent_current_nonuniformity(
    particle_group, bins=None, method="res2", **kwargs
):
    return rms_equivalent_1d_nonuniformity(
        particle_group, "t", bins=bins, method=method, **kwargs
    )


# --------------------------------------------------------------------
