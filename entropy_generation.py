"""
Shock entropy generation rate as a pynbody derived quantity.

Estimates dK/dt per particle based on the local compression Mach number,
following the SPHENIX artificial viscosity formalism (Borrow et al. 2022,
MNRAS 511, 2367) as used in FLAMINGO (Schaye et al. 2023).

In SPHENIX, the viscous interaction strength is:

    zeta_ij = alpha * |mu_ij| * v_sig / (2 * rho_bar)

where v_sig = c_i + c_j - beta_V * mu_ij, and beta_V = 3.0 is a
compile-time constant.  Crucially, alpha multiplies BOTH the linear
and quadratic terms:

    zeta ~ alpha * |mu| * (2*c_s + beta_V * |mu|) / (2*rho)

In the per-particle approximation (|mu| ~ h*|theta|, M = h*|theta|/c_s):

    dK/dt = (gamma-1) * K * |theta| * M * alpha * (1 + beta_V * M)

Only applied to particles in converging flow (theta < 0).
"""

import numpy as np
import pynbody
from pynbody.array import SimArray

GAMMA = 5.0 / 3.0
BETA_V = 3.0  # SPHENIX compile-time constant (viscosity_beta)


@pynbody.derived_array
def entropy_function(sim):
    """K = P / rho^gamma"""
    return sim["p"] / sim["rho"] ** GAMMA


@pynbody.derived_array
def sound_speed(sim):
    """c_s = sqrt(gamma * P / rho)"""
    return np.sqrt(GAMMA * sim["p"] / sim["rho"])


@pynbody.derived_array
def compression_mach(sim):
    """
    Local compression Mach number: M = h * |theta| / c_s.

    Set to zero for non-converging regions (theta >= 0).
    """
    theta = sim["VelocityDivergences"]
    cs = sim["sound_speed"]
    h = sim["smooth"]

    mach = SimArray(np.zeros(len(sim)), units=h.units * theta.units / cs.units)
    compressing = theta < 0
    mach[compressing] = h[compressing] * np.abs(theta[compressing]) / cs[compressing]
    return mach.in_units("1")


@pynbody.derived_array
def entropy_generation_rate(sim):
    """
    dK/dt per particle from shock dissipation (SPHENIX formalism).

    Uses the per-particle Cullen & Dehnen viscosity switch (ViscosityParameters)
    as alpha_V.  In SPHENIX, alpha multiplies both the linear and quadratic
    viscous terms, so dK/dt = (gamma-1) * K * |theta| * M * alpha * (1 + beta_V * M).
    """
    K = sim["entropy_function"]
    theta = sim["VelocityDivergences"]
    mach = sim["compression_mach"]
    alpha = sim["ViscosityParameters"]

    compressing = theta < 0

    kdot = SimArray(np.zeros(len(sim)), units=K.units * theta.units)
    kdot[compressing] = (
        (GAMMA - 1.0)
        * K[compressing]
        * np.abs(theta[compressing])
        * mach[compressing]
        * alpha[compressing]
        * (1.0 + BETA_V * mach[compressing])
    )

    return kdot