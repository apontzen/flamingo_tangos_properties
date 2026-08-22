"""
Exact per-particle numerical entropy generation rates for the SPHENIX SPH
scheme (Borrow et al. 2022, MNRAS 511, 2367), as used in FLAMINGO.

The dissipation sums are evaluated directly from the paper's pairwise
equations, using pynbody's KDTree.pair_reduce to walk the neighbour pairs in
C++ and accumulate them onto particles.  The per-pair algebra it calls back
for is a numba kernel, one call per block of pairs.

The velocity divergence and curl needed by the Balsara switch are not done
that way: they are ordinary linear SPH sums, so they go through pynbody's own
KDTree.sph_divergence and sph_curl, which stay in threaded C++ throughout and
run some 30x faster than the equivalent pair walk.  See _div_curl.

Two mechanisms are implemented:

    artificial conduction   Eqns 25-27
    artificial viscosity    Eqns 14-20

Both are converted to dK/dt via the fact that the adiabatic part of the
equation of motion conserves K = P rho^-gamma identically in continuous time,
so every dissipative term in du/dt maps onto dK/dt as

    dK/dt = (K / u) * (du/dt)_diss                      (at fixed rho)

Sign conventions
----------------
Let d_ij = x_j - x_i, r = |d_ij|, dhat = d_ij / r.  With W decreasing in r,

    grad_i W(r, h_i) = |dW/dr|(r, h_i) * dhat

i.e. the gradient with respect to x_i points from i towards j.  This is the
convention under which Eqn 10 gives a repulsive pressure force, and is the one
used throughout here.

Note that pynbody's ``kernel.gradient`` returns the signed dW/dr, which is
negative inside the kernel, so it appears negated wherever |dW/dr| is wanted.

NOTE ON EQN 25.  Under the above convention rhat_ij . grad_i W_ij > 0, so
Eqn 25 as printed gives du_i/dt > 0 for u_i > u_j -- the hotter particle heats
further.  That is unphysical and anti-diffusive, so the sign is flipped here.
The form implemented,

    du_i/dt = sum_j m_j v_D,ij (u_j - u_i) G_ij
    G_ij    = f_ij |W'|(r, h_i) / rho_i  +  f_ji |W'|(r, h_j) / rho_j

has G_ij symmetric under i <-> j, so m_i du_i/dt + m_j du_j/dt = 0 pairwise
and total energy is conserved exactly.  This now holds by construction rather
than by arithmetic accident: the symmetric pair set visits each unordered pair
once and both ends are accumulated from that single visit, so an antisymmetric
contribution makes sum_i m_i du_i/dt = 0 an algebraic identity.
check_energy_conservation() verifies it anyway, as a check on the signs.

Also note Eqn 26 is printed with denominator (rho_j + rho_j); read as
(rho_i + rho_j), which is what is used here.

Grad-h terms
------------
The f_ij factors of Eqn 11 require dh/drho, which is not in the snapshot.
They are O(1) and are set to unity by default (FIXED_GRADH).  This is the only
approximation remaining in the conduction term.

Kernel
------
Wendland C2, which is what SWIFT uses for FLAMINGO, evaluated through
pynbody's own kernel object so that the normalisation cannot drift from the
one the library uses.

Two h conventions are in play.  Eqn 6 is written with a kernel smoothing
length h_k whose support is at 2 h_k, whereas the kernel_gamma quoted in the
paper (and used by SWIFT) refers to the eta = 1.2 definition, in which the
support radius is GAMMA_K * h.  They are reconciled by evaluating the kernel
with h_k = GAMMA_K * h / 2, so that the support sits at GAMMA_K * h.
GAMMA_K = 1.936492 is the Wendland C2 value for eta = 1.2.

pynbody's built-in operators take that h_k from the tree rather than as an
argument, so the two conventions have to be reconciled by what gets bound to
the tree: _pair_context binds h and does the conversion when it evaluates the
kernel itself, whereas _operator_context binds h_k directly.  validate_density
checks the identification end to end.

Neighbour completeness
----------------------
Every walk here binds h_k to the tree, so the pair set runs out to 2 h_k,
which is GAMMA_K * h -- the kernel support exactly.  Nothing inside the
support is missed and nothing outside it is visited.  The neighbour-count and
truncation-warning machinery this module used to carry is therefore gone:
there is nothing left to truncate.

Units
-----
Convert with the units object a snapshot array carries, never with
Unit(str(that object)).  str() rounds a unit's prefactor to three significant
figures, so the round trip yields a subtly different unit: a SWIFT length unit
prints as "3.09e+24 cm a" but is exactly Mpc a, and going via the string
inflates every length by a factor 1.0014.  An earlier version of this module
built its distance conversion that way, which put a 0.14 per cent error into
r/h and hence into every kernel evaluation here.

Periodicity
-----------
The displacements supplied by pair_reduce are already minimum-imaged, so pairs
straddling a periodic boundary are handled correctly.  (A previous version of
this module computed pos[j] - pos[i] directly while taking r from the tree,
which silently produced separations of order the box size for such pairs.)

Cost
----
The neighbour walk runs in C++ and the pair algebra in numba kernels that take
a whole block of pairs in one pass, so this runs comfortably on subvolumes of
a few 1e6 particles.  Each du/dt function performs one pair walk; the
viscosity additionally walks twice for the Balsara switch.

The pair algebra is not where the time goes.  Writing the same equations as
numpy array expressions -- the _numpy variants below -- costs 6 to 10 times
more overall, and 8 to 16 times more counting only the algebra itself,
because every intermediate becomes a freshly allocated array of the block
length and every particle property is gathered anew at each of the several
places it appears.

What is left divides roughly evenly between the neighbour walk, which pynbody
runs across several threads, and pair_reduce accumulating the contributions
back onto particles, which is numpy and single threaded.  Doing that scatter
here instead, in numba, would save about a third; it is left to pynbody so
that this module stays a straightforward use of the public API.  On 1e6
particles with 33 pairs each the whole conduction term takes about a second.

Required fields
---------------
    pos, vel, mass, rho, p, u, smooth,
    ViscosityParameters, DiffusionParameters
    VelocityDivergences        (validation only)
    Entropies                  (optional; else K = p / rho**gamma)
"""

import numpy as np
from numba import njit, prange

import pynbody
from pynbody.array import SimArray
from pynbody.kdtree import buffered_kernel
from pynbody.sph.kernels import WendlandC2Kernel

GAMMA = 5.0 / 3.0
GAMMA_K = 1.936492          # Wendland C2 kernel gamma at eta = 1.2

N_NEIGHBOURS = 65
# Nominal neighbour count for eta = 1.2 with Wendland C2.  Nothing here selects
# neighbours by count -- the smoothing lengths come from the snapshot and the
# gather is a fixed-radius ball -- so this only sizes pynbody's internal
# neighbour buffer, which grows on demand if it turns out to be too small.

BETA_V = 3.0                # Eqn 17, viscosity_beta
ALPHA_V_MAX = 2.0           # Eqn 23
FIXED_GRADH = 1.0           # f_ij, see module docstring

_BLOCKSIZE = 1 << 21
# Pairs handed to the numba kernels at a time.  Larger than pair_reduce's own
# default, because with a compiled kernel doing the algebra it is the walk and
# the scatter back onto particles that dominate, and both prefer long blocks.
# Worth perhaps 30 per cent over the default here; beyond this it flattens off
# and only the memory grows (48 bytes per pair, plus 16 for the two
# contributions).

INCLUDE_HUBBLE_FLOW = True
# Applied to the velocity difference in the divergence and curl estimates
# below, as it was in the original version of this module.  Note it is *not*
# applied to mu_ij in the dissipation terms, which is inconsistent; that
# inconsistency is inherited deliberately rather than silently changed, since
# fixing it would alter the physics.

_KERNEL = WendlandC2Kernel()

# Internal working units.  Chosen mutually consistent so that
# p = (gamma-1) * u * rho holds numerically.
_U_LEN = "kpc"
_U_MASS = "Msol"
_U_VEL = "kpc s**-1"
_U_RHO = "Msol kpc**-3"
_U_U = "kpc**2 s**-2"
_U_P = "Msol kpc**-1 s**-2"


# ---------------------------------------------------------------------------
# Snapshot access
# ---------------------------------------------------------------------------

def _arrays(sim):
    """Snapshot fields as plain float64 arrays in the internal unit system."""
    sim.physical_units()
    sim["pos"].convert_units(_U_LEN)
    sim["vel"].convert_units(_U_VEL)
    sim["mass"].convert_units(_U_MASS)
    sim["rho"].convert_units(_U_RHO)
    sim["p"].convert_units(_U_P)
    sim["u"].convert_units(_U_U)
    sim["smooth"].convert_units(_U_LEN)
    return dict(
        pos=sim['pos'],
        vel=sim['vel'],
        mass=sim['mass'],
        rho=sim['rho'],
        p=sim['p'],
        u=sim['u'],
        h=sim['smooth']
    )


def _pair_context(sim):
    """Prepare the tree for pair iteration.

    Returns the tree, the factor converting distances as reported by
    pair_blocks into the internal length unit, and the kernel smoothing length
    h_k in that unit.

    The snapshot's own smoothing lengths are bound to the tree, since the
    SPHENIX equations are defined with the h the simulation itself used.  Left
    to itself, pynbody would recompute h from a fixed neighbour count.

    What is bound is h_k, not h, exactly as in _operator_context: pair_blocks
    then emits pairs out to 2 h_k = GAMMA_K * h, which is the kernel support.
    Binding h instead would emit everything out to 2h, of which the outermost
    shell contributes identically zero -- about 10 per cent of the pairs, for
    nothing.
    """
    sim.build_tree()
    tree = sim.kdtree
    tree.set_kernel(_KERNEL)

    pos_units = sim["pos"].units
    h_tree = np.asarray(GAMMA_K * sim["smooth"].in_units(pos_units) / 2.0,
                        dtype=sim["pos"].dtype)
    tree.set_array_ref("smooth", h_tree)

    # Use the units object itself, not Unit(str(...)).  str() of a unit
    # rounds its prefactor to three significant figures, so the round trip
    # silently changes the unit: on a SWIFT snapshot pos.units prints as
    # "3.09e+24 cm a" but is exactly Mpc a, and the round trip inflates every
    # length by 1.0014.
    to_len = float(sim["pos"].units.ratio(_U_LEN, **sim.conversion_context()))

    h_k = GAMMA_K * np.asarray(sim["smooth"].in_units(_U_LEN),
                               dtype=np.float64) / 2.0

    return tree, to_len, h_k


def _operator_context(sim):
    """Prepare the tree for pynbody's built-in C++ smoothing operators.

    Those operators do not take a kernel smoothing length as an argument: they
    read it from the tree's ``smooth`` slot, evaluate W at r/h_k, and stop the
    neighbour gather at 2 h_k.  So h_k -- not the snapshot's h -- is what has
    to be bound here, which is the one difference from _pair_context above.
    With h_k = GAMMA_K * h / 2 the gather radius is exactly GAMMA_K * h, the
    kernel support, so nothing inside the support is dropped and nothing
    outside it is visited.

    mass and rho are bound too, because the operators form the volume element
    m_j / rho as a bare ratio of the stored numbers.  For that to be a volume
    in the same units as pos**3, rho has to be given in mass_units/pos_units**3
    whatever the snapshot happens to store it as.

    Returns the tree and the number of neighbours to declare (used only to
    size the internal neighbour buffer, which grows on demand anyway).
    """
    sim.build_tree()
    tree = sim.kdtree
    tree.set_kernel(_KERNEL)

    pos_units = sim["pos"].units
    dtype = sim["pos"].dtype

    h_k = SimArray(
        np.asarray(GAMMA_K * sim["smooth"].in_units(pos_units) / 2.0,
                   dtype=dtype),
        pos_units)
    tree.set_array_ref("smooth", h_k)

    tree.set_array_ref(
        "mass",
        SimArray(np.asarray(sim["mass"], dtype=dtype), sim["mass"].units))
    rho_units = sim["mass"].units / pos_units ** 3
    tree.set_array_ref(
        "rho",
        SimArray(np.asarray(sim["rho"].in_units(rho_units), dtype=dtype),
                 rho_units))

    return tree, N_NEIGHBOURS


# ---------------------------------------------------------------------------
# Gather-form operators (needed for the Balsara switch)
# ---------------------------------------------------------------------------

def _hubble(sim):
    """H(a) in 1/s, or zero if the Hubble flow is being left out."""
    if not INCLUDE_HUBBLE_FLOW:
        return 0.0
    return float(pynbody.analysis.cosmology.H(sim).in_units("s**-1"))


def _div_curl(sim):
    """
    SPH divergence and curl of the velocity field, in SWIFT's gather form

        div v_i  =  (1/rho_i) sum_j m_j (v_j - v_i) . grad_i W_ij
        curl v_i = -(1/rho_i) sum_j m_j (v_j - v_i) x grad_i W_ij

    Returned in 1/s.  Compare div against the snapshot's VelocityDivergences
    with validate_divergence().

    These are exactly what pynbody's sph_divergence and sph_curl compute, once
    they are asked for the mass-weighted average (weighting="mass", i.e. the
    volume element m_j / rho_i rather than m_j / rho_j).  Both therefore run
    threaded in C++, some 30x faster than the equivalent Python pair walk,
    which they were checked against particle by particle.

    Hubble flow
    -----------
    v here is the peculiar velocity, so the divergence of the full velocity
    field carries an extra H * (1/rho_i) sum_j m_j d_ij . grad_i W_ij.  That
    sum is a purely geometric quantity, independent of the velocity field, and
    equals 3 in the continuum, so it is added as 3H.  The discrete sum is a
    poor estimate of that (28 per cent RMS scatter on FLAMINGO, biased low),
    and using it instead moves div v away from the snapshot's own
    VelocityDivergences rather than towards them, which is consistent with
    SWIFT also adding the continuum value.  The curl needs no such term:
    d_ij x grad_i W_ij vanishes identically, the two being parallel.
    """
    tree, nn = _operator_context(sim)

    vel = SimArray(
        np.asarray(sim["vel"].in_units(_U_VEL), dtype=sim["pos"].dtype),
        _U_VEL)

    ctx = sim.conversion_context()
    div = np.asarray(
        tree.sph_divergence(vel, nsmooth=nn, weighting="mass")
        .in_units("s**-1", **ctx), dtype=np.float64)
    curl = np.asarray(
        tree.sph_curl(vel, nsmooth=nn, weighting="mass")
        .in_units("s**-1", **ctx), dtype=np.float64)

    return div + 3.0 * _hubble(sim), curl

def balsara(sim):
    """B_i, Eqn 20."""
    a = _arrays(sim)
    div, curl = _div_curl(sim)
    cs = np.sqrt(GAMMA * a["p"] / a["rho"])
    adiv = np.abs(div)
    return adiv / (adiv + np.linalg.norm(curl, axis=1) + 1e-4 * cs / a["h"])


# ---------------------------------------------------------------------------
# Dissipation terms
#
# Each comes in two forms.  The public one calls a numba kernel that runs the
# whole pair block in one pass; the _numpy one immediately below it is the
# same equations written as array expressions, kept as the readable statement
# of the physics and checked against by validate_du_dt().  The numba versions
# are around 25x faster on the pair algebra, which matters because there are
# tens of pairs per particle; see the Cost section of the module docstring.
# ---------------------------------------------------------------------------

@njit(cache=True, nogil=True, parallel=True)
def _conduction_pairs(i, j, dx, r, p, p_alpha_d, rho, u, vel, mass,
                      inv_h, grad_c, fixed_gradh, out_i, out_j):
    """Per-pair conduction contribution; see conduction_du_dt_numpy.

    inv_h[k] converts a tree-frame separation to r/h_k for particle k, and
    grad_c[k] is 105 / (16 pi h_k**4 rho_k), so that the magnitude of the
    kernel gradient over rho is grad_c * q * (1 - q/2)**3.

    Fills the two contributions pair_reduce expects, one for each end of the
    pair.  They are equal and opposite, conduction moving energy between the
    two rather than creating it.
    """
    for k in prange(i.shape[0]):
        ia = i[k]
        jb = j[k]
        rk = r[k]

        # (v_j - v_i) . dhat, with the 1/r of dhat applied at the end; the
        # ratio is dimensionless, so no unit conversion is needed
        dot = ((vel[jb, 0] - vel[ia, 0]) * dx[k, 0]
               + (vel[jb, 1] - vel[ia, 1]) * dx[k, 1]
               + (vel[jb, 2] - vel[ia, 2]) * dx[k, 2])
        v_vel = abs(dot) / rk

        pa = p[ia]
        pb = p[jb]
        alpha_ij = (p_alpha_d[ia] + p_alpha_d[jb]) / (pa + pb)   # Eqn 27
        v_press = np.sqrt(2.0 * abs(pa - pb) / (rho[ia] + rho[jb]))
        v_d = 0.5 * alpha_ij * (v_vel + v_press)                 # Eqn 26

        qa = rk * inv_h[ia]
        ta = 1.0 - 0.5 * qa
        ga = grad_c[ia] * qa * ta * ta * ta if ta > 0.0 else 0.0

        qb = rk * inv_h[jb]
        tb = 1.0 - 0.5 * qb
        gb = grad_c[jb] * qb * tb * tb * tb if tb > 0.0 else 0.0

        c = v_d * (u[jb] - u[ia]) * fixed_gradh * (ga + gb)
        out_i[k] = mass[jb] * c
        out_j[k] = -mass[ia] * c


@njit(cache=True, nogil=True, parallel=True)
def _viscosity_pairs(i, j, dx, r, rho, vel, cs, alpha_v, balsara_b, mass,
                     inv_h, grad_c, beta_v, fixed_gradh, out_i, out_j):
    """Per-pair viscous contribution; see viscosity_du_dt_numpy.

    Both ends gain, viscosity dissipating kinetic energy into heat rather
    than moving it about, so the two contributions have the same sign.
    """
    for k in prange(i.shape[0]):
        ia = i[k]
        jb = j[k]
        rk = r[k]

        dot = ((vel[jb, 0] - vel[ia, 0]) * dx[k, 0]
               + (vel[jb, 1] - vel[ia, 1]) * dx[k, 1]
               + (vel[jb, 2] - vel[ia, 2]) * dx[k, 2])
        mu = dot / rk
        if mu > 0.0:
            mu = 0.0                                             # Eqn 16

        v_sig = cs[ia] + cs[jb] - beta_v * mu                    # Eqn 17
        alpha_ij = 0.25 * (alpha_v[ia] + alpha_v[jb]) \
            * (balsara_b[ia] + balsara_b[jb])                    # Eqn 19
        zeta = -alpha_ij * mu * v_sig / (rho[ia] + rho[jb])      # Eqn 15

        qa = rk * inv_h[ia]
        ta = 1.0 - 0.5 * qa
        ga = grad_c[ia] * qa * ta * ta * ta if ta > 0.0 else 0.0

        qb = rk * inv_h[jb]
        tb = 1.0 - 0.5 * qb
        gb = grad_c[jb] * qb * tb * tb * tb if tb > 0.0 else 0.0

        c = -0.25 * zeta * mu * fixed_gradh * (ga + gb)          # Eqn 14
        out_i[k] = mass[jb] * c
        out_j[k] = mass[ia] * c


def _kernel_coefficients(h_k, to_len, per_rho=None):
    """inv_h and grad_c as used by the numba kernels.

    h_k is in the internal length unit and to_len converts a tree-frame
    separation into it, so inv_h = to_len / h_k turns r into q = r / h_k.
    grad_c collects everything in |dW/dr| = 105 q (1 - q/2)**3 / (16 pi h_k**4)
    that does not depend on r, optionally divided through by per_rho.
    """
    grad_c = 105.0 / (16.0 * np.pi * h_k ** 4)
    if per_rho is not None:
        grad_c = grad_c / per_rho
    return to_len / h_k, grad_c


def conduction_du_dt(sim):
    """
    (du/dt) from artificial conduction, Eqns 25-27, in kpc**2 s**-3.

    Sign-indefinite per particle; conserves total energy pairwise by
    construction.
    """
    a = _arrays(sim)
    tree, to_len, h_k = _pair_context(sim)
    alpha_d = np.asarray(sim["DiffusionParameters"], dtype=np.float64)

    inv_h, grad_c = _kernel_coefficients(h_k, to_len, per_rho=a["rho"])

    return tree.pair_reduce(
        buffered_kernel(_conduction_pairs,
                        a["p"], a["p"] * alpha_d, a["rho"], a["u"],
                        np.ascontiguousarray(a["vel"]), a["mass"],
                        inv_h, grad_c, FIXED_GRADH),
        mode="symmetric", blocksize=_BLOCKSIZE)


def viscosity_du_dt(sim):
    """
    (du/dt) from artificial viscosity, Eqns 14-20, in kpc**2 s**-3.

    Positive definite.  Uses the Balsara switch computed from this module's
    own curl estimate, so alpha_V,ij = (1/4)(alpha_i + alpha_j)(B_i + B_j)
    is complete rather than the B = 1 limit.
    """
    a = _arrays(sim)
    b = balsara(sim)                    # two tree walks, before the pair loop
    tree, to_len, h_k = _pair_context(sim)
    alpha_v = np.asarray(sim["ViscosityParameters"], dtype=np.float64)
    cs = np.sqrt(GAMMA * a["p"] / a["rho"])

    inv_h, grad_c = _kernel_coefficients(h_k, to_len)

    return tree.pair_reduce(
        buffered_kernel(_viscosity_pairs,
                        a["rho"], np.ascontiguousarray(a["vel"]), cs, alpha_v,
                        b, a["mass"], inv_h, grad_c, BETA_V, FIXED_GRADH),
        mode="symmetric", blocksize=_BLOCKSIZE)


def conduction_du_dt_numpy(sim):
    """
    conduction_du_dt as array expressions.  Slower; the readable statement of
    Eqns 25-27, and what validate_du_dt() checks the numba kernel against.
    """
    a = _arrays(sim)
    tree, to_len, h_k = _pair_context(sim)
    mass, rho, p, u, vel = a["mass"], a["rho"], a["p"], a["u"], a["vel"]

    alpha_d = np.asarray(sim["DiffusionParameters"], dtype=np.float64)

    def pair(i, j, dx, r):
        r_len = r * to_len
        dhat = dx / r[:, None]          # a ratio, so needs no unit conversion

        # Eqn 27: pressure-weighted conduction coefficient
        alpha_ij = ((p[i] * alpha_d[i] + p[j] * alpha_d[j])
                    / (p[i] + p[j]))

        # Eqn 26: conduction speed.  Denominator read as rho_i + rho_j.
        v_vel = np.abs(np.einsum("kl,kl->k", vel[j] - vel[i], dhat))
        v_press = np.sqrt(2.0 * np.abs(p[i] - p[j]) / (rho[i] + rho[j]))
        v_d = 0.5 * alpha_ij * (v_vel + v_press)

        # symmetric geometric factor; see the sign note in the module docstring
        g_sym = FIXED_GRADH * (-_KERNEL.gradient(r_len, h_k[i]) / rho[i]
                               - _KERNEL.gradient(r_len, h_k[j]) / rho[j])

        contrib = v_d * (u[j] - u[i]) * g_sym
        return mass[j] * contrib, -mass[i] * contrib

    return tree.pair_reduce(pair, mode="symmetric")


def viscosity_du_dt_numpy(sim):
    """
    viscosity_du_dt as array expressions.  Slower; the readable statement of
    Eqns 14-20, and what validate_du_dt() checks the numba kernel against.
    """
    a = _arrays(sim)
    b = balsara(sim)                    # two tree walks, before the pair loop
    tree, to_len, h_k = _pair_context(sim)
    mass, rho, vel = a["mass"], a["rho"], a["vel"]

    alpha_v = np.asarray(sim["ViscosityParameters"], dtype=np.float64)
    cs = np.sqrt(GAMMA * a["p"] / a["rho"])

    def pair(i, j, dx, r):
        r_len = r * to_len
        dhat = dx / r[:, None]

        # Eqn 16: only converging pairs dissipate
        mu = np.minimum(np.einsum("kl,kl->k", vel[j] - vel[i], dhat), 0.0)

        v_sig = cs[i] + cs[j] - BETA_V * mu                     # Eqn 17
        alpha_ij = 0.25 * (alpha_v[i] + alpha_v[j]) * (b[i] + b[j])  # Eqn 19
        zeta = -alpha_ij * mu * v_sig / (rho[i] + rho[j])       # Eqn 15

        g_sym = FIXED_GRADH * (-_KERNEL.gradient(r_len, h_k[i])
                               - _KERNEL.gradient(r_len, h_k[j]))

        # Eqn 14: du_i/dt = -(1/2) sum_j m_j zeta_ij v_ij . [grad terms]
        contrib = -0.25 * zeta * mu * g_sym
        return mass[j] * contrib, mass[i] * contrib

    return tree.pair_reduce(pair, mode="symmetric")


# ---------------------------------------------------------------------------
# Derived arrays
# ---------------------------------------------------------------------------

def _entropy_function(sim):
    if "Entropies" in sim.loadable_keys():
        return sim["Entropies"]
    return sim["p"] / sim["rho"] ** GAMMA


def _to_kdot(sim, du_dt):
    """dK/dt = (K / u) du/dt, with du_dt a bare array in kpc**2 s**-3."""
    k = _entropy_function(sim)
    u = np.asarray(sim["u"].in_units(_U_U), dtype=np.float64)
    return SimArray(np.asarray(k) * du_dt / u,
                    units=k.units / pynbody.units.Unit("s"))


@pynbody.snapshot.simsnap.SimSnap.stable_derived_array
def conduction_entropy_rate(sim):
    """dK/dt from artificial conduction. Sign-indefinite; see module docstring."""
    return _to_kdot(sim, conduction_du_dt(sim))


@pynbody.snapshot.simsnap.SimSnap.stable_derived_array
def viscous_entropy_rate(sim):
    """dK/dt from artificial viscosity. Positive definite."""
    return _to_kdot(sim, viscosity_du_dt(sim))


@pynbody.snapshot.simsnap.SimSnap.stable_derived_array
def balsara_switch(sim):
    """B_i, Eqn 20, from this module's own curl estimate."""
    return SimArray(balsara(sim), units="1")


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def check_energy_conservation(sim):
    """
    sum_i m_i (du_i/dt)_cond should vanish to roundoff.

    On the symmetric pair set this is an algebraic identity rather than a
    numerical coincidence, so it now mainly checks that the two contributions
    _conduction_pairs returns really are equal and opposite.  Returns the
    residual normalised by the sum of
    |m_i du_i/dt|, so a correct implementation gives ~1e-16 or better.
    """
    a = _arrays(sim)
    terms = a["mass"] * conduction_du_dt(sim)
    return float(np.abs(terms.sum()) / np.abs(terms).sum())


def check_positivity(sim):
    """
    Fraction of particles with (du/dt)_visc < 0.  Should be exactly zero:
    viscous dissipation is positive definite.
    """
    return float(np.mean(viscosity_du_dt(sim) < 0.0))


def validate_density(sim):
    """
    Recompute rho = sum_j m_j W(r, h_i) and compare to the snapshot value.
    Returns the fractional residual per particle.

    This tests the kernel normalisation and, in particular, the h-convention
    identification h_k = GAMMA_K * h / 2 described in the module docstring.
    A systematic offset of a few per cent points at that identification.
    """
    a = _arrays(sim)
    tree, to_len, h_k = _pair_context(sim)
    mass = a["mass"]

    rho = tree.pair_reduce(
        lambda i, j, dx, r: mass[j] * _KERNEL.value(r * to_len, h_k[i]),
        mode="gather")
    rho += mass * _KERNEL.value(np.zeros(len(sim)), h_k)  # self-contribution

    return rho / a["rho"] - 1.0

def validate_du_dt(sim):
    """
    Compare the numba dissipation kernels against the array-expression forms.

    Returns (conduction_err, viscosity_err), each the deviation normalised by
    the RMS of the numpy result.  The two evaluate the same expressions with
    the factors grouped differently -- the numba kernels hoist everything
    r-independent into per-particle coefficients -- so the deviations should
    sit at a few times the float64 epsilon.

    Only worth running on a subvolume: the numpy forms are the slow path.
    """
    def dev(fast, slow):
        return (fast - slow) / np.sqrt(np.mean(slow ** 2))

    return (dev(conduction_du_dt(sim), conduction_du_dt_numpy(sim)),
            dev(viscosity_du_dt(sim), viscosity_du_dt_numpy(sim)))


def validate_divergence(sim):
    """
    Compare the recomputed div(v) against the snapshot's VelocityDivergences.
    """
    div, _ = _div_curl(sim)
    stored = np.asarray(sim["VelocityDivergences"].in_units("s**-1"),
                        dtype=np.float64)
    return div, stored
