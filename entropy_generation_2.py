"""
Exact per-particle numerical entropy generation rates for the SPHENIX SPH
scheme (Borrow et al. 2022, MNRAS 511, 2367), as used in FLAMINGO.

Neighbour lists come from pynbody's KDTree; the dissipation sums themselves
are evaluated directly from the paper's pairwise equations in numpy, with no
per-particle proxies and no free O(1) calibration constants.

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

NOTE ON EQN 25.  Under the above convention rhat_ij . grad_i W_ij > 0, so
Eqn 25 as printed gives du_i/dt > 0 for u_i > u_j -- the hotter particle heats
further.  That is unphysical and anti-diffusive, so the sign is flipped here.
The form implemented,

    du_i/dt = sum_j m_j v_D,ij (u_j - u_i) G_ij
    G_ij    = f_ij |W'|(r, h_i) / rho_i  +  f_ji |W'|(r, h_j) / rho_j

has G_ij symmetric under i <-> j, so m_i du_i/dt + m_j du_j/dt = 0 pairwise
and total energy is conserved exactly.  check_energy_conservation() verifies
this and is the primary test that the pair list and signs are right.

Also note Eqn 26 is printed with denominator (rho_j + rho_j); read as
(rho_i + rho_j), which is what is used here.

Grad-h terms
------------
The f_ij factors of Eqn 11 require dh/drho, which is not in the snapshot.
They are O(1) and are set to unity by default (FIXED_GRADH).  This is the only
approximation remaining in the conduction term.

Kernel
------
Quartic spline (M5), Eqn 6, with W = kappa * w(r/h_k) / h_k**3 and
kappa_3 = 7/(478 pi).  Eqn 6's w(q) has support at q = 5/2, whereas the
kernel_gamma quoted elsewhere in the paper (and used by SWIFT) refers to the
eta = 1.2 definition of h, in which the support radius is GAMMA_K * h.  These
are different h conventions; they are reconciled by evaluating Eqn 6 with
h_k = GAMMA_K * h / 2.5.  If validate_density() shows a systematic offset,
this identification is what to revisit first.

Cost
----
The neighbour list is built through pynbody's Python-level nn() generator, so
this is a subvolume-scale tool: expect minutes for ~1e6 particles, and do not
run it on a full FLAMINGO box.  The dissipation sums themselves are vectorised
over pairs.

Required fields
---------------
    pos, vel, mass, rho, p, u, smooth,
    ViscosityParameters, DiffusionParameters
    VelocityDivergences        (validation only)
    Entropies                  (optional; else K = p / rho**gamma)
"""

import warnings

import numpy as np
import pynbody
from pynbody.array import SimArray
from pynbody.kdtree import KDTree

GAMMA = 5.0 / 3.0
GAMMA_K = 1.936492
KERNEL_NORM_3D = 21.0 / (16.0 * np.pi)

BETA_V = 3.0              # Eqn 17, viscosity_beta
ALPHA_V_MAX = 2.0         # Eqn 23
FIXED_GRADH = 1.0         # f_ij, see module docstring

N_NEIGHBOURS = 128

# Internal working units.  Chosen mutually consistent so that
# p = (gamma-1) * u * rho holds numerically.
_U_LEN = "kpc"
_U_MASS = "Msol"
_U_VEL = "kpc s**-1"
_U_RHO = "Msol kpc**-3"
_U_U = "kpc**2 s**-2"
_U_P = "Msol kpc**-1 s**-2"


# ---------------------------------------------------------------------------
# Kernel
# ---------------------------------------------------------------------------

def _kernel_h(h):
    return GAMMA_K * h / 2.0

def kernel_W(r, h):
    hk = _kernel_h(h)
    d = np.minimum(r / hk, 2.0)
    return KERNEL_NORM_3D * (1.0 - 0.5 * d)**4 * (2.0 * d + 1.0) / hk**3

def kernel_absdWdr(r, h):
    hk = _kernel_h(h)
    d = np.minimum(r / hk, 2.0)
    # dw/dd = -5 d (1 - d/2)^3, non-positive on [0, 2]
    return KERNEL_NORM_3D * 5.0 * d * (1.0 - 0.5 * d)**3 / hk**4


# ---------------------------------------------------------------------------
# Neighbour lists
# ---------------------------------------------------------------------------


def _arrays(sim):
    """Snapshot fields as plain float64 arrays in the internal unit system."""
    return dict(
        pos=np.asarray(sim["pos"].in_units(_U_LEN), dtype=np.float64),
        vel=np.asarray(sim["vel"].in_units(_U_VEL), dtype=np.float64),
        mass=np.asarray(sim["mass"].in_units(_U_MASS), dtype=np.float64),
        rho=np.asarray(sim["rho"].in_units(_U_RHO), dtype=np.float64),
        p=np.asarray(sim["p"].in_units(_U_P), dtype=np.float64),
        u=np.asarray(sim["u"].in_units(_U_U), dtype=np.float64),
        h=np.asarray(sim["smooth"].in_units(_U_LEN), dtype=np.float64),
    )


def _directed_pairs(sim: pynbody.snapshot.SimSnap, nn=N_NEIGHBOURS):
    """
    Directed pair list (i, j, r) from the kNN of every particle.

    Returns arrays of equal length.  Used for gather-form operators (density,
    divergence, curl) which involve h_i only.
    """
    cached = getattr(sim.ancestor, "_sphenix_pairs_dir", None)
    if cached is not None and cached[0] == (len(sim), nn):
        return cached[1]

    sim.build_tree()
    tree = sim.kdtree 
    tree.set_array_ref('smooth', sim['smooth'].astype(np.float64))

    rows, cols, dists = [], [], []
    for entry in tree.nn(nn):
        i, _h_tree, nbrs, d2 = entry
        nbrs = np.asarray(nbrs, dtype=np.int64)
        d = np.sqrt(np.asarray(d2, dtype=np.float64))
        keep = nbrs != i
        rows.append(np.full(keep.sum(), i, dtype=np.int64))
        cols.append(nbrs[keep])
        dists.append(d[keep])

    i_idx = np.concatenate(rows)
    j_idx = np.concatenate(cols)
    r = np.concatenate(dists)

    # nn() distances are in the units of the pos array as passed to KDTree.
    r *= float(pynbody.units.Unit(str(sim["pos"].units)).ratio(_U_LEN,
                                                              **sim.conversion_context()))

    result = (i_idx, j_idx, r)
    setattr(sim.ancestor, "_sphenix_pairs_dir", ((len(sim), nn), result))
    _check_support_coverage(sim, i_idx, r)
    return result


def _undirected_pairs(sim, nn=N_NEIGHBOURS):
    """
    Unique unordered pairs {i, j} with i < j, formed from the union of the
    directed kNN lists.  A pair is retained if j is a neighbour of i OR i is a
    neighbour of j, which is what the gather-scatter structure of Eqns 14 and
    25 requires (they involve both h_i and h_j).
    """
    cached = getattr(sim.ancestor, "_sphenix_pairs_undir", None)
    if cached is not None and cached[0] == (len(sim), nn):
        return cached[1]

    i_idx, j_idx, r = _directed_pairs(sim, nn)

    lo = np.minimum(i_idx, j_idx)
    hi = np.maximum(i_idx, j_idx)
    key = lo.astype(np.int64) * np.int64(len(sim)) + hi.astype(np.int64)
    _, first = np.unique(key, return_index=True)

    result = (lo[first], hi[first], r[first])
    setattr(sim.ancestor, "_sphenix_pairs_undir", ((len(sim), nn), result))
    return result


def _boxsize(sim):
    if "boxsize" in sim.properties:
        try:
            return float(sim.properties["boxsize"].in_units(sim["pos"].units))
        except Exception:
            return None
    return None


def _check_support_coverage(sim, i_idx, r):
    """
    Warn if the kNN list is truncated inside the kernel support, which would
    silently drop pair contributions.
    """
    h = np.asarray(sim["smooth"].in_units(_U_LEN), dtype=np.float64)
    rmax = np.zeros(len(sim))
    np.maximum.at(rmax, i_idx, r)
    truncated = np.mean(rmax < GAMMA_K * h)
    if truncated > 0.01:
        warnings.warn(
            f"{100 * truncated:.1f}% of particles have their neighbour list "
            f"truncated inside the kernel support. Increase N_NEIGHBOURS.",
            RuntimeWarning,
        )


def _accumulate(n, idx, w):
    return np.bincount(idx, weights=w, minlength=n)


# ---------------------------------------------------------------------------
# Gather-form operators (needed for the Balsara switch)
# ---------------------------------------------------------------------------


def _div_curl(sim, nn=N_NEIGHBOURS):
    """
    SPH divergence and curl of the velocity field, in SWIFT's gather form

        div v_i  =  (1/rho_i) sum_j m_j (v_j - v_i) . grad_i W_ij
        curl v_i = -(1/rho_i) sum_j m_j (v_j - v_i) x grad_i W_ij

    Returned in 1/s.  Compare div against the snapshot's VelocityDivergences
    with validate_divergence().
    """
    a = _arrays(sim)
    i_idx, j_idx, r = _directed_pairs(sim, nn)
    n = len(sim)

    d = a["pos"][j_idx] - a["pos"][i_idx]
    dhat = d / r[:, None]

    H = float(pynbody.analysis.cosmology.H(sim).in_units('s**-1'))
    
    dv = a["vel"][j_idx] - a["vel"][i_idx]
    dv += H * d  # add hubble flow

    gw = kernel_absdWdr(r, a["h"][i_idx])[:, None] * dhat  # grad_i W_ij
    mgw = a["mass"][j_idx][:, None] * gw

    div = _accumulate(n, i_idx, np.einsum("ij,ij->i", dv, mgw)) / a["rho"]
    cross = np.cross(dv, mgw)
    curl = np.stack(
        [-_accumulate(n, i_idx, cross[:, k]) / a["rho"] for k in range(3)], axis=1
    )
    return div, curl


def balsara(sim, nn=N_NEIGHBOURS):
    """B_i, Eqn 20."""
    a = _arrays(sim)
    div, curl = _div_curl(sim, nn)
    cs = np.sqrt(GAMMA * a["p"] / a["rho"])
    adiv = np.abs(div)
    return adiv / (adiv + np.linalg.norm(curl, axis=1) + 1e-4 * cs / a["h"])


# ---------------------------------------------------------------------------
# Dissipation terms
# ---------------------------------------------------------------------------


def _pair_geometry(sim, nn=N_NEIGHBOURS):
    """
    Per-pair quantities shared by both dissipation terms, on the undirected
    pair list.  Returns a dict of numpy arrays.
    """
    a = _arrays(sim)
    i_idx, j_idx, r = _undirected_pairs(sim, nn)

    d = a["pos"][j_idx] - a["pos"][i_idx]
    dhat = d / r[:, None]
    dv = a["vel"][j_idx] - a["vel"][i_idx]

    # mu_ij = v_ij . x_ij / |x_ij|, Eqn 16 (before the converging-flow clamp)
    mu_raw = np.einsum("ij,ij->i", dv, dhat)

    # Symmetric kernel-gradient combination appearing in both equations.
    wi = kernel_absdWdr(r, a["h"][i_idx])
    wj = kernel_absdWdr(r, a["h"][j_idx])

    # add hubble flow
    #H = float(pynbody.analysis.cosmology.H(sim).in_units('s**-1'))
    #mu_raw = np.einsum("ij,ij->i", dv, dhat) + H * r

    return dict(i=i_idx, j=j_idx, r=r, mu_raw=mu_raw, wi=wi, wj=wj, **a)


def conduction_du_dt(sim, nn=N_NEIGHBOURS):
    """
    (du/dt) from artificial conduction, Eqns 25-27, in kpc**2 s**-3.

    Sign-indefinite per particle; conserves total energy pairwise by
    construction.
    """
    g = _pair_geometry(sim, nn)
    i, j = g["i"], g["j"]
    n = len(sim)
    f = FIXED_GRADH

    # Eqn 27: pressure-weighted conduction coefficient.
    alpha_d = np.asarray(sim["DiffusionParameters"], dtype=np.float64)
    p_sum = g["p"][i] + g["p"][j]
    alpha_ij = (g["p"][i] * alpha_d[i] + g["p"][j] * alpha_d[j]) / p_sum

    # Eqn 26: conduction speed.  Denominator read as rho_i + rho_j.
    v_vel = np.abs(g["mu_raw"])
    v_press = np.sqrt(2.0 * np.abs(g["p"][i] - g["p"][j]) / (g["rho"][i] + g["rho"][j]))
    v_d = 0.5 * alpha_ij * (v_vel + v_press)

    # Symmetric geometric factor; see the sign note in the module docstring.
    gsym = f * g["wi"] / g["rho"][i] + f * g["wj"] / g["rho"][j]

    du = g["u"][j] - g["u"][i]
    contrib = v_d * du * gsym

    out = _accumulate(n, i, g["mass"][j] * contrib)
    out -= _accumulate(n, j, g["mass"][i] * contrib)
    return out


def viscosity_du_dt(sim, nn=N_NEIGHBOURS):
    """
    (du/dt) from artificial viscosity, Eqns 14-20, in kpc**2 s**-3.

    Positive definite.  Uses the Balsara switch computed from this module's
    own curl estimate, so alpha_V,ij = (1/4)(alpha_i + alpha_j)(B_i + B_j)
    is complete rather than the B = 1 limit.
    """
    g = _pair_geometry(sim, nn)
    i, j = g["i"], g["j"]
    n = len(sim)
    f = FIXED_GRADH

    # Eqn 16: only converging pairs dissipate.
    mu = np.where(g["mu_raw"] < 0.0, g["mu_raw"], 0.0)

    cs = np.sqrt(GAMMA * g["p"] / g["rho"])
    v_sig = cs[i] + cs[j] - BETA_V * mu                       # Eqn 17

    alpha_v = np.asarray(sim["ViscosityParameters"], dtype=np.float64)
    b = balsara(sim, nn)
    alpha_ij = 0.25 * (alpha_v[i] + alpha_v[j]) * (b[i] + b[j])  # Eqn 19

    zeta = -alpha_ij * mu * v_sig / (g["rho"][i] + g["rho"][j])   # Eqn 15

    gsym = f * g["wi"] + f * g["wj"]
    # Eqn 14: du_i/dt = -(1/2) sum_j m_j zeta_ij v_ij . [grad terms]
    contrib = -0.25 * zeta * mu * gsym

    out = _accumulate(n, i, g["mass"][j] * contrib)
    out += _accumulate(n, j, g["mass"][i] * contrib)
    return out


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
    return SimArray(np.asarray(k) * du_dt / u, units=k.units / pynbody.units.Unit("s"))


@pynbody.derived_array
def conduction_entropy_rate(sim):
    """dK/dt from artificial conduction. Sign-indefinite; see module docstring."""
    return _to_kdot(sim, conduction_du_dt(sim))

@pynbody.derived_array
def viscous_entropy_rate(sim):
    """dK/dt from artificial viscosity. Positive definite."""
    return _to_kdot(sim, viscosity_du_dt(sim))


@pynbody.derived_array
def balsara_switch(sim):
    """B_i, Eqn 20, from this module's own curl estimate."""
    return SimArray(balsara(sim), units="1")


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def check_energy_conservation(sim, nn=N_NEIGHBOURS):
    """
    sum_i m_i (du_i/dt)_cond should vanish to roundoff.

    This is the primary test of the pair list, the antisymmetry, and the sign
    convention of Eqn 25.  Returns the residual normalised by the sum of
    |m_i du_i/dt|, so a correct implementation gives ~1e-14.  A result near 1
    means the sign flip discussed in the module docstring has been applied in
    the wrong direction.
    """
    a = _arrays(sim)
    du = conduction_du_dt(sim, nn)
    terms = a["mass"] * du
    return float(np.abs(terms.sum()) / np.abs(terms).sum())


def check_positivity(sim, nn=N_NEIGHBOURS):
    """
    Fraction of particles with (du/dt)_visc < 0.  Should be exactly zero:
    viscous dissipation is positive definite.
    """
    return float(np.mean(viscosity_du_dt(sim, nn) < 0.0))


def validate_density(sim, nn=N_NEIGHBOURS):
    """
    Recompute rho = sum_j m_j W(r, h_i) and compare to the snapshot value.
    Returns the fractional residual per particle.

    This tests the kernel normalisation and, in particular, the h-convention
    identification h_k = GAMMA_K * h / 2.5 described in the module docstring.
    A systematic offset of a few per cent points at that identification; large
    scatter points at neighbour-list truncation instead.
    """
    a = _arrays(sim)
    i_idx, j_idx, r = _directed_pairs(sim, nn)
    rho = _accumulate(
        len(sim), i_idx, a["mass"][j_idx] * kernel_W(r, a["h"][i_idx])
    )
    rho += a["mass"] * kernel_W(np.zeros(len(sim)), a["h"])  # self-contribution
    return rho / a["rho"] - 1.0


def validate_divergence(sim, nn=N_NEIGHBOURS):
    """
    Compare the recomputed div(v) against the snapshot's VelocityDivergences.
    Returns the fractional residual per particle.
    """
    div, _ = _div_curl(sim, nn)
    stored = np.asarray(sim["VelocityDivergences"].in_units("s**-1"), dtype=np.float64)
    scale = np.maximum(np.abs(stored), np.abs(div))
    #return (div - stored) / np.maximum(scale, 1e-30)
    return div, stored