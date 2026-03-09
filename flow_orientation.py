import argparse
import math
import pathlib
import numpy as np
import pynbody
import matplotlib.pyplot as plt

_SUPERSCRIPTS = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")

def _sci(value):
    """Format a number as e.g. 3.2 × 10¹³."""
    exp = int(math.floor(math.log10(abs(float(value)))))
    mantissa = float(value) / 10**exp
    return f"{mantissa:.2g} × 10{str(exp).translate(_SUPERSCRIPTS)}"

def _sci_latex(value):
    """Format a number as LaTeX e.g. 3.2 \\times 10^{13}."""
    exp = int(math.floor(math.log10(abs(float(value)))))
    mantissa = float(value) / 10**exp
    return rf"{mantissa:.2g} \times 10^{{{exp}}}"

def fit_quadratic_form(r, q):
    """Fit a symmetric 3x3 matrix Q such that q ≈ r @ Q @ r for each row of r.

    Parameters
    ----------
    r : (N, 3) array  — positions on the unit sphere
    q : (N,) array    — scalar quantity at each position

    Returns
    -------
    Q : (3, 3) ndarray — best-fit symmetric quadratic form
    """
    r = np.asarray(r)
    q = np.asarray(q)
    x, y, z = r[:, 0], r[:, 1], r[:, 2]
    # Design matrix for the 6 independent entries of a symmetric matrix:
    # Q_00 x² + Q_11 y² + Q_22 z² + 2Q_01 xy + 2Q_02 xz + 2Q_12 yz
    A = np.column_stack([x**2, y**2, z**2, 2*x*y, 2*x*z, 2*y*z])
    coeffs, _, _, _ = np.linalg.lstsq(A, q, rcond=None)
    Q = np.array([
        [coeffs[0], coeffs[3], coeffs[4]],
        [coeffs[3], coeffs[1], coeffs[5]],
        [coeffs[4], coeffs[5], coeffs[2]],
    ])
    return Q

def get_gas_flow_quadratic_form(f: pynbody.snapshot.SimSnap, radius):
    filt = pynbody.filt.Annulus(radius*0.9, radius*1.1)
    sub = f.gas[filt]
    with sub.immediate_mode:
        r = sub['pos']/sub['r'][:, np.newaxis]
        vr = sub['vr']
        Q = fit_quadratic_form(r, vr)
    return Q

def gas_flow_alignment(f: pynbody.snapshot.SimSnap, radius):
   
    Q = get_gas_flow_quadratic_form(f, radius)
    eigvals, eigvecs = np.linalg.eigh(Q)

    # want to align such that y axis is max, and x axis is min eigval
    # eigh guarantees eigvals are in ascending order, so this is just:
    z_vec = eigvecs[:, 1] # intermediate direction
    y_vec = eigvecs[:, 2] # outflow direction
    x_vec = eigvecs[:, 0] # inflow direction

    # now we need a rotation matrix such that x_vec maps to (1,0,0) 
    # and y_vec maps to (0,1,0)

    R = np.column_stack([x_vec, y_vec, z_vec]).T
    return f.rotate(R)




def process_halo(halo, number, label):
    print(f"Halo {number}")
    f = halo.ancestor
    
    with pynbody.analysis.halo.center(halo):
        try:
            virial_radius = pynbody.analysis.halo.virial_radius(f, overden=200)
        except ValueError:
            return
        print(f"  Virial radius: {_sci(virial_radius)} kpc")
        virial_mass = f['mass'][f['r'] < virial_radius].sum()
        print(f"  Virial mass:   {_sci(virial_mass)} M☉")
        plt.clf()
        with gas_flow_alignment(f, virial_radius):
            pynbody.plot.sph.velocity_image(f.gas, qty='Entropies', width=virial_radius*4,
                                            key_length="500 km s**-1", vector_scale="5000 km s**-1",
                                            cmap='magma', vmin=1e2/2, vmax=1e4/2)

        ax = plt.gca()
        for scale in [0.5, 1.0, 2.0]:
            ax.add_patch(plt.Circle((0, 0), scale * virial_radius, fill=False,
                                    color='white', linestyle='--', linewidth=1))
        ax.set_title(f"{label}  — id $= {number}$ — $M_{{200}} = {_sci_latex(virial_mass)}\\ M_{{\\odot}}$")

        folder_name : pathlib.Path = f.filename.parent
        plt.savefig(folder_name / f"halo_{number}_portrait.png",
                    bbox_inches='tight', dpi=144)
        
def make_test(N=100_000):
    f = pynbody.new(gas=N)
    f.gas['pos'] = np.random.normal(size=(N, 3))
    flow_matrix = np.array([[1, 0, 0], [0, -2, 0], [0, 0, 0]])
    r = f.gas['pos']
    # Random rotation matrix from Haar measure via QR decomposition
    haar, _ = np.linalg.qr(np.random.normal(size=(3, 3)))
    flow_matrix_rotated = haar @ flow_matrix @ haar.T

    f.gas['vr'] = np.einsum('ni,ij,nj->n', r, flow_matrix_rotated, r)

    print("True flow matrix (rotated):")
    print(flow_matrix_rotated)
    print("True rotation matrix:")
    print(haar)
    print("Original flow matrix:")
    print(flow_matrix)
    with gas_flow_alignment(f, radius=1):
        Q = get_gas_flow_quadratic_form(f, radius=1)
    print("Fitted flow matrix after rotation:")
    print(Q)
    


def process_snapshot(filename, label):
    f = pynbody.load(filename)
    f.physical_units()
    pynbody.analysis.cosmology.add_hubble(f)
    h = f.halos()
    h.load_all()
    hi = 1
    for hi in range(100,1000,50):
        process_halo(h[hi], hi, label)

def main():
    parser = argparse.ArgumentParser(description="Process a flamingo snapshot.")
    parser.add_argument("snapshot", help="Path to the flamingo snapshot file")
    parser.add_argument("label", help="Simulation label shown in the plot title")
    args = parser.parse_args()
    process_snapshot(args.snapshot, args.label)


if __name__ == "__main__":
    main()
