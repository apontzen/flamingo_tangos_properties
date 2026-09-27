"""Export per-halo gas pressure profiles (physical radial bins) for L0200N0720 fiducial, z=0.4.

Run from the repository root:

    python pressure_profiles_L0200N0720_z0.4/export_pressure_profiles.py

See README.md in this folder for the meaning of every field in the output JSON.
"""

import datetime
import json
import os

import numpy as np
import pynbody
import tangos as db

from flamingo_tangos import FlamingoDensityProfileAbsolute

DB_PATH = 'data.db'
TIMESTEP = '%720%FIDUCIAL%/%4.hdf5'
OUTPUT = os.path.join(os.path.dirname(__file__), 'L0200N0720_FIDUCIAL_z0.4_gas_pressure_profiles.json')

#: Thermal pressure, database units (Msol km^2 s^-2 kpc^-3) -> erg cm^-3.
PRESSURE_TO_ERG_CM3 = float(pynbody.units.Unit('Msol km^2 s^-2 kpc^-3').in_units('erg cm^-3'))

#: FLAMINGO fiducial D3A (DES-Y3) cosmology, Schaye et al. 2023 Tables 3-4.
COSMOLOGY = {
    "name": "FLAMINGO fiducial D3A (DES-Y3)",
    "reference": "Schaye et al. 2023 (arXiv:2306.04024), Tables 3-4",
    "h": 0.681, "Omega_m": 0.306, "Omega_Lambda": 0.694, "Omega_b": 0.0486,
    "sum_m_nu_eV": 0.06, "A_s": 2.099e-9, "n_s": 0.967, "sigma_8": 0.807,
}


def _nan_to_none(values):
    return [None if np.isnan(v) else float(v) for v in values]


def main():
    db.init_db(DB_PATH)
    ts = db.get_timestep(TIMESTEP)

    pressure, M200m, r200m, halo_number, finder_id, centre = ts.calculate_all(
        'gas_p', 'M200m()', 'r200m', 'halo_number()', 'finder_id()', 'shrink_center')
    order = np.argsort(halo_number)

    # Radial shells exactly as constructed by the profile calculation (pynbody type='log')
    r_min, r_max, n_bins = (FlamingoDensityProfileAbsolute._min_rad,
                            FlamingoDensityProfileAbsolute._max_rad,
                            FlamingoDensityProfileAbsolute._nbins)
    assert pressure.shape[1] == n_bins
    edges = np.logspace(np.log10(r_min), np.log10(r_max), n_bins + 1)
    centres = np.sqrt(edges[:-1] * edges[1:])

    halos = []
    for i in order:
        halos.append({
            'halo_number': int(halo_number[i]),
            'hbt_row_index': int(finder_id[i]),
            'M200m_Msol': float(M200m[i]),
            'log10_M200m_Msol': float(np.log10(M200m[i])),
            'r200m_kpc': float(r200m[i]),
            'centre_kpc': [float(c) for c in centre[i]],
            'pressure_erg_cm3': _nan_to_none(pressure[i] * PRESSURE_TO_ERG_CM3),
        })

    document = {
        'description': "Spherically averaged, volume-weighted thermal gas pressure profiles of "
                       "individual haloes in FLAMINGO L0200N0720_HYDRO_FIDUCIAL at z = 0.4, "
                       "in fixed physical radial shells. One profile per halo; no stacking. "
                       "See README.md for full documentation.",
        'generated_by': "pressure_profiles_L0200N0720_z0.4/export_pressure_profiles.py",
        'generated_on': datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d'),
        'simulation': ts.simulation.basename,
        'snapshot': ts.extension,
        'redshift': round(float(ts.redshift), 6),
        'scale_factor': round(1.0 / (1.0 + float(ts.redshift)), 6),
        'cosmology': COSMOLOGY,
        'units': {
            'pressure': 'erg cm^-3 (physical)',
            'radius': 'kpc (physical)',
            'mass': 'Msol (no h factors)',
            'centre': 'kpc (physical)',
        },
        'pressure_conversion_from_database_units': {
            'database_unit': 'Msol km^2 s^-2 kpc^-3',
            'multiply_by_to_get_erg_cm3': PRESSURE_TO_ERG_CM3,
        },
        'radial_bins': {
            'n_bins': n_bins,
            'spacing': 'logarithmic',
            'r_min_kpc': r_min,
            'r_max_kpc': r_max,
            'edges_kpc': edges.tolist(),
            'centres_kpc': centres.tolist(),
            'centre_definition': 'geometric mean of the two edges, sqrt(r_lo * r_hi)',
        },
        'n_halos': len(halos),
        'halos': halos,
    }

    with open(OUTPUT, 'w') as f:
        json.dump(document, f, indent=1)

    n_null = int(np.isnan(pressure).sum())
    print(f"Wrote {len(halos)} haloes to {OUTPUT}; {n_null} empty shells written as null")


if __name__ == '__main__':
    main()
