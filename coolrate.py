"""Very simple cooling rate / times estimator using the Ploeckinger/Schaye tables"""

import numpy as np
import h5py
import scipy.interpolate
import pynbody

table_url = "https://dataverse.harvard.edu/api/access/datafile/3985215"
m_H = 1.6735575e-24 # g
X_H = 0.76 # this should probably be metal-dependent but we are only going for rough timescales not precision
Z_sol = 0.0134 # Asplund et al. 2009, this is the solar metallicity by mass, which is what the PS20 tables use

def entropy_to_density(K, T):
    """Convert adiabatic invariant K and temperature T to density.

    Parameters
    ----------
    K : float or array
        Adiabatic invariant K = p * rho^(-5/3), in units Msol^(-2/3) kpc^2 km^2 s^-2
    T : float or array
        Temperature in K

    Returns
    -------
    float or array
        Density in Msol kpc^-3

    Notes
    -----
    Assumes a fully ionised, primordial-composition plasma (X_H = 0.76).
    For an ideal gas, p = rho k_B T / (mu m_H), giving
    rho = (k_B T / (K mu m_H))^(3/2).
    """
    k_B = 1.380649e-16  # erg/K (CGS)

    # Mean molecular weight for fully ionised H + He: mu = 4 / (5 X + 3)
    mu = 4.0 / (5.0 * X_H + 3.0)  # ~ 0.588

    # Unit conversion factors
    Msol_g = 1.989e33   # g per Msol
    kpc_cm = 3.0857e21  # cm per kpc
    km_cm  = 1.0e5      # cm per km

    # Convert K from [Msol^(-2/3) kpc^2 km^2 s^-2] to CGS [g^(-2/3) cm^4 s^-2]
    K_cgs = K * Msol_g**(-2.0/3.0) * kpc_cm**2 * km_cm**2

    # Solve rho = (k_B T / (K mu m_H))^(3/2) in g/cm^3
    rho_cgs = (k_B * T / (K_cgs * mu * m_H))**1.5

    # Convert g/cm^3 -> Msol/kpc^3
    return rho_cgs * kpc_cm**3 / Msol_g


class PloeckingerSchayeCoolingRate:
    def __init__(self, filename=None):
        if filename is None:
            filename = __file__.replace("coolrate.py", "UVB_dust1_CR1_G1_shield1.hdf5")
            self._get_file_if_needed(filename)

        self._f = h5py.File(filename, "r")
        self._cool = np.log10(np.sum(10.**self._f["Tdep/Cooling"][..., 20:], axis=-1))
        self._heat = np.log10(np.sum(10.**self._f["Tdep/Heating"][..., 22:], axis=-1))
        self._u = self._f["Tdep/U_from_T"][:]

        self._redshifts = self._f['TableBins/RedshiftBins'][:]
        self._log_temperatures = self._f['TableBins/TemperatureBins'][:]
        self._log_metallicities = self._f['TableBins/MetallicityBins'][:]
        self._log_nH = self._f['TableBins/DensityBins'][:]
        
        expected_shape = ((len(self._redshifts), len(self._log_temperatures), len(self._log_metallicities), len(self._log_nH)))

        assert self._cool.shape == expected_shape
        assert self._heat.shape == expected_shape

        self._cooling_interpolator = self._build_interpolator(self._cool)
        self._heating_interpolator = self._build_interpolator(self._heat)
        self._u_interpolator = self._build_interpolator(self._u)

    def _build_interpolator(self, values):
        return scipy.interpolate.RegularGridInterpolator(
            (self._redshifts, self._log_temperatures, self._log_metallicities, self._log_nH),
            values,
            bounds_error=False,
            fill_value=None,
        )
    
    @classmethod
    def _get_file_if_needed(cls, filename):
        import os, ssl, urllib.request, shutil
        if not os.path.exists(filename):
            print(f"Downloading PS20 cooling table from {table_url}...")
            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE # IKR: this is not great, but macos certificate store is broken
            req = urllib.request.Request(table_url, headers={"User-Agent": "Mozilla/5.0"}) # dataverse wants a user agent
            with urllib.request.urlopen(req, context=ctx) as response, open(filename, 'wb') as f:
                shutil.copyfileobj(response, f)
            print("Download complete.")

    def __call__(self, redshift, temp_K, metallicity_Zsun, nH_cm3):
        metallicity_Zsun = np.maximum(metallicity_Zsun, 1e-10) 
        cooling = 10.**self._cooling_interpolator((redshift, np.log10(temp_K), np.log10(metallicity_Zsun), np.log10(nH_cm3)))
        #heating = 10.**self._heating_interpolator((redshift, np.log10(temp_K), np.log10(metallicity_Zsun), np.log10(nH_cm3)))
        return cooling #- heating # erg cm^3 s^-1
    
    def internal_energy(self, redshift, temp_K, metallicity_Zsun, nH_cm3):
        metallicity_Zsun = np.maximum(metallicity_Zsun, 1e-10) 
        return 10.**self._u_interpolator((redshift, np.log10(temp_K), np.log10(metallicity_Zsun), np.log10(nH_cm3))) # erg g^-1

    
@pynbody.snapshot.simsnap.SimSnap.derived_quantity
def ps20_cooling_rate(snap):
    cooling_rate = PloeckingerSchayeCoolingRate()
    redshift = snap.properties['z']
    temp_K = snap.gas['temp'].in_units("K")
    metallicity_Zsun = snap.gas['metals'] / Z_sol
    nH_cm3 = snap.gas['rho'].in_units('m_p cm^-3') * X_H
    # cooling_rate is in erg cm^3 s^-1, multiplying by nH_cm3^2 gives erg cm^-3 s^-1, but then divide by nH_cm3 * X_H / m_H to give erg g^-1 s^-1
    result = (cooling_rate(redshift, temp_K, metallicity_Zsun, nH_cm3)) * nH_cm3 * X_H / m_H
    return pynbody.array.SimArray(result, "erg g^-1 s^-1")

@pynbody.snapshot.simsnap.SimSnap.derived_quantity
def approx_cooling_rate(snap):
    n_e = snap.gas['rho'].in_units('m_p cm^-3') * (X_H) # electrons per cm^3, assuming fully ionised H and He
    temp_K = snap.gas['temp'].in_units("K")

    Lambda_approx = 1e-27 * np.sqrt(temp_K) * n_e**2 # erg cm^-3 s^-1

    print(n_e,Lambda_approx, snap.gas['rho'].in_units('g cm^-3'))
    result = Lambda_approx.astype(np.float64) / (snap.gas['rho'].in_units('g cm^-3')) # erg g^-1 s^-1
    print(result)
    return pynbody.array.SimArray(result, "erg g^-1 s^-1")


@pynbody.snapshot.swift.SwiftSnap.derived_quantity
def ps20_cooling_time(snap):
    u = snap.gas['u'] 
    # NB this should probably have been enthalpy not internal energy; a correction
    # is made in the plotting of cooling rates by multiplying by 5/3.
    coolrate = snap.gas['ps20_cooling_rate']
    result = u / coolrate
    return result

@pynbody.snapshot.swift.SwiftSnap.derived_quantity
def ps20_u_from_T(snap): 
    # this isn't really needed - u is already stored in the file - just here for testing purposes
    cooling_rate = PloeckingerSchayeCoolingRate()
    redshift = snap.properties['z']
    temp_K = snap.gas['temp'].in_units("K")
    metallicity_Zsun = snap.gas['metals'] / Z_sol
    nH_cm3 = snap.gas['rho'].in_units('m_p cm^-3') * X_H
    result = cooling_rate.internal_energy(redshift, temp_K, metallicity_Zsun, nH_cm3)
    return pynbody.array.SimArray(result, "erg g^-1")

def main():
    print("Testing the Ploeckinger & Schaye cooling rate interpolator...")
    cooling_rate = PloeckingerSchayeCoolingRate()
    redshift = 0.0
    temp_K = 1e6
    metallicity_Zsun = 1.0
    nH_cm3 = 1e2
    net_cooling = cooling_rate(redshift, temp_K, metallicity_Zsun, nH_cm3)
    print(f"Net cooling rate: {net_cooling:.3e} erg cm^3 s^-1")

if __name__ == "__main__":
    main()
