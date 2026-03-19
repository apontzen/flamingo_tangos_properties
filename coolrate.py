import numpy as np
import h5py
import scipy.interpolate

class PloeckingerSchayeCoolingRate:
    def __init__(self, filename="/Users/app/Downloads/UVB_dust1_CR1_G1_shield1.hdf5"):
        self._f = h5py.File(filename, "r")
        self._cool = np.sum(self._f["Tdep/Cooling"][..., 20:], axis=-1)
        self._heat = np.sum(self._f["Tdep/Heating"][..., 22:], axis=-1)

        self._redshifts = self._f['TableBins/RedshiftBins'][:]
        self._log_temperatures = self._f['TableBins/TemperatureBins'][:]
        self._log_metallicities = self._f['TableBins/MetallicityBins'][:]
        self._log_nH = self._f['TableBins/DensityBins'][:]
        
        expected_shape = ((len(self._redshifts), len(self._log_temperatures), len(self._log_metallicities), len(self._log_nH)))

        print("expect:", expected_shape)
        print("cool:", self._cool.shape)
        print("heat:", self._heat.shape)

        assert self._cool.shape == expected_shape
        assert self._heat.shape == expected_shape

        self._cooling_interpolator = scipy.interpolate.RegularGridInterpolator(
            (self._redshifts, self._log_temperatures, self._log_metallicities, self._log_nH),
            self._cool,
            bounds_error=False,
            fill_value=None,
        )

        self._heating_interpolator = scipy.interpolate.RegularGridInterpolator(
            (self._redshifts, self._log_temperatures, self._log_metallicities, self._log_nH),
            self._heat,
            bounds_error=False,
            fill_value=None,
        )



    def __call__(self, redshift, temp_K, metallicity_Zsun, nH_cm3):
        cooling = 10.**self._cooling_interpolator((redshift, np.log10(temp_K), np.log10(metallicity_Zsun), np.log10(nH_cm3)))
        heating = 10.**self._heating_interpolator((redshift, np.log10(temp_K), np.log10(metallicity_Zsun), np.log10(nH_cm3)))
        return cooling, heating


def main():
    print("Testing the Ploeckinger & Schaye cooling rate interpolator...")
    cooling_rate = PloeckingerSchayeCoolingRate()
    redshift = 1.0
    temp_K = 1e6
    metallicity_Zsun = 0.1
    nH_cm3 = 1e2
    cooling, heating = cooling_rate(redshift, temp_K, metallicity_Zsun, nH_cm3)
    print(f"Cooling rate: {cooling:.3e} erg cm^3 s^-1")
    print(f"Heating rate: {heating:.3e} erg cm^3 s^-1")

if __name__ == "__main__":
    main()
