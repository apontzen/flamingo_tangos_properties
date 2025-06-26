import tangos, tangos.input_handlers.pynbody
from tangos.properties.pynbody import spherical_region
from tangos.properties import LiveHaloProperties
from tangos.properties.pynbody.centring import centred_calculation
import pynbody, pynbody.halo

__version__ = "0.1.0"

class FlamingoInputHandler(tangos.input_handlers.pynbody.Gadget4HDFSubfindInputHandler):
    patterns = ['flamingo_*.hdf5']
    auxiliary_file_patterns = ['fof_output_*.hdf5']
    snap_class_name = "pynbody.snapshot.swift.SwiftSnap"
    catalogue_class_name = "pynbody.halo.number_array.HaloNumberCatalogue"

class M200m(LiveHaloProperties):
    names = "M200m"

    def calculate(self, data, existing_properties):
        # convert r200m to M200m
        r200m = existing_properties['r200m']
        # Critical density in units of Msun/kpc^3
        G = 4.30091e-6  # gravitational constant in (kpc/Msun)*(km/s)^2
        H0 = 68.0
        H0_kpc = H0 / 1e3  # convert H0 to km/s/kpc
        rho_crit = 3 * (H0_kpc)**2 / (8 * 3.141592653589793 * G)  # Msun/kpc^3
        return (4./3) * 3.141592653589793 * r200m**3 * 200 * rho_crit
    
    def requires_property(self):
        return super().requires_property() + ['r200m']

class FlamingoDensityProfile(spherical_region.SphericalRegionPropertyCalculation):
    names = "gas_rho_r200m_relative", "gas_p_r200m_relative", "gas_entropy_r200m_relative"
        
    _max_rad = 2.0  # Maximum radius in units of r200m
    _radius_name = "r200m"  # Name of the radius property to use for scaling

    @centred_calculation
    def calculate(self, data, existing_properties):
        delta = self.plot_xdelta() * existing_properties[self._radius_name]
        maxrad = existing_properties[self._radius_name] * self._max_rad
        nbins = int(maxrad / delta)
        maxrad = delta * nbins

        pro = pynbody.analysis.profile.Profile(data.gas, type='lin', ndim=3,
                                               min=0, max=maxrad, nbins=nbins)
        
        data.gas['Entropies']

        rho = pro['density']
        p = pro['p']
        entropy = pro['Entropies']

        return rho, p, entropy

    
    def plot_xlabel(self):
        return "r/r_200m"
    
    def plot_xdelta(self):
        return 0.1

    def plot_ylabel(self):
        return r"$\rho/M_{\odot}\,kpc^{-3}$", r"pressure/$M_{\odot} km^2 s^{-2} kpc^{-3}$", \
               r"entropy/$M_{\odot}^{-2/3} kpc^2 km^2 s^{-2}$" 
    
    def region_specification(self, db_data):
        return pynbody.filt.Sphere(db_data[self._radius_name]*self._max_rad, 
                                   db_data['shrink_center'])

    def requires_property(self):
        return ["shrink_center", self._radius_name]+super().requires_property()
