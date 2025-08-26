import tangos, tangos.input_handlers.pynbody
from tangos.properties.pynbody import spherical_region
from tangos.properties import LiveHaloProperties
from tangos.properties.pynbody.centring import centred_calculation
import pynbody, pynbody.halo

import numpy as np

__version__ = "0.1.0"

class FlamingoInputHandler(tangos.input_handlers.pynbody.Gadget4HDFSubfindInputHandler):
    patterns = ['flamingo_00??.hdf5']
    auxiliary_file_patterns = ['fof_output_*.hdf5']
    snap_class_name = "pynbody.snapshot.swift.SwiftSnap"
    catalogue_class_name = "pynbody.halo.number_array.HaloNumberCatalogue"

    def load_timestep_without_caching(self, ts_extension, mode=None) -> pynbody.snapshot.simsnap.SimSnap:
        f = super().load_timestep_without_caching(ts_extension, mode)
        if mode is None:
            # hackety hack! Seems like swift snapshot not fully wrapped?
            f._shared_arrays = True
            f.wrap()
        return f

class M200m(LiveHaloProperties):
    names = "M200m"

    def calculate(self, data, existing_properties):
        # convert r200m to M200m
        r200m = existing_properties['r200m']
        # Critical density in units of Msun/kpc^3
        G = 4.30091e-6  # gravitational constant in (kpc/Msun)*(km/s)^2
        H0 = 68.0
        H0_kpc = H0 / 1e3  # convert H0 to km/s/kpc
        OmegaM0 = 0.308
        rho_crit = 3 * (H0_kpc)**2 / (8 * 3.141592653589793 * G)  # Msun/kpc^3
        return (4./3) * 3.141592653589793 * r200m**3 * 200 * rho_crit * OmegaM0 * \
            (1+existing_properties.timestep.redshift)**3
    
    def requires_property(self):
        return super().requires_property() + ['r200m']

class FlamingoDensityProfileBase(spherical_region.SphericalRegionPropertyCalculation):
    names = "_gas_density", "_gas_p", "_gas_entropy", \
            "_gas_temp", "_gas_rho", "_gas_vr", "_gas_vr_disp", "_gas_mass_enclosed", "_gas_mass_enclosed_2d", \
            "_dm_mass_enclosed", "_dm_mass_enclosed_2d", "_dm_vr", "_dm_vr_disp", \
            "_gas_mdot", "_gas_mdot_inflow", "_gas_mdot_outflow", \
            "_dm_mdot", "_dm_mdot_inflow", "_dm_mdot_outflow", "_gas_entropy_outflow", "_gas_entropy_inflow", \
            "_gas_temp_outflow", "_gas_temp_inflow", "_gas_rho_outflow", "_gas_rho_inflow"


    _nbins = 50

    def _get_velocity_centre(self, data, region_sizes=['25 kpc', '50 kpc', '200 kpc']):
        for region_size in region_sizes:
            try:
                region = data[pynbody.filt.Sphere(region_size)]
                vel_centre = np.average(region['vel'], axis=0, weights=region['mass'])
                return vel_centre
            except ZeroDivisionError:
                pass
        return np.average(data['vel'], axis=0, weights=data['mass'])

    @centred_calculation
    def calculate(self, data: pynbody.snapshot.SimSnap, existing_properties):
        minrad, maxrad  = self._get_min_max_radius(existing_properties)
        vel_centre = self._get_velocity_centre(data)

        try:
            data['vel']-=vel_centre
            pynbody.analysis.cosmology.add_hubble(data) 

            data.gas['vol'] = data.gas['smooth']**3  # Volume for weighting

            pro_vol_weighted = self._make_vol_weighted_profile(data.gas, minrad, maxrad)
            
            # pre-load data (unclear why this is necessary, but it is)
            data.gas['Entropies']
            data.gas['temp']
            data.gas['rho']
            data.gas['p']

            den = pro_vol_weighted['density']
            p = pro_vol_weighted['p']
            entropy = pro_vol_weighted['Entropies']
            temp = pro_vol_weighted['temp']
            rho = pro_vol_weighted['rho']

            filt_outflow = pynbody.filt.HighPass('vr', 0)
            pro_vol_weighted_out = self._make_mdot_weighted_profile(data.gas[filt_outflow], minrad, maxrad)
            entropy_out = pro_vol_weighted_out['Entropies']
            temp_out = pro_vol_weighted_out['temp']
            rho_out = pro_vol_weighted_out['rho']

            filt_inflow = pynbody.filt.LowPass('vr', 0)
            pro_vol_weighted_in = self._make_mdot_weighted_profile(data.gas[filt_inflow], minrad, maxrad)
            entropy_in = pro_vol_weighted_in['Entropies']
            temp_in = pro_vol_weighted_in['temp']
            rho_in = pro_vol_weighted_in['rho']

            vr, vr_disp, mass_enc, mdot, mdot_inflow, mdot_outflow, mass_enc_2d = self._get_profiles(data.gas, minrad, maxrad)


            vr_dm, vr_disp_dm, mass_enc_dm, mdot_dm, mdot_inflow_dm, mdot_outflow_dm, mass_enc_2d_dm = self._get_profiles(data.dm, minrad, maxrad)


        finally:
            data['vel'] += vel_centre  

        return den, p, entropy, temp, rho, vr, vr_disp, mass_enc, mass_enc_2d, mass_enc_dm, mass_enc_2d_dm, vr_dm, vr_disp_dm, \
                mdot, mdot_inflow, mdot_outflow, mdot_dm, mdot_inflow_dm, mdot_outflow_dm, entropy_out, entropy_in, temp_out, \
                temp_in, rho_out, rho_in 


    def _make_vol_weighted_profile(self, data, minrad, maxrad):
        return pynbody.analysis.profile.Profile(data, type='log', ndim=3,
                                                min=minrad, max=maxrad, nbins=self._nbins,
                                                weight_by='vol')

    def _make_mdot_weighted_profile(self, data, minrad, maxrad):
        return pynbody.analysis.profile.Profile(data, type='log', ndim=3,
                                                min=minrad, max=maxrad, nbins=self._nbins,
                                                weight_by='vr') # assuming ~const particle mass

    def _get_profiles(self, data, minrad, maxrad):
        pro_2d = pynbody.analysis.profile.Profile(data, type='log', ndim=3,
                                                min=minrad, max=maxrad, nbins=self._nbins)
        vr = pro_2d['vr']
        vr_disp = pro_2d['vr_disp']
        mass_enc = pro_2d['mass_enc']
        mdot = pro_2d['mdot']

        filt_inflow = pynbody.filt.LowPass('vr', 0)
        pro_inflow = pynbody.analysis.profile.Profile(data[filt_inflow], type='log', ndim=3,
                                                          min=minrad, max=maxrad, nbins=self._nbins)
        mdot_inflow = pro_inflow['mdot']
            
        filt_outflow = pynbody.filt.HighPass('vr', 0)
        pro_outflow = pynbody.analysis.profile.Profile(data[filt_outflow], type='log', ndim=3,
                                                           min=minrad, max=maxrad, nbins=self._nbins)
        mdot_outflow = pro_outflow['mdot']

        pro_2d = pynbody.analysis.profile.Profile(data, type='log', ndim=2,
                                                min=minrad, max=maxrad, nbins=self._nbins)
        mass_enc_2d = pro_2d['mass_enc']

        return vr,vr_disp,mass_enc,mdot,mdot_inflow,mdot_outflow,mass_enc_2d

    def _get_min_max_radius(self, existing_properties):
        raise NotImplementedError("Subclasses must implement _get_min_max_radius method")
    
    def plot_xlabel(self):
        return "log_10 r/r_200m"
    
    def plot_x0(self):
        return np.log10(self._min_rad)
    
    def plot_xdelta(self):
        return np.log10(self._max_rad/self._min_rad)/self._nbins

    def plot_ylabel(self):
        return r"$\rho/M_{\odot}\,kpc^{-3}$", r"pressure/$M_{\odot} km^2 s^{-2} kpc^{-3}$", \
               r"entropy/$M_{\odot}^{-2/3} kpc^2 km^2 s^{-2}$", r"temperature/$K$",  \
               r"$\rho/M_{\odot}\,kpc^{-3}$", r"velocity/$km/s$", r"vel dispersion/$km/s$", r"$M_{gas}/M_{\odot}$", r"$M_{gas,2D}/M_{\odot}$", \
               r"$M_{dm}/M_{\odot}$", r"$M_{dm,2D}/M_{\odot}$", r"DM velocity/$km/s$", r"DM vel dispersion/$km/s$", \
                r"$\dot{M}_{gas}/M_{\odot} yr^{-1}$", r"$\dot{M}_{gas,inflow}/M_{\odot} yr^{-1}$", r"$\dot{M}_{gas,outflow}/M_{\odot} yr^{-1}$", \
                r"$\dot{M}_{dm}/M_{\odot} yr^{-1}$", r"$\dot{M}_{dm,inflow}/M_{\odot} yr^{-1}$", r"$\dot{M}_{dm,outflow}/M_{\odot} yr^{-1}$", \
                r"entropy/$_{\rm outflow}M_{\odot}^{-2/3} kpc^2 km^2 s^{-2}$", r"entropy/$_{\rm inflow}M_{\odot}^{-2/3} kpc^2 km^2 s^{-2}$", \
                r"T$_{\rm outflow}/K$", r"T$_{\rm inflow}/K$", r"$\rho_{\rm outflow}/M_{\odot} kpc^{-3}$", r"$\rho_{\rm inflow}/M_{\odot} kpc^{-3}$"

    def plot_xlog(self):
        return False
    
    def region_specification(self, db_data):
        TOLERANCE = 1.1
        _, max_rad = self._get_min_max_radius(db_data) 
        return pynbody.filt.Sphere(max_rad*TOLERANCE, db_data['shrink_center'])

    def requires_property(self):
        return ["shrink_center", self._radius_name]+super().requires_property()
        
class FlamingoDensityProfileRelative(FlamingoDensityProfileBase):
    _min_rad = 0.05 # Minimum radius in units of r200m 
    _max_rad = 5.0  # Maximum radius in units of r200m
    _radius_name = "r200m"  # Name of the radius property to use for scaling
    names = [n[1:]+"_r200m_relative" for n in FlamingoDensityProfileBase.names]

    def _get_min_max_radius(self, existing_properties):
        maxrad = existing_properties[self._radius_name] * self._max_rad
        minrad = existing_properties[self._radius_name] * self._min_rad
        return minrad, maxrad
    
class FlamingoDensityProfileAbsolute(FlamingoDensityProfileBase):
    _min_rad = 50.0  # Minimum radius in kpc
    _max_rad = 5000.0  # Maximum radius in kpc
    _radius_name = "shrink_center"  # Use the shrink center for absolute radius
    names = [n[1:] for n in FlamingoDensityProfileBase.names]

    def _get_min_max_radius(self, existing_properties):
        return self._min_rad, self._max_rad
    
    def plot_xlabel(self):
        return "log_10 r/Mpc"
    
    def plot_x0(self):
        return np.log10(self._min_rad*1e-3) + np.log10(self._max_rad/self._min_rad)/self._nbins # outer bin
    
    def plot_xdelta(self):
        return np.log10(self._max_rad/self._min_rad)/self._nbins
    
def _filter_out_other_halos(data, existing_properties):
    halo_number = np.median(data[pynbody.filt.Sphere('10 kpc', existing_properties['shrink_center'])]['grp'])
    data_exclusive = data[(data['grp'] == halo_number) | (data['grp'] == 2**31 - 1)]
    return data_exclusive

class FlamingoExclusiveDensityProfileRelative(FlamingoDensityProfileRelative):
    names = [n+"_exclusive" for n in FlamingoDensityProfileRelative.names]

    def calculate(self, data, existing_properties):
        data_exclusive = _filter_out_other_halos(data, existing_properties)
        return super().calculate(data_exclusive, existing_properties)
    
    
class FlamingoExclusiveDensityProfileAbsolute(FlamingoDensityProfileAbsolute):
    names = [n+"_exclusive" for n in FlamingoDensityProfileAbsolute.names]

    def calculate(self, data, existing_properties):
        data_exclusive = _filter_out_other_halos(data, existing_properties)
        return super().calculate(data_exclusive, existing_properties)
    
class FlamingoPrimordialBaryonicMassDeficit(spherical_region.SphericalRegionHaloProperties):
    names = "primordial_baryonic_massfrac_deficit"

    @centred_calculation
    def calculate(self, data, existing_properties):
        reference_mass = 5.65006349e+09 # Reference mass in Msun
        mean_dm_mass = data.dm['mass'].mean()

        # now work out the simulation particle mass, which is 2^n times reference_mass where n
        # is an integer (but may be negative/zero/positive)
        n = np.round(np.log2(mean_dm_mass / reference_mass), decimals=0)
        sim_particle_mass = reference_mass * (2**n)

        sim_gas_particle_mass = sim_particle_mass * 0.19565705 # OmB / OmC

        # Calculate the primordial baryonic mass deficit
        primordial_baryonic_massfrac_deficit = (mean_dm_mass - sim_particle_mass) / sim_gas_particle_mass

        return primordial_baryonic_massfrac_deficit
    
    def requires_property(self):
        return ["shrink_center", 'r200m'] + super().requires_property()
    
    def region_specification(self, db_data):
        return pynbody.filt.Sphere(db_data['r200m'], db_data['shrink_center'])
    



@pynbody.analysis.profile.Profile.profile_property
def mdot(profile: pynbody.analysis.profile.Profile):
    # mdot = integral rho v_r r^2 d omega
    # estimate in a spherical shell of thickness delta r:
    # mdot = integral_r0^(r0+delta r) dr r^2 d omega (rho v_r) / delta r
    #      = integral dV (rho v_r) / delta r
    #      = sum m v_r / delta r
    #
    # profile['vr'] gives mass-weighted mean v_r, while profile['mass'] gives the mass in each shell,
    # so the product is sum m v_r in each shell.
    ar = profile['vr'] * profile['mass'] / np.diff(profile['bin_edges'])
    ar.units = profile['mass'].units * profile['vr'].units / profile['bin_edges'].units
    return ar.in_units('Msol yr^-1')

@pynbody.derived_array
def vr_smoothed(f: pynbody.snapshot.SimSnap):
    """Smoothed radial velocity field, using SPH smoothing."""
    ar = f['vr']
    f.gas.build_tree()
    f.gas['smooth']
    ar_gas = f.gas.kdtree.sph_mean(f.gas['vr'])
    ar[f._get_family_slice(pynbody.family.gas)] = ar_gas
    return ar