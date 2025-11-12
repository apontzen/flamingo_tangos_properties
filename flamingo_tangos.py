from tangos.config import min_halo_particles
import tangos, tangos.input_handlers.pynbody
from tangos.properties.pynbody import spherical_region
from tangos.properties import LiveHaloProperties
from tangos.properties.pynbody.centring import centred_calculation
import pynbody, pynbody.halo

import numpy as np

import pathlib
import tqdm

from typing import Iterator, Union
from collections import defaultdict 

__version__ = "0.1.0"

def _cosmic_hubble(redshift):
    H0 = 68.1
    OmegaM0 = 0.306
    OmegaL0 = 1.-OmegaM0

    H_z = H0 * np.sqrt(OmegaM0 * (1 + redshift)**3 + OmegaL0)
    return H_z

def _cosmic_crit_density_0():
    G = 4.30091e-6  # gravitational constant in (kpc/Msun)*(km/s)^2
    H0 = 68.1
    H0_kpc = H0 / 1e3  # convert H0 to km/s/kpc
    return 3 * (H0_kpc)**2 / (8 * 3.141592653589793 * G)  # Msun/kpc^3

def _cosmic_mean_density(redshift):
    OmegaM0 = 0.306
    rho_crit = _cosmic_crit_density_0()
    return rho_crit * OmegaM0 * (1 + redshift)**3


def _cosmic_dm_density(redshift):
    OmegaC0 = 0.306 - 0.0486
    rho_crit = _cosmic_crit_density_0()
    return rho_crit * OmegaC0 * (1 + redshift)**3

def _cosmic_baryon_density(redshift):
    OmegaB0 = 0.0486
    rho_crit = _cosmic_crit_density_0()
    return rho_crit * OmegaB0 * (1 + redshift)**3


class FlamingoInputHandler(tangos.input_handlers.pynbody.Gadget4HDFSubfindInputHandler):
    patterns = ['flamingo_00??.hdf5']
    auxiliary_file_patterns = ['fof_output_*.hdf5']
    snap_class_name = "pynbody.snapshot.swift.SwiftSnap"
    catalogue_class_name = "pynbody.halo.hbtplus.HBTPlusCatalogue"

    _sub_parent_names = [] # although HBTplus stores this as 'HostHaloId', pynbody already translates it to 'parent'

    def load_timestep_without_caching(self, ts_extension, mode=None) -> pynbody.snapshot.simsnap.SimSnap:
        f = super().load_timestep_without_caching(ts_extension, mode)
        if mode is None:
            # hackety hack! Seems like swift snapshot not fully wrapped?
            f._shared_arrays = True
            f.wrap()
        return f
    
    def enumerate_objects(self, ts_extension, object_typetag="halo", min_halo_particles=100):
        if object_typetag != 'halo':
            return # not implementing groups or subhalos at this time
        h = self.get_catalogue(ts_extension, object_typetag)
        for hi in tqdm.tqdm(h):
            try:
                i = hi.properties['halo_number']
                is_halo = hi.properties['Depth']<=0
                if len(hi.dm) + len(hi.star) + len(hi.gas) >= min_halo_particles and is_halo:
                    yield i, i, len(hi.dm), len(hi.star), len(hi.gas)
            except (ValueError, KeyError) as e:
                pass

    def _is_able_to_load(self, ts_extension):
        filepath = self._extension_to_filename(ts_extension)
        try:
            f = pynbody.snapshot.swift.SwiftSnap(filepath)
            h = self._construct_pynbody_halos(f)
            return True
        except OSError:
            return False
        
    @classmethod
    def _construct_pynbody_halos(cls, sim, *args, **kwargs):
        h = pynbody.halo.hbtplus.HBTPlusCatalogue(sim,
                                                   filename=cls._sim_filename_to_hbt_filename(sim.filename),)
        return h

    @classmethod
    def _sim_filename_to_hbt_filename_candidates(cls, sim_filename : Union[pathlib.Path, str]) -> Iterator[pathlib.Path]:
        """e.g. /path/to/simname_0004.hdf5 -> /path/to/HBT/004/SubSnap_004"""
        if isinstance(sim_filename, str):
            sim_filename = pathlib.Path(sim_filename)
        snapnum = int(sim_filename.stem.split("_")[-1])
        yield sim_filename.parent / f"SubSnap_{snapnum:03d}"
        yield sim_filename.parent / "HBT" / f"SubSnap_{snapnum:03d}"
        yield sim_filename.parent / "HBT" / f"{snapnum:03d}" / f"SubSnap_{snapnum:03d}"

    @classmethod
    def _sim_filename_to_hbt_filename(cls, sim_filename : pathlib.Path) -> pathlib.Path:
        for candidate in cls._sim_filename_to_hbt_filename_candidates(sim_filename):
            if candidate.with_suffix(".0.hdf5").exists() or candidate.with_suffix(".hdf5").exists():
                return candidate
        raise ValueError(f"Could not find HBT+ halo catalogue corresponding to simulation file {sim_filename}")

    
    def _make_track_id_to_min_depth_track_id(self, props: np.recarray):
        """Make a mapping from TrackId to the TrackId of the depth-0 (or -1) ancestor of each halo.
        
        This is basically the way of identifying mergers.
        """
        input_trackids = props['TrackId']
        input_trackids_sorter = np.argsort(input_trackids)
        
        trackid_to_mindepth_trackid = np.repeat(-1, np.max(input_trackids)+1)
        trackid_to_mindepth_trackid[input_trackids] = input_trackids
        depth_current_map = props['Depth']

        while depth_current_map.max() > 0:
            mask = depth_current_map > 0
            problematic_trackids = input_trackids[mask]
            offsets_of_problematic_trackids = input_trackids_sorter[
                np.searchsorted(input_trackids, problematic_trackids, sorter=input_trackids_sorter)
            ]
            parent_trackids = props['NestedParentTrackId'][offsets_of_problematic_trackids]

            trackid_to_mindepth_trackid[problematic_trackids] = parent_trackids

            depth_current_map[mask] -= 1

        return trackid_to_mindepth_trackid
    
    @classmethod
    def create_offset_mapping(cls, offset1_to_trackid, offset2_to_trackid):
        # Group offset2 indices by their track IDs
        trackid_to_offset2 = defaultdict(list)
        for offset2, trackid in enumerate(offset2_to_trackid):
            trackid_to_offset2[trackid].append(offset2)
        
        # Build the mapping from offset1 to matching offset2 values
        offset1_to_offset2 = {}
        for offset1, trackid in enumerate(offset1_to_trackid):
            offset1_to_offset2[offset1] = trackid_to_offset2[trackid]
        
        return dict(offset1_to_offset2)
    
    @classmethod
    def offset_mapping_to_number_mapping_with_fractions(cls, offset_mapping, offset_to_number1, offset_to_number2, offset2_mbound):
        number_mapping = {}
        for offset1, offset2_list in offset_mapping.items():
            number1 = offset_to_number1(offset1)
            mbounds = [offset2_mbound[offset2] for offset2 in offset2_list]
            total_mbound = np.sum(mbounds)
            number2_list = [(offset_to_number2(offset2), mbound2/total_mbound) for offset2, mbound2 in zip(offset2_list, mbounds) if mbound2>0]
            number_mapping[number1] = number2_list
        return number_mapping

    def match_objects(self, ts1, ts2, halo_min, halo_max,
                      dm_only=False, threshold=0.005, object_typetag='halo',
                      output_handler_for_ts2=None,
                      fuzzy_match_kwa={}):

        if object_typetag=='halo' and output_handler_for_ts2 is self:
            # specialised case
            f1 = self.load_timestep(ts1)
            f2 = self.load_timestep(ts2)

            if f1.properties['z'] == f2.properties['z']:
                raise ValueError("Cannot match HBT+ halos between identical snapshots")
            elif f1.properties['z'] < f2.properties['z']:
                reverse_map = True 
                f1, f2 = f2, f1
                ts1, ts2 = ts2, ts1
            else:
                reverse_map = False
                

            assert f1.properties['z'] > f2.properties['z'], "f1 should be later than f2"
            h1 = self.get_catalogue(ts1, 'halo')
            h2 = self.get_catalogue(ts2, 'halo')

            offset1_to_number1 = h1.number_mapper.index_to_number
            offset2_to_number2 = h2.number_mapper.index_to_number

            props1 = h1.get_properties_all_halos()
            props2 = h2.get_properties_all_halos()
            
            trackid_to_trackid2 = self._make_track_id_to_min_depth_track_id(props2) # many-to-one (early-to-late) mapping

            offset1_to_trackid = trackid_to_trackid2[props1['TrackId']] # many-to-one
            offset2_to_trackid = props2['TrackId'] # one-to-one

            if reverse_map:
                result = self.create_offset_mapping(offset2_to_trackid, offset1_to_trackid)
                result = self.offset_mapping_to_number_mapping_with_fractions(result, offset2_to_number2, offset1_to_number1, props1['Mbound'])
            else:
                result = self.create_offset_mapping(offset1_to_trackid, offset2_to_trackid)
                result = self.offset_mapping_to_number_mapping_with_fractions(result, offset1_to_number1, offset2_to_number2, props2['Mbound'])

            return result


        else:
            return super().match_objects(ts1, ts2, halo_min, halo_max, dm_only, threshold, object_typetag,
                                         output_handler_for_ts2, fuzzy_match_kwa)

    def _soap_filename(self, ts_extension):
        filepath = pathlib.Path(self._extension_to_filename(ts_extension))
        parent_dir = filepath.parent
        snapnum = int(filepath.stem.split("_")[-1])
        soap_filename = parent_dir / "SOAP-HBT" / f"halo_properties_{snapnum:04d}.hdf5"
        return str(soap_filename)
    
    def iterate_object_properties_for_timestep(self, ts_extension, object_typetag, property_names):
        import h5py
        if object_typetag != 'halo':
            return super().iterate_object_properties_for_timestep(ts_extension, object_typetag, property_names)
        property_names = [p.replace('_','/') for p in property_names]
        with h5py.File(self._soap_filename(ts_extension), 'r') as f_soap:
            h5py_names = ["InputHalos/HaloCatalogueIndex", "InputHalos/HaloCatalogueIndex"] + property_names
            h5py_datasets = [f_soap[name] for name in h5py_names]
            yield from zip(*h5py_datasets)

class StarForm(spherical_region.SphericalRegionPropertyCalculation):
    names = "central_SFR", "central_Mstar"

    @centred_calculation
    def calculate(self, ptcls, existing_properties):
        ptcls = ptcls.star

        tform = ptcls['tform'].in_units("Gyr")
        t_now = pynbody.analysis.cosmology.age(ptcls)

        age = t_now - tform

        # get a SFR over either the last 0.1 Gyr, 0.5 Gyr, 1 Gyr or 2 Gyr, whichever is shortest with >2 particles:
        candidate_times = [0.1, 0.5, 1.0, 2.0, 5.0]  # in Gyr
        sfr = 0.0 # default
        require_particles = 2
        for dt in candidate_times:
            mask = age < dt
            if mask.sum() > require_particles:
                sfr = ptcls['mass'][mask].sum() / (dt * 1e9)  # Msol/yr
                break

        return sfr, ptcls['mass'].sum()
    
    def region_specification(self, db_data):
        return pynbody.filt.Sphere("30 kpc", db_data['shrink_center'])
    


class M200m(LiveHaloProperties):
    names = "M200m"

    def calculate(self, data, existing_properties):
        # convert r200m to M200m
        r200m = existing_properties['r200m']
        return (4./3) * 3.141592653589793 * r200m**3 * 200 * \
              _cosmic_mean_density(existing_properties.timestep.redshift)

    def requires_property(self):
        return super().requires_property() + ['r200m']
    
class SSFR(LiveHaloProperties):
    names = "sSFR", 

    def calculate(self, data, existing_properties):
        sfr = existing_properties['central_SFR']  # in Msol/yr
        mstar = existing_properties['central_Mstar']     # in Msol
        if mstar > 0:
            return sfr / mstar,   # in yr^-1
        else:
            return 0.0, 

    def requires_property(self):
        return ['central_SFR', 'central_Mstar'] + super().requires_property()


class RadialVelocityProfile(spherical_region.SphericalRegionPropertyCalculation):
    names = "gas_vr", "gas_vr_disp"

    @centred_calculation
    def calculate(self, particle_data: pynbody.snapshot.SimSnap, halo_entry):
        minrad = 10.0; maxrad = 3000.0
        pro = pynbody.analysis.profile.Profile(particle_data.gas, type='log', ndim=3,
                                            min=minrad, max=maxrad, nbins=50,
                                            weight_by='vol')

        return pro['vr'], pro['vr_disp']

    
class FlamingoDensityProfileBase(spherical_region.SphericalRegionPropertyCalculation):
    names = "_gas_density", "_gas_p", "_gas_entropy", \
            "_gas_temp", "_gas_rho", "_gas_vr", "_gas_vr_disp", "_gas_mass_enclosed", "_gas_mass_enclosed_2d", \
            "_dm_mass_enclosed", "_dm_mass_enclosed_2d", "_dm_vr", "_dm_vr_disp", \
            "_gas_mdot", "_gas_mdot_inflow", "_gas_mdot_outflow", \
            "_dm_mdot", "_dm_mdot_inflow", "_dm_mdot_outflow", "_gas_entropy_outflow", "_gas_entropy_inflow", \
            "_gas_temp_outflow", "_gas_temp_inflow", "_gas_density_outflow", "_gas_density_inflow", \
            "_gas_vr_inflow", "_gas_vr_outflow", "_gas_cs"


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
            data.gas['abs_vr'] = abs(data.gas['vr'])  # |radial velocity| for weighting

            pro_vol_weighted = self._make_vol_weighted_profile(data.gas, minrad, maxrad)
            
            # pre-load data (unclear why this is necessary, but it is)
            data.gas['Entropies']
            data.gas['temp']
            data.gas['rho']
            data.gas['p']
            data.gas['cs'].convert_units('km s^-1')

            den = pro_vol_weighted['density']
            p = pro_vol_weighted['p']
            entropy = pro_vol_weighted['Entropies']
            temp = pro_vol_weighted['temp']
            rho = pro_vol_weighted['rho']
            cs = pro_vol_weighted['cs']

            filt_outflow = pynbody.filt.HighPass('vr', 0)
            pro_outflow_vr_weighted = self._make_vr_weighted_profile(data.gas[filt_outflow], minrad, maxrad)
            entropy_out = pro_outflow_vr_weighted['Entropies']
            temp_out = pro_outflow_vr_weighted['temp']
            den_out = pro_outflow_vr_weighted['density']
            vr_out = pro_outflow_vr_weighted['vr']

            filt_inflow = pynbody.filt.LowPass('vr', 0)
            pro_inflow_vr_weighted = self._make_vr_weighted_profile(data.gas[filt_inflow], minrad, maxrad)
            entropy_in = pro_inflow_vr_weighted['Entropies']
            temp_in = pro_inflow_vr_weighted['temp']
            den_in = pro_inflow_vr_weighted['density']
            vr_in = pro_inflow_vr_weighted['vr']

            vr, vr_disp, mass_enc, mdot, mdot_inflow, mdot_outflow, mass_enc_2d = self._get_profiles(data.gas, minrad, maxrad)


            vr_dm, vr_disp_dm, mass_enc_dm, mdot_dm, mdot_inflow_dm, mdot_outflow_dm, mass_enc_2d_dm = self._get_profiles(data.dm, minrad, maxrad)


        finally:
            data['vel'] += vel_centre  

        return den, p, entropy, temp, rho, vr, vr_disp, mass_enc, mass_enc_2d, mass_enc_dm, mass_enc_2d_dm, vr_dm, vr_disp_dm, \
                mdot, mdot_inflow, mdot_outflow, mdot_dm, mdot_inflow_dm, mdot_outflow_dm, entropy_out, entropy_in, temp_out, \
                temp_in, den_out, den_in, vr_in, vr_out, cs


    def _make_vol_weighted_profile(self, data, minrad, maxrad):
        return pynbody.analysis.profile.Profile(data, type='log', ndim=3,
                                                min=minrad, max=maxrad, nbins=self._nbins,
                                                weight_by='vol')

    def _make_vr_weighted_profile(self, data, minrad, maxrad):
        return pynbody.analysis.profile.Profile(data, type='log', ndim=3,
                                                min=minrad, max=maxrad, nbins=self._nbins,
                                                weight_by='abs_vr') # assuming ~const particle mass

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
        return np.log10(self._min_rad) + self.plot_xdelta()/2
    
    def plot_xdelta(self):
        return np.log10(self._max_rad/self._min_rad)/(self._nbins)

    def plot_ylabel(self):
        return r"$\rho/M_{\odot}\,kpc^{-3}$", r"pressure/$M_{\odot} km^2 s^{-2} kpc^{-3}$", \
               r"entropy/$M_{\odot}^{-2/3} kpc^2 km^2 s^{-2}$", r"temperature/$K$",  \
               r"$\rho/M_{\odot}\,kpc^{-3}$", r"velocity/$km/s$", r"vel dispersion/$km/s$", r"$M_{gas}/M_{\odot}$", r"$M_{gas,2D}/M_{\odot}$", \
               r"$M_{dm}/M_{\odot}$", r"$M_{dm,2D}/M_{\odot}$", r"DM velocity/$km/s$", r"DM vel dispersion/$km/s$", \
                r"$\dot{M}_{gas}/M_{\odot} yr^{-1}$", r"$\dot{M}_{gas,inflow}/M_{\odot} yr^{-1}$", r"$\dot{M}_{gas,outflow}/M_{\odot} yr^{-1}$", \
                r"$\dot{M}_{dm}/M_{\odot} yr^{-1}$", r"$\dot{M}_{dm,inflow}/M_{\odot} yr^{-1}$", r"$\dot{M}_{dm,outflow}/M_{\odot} yr^{-1}$", \
                r"entropy/$_{\rm outflow}M_{\odot}^{-2/3} kpc^2 km^2 s^{-2}$", r"entropy/$_{\rm inflow}M_{\odot}^{-2/3} kpc^2 km^2 s^{-2}$", \
                r"T$_{\rm outflow}/K$", r"T$_{\rm inflow}/K$", r"$\rho_{\rm outflow}/M_{\odot} kpc^{-3}$", r"$\rho_{\rm inflow}/M_{\odot} kpc^{-3}$", \
                r"v$_{\rm inflow}/km s^{-1}$", r"v$_{\rm outflow}/km s^{-1}$", r"c$_s/km s^{-1}$"

    def plot_xlog(self):
        return False
    
    def region_specification(self, db_data):
        TOLERANCE = 1.1
        _, max_rad = self._get_min_max_radius(db_data) 
        return pynbody.filt.Sphere(max_rad*TOLERANCE, db_data['shrink_center'])

    def requires_property(self):
        return ["shrink_center", self._radius_name]+super().requires_property()

class FlamingoEntropyRadiusHistogram(FlamingoDensityProfileBase):
    _nbins = 30 # in both dimensions
    _min_rad = 0.05 # Mpc
    _max_rad = 5.0  # Mpc
    _min_entropy = 10.0 # in sim units
    _max_entropy = 1e5 # in sim units

    names = "gas_entropy_radius_histogram", "gas_entropy_radius_histogram_outflow", "gas_entropy_radius_histogram_inflow"

    @centred_calculation
    def calculate(self, data, existing_properties):
        vel_centre = self._get_velocity_centre(data)

        minrad = self._min_rad * 1e3  # convert to kpc
        maxrad = self._max_rad * 1e3

        gas = data.gas
        mask = (gas['r'] >= minrad) & (gas['r'] <= maxrad) 
        gas = gas[mask]

        radial_bins = np.logspace(np.log10(minrad), np.log10(maxrad), self._nbins+1)
        entropy_bins = np.logspace(np.log10(self._min_entropy), np.log10(self._max_entropy), self._nbins+1)

        try:
            data['vel']-=vel_centre
            outflow_mask = gas['vr'] > 0
            inflow_mask = gas['vr'] < 0

            entrops = gas['Entropies']  

            weights = gas['smooth']**3 # weight by volume for overall histogram

            histogram, _, _ = np.histogram2d(gas['r'], entrops, bins=[radial_bins, entropy_bins], weights=weights)

            weights = abs(gas['vr']) # weight by mass flux for outflow/inflow histograms

            histogram_outflow, _, _ = np.histogram2d(gas['r'][outflow_mask], entrops[outflow_mask],
                                                    bins=[radial_bins, entropy_bins], weights=weights[outflow_mask])
            histogram_inflow, _, _ = np.histogram2d(gas['r'][inflow_mask], entrops[inflow_mask],
                                                    bins=[radial_bins, entropy_bins], weights=weights[inflow_mask])
        
        finally:
            data['vel'] += vel_centre
        

        return histogram, histogram_outflow, histogram_inflow

    def region_specification(self, db_data):
        TOLERANCE = 1.1
        return pynbody.filt.Sphere(self._max_rad*1000*TOLERANCE, db_data['shrink_center'])
    
    def requires_property(self):
        return ["shrink_center"] 

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
        return np.log10(self._min_rad*1e-3) + self.plot_xdelta()/2
    
    def plot_xdelta(self):
        return np.log10(self._max_rad/self._min_rad)/self._nbins
    

class RelativeInflowEquivalentRate(LiveHaloProperties, FlamingoDensityProfileRelative):
    names = "gas_inflow_equivalent_rate_r200m_relative",
    def calculate(self, data, existing_properties):
        return existing_properties['gas_mdot_inflow_r200m_relative'] * existing_properties['gas_entropy_inflow_r200m_relative'], 
    
    def requires_property(self):
        return ['gas_mdot_inflow_r200m_relative', 'gas_entropy_inflow_r200m_relative'] + super().requires_property()
    
class AbsoluteInflowEquivalentRate(LiveHaloProperties, FlamingoDensityProfileAbsolute):
    names = "gas_inflow_equivalent_rate",
    def calculate(self, data, existing_properties):
        return existing_properties['gas_mdot_inflow'] * existing_properties['gas_entropy_inflow'],
    
    def requires_property(self):
        return ['gas_mdot_inflow', 'gas_entropy_inflow'] + super().requires_property()

class RelativeOutflowEquivalentRate(LiveHaloProperties, FlamingoDensityProfileRelative):
    names = "gas_outflow_equivalent_rate_r200m_relative",
    def calculate(self, data, existing_properties):
        return existing_properties['gas_mdot_outflow_r200m_relative'] * existing_properties['gas_entropy_outflow_r200m_relative'],
    
    def requires_property(self):
        return ['gas_mdot_outflow_r200m_relative', 'gas_entropy_outflow_r200m_relative'] + super().requires_property()    
class AbsoluteOutflowEquivalentRate(LiveHaloProperties, FlamingoDensityProfileAbsolute):
    names = "gas_outflow_equivalent_rate",
    def calculate(self, data, existing_properties):
        return existing_properties['gas_mdot_outflow'] * existing_properties['gas_entropy_outflow'], 

    def requires_property(self):
        return ['gas_mdot_outflow', 'gas_entropy_outflow'] + super().requires_property()

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