from tangos.config import min_halo_particles
import tangos, tangos.input_handlers.pynbody
from tangos.properties.pynbody import spherical_region
from tangos.properties import LiveHaloProperties
from tangos.properties.pynbody.centring import centred_calculation
import pynbody, pynbody.halo
import coolrate # for derived property ps20_cooling_time
import entropy_generation # for derived properties entropy_generation_rate

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

def _cosmic_crit_density(redshift):
    G = 4.30091e-6  # gravitational constant in (kpc/Msun)*(km/s)^2
    H0 = _cosmic_hubble(redshift)
    H0_kpc = H0 / 1e3  # convert H0 to km/s/kpc
    return 3 * (H0_kpc)**2 / (8 * 3.141592653589793 * G)  # Msun/kpc^3

def _cosmic_crit_density_0():
    return _cosmic_crit_density(redshift=0.0)

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

            # Also, there is a big table lookup for the following calculation, which we don't want to do separately
            # on each process, so we pre-calculate the cooling rates here

            _ = f.gas['ps20_cooling_time']
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
        yield sim_filename.parent.parent / "HBT" / f"{snapnum:03d}" / f"SubSnap_{snapnum:03d}"

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
        if soap_filename.exists():
            return str(soap_filename)
        soap_filename = parent_dir.parent / "SOAP-HBT" / f"halo_properties_{snapnum:04d}.hdf5"
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
    
class FlowImageAligned(spherical_region.SphericalRegionPropertyCalculation):
    names = "aligned_13_entropy_image", "aligned_13_density_image", "aligned_13_vx_image", "aligned_13_vy_image", "aligned_13_entropy_creation_image", \
            "aligned_12_entropy_image", "aligned_12_density_image", "aligned_12_vx_image", "aligned_12_vy_image", "aligned_12_entropy_creation_image", \
            "flow_alignment_eigvals"
    
    @classmethod
    def plot_extent(cls):
        return 4.0 # units of virial radius

    def plot_xlabel(self):
        return r"x/$r_{200m}$"

    def plot_ylabel(self):
        return r"y/$r_{200m}$"

    #def plot_clabel(self):
    #    return "K", r"$\rho / M_{\odot} \mathrm{kpc}^{-3}$", r"$v_x$ (km/s)", r"$v_y$ (km/s)"
    
    def region_specification(self, db_data):
        return pynbody.filt.Sphere(db_data['r200m']*3.0, db_data['shrink_center'])
    
    @classmethod
    def _make_image_set(cls, ptcls, radius):
        ptcls.gas['entropy_generation_rate'].convert_units('Msol^-2/3 kpc^2 km^2 s^-2 Myr^-1')
        entropy_image = pynbody.plot.sph.image(ptcls, qty='Entropies', width=radius*4, return_data=True, noplot=True, resolution=200)
        entropy_creation_image = pynbody.plot.sph.image(ptcls, qty='entropy_generation_rate', width=radius*4, return_data=True, noplot=True, resolution=200)
        density_image = pynbody.plot.sph.image(ptcls, qty='rho', width=radius*4, return_data=True, noplot=True, resolution=200)
        vx_image = pynbody.plot.sph.image(ptcls, qty='vx', width=radius*4, return_data=True, noplot=True, resolution=50)
        vy_image = pynbody.plot.sph.image(ptcls, qty='vy', width=radius*4, return_data=True, noplot=True, resolution=50)

        return entropy_image, density_image, vx_image, vy_image, entropy_creation_image
        
    @centred_calculation
    def calculate(self, ptcls, existing_properties):
        vel_centre = _get_velocity_centre(ptcls)
        ptcls['vel']-=vel_centre
        pynbody.analysis.cosmology.add_hubble(ptcls)  

        from flow_orientation import gas_flow_alignment, get_gas_flow_quadratic_form
        radius = existing_properties['r200m']
        ptcls = ptcls.gas

        Q = get_gas_flow_quadratic_form(ptcls, radius)
        eigvals, _ = np.linalg.eigh(Q)

        with gas_flow_alignment(ptcls, radius):
            imageset_13 = self._make_image_set(ptcls, radius)
            with ptcls.rotate_y(90):
                imageset_12 = self._make_image_set(ptcls, radius)

        return imageset_13 + imageset_12 + (eigvals,)
    
    def requires_property(self):
        return super().requires_property() + ['r200m']
    
class CentralDensity(spherical_region.SphericalRegionPropertyCalculation):
    names = "central_density",

    @centred_calculation
    def calculate(self, ptcls, existing_properties):
        ptcls = ptcls.gas

        index = ptcls['r'].argmin()

        return ptcls['rho'][index], 

    def region_specification(self, db_data):
        return pynbody.filt.Sphere("100 kpc", db_data['shrink_center'])


class M200m(LiveHaloProperties):
    names = "M200m"

    def calculate(self, data, existing_properties):
        # convert r200m to M200m
        r200m = existing_properties['r200m']
        return (4./3) * 3.141592653589793 * r200m**3 * 200 * \
              _cosmic_mean_density(existing_properties.timestep.redshift)

    def requires_property(self):
        return super().requires_property() + ['r200m']

class R200mDot(LiveHaloProperties):
    names = "r200m_dot", "r200m_dot_denominator"

    def calculate(self, data, existing_properties):
        M_profile = existing_properties['all_mass_enclosed_r200m_relative']
        Mdot_profile = -existing_properties['all_mdot_r200m_relative'] # mdot is negative for growth (i.e. measures outflow not inflow, confusingly)
        r200m = existing_properties['r200m']

        pro = FlamingoDensityProfileRelative(None)

        M200m = pro.get_interpolated_value(0.0, M_profile)

        # dM/dr could be expressed in terms of density quite neatly, but we didn't calculate that so let's do it numerically -
        # it'd amount to the same thing anyway
        delta_r = (10**0.05-10**-0.05) * existing_properties['r200m']  # small delta r in kpc
        dM200m_dr = (pro.get_interpolated_value(0.05, M_profile) - pro.get_interpolated_value(-0.05, M_profile)) / delta_r

        Mdot_200m = -pro.get_interpolated_value(0.0, Mdot_profile) # minus because we want Mdot inward directed but profile is outflow-directed

        H_of_z = _cosmic_hubble(existing_properties.timestep.redshift) # in km/s/Mpc
        H_of_z_per_yr = H_of_z / (3.086e19) * 3.154e7  # convert to 1/yr

        rho_mean = _cosmic_mean_density(existing_properties.timestep.redshift) # in Msun/kpc^3 physical

        numerator = 3 * M200m * H_of_z_per_yr + Mdot_200m 
        denominator = 4*np.pi*r200m**2 * 200 * rho_mean - dM200m_dr#, should be subdominant
        return numerator / denominator, denominator  # -> kpc/yr
    
    def requires_property(self):
        return ['all_mass_enclosed_r200m_relative', 'all_mdot_r200m_relative', 'r200m'] + super().requires_property()

    
class FgasGradient(LiveHaloProperties):
    names = "dfgas_dr"

    def calculate(self, data, existing_properties):
        Mgas_profile = existing_properties['gas_mass_enclosed_r200m_relative']
        dm_profile = existing_properties['dm_mass_enclosed_r200m_relative']

        pro = FlamingoDensityProfileRelative(None)

        fgas_upper = pro.get_interpolated_value(0.1, Mgas_profile) / pro.get_interpolated_value(0.1, dm_profile)
        fgas_lower = pro.get_interpolated_value(-0.1, Mgas_profile) / pro.get_interpolated_value(-0.1, dm_profile)
        delta_r = (10**0.1 - 10**-0.1) * existing_properties['r200m']  # small delta r in kpc

        return (fgas_upper - fgas_lower) / delta_r  
    
    def requires_property(self):
        return ['gas_mass_enclosed_r200m_relative', 'dm_mass_enclosed_r200m_relative', 'r200m'] + super().requires_property()
    
class SSFR(LiveHaloProperties):
    names = "sSFR", 

    def calculate(self, data, existing_properties):
        return 0.0102*existing_properties['InclusiveSphere_30kpc_StarFormationRate']/(1e10*existing_properties['InclusiveSphere_30kpc_StellarMass']), 

    def requires_property(self):
        return ['InclusiveSphere_30kpc_StarFormationRate', 'InclusiveSphere_30kpc_StellarMass'] + super().requires_property()
    
def _get_velocity_centre(data, region_sizes=['25 kpc', '50 kpc', '200 kpc']):
    for region_size in region_sizes:
        try:
            region = data[pynbody.filt.Sphere(region_size)]
            vel_centre = np.average(region['vel'], axis=0, weights=region['mass'])
            return vel_centre
        except ZeroDivisionError:
            pass
    return np.average(data['vel'], axis=0, weights=data['mass'])

class FlamingoDensityProfileBase(spherical_region.SphericalRegionPropertyCalculation):
    names = "_gas_density", "_gas_p", "_gas_entropy", \
            "_gas_temp", "_gas_rho", "_gas_vr", "_gas_vr_disp", "_gas_mass_enclosed", "_gas_mass_enclosed_2d", \
            "_dm_mass_enclosed", "_dm_mass_enclosed_2d", "_dm_vr", "_dm_vr_disp", \
            "_gas_mdot", "_gas_mdot_inflow", "_gas_mdot_outflow", \
            "_dm_mdot", "_dm_mdot_inflow", "_dm_mdot_outflow", "_gas_entropy_outflow", "_gas_entropy_inflow", \
            "_gas_temp_outflow", "_gas_temp_inflow", "_gas_density_outflow", "_gas_density_inflow", \
            "_gas_vr_inflow", "_gas_vr_outflow", "_gas_cs", \
            "_gas_energy_inflow", "_gas_energy_outflow", "_dm_energy_inflow", "_dm_energy_outflow", \
            "_gas_cooltime", "_gas_cooltime_inflow", "_gas_cooltime_outflow", \
            "_all_mass_enclosed", "_all_mdot", "_gas_entropy_generation"


    _nbins = 50

    
    @centred_calculation
    def calculate(self, data: pynbody.snapshot.SimSnap, existing_properties):
        minrad, maxrad  = self._get_min_max_radius(existing_properties)
        vel_centre = _get_velocity_centre(data)

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
            data.gas['entropy_generation_rate'].convert_units('Msol^-2/3 kpc^2 km^2 s^-2 Myr^-1')
            data.gas['ps20_cooling_time'].convert_units('Gyr')

            data['energy_flow_integrand'] = (data['vr'] * (data['vel']**2).sum(axis=1)/2).in_units("erg kpc Myr^-1 Msol^-1")

            # for gas, add thermal energy term
            mu = 0.59 # mean molecular weight, fully ionized, primordial
            data.gas['energy_flow_integrand'] = (data.gas['vr'] * ( 
                (data.gas['vel']**2).sum(axis=1)/2 + 1.5 * data.gas['temp'] * pynbody.units.k / (mu*pynbody.units.m_p) 
                )).in_units("erg kpc Myr^-1 Msol^-1")

            

            den = pro_vol_weighted['density']
            p = pro_vol_weighted['p']
            entropy = pro_vol_weighted['Entropies']
            temp = pro_vol_weighted['temp']
            rho = pro_vol_weighted['rho']
            cs = pro_vol_weighted['cs']
            cooltime = pro_vol_weighted['ps20_cooling_time']

            filt_outflow = pynbody.filt.HighPass('vr', 0)
            pro_outflow_vr_weighted = self._make_vr_weighted_profile(data.gas[filt_outflow], minrad, maxrad)
            entropy_out = pro_outflow_vr_weighted['Entropies']
            temp_out = pro_outflow_vr_weighted['temp']
            den_out = pro_outflow_vr_weighted['density']
            vr_out = pro_outflow_vr_weighted['vr']
            cooltime_out = pro_outflow_vr_weighted['ps20_cooling_time']

            filt_inflow = pynbody.filt.LowPass('vr', 0)
            pro_inflow_vr_weighted = self._make_vr_weighted_profile(data.gas[filt_inflow], minrad, maxrad)
            entropy_in = pro_inflow_vr_weighted['Entropies']
            temp_in = pro_inflow_vr_weighted['temp']
            den_in = pro_inflow_vr_weighted['density']
            vr_in = pro_inflow_vr_weighted['vr']
            cooltime_in = pro_inflow_vr_weighted['ps20_cooling_time']

            vr, vr_disp, mass_enc, mdot, mdot_inflow, mdot_outflow, mass_enc_2d, energy_outflow, energy_inflow, entropy_generation = self._get_profiles(data.gas, minrad, maxrad)


            vr_dm, vr_disp_dm, mass_enc_dm, mdot_dm, mdot_inflow_dm, mdot_outflow_dm, mass_enc_2d_dm, energy_outflow_dm, energy_inflow_dm, _ = self._get_profiles(data.dm, minrad, maxrad)

            _, _, mass_enc_all, mdot_all, _, _, _, _, _, _ = self._get_profiles(data, minrad, maxrad)

        finally:
            data['vel'] += vel_centre  

        return den, p, entropy, temp, rho, vr, vr_disp, mass_enc, mass_enc_2d, mass_enc_dm, mass_enc_2d_dm, vr_dm, vr_disp_dm, \
                mdot, mdot_inflow, mdot_outflow, mdot_dm, mdot_inflow_dm, mdot_outflow_dm, entropy_out, entropy_in, temp_out, \
                temp_in, den_out, den_in, vr_in, vr_out, cs, energy_inflow, energy_outflow, energy_inflow_dm, energy_outflow_dm, \
                cooltime, cooltime_in, cooltime_out, \
                mass_enc_all, mdot_all, entropy_generation


    def _make_vol_weighted_profile(self, data, minrad, maxrad):
        return pynbody.analysis.profile.Profile(data, type='log', ndim=3,
                                                min=minrad, max=maxrad, nbins=self._nbins,
                                                weight_by='vol')

    def _make_vr_weighted_profile(self, data, minrad, maxrad):
        return pynbody.analysis.profile.Profile(data, type='log', ndim=3,
                                                min=minrad, max=maxrad, nbins=self._nbins,
                                                weight_by='abs_vr') # assuming ~const particle mass

    def _get_profiles(self, data, minrad, maxrad):
        pro = pynbody.analysis.profile.Profile(data, type='log', ndim=3,
                                                min=minrad, max=maxrad, nbins=self._nbins)
        vr = pro['vr']
        vr_disp = pro['vr_disp']
        mass_enc = pro['mass_enc']
        mdot = pro['mdot']
        if len(data.dm) == 0 and len(data.gas) > 0:
            entropy_generation = pro['entropy_generation_rate']
        else:
            entropy_generation = np.zeros_like(mass_enc)

        filt_inflow = pynbody.filt.LowPass('vr', 0)
        pro_inflow = pynbody.analysis.profile.Profile(data[filt_inflow], type='log', ndim=3,
                                                          min=minrad, max=maxrad, nbins=self._nbins)
        mdot_inflow = pro_inflow['mdot']

        energy_inflow = abs(pro_inflow['energy_flux']) 
        # NB above calc neglects potential; added in gas_inflow_energy_with_potential calc
            
        filt_outflow = pynbody.filt.HighPass('vr', 0)
        pro_outflow = pynbody.analysis.profile.Profile(data[filt_outflow], type='log', ndim=3,
                                                           min=minrad, max=maxrad, nbins=self._nbins)
        mdot_outflow = pro_outflow['mdot']
        energy_outflow = abs(pro_outflow['energy_flux'])
        # NB above calc neglects potential; added in gas_inflow_energy_with_potential calc

        pro = pynbody.analysis.profile.Profile(data, type='log', ndim=2,
                                                min=minrad, max=maxrad, nbins=self._nbins)
        mass_enc_2d = pro['mass_enc']

        return vr,vr_disp,mass_enc,mdot,mdot_inflow,mdot_outflow,mass_enc_2d, energy_outflow, energy_inflow, entropy_generation

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
                r"v$_{\rm inflow}/km s^{-1}$", r"v$_{\rm outflow}/km s^{-1}$", r"c$_s/km s^{-1}$", \
                r"energy flow/$_{\rm inflow}/erg Myr^{-1}$", r"energy flow/$_{\rm outflow}/erg Myr^{-1}$", \
                r"DM energy flow/$_{\rm inflow}/erg Myr^{-1}$", r"DM energy flow/$_{\rm outflow}/erg Myr^{-1}$", \
                r"cooling time/$_{\rm total}/Gyr$", r"cooling time/$_{\rm inflow}/Gyr$", r"cooling time/$_{\rm outflow}/Gyr$", \
                r"$M(<r)/M_{\odot}$", r"$\dot{M}/M_{\odot} yr^{-1}$", r"$\dot{K}/{\rm M_{\odot}^{-2/3} kpc^2 km^2 s^{-2} Myr^{-1}}$"

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
        vel_centre = _get_velocity_centre(data)

        minrad = self._min_rad * 1e3  # convert to kpc
        maxrad = self._max_rad * 1e3

        gas = data.gas
        mask = (gas['r'] >= minrad) & (gas['r'] <= maxrad) 
        gas = gas[mask]

        radial_bins = np.logspace(np.log10(minrad), np.log10(maxrad), self._nbins+1)
        entropy_bins = np.logspace(np.log10(self._min_entropy), np.log10(self._max_entropy), self._nbins+1)

        try:
            data['vel']-=vel_centre
            pynbody.analysis.cosmology.add_hubble(data)  
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
    
class FlamingoDensityFromMassProfileRelative(LiveHaloProperties, FlamingoDensityProfileRelative):
    particle_name = "gas"
    names = "gas_density_from_mass_r200m_relative", 

    def calculate(self, data, existing_properties):
        r200m = existing_properties['r200m']
        gas_rho = self._get_density_estimate_from_mass_profile(existing_properties[f'{self.particle_name}_mass_enclosed_r200m_relative'], r200m)
        return gas_rho,

    def _get_density_estimate_from_mass_profile(self, mass_enclosed, r200m):
        radii = np.logspace(np.log10(self._min_rad*r200m), np.log10(self._max_rad*r200m), self._nbins + 1)

        shell_volume = 4/3 * np.pi * (radii[1:]**3 - radii[:-1]**3)

        shell_mass = np.diff(np.concatenate(([0], mass_enclosed)))
        density_estimate = shell_mass / shell_volume
        return density_estimate 
    
    def requires_property(self):
        return [f'{self.particle_name}_mass_enclosed_r200m_relative', 'r200m'] + super().requires_property()
    

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
    
    
    
class FlamingoEnclosedCoolingRateProfile(FlamingoDensityProfileAbsolute):
    names = "gas_enclosed_cooling_rate", "gas_enclosed_enthalpy"

    @centred_calculation
    def calculate(self, data, existing_properties):
        # cooling rate in ergs Myr^-1
        pro = pynbody.analysis.profile.Profile(data.gas, type='log', ndim=3,
                                                min=self._min_rad, max=self._max_rad, nbins=self._nbins)
        
        return pro['ps20_cooling_cumulative'], pro['enthalpy_cumulative']


    def plot_ylabel(self):
        return r"$\dot{E}_{\rm cool}/{\rm ergs Myr^{-1}}$", r"$H/{\rm ergs}$"


class FlamingoDensityFromMassProfileAbsolute(LiveHaloProperties, FlamingoDensityProfileAbsolute):
    particle_name = "gas"
    names = "gas_density_from_mass", 

    def calculate(self, data, existing_properties):
        gas_rho = self._get_density_estimate_from_mass_profile(existing_properties[f'{self.particle_name}_mass_enclosed'])
        return gas_rho,

    def _get_density_estimate_from_mass_profile(self, mass_enclosed):
        radii = np.logspace(np.log10(self._min_rad), np.log10(self._max_rad), self._nbins + 1)

        shell_volume = 4/3 * np.pi * (radii[1:]**3 - radii[:-1]**3)

        shell_mass = np.diff(np.concatenate(([0], mass_enclosed)))
        density_estimate = shell_mass / shell_volume
        return density_estimate 
    
    def requires_property(self):
        return [f'{self.particle_name}_mass_enclosed'] + super().requires_property()
    
class FlamingoDmDensityFromMassProfileRelative(FlamingoDensityFromMassProfileRelative):
    particle_name = "dm"
    names = "dm_density_from_mass_r200m_relative",

class FlamingoDmDensityFromMassProfileAbsolute(FlamingoDensityFromMassProfileAbsolute):
    particle_name = "dm"
    names = "dm_density_from_mass",

class FlamingoAllDensityFromMassProfileRelative(FlamingoDensityFromMassProfileRelative):
    particle_name = "all"
    names = "all_density_from_mass_r200m_relative",

class FlamingoAllDensityFromMassProfileAbsolute(FlamingoDensityFromMassProfileAbsolute):
    particle_name = "all"
    names = "all_density_from_mass",

class FlamingoPotentialFromMassProfileAbsolute(LiveHaloProperties, FlamingoDensityProfileAbsolute):
    names = "all_potential_from_mass",

    def calculate(self, data, existing_properties):
        G = pynbody.units.G.in_units("kpc erg Msol^-2")
        mass_enclosed = existing_properties['all_mass_enclosed']
        radii = np.logspace(np.log10(self._min_rad), np.log10(self._max_rad), self._nbins+1)[1:]

        # potential with zero at r=0
        # derivation assumes r^-2 density profile on average
        # see notes 1/1/26
        mass_enclosed_inner_edge = np.concatenate(([0], mass_enclosed[:-1]))
        delta_mass_enclosed = mass_enclosed - mass_enclosed_inner_edge

        #G = 8.552e37 # kpc ergs / Msol^2

        r_inner = np.concatenate(([0], radii[:-1]))
        r_outer = radii

        t1 = G*mass_enclosed_inner_edge*(1./r_inner - 1./r_outer)
        t1[0] = 0.0 # correction for r=0, M=0 endpoint
        t2 = G*delta_mass_enclosed / r_outer

        potential = (t1+t2).cumsum() + t2/2
        return potential,

    def requires_property(self):
        return ['all_mass_enclosed'] + super().requires_property()


class FlamingoFlowEnergyWithPotentialAbsolute(FlamingoPotentialFromMassProfileAbsolute):
    names = "gas_energy_with_potential_inflow", "gas_energy_with_potential_outflow"

    def calculate(self, data, existing_properties):
        potential, = super().calculate(data, existing_properties)
        # 1e6 below converts from yr^-1 (mdot properties stored this way) to Myr^-1 (energy properties)
        energy_inflow = existing_properties['gas_energy_inflow'] - 1e6 * potential * existing_properties['gas_mdot_inflow']
        energy_outflow = existing_properties['gas_energy_outflow'] + 1e6 * potential * existing_properties['gas_mdot_outflow']
        return energy_inflow, energy_outflow
    
    def requires_property(self):
        return ['gas_energy_inflow', 'gas_mdot_inflow', 'gas_energy_outflow', 'gas_mdot_outflow'] + super().requires_property()


class EntropyCoolingRate(LiveHaloProperties, FlamingoDensityProfileRelative):
    names = "gas_entropy_cooling_rate_r200m_relative",

    def calculate(self, data, existing_properties):
        # cooling rate in units of entropy per time, i.e. K Myr^-1
        # this is mdot * T / rho^(2/3)
        return existing_properties['gas_entropy_r200m_relative'] / existing_properties['gas_cooltime_r200m_relative'],

    def requires_property(self):
        return ['gas_entropy_r200m_relative', 'gas_cooltime_r200m_relative'] + super().requires_property()

class EnthalpyProfile(LiveHaloProperties):
    names = "_",

    def requires_property(self):
        names = ['gas_energy_inflow', 'gas_mdot_inflow', 'gas_temp_inflow', 
                 'gas_energy_outflow', 'gas_mdot_outflow', 'gas_temp_outflow']
        if self._is_relative:
            names = [n+"_r200m_relative" for n in names]
        return names + super().requires_property()
    
    def calculate(self, data, existing_properties):
        # factor is k / (mu * m_p) where mu = 0.59
        factor = (pynbody.units.k / (0.59 * pynbody.units.m_p)).in_units("erg Msol^-1 K^-1")
        factor *= 1e6 # convert from yr^-1 to Myr^-1 for consistency with other energy properties

        post = "_r200m_relative" if self._is_relative else ""

        # NB temp_inflow is weighted by mass flux, so multiplying by mdot_inflow gives back the pressure part
        # of the enthalpy flow rate exactly. 
        enthalpy_inflow = existing_properties['gas_energy_inflow' + post] - \
             factor * existing_properties['gas_temp_inflow' + post] * existing_properties['gas_mdot_inflow' + post]
        enthalpy_outflow = existing_properties['gas_energy_outflow' + post] + \
             factor * existing_properties['gas_temp_outflow' + post] * existing_properties['gas_mdot_outflow' + post]
        return enthalpy_inflow, enthalpy_outflow
    
    def plot_ylabel(self):
        return r"Enthalpy flow/$_{\rm inflow}/erg Myr^{-1}$", r"Enthalpy flow/$_{\rm outflow}/erg Myr^{-1}$"
    
class EnthalpyProfileRelative(EnthalpyProfile, FlamingoDensityProfileRelative):
    names = "gas_enthalpy_inflow_r200m_relative", "gas_enthalpy_outflow_r200m_relative"
    _is_relative = True 

class EnthalpyProfileAbsolute(EnthalpyProfile, FlamingoDensityProfileAbsolute):
    names = "gas_enthalpy_inflow", "gas_enthalpy_outflow"
    _is_relative = False

class GravitationalHeatingRate(LiveHaloProperties, FlamingoDensityProfileAbsolute):
    names = "gas_gravitational_heating_enclosed", "gas_pressure_heating_enclosed", "gas_net_heating_enclosed"

    conv_ratio_grav_term = pynbody.units.Unit("G Msol^2 yr^-1 kpc^-1").in_units("erg Myr^-1")
    conv_ratio_pressure_term = pynbody.units.Unit("Msol km^3 s^-3 kpc^-1").in_units("erg Myr^-1")

    def calculate(self, _, existing_properties):
        net_mdot = existing_properties['gas_mdot_inflow'] + existing_properties['gas_mdot_outflow']
        net_mdot[np.isnan(net_mdot)] = 0.0 # handle zero mdot cases, which would otherwise cause issues below

        mass_cumulative = existing_properties['all_mass_enclosed']
        mass_cumulative_mid_shell = (mass_cumulative + np.concatenate(([0], mass_cumulative[:-1]))) / 2

        r = np.logspace(np.log10(self._min_rad), np.log10(self._max_rad), self._nbins + 1) # kpc
        r_central = np.sqrt(r[:-1]*r[1:]) # geometric mean radius of each shell
        dr = np.diff(r) 

        # 1e6 below converts from yr^-1 (mdot properties stored this way) to Myr^-1 (energy properties)
        per_shell_grav = (self.conv_ratio_grav_term * net_mdot * mass_cumulative_mid_shell * dr / r_central**2).view(np.ndarray)
        

        pressure = existing_properties['gas_p']
        d_pressure_dr = np.gradient(pressure, r_central)
        vr = existing_properties['gas_vr'] 

        per_shell_pressure = (4*np.pi * r_central**2 * dr * (vr * d_pressure_dr) * self.conv_ratio_pressure_term).view(np.ndarray)

        enclosed = ((per_shell_pressure-per_shell_grav )).cumsum()
        

        return per_shell_grav.cumsum(), per_shell_pressure.cumsum(), enclosed
    
    def requires_property(self):
        return ['all_mass_enclosed', 'gas_mdot_inflow', 'gas_mdot_outflow', 'gas_density', 'gas_p', 'gas_vr']


class ShellFlippedMdot(LiveHaloProperties, FlamingoDensityProfileRelative):
    names = 'dm_mdot_alt_r200m_relative',

    def calculate(self, data, existing_properties):
        mdot_alt = np.zeros_like(existing_properties['dm_mdot_outflow_r200m_relative'])
        mdot_alt[1:] = -existing_properties['dm_mdot_outflow_r200m_relative'][:-1]-existing_properties['dm_mdot_inflow_r200m_relative'][1:]
        
        return mdot_alt,

    def requires_property(self):
        return ['dm_mdot_outflow_r200m_relative', 'dm_mdot_inflow_r200m_relative'] + super().requires_property()
    
    def plot_ylabel(self):
        return r"Alt DM Mdot",

class AGNEnergyRate(LiveHaloProperties):
    names = "agn_energy_rate",

    _soap_mass_rate_unit = pynbody.units.Unit("1.988e43 g") / pynbody.units.Unit("3.086e19 s")
    _energy_per_mass_unit = pynbody.units.Unit("0.015 c^2") # efficiency factor

    _soap_mass_rate_to_ergs_per_myr = (_soap_mass_rate_unit * _energy_per_mass_unit).in_units("erg Myr^-1")

    def calculate(self, _, existing_properties):
        return existing_properties['InclusiveSphere_30kpc_MostMassiveBlackHoleAccretionRate'] * self._soap_mass_rate_to_ergs_per_myr, 

    def requires_property(self):
        return ['InclusiveSphere_30kpc_MostMassiveBlackHoleAccretionRate']

class SNEnergyRate(LiveHaloProperties):
    names = "sne_energy_rate",
    _soap_mass_rate_unit = pynbody.units.Unit("1.988e43 g") / pynbody.units.Unit("3.086e19 s")
    _energy_per_mass_unit = pynbody.units.Unit("1.18e49 erg Msol^-1") # Schaye 23, 2.3.4 
    _fSN = 0.238 # alert! should change this for different sims, this is for fiducial m9

    _soap_mass_rate_to_ergs_per_myr = (_soap_mass_rate_unit * _energy_per_mass_unit).in_units("erg Myr^-1")

    def calculate(self, _, existing_properties):
        return existing_properties['InclusiveSphere_30kpc_StarFormationRate'] * self._soap_mass_rate_to_ergs_per_myr, 
    
    def requires_property(self):
        return ['InclusiveSphere_30kpc_StarFormationRate']

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
def ps20_cooling(profile: pynbody.analysis.profile.Profile):
    sim = profile.sim 
    cooling_erg_per_s_per_particle = (sim['ps20_cooling_rate']*sim['mass']).in_units('erg Myr^-1')
    cooling_erg_per_s_per_bin = pynbody.array.SimArray(np.zeros(profile.nbins), 'erg Myr^-1')
    nH = sim['rho'].in_units('m_p cm^-3')*0.76 # num hydrogens per cm^3, assuming primordial composition
    temp_floor = 8000.*(nH/0.1)**0.333333 # Schaye eq 1.
    temp_floor.units="K"
    mask = sim['temp'] > 1e5 
    cooling_erg_per_s_per_particle*=mask

    for i in range(profile.nbins):
        cooling_erg_per_s_per_bin[i] = cooling_erg_per_s_per_particle[profile.binind[i]].sum()

    return cooling_erg_per_s_per_bin.in_units('erg Myr^-1')

@pynbody.analysis.profile.Profile.profile_property
def ps20_cooling_cumulative(profile: pynbody.analysis.profile.Profile):
    cooling_rate = profile['ps20_cooling']
    cumulative_cooling = np.cumsum(cooling_rate)
    return cumulative_cooling

@pynbody.analysis.profile.Profile.profile_property
def enthalpy_cumulative(profile: pynbody.analysis.profile.Profile):
    sim = profile.sim
    enthalpy_per_particle = (5./3.) * (sim['u'] * sim['mass']).in_units('erg')
    enthalpy_per_bin = np.array([enthalpy_per_particle[profile.binind[i]].sum() for i in range(profile.nbins)])
    return enthalpy_per_bin.cumsum()

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

@pynbody.analysis.profile.Profile.profile_property
def energy_flux(profile: pynbody.analysis.profile.Profile):
    """
    Energy flux = integral rho v_r (v^2/2 + 3/2 kT/(mu m_p)) r^2 d omega

    Estimate in a spherical shell of thickness delta r:
    energy flux = integral_r0^(r0+delta r) dr r^2 d omega (rho v_r (v^2/2 + 3/2 kT/(mu m_p))) / delta r
                = integral dV (rho v_r (v^2/2 + 3/2 kT/(mu m_p))) / delta r
                = sum m v_r (v^2/2 + 3/2 kT/(mu m_p)) / delta r
    
    profile['energy_flow_integrand'] gives mass-weighted mean of v_r (v^2/2 + 3/2 kT/(mu m_p)), while profile['mass'] gives the mass in each shell,
    so the product is sum m v_r (v^2/2 + 3/2 kT/(mu m_p)) in each shell.
    
    Note gravitational energy is neglected here; to include it, see FlamingoFlowEnergyWithPotentialAbsolute
    
    Also note that this is pure internal energy flux, NOT enthalpy flux. The enthalpy flux is obtained by
    adding mdot * kT/(mu m_p), as done in EnthalpyProfile.
    """
    ar = profile['energy_flow_integrand'] * profile['mass'] / np.diff(profile['bin_edges'])
    ar.units = profile['mass'].units * profile['energy_flow_integrand'].units / profile['bin_edges'].units
    return ar.in_units('erg Myr^-1')

@pynbody.derived_array
def vr_smoothed(f: pynbody.snapshot.SimSnap):
    """Smoothed radial velocity field, using SPH smoothing."""
    ar = f['vr']
    f.gas.build_tree()
    f.gas['smooth']
    ar_gas = f.gas.kdtree.sph_mean(f.gas['vr'])
    ar[f._get_family_slice(pynbody.family.gas)] = ar_gas
    return ar
