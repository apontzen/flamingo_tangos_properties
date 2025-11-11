import matplotlib.pyplot as p
import numpy as np
import tangos as db
import flamingo_tangos as ft


class NoHalosInStackError(ValueError):
    pass

def get_xs(ts, property_name, profile):
    try:
        prop = ts.halos[2].get_description(property_name)
        return prop.plot_x_values(profile)
    except:
        num_bins = len(profile)
        return np.linspace(np.log10(0.01), np.log10(3.0), num_bins)
    
def get_labels(ts, property_name):
    try:
        prop = ts.halos[2].get_description(property_name)
        ylabs = prop.plot_ylabel()
        xlab = prop.plot_xlabel()
        return xlab, ylabs[prop.index_of_name(property_name)]
    except:
        return "?", "?"
def get_stack(property_name, M_min, M_max, cut=None, earlier=None,
            use_log=False, timestep_name="L0200%HYDRO%/%8%",
              use_percentile=None,weight_by=None):
    ts = db.get_timestep(timestep_name)

    if cut is not None:
        # cut should be of the form: (variable, 'upper' or 'lower')
        # verify:
        if not (isinstance(cut, tuple) and (len(cut) == 2 or len(cut) == 3)):
            raise ValueError("Cut must be of the form (variable, 'upper', [value]) or (variable, 'lower', [value])")
        if len(cut) == 2:
            cut_variable, cut_upper_or_lower = cut
            cut_value = None
        else:
            cut_variable, cut_upper_or_lower, cut_value = cut
        M_and_cutvar = 'M200m()', cut_variable
    else:
        M_and_cutvar = 'M200m()',

    if earlier is not None:
        if earlier>0:
            property_name_with_rel = f"earlier({earlier}).{property_name}"
        elif earlier<0:
            property_name_with_rel = f"later({-earlier}).{property_name}"
        else:
            property_name_with_rel = property_name
    else:
        property_name_with_rel = property_name

    if weight_by:
        profiles, weights, *M_and_cutvar = ts.calculate_all(property_name_with_rel, weight_by, *M_and_cutvar)
        profiles *= weights
    else:
        profiles, *M_and_cutvar = ts.calculate_all(property_name_with_rel, *M_and_cutvar)

    if cut is not None:
        M200m, cut_var = M_and_cutvar
        mask = (M200m>10**M_min)*(M200m<10**M_max)
        if cut_value is None:
            cut_value = np.median(cut_var[mask])
            print(f"Median {cut_variable} for halos with 10^{M_min} < M200m/Msol < 10^{M_max} is {cut_value:.3e}")
        if cut_upper_or_lower == 'upper':
            mask *= (cut_var > cut_value)
        elif cut_upper_or_lower == 'lower':
            mask *= (cut_var <= cut_value)
        else:
            raise ValueError("Cut must be of the form (variable, 'upper') or (variable, 'lower')")
    else:
        M200m, = M_and_cutvar
        mask = (M200m>10**M_min)*(M200m<10**M_max)

    num_included = mask.sum()
    if num_included == 0 :
        raise NoHalosInStackError("No halos in stack")

    if use_percentile is not None:
        mean_profile = np.nanpercentile([p for p in profiles[mask]], use_percentile, axis=0)
        err_profile = 0.0
    elif 'rho' in property_name:
        # zeros should be counted, otherwise biased mass estimator
        if use_log:
            mean_profile = np.exp(np.nansum([p for p in log_profiles], axis=0)/num_included)
        else:
            mean_profile = np.nansum([p for p in profiles[mask]], axis=0)/num_included
        err_profile = mean_profile/np.sqrt(num_included)
    else:
        # nan bins should not be counted
        mean_profile, err_profile = _get_mean_of_profiles(profiles, use_log, mask)
    
    xs = get_xs(ts, property_name, mean_profile)
    labels = get_labels(ts, property_name)
    return mean_profile, err_profile, xs, labels

def _get_mean_of_profiles(profiles, use_log, mask):
    log_profiles = []
    for p in profiles[mask]:
        if (p<0).sum() > (p>0).sum():
            p = -p
        ln_p = np.log(p)
        ln_p[ln_p==-np.inf] = np.nan
        log_profiles.append(ln_p)
    if use_log:
        mean_profile = np.exp(np.nanmean([p for p in log_profiles], axis=0))
    else:
        mean_profile = np.nanmean([p for p in profiles[mask]], axis=0)
    num_included = mask.sum()

    err_log_profile = (np.nanstd([p for p in log_profiles], axis=0)/np.sqrt(num_included))
    err_profile = mean_profile * err_log_profile
    return mean_profile, err_profile

def make_flow_ratio_plot(prop_name = 'gas_mdot_inflow', M_min=12.5, M_max=13.0, box1="L0200N0360_HYDRO_STRONGEST_AGN", box2="L0200N0360_HYDRO_WEAK_AGN", tsnum=1):
    try:
        profile1, uncertainty1, xs, labels = get_stack(prop_name, M_min, M_max, timestep_name=f"{box1}/%{tsnum}.hdf5")
        profile2, uncertainty2, _, _ = get_stack(prop_name, M_min, M_max, timestep_name=f"{box2}/%{tsnum}.hdf5")
    except NoHalosInStackError:
        print(f"No halos in stack for {(M_min, M_max)}")
        return 
    
    r = 10**xs

    if 'inflow' in prop_name:
        ratio_profile = profile2 - profile1
    else:
        ratio_profile = profile1 - profile2
    ratio_uncertainty = ratio_profile * np.sqrt((uncertainty1/profile1)**2 + (uncertainty2/profile2)**2)
    
    p.plot(r, ratio_profile, label=f"$10^{{{M_min}}} < M_{{200m}} / M_{{\\odot}} < 10^{{{M_max}}}$")
    p.fill_between(r, ratio_profile - ratio_uncertainty, ratio_profile + ratio_uncertainty, alpha=0.2)
    p.xlabel(labels[0])
    p.ylabel(labels[1])

def make_flow_ratio_plots(prop_name = 'gas_mdot_inflow', tsnum=1):
    p.figure(figsize=(8, 6))
    for ra in ranges:
        make_flow_ratio_plot(M_min=ra[0], M_max=ra[1], prop_name=prop_name, tsnum=tsnum)
    p.title("Inflow Ratio Profile")
    p.legend()

def make_plot(name='rho', M_min=12.5, M_max=13.0, with_guide=False,
              relative=True, exclusive=False, with_exclusive=False,
              with_alternative_ts=None, particle='gas',
              get_stack_kwargs={}, 
              plot_kwargs={}, norm_guide=False):
    
    if name.endswith("()"):
        is_function = True
        name = name[:-2]
    else:
        is_function = False
    
    # Determine base property name
    if relative:
        prop_name = f'{name}_r200m_relative'
    else:
        prop_name = name
    if exclusive:
        prop_name += "_exclusive"

    
    if particle == 'ratio':
        # Create gas and dm property names
        gas_prop_name = f'gas_{prop_name}'
        dm_prop_name = f'dm_{prop_name}'
        
        try:
            gas_profile, gas_uncertainty, xs, labels = get_stack(gas_prop_name, M_min, M_max, **get_stack_kwargs)
            dm_profile, dm_uncertainty, _, _ = get_stack(dm_prop_name, M_min, M_max, **get_stack_kwargs)
        except NoHalosInStackError:
            print(f"No halos in stack for {(M_min, M_max)}")
            return 
        
        # Calculate ratio
        profile = gas_profile / dm_profile
        # Propagate uncertainty (assuming independent errors)
        uncertainty = profile * np.sqrt((gas_uncertainty/gas_profile)**2 + (dm_uncertainty/dm_profile)**2)
        
    else:
        prop_name = f'{particle}_{prop_name}'

        if is_function:
            prop_name += '()'
        
        try:
            profile, uncertainty, xs, labels = get_stack(prop_name, M_min, M_max, **get_stack_kwargs)
        except NoHalosInStackError:
            print(f"No halos in stack for {(M_min, M_max)}")
            return 

    r = 10**xs

    if norm_guide:
        if name == 'entropy':
            pro_ks = (r)**(1.1)
        else:
            pro_ks = (r)**-2
        profile/=pro_ks
        uncertainty /= pro_ks
    
    if (profile<=0).all():
        profile = -profile

    plot_kwargs = {'label': f"$10^{{{M_min}}} < M_{{200m}} / M_{{\\odot}} < 10^{{{M_max}}}$"} | plot_kwargs
    if name == 'mdot':
        main_line = p.plot(r, profile, **plot_kwargs)
        p.plot(r, -profile, color=main_line[0].get_color(), 
               **(plot_kwargs | {'alpha': 0.2, 'label': '_nolegend_'}))
    else:
        main_line = p.plot(r, profile, **plot_kwargs)
    p.fill_between(r, profile-uncertainty, profile+uncertainty, alpha=0.2)

    if name == 'vr':
        p.semilogx()
    else:
        p.loglog()

    if name == 'mdot':
        p.ylim(bottom=10.0)

    
    
    if with_guide and not norm_guide:
        if name == 'entropy':
            pro_ks = profile[-10] * (r/r[-10])**(1.1)
            p.plot(r, pro_ks, ':', color='grey', label=r"$\propto r^{1.1}$")
        else:
            pro_rm2 = profile[-10] * (r/r[-10])**-2
            p.plot(r, pro_rm2, ':', color='grey', label=r"$\propto r^{-2}$")
    
    xlabel, ylabel = labels

    p.xlabel(xlabel)

    if norm_guide:
        ylabel = f"$r^{2}$\\, {ylabel}$"
    p.ylabel(ylabel)


    if with_exclusive:
        make_plot(name,M_min,M_max,False,relative,True,
                  plot_kwargs={'color': main_line[0].get_color(),
                               'linestyle': '--', 
                               'label': None},
                  get_stack_kwargs=get_stack_kwargs,
                  norm_guide=norm_guide,
                  particle=particle)
    
    if with_alternative_ts:
        make_plot(name, M_min, M_max, False, relative, exclusive,
                  plot_kwargs={'color': main_line[0].get_color(),
                               'alpha': 0.2,
                               'label': None},
                  get_stack_kwargs= {"timestep_name": "L0100%/%8%"} | get_stack_kwargs,
                  norm_guide=norm_guide,
                  particle=particle
                  )

#ranges = [(11.8, 12.2), (12.6, 13.0), (13.0, 13.5), (13.5, 14.0), (14.0, 15.0)]
#ranges = [(12.5, 13.0), (13.0, 13.5), (13.5, 14.0), (14.0, 15.0)]
#ranges = [(12.0, 12.5), (13.0, 13.5), (14.0, 14.5)]
ranges = [(12.6, 12.8), (12.9, 13.1), (13.2, 13.4), (13.5, 13.7)]
vars = ['density', 'entropy', 'temp', 'p']
plot_guides_for = ['density', 'entropy', 'temp', 'p']

def make_histogram(histogram_property, tsnum=8, box="L0200N0720_HYDRO_FIDUCIAL"):
    timestep_name = f"{box}/%{tsnum}.hdf5"
    ts = db.get_timestep(timestep_name)
    mass, property = ts.calculate_all('M200m()', histogram_property)
    property[~np.isfinite(property)] = np.min(property[np.isfinite(property)]) 
    for i, ra in enumerate(ranges):
        mask = (mass > 10**ra[0]) & (mass < 10**ra[1]) 
        p.hist(property[mask], bins=30, histtype='step', label=f"$10^{{{ra[0]}}} < M_{{200m}} / M_{{\\odot}} < 10^{{{ra[1]}}}$")
    p.xlabel(histogram_property)
    p.ylabel("Number of halos")

def make_profile_plots(v, tsnum=8, box="L0200N0720_HYDRO_FIDUCIAL", 
                       newfig=True, with_exclusive=False, norm_guide=False, 
                       particle='gas', plot_kwargs={}, get_stack_kwargs={}):
    timestep_name = f"{box}/%{tsnum}.hdf5"
    z = db.get_timestep(timestep_name).redshift
    print(f"Plotting {v} profiles for {timestep_name}")
    if newfig:
        p.figure(figsize=(12, 5))
    p.subplot(121)

    redshift_label = f"$z={z:.1f}$"
    if 'earlier' in get_stack_kwargs:
        earlier = int(get_stack_kwargs['earlier'])
        tsnum_earlier = tsnum - earlier
        z_earlier = db.get_timestep(f"{box}/%{tsnum_earlier}.hdf5").redshift
        redshift_label = f"sel@{redshift_label}, plot@${z_earlier:.1f}$"

    p.title(f"Relative radius profiles ({redshift_label})")
    p.gca().set_prop_cycle(None)
    for i, ra in enumerate(ranges):
        with_guide = i == 3 and v in plot_guides_for
        make_plot(v, ra[0], ra[1], with_guide=with_guide, with_exclusive=with_exclusive, relative=True,
                  with_alternative_ts=False, 
                  get_stack_kwargs=get_stack_kwargs | {'timestep_name': timestep_name},
                  norm_guide=norm_guide, particle=particle, plot_kwargs=plot_kwargs)
    if newfig:
        p.legend()
    p.subplot(122)
    p.gca().set_prop_cycle(None)
    p.title(f"Absolute radius profiles ({redshift_label})")
    for i, ra in enumerate(ranges):
        with_guide = i == 3 and v in plot_guides_for
        make_plot(v, ra[0], ra[1], with_guide=with_guide, with_exclusive=with_exclusive, relative=False,
                  with_alternative_ts=False, get_stack_kwargs=get_stack_kwargs|{'timestep_name': timestep_name},
                  norm_guide=norm_guide, particle=particle, plot_kwargs=plot_kwargs)

    if newfig:
        p.legend()

def make_profile_plots_with_cut(v, cut_variable, cut_value=None, **kwargs):
    if cut_value is not None:
        cut_upper = (cut_variable, 'upper', cut_value)
        cut_lower = (cut_variable, 'lower', cut_value)
    else:
        cut_upper = (cut_variable, 'upper')
        cut_lower = (cut_variable, 'lower')

    get_stack_kwargs = kwargs.get('get_stack_kwargs', {}) | {'cut': cut_upper}
    make_profile_plots(v, **kwargs | {'get_stack_kwargs': get_stack_kwargs})
    get_stack_kwargs = kwargs.get('get_stack_kwargs', {}) | {'cut': cut_lower}
    make_profile_plots(v, **kwargs | {'get_stack_kwargs': get_stack_kwargs, 'plot_kwargs': {'linestyle': '--'}, 'newfig': False})

def cosmic_density(redshift, particle):
    match particle:
        case 'gas':
            mean_den = ft._cosmic_baryon_density(redshift)
        case 'dm':
            mean_den = ft._cosmic_dm_density(redshift)
        case 'total':
            mean_den = ft._cosmic_mean_density(redshift)
        case _:
            raise ValueError("particle must be 'gas', 'dm', or 'total'")
    return mean_den
        
def cosmic_hubble_flow(redshift, box):
    h_z = ft._cosmic_hubble(redshift)
    r_min, r_max = p.gca().get_xlim()
    r_vals = np.logspace(np.log10(r_min), np.log10(r_max), 100)
    v_vals = h_z * r_vals
    return r_vals, v_vals

def add_cosmic_mean_density(tsnum=8, box="L0200N0360_HYDRO_FIDUCIAL", particle=None):
    redshift = db.get_timestep(f"{box}/%{tsnum}.hdf5").redshift
    mean_den = cosmic_density(redshift, particle)
    p.axhline(mean_den, color='grey', linestyle=':', label="Cosmic Mean Density")

def add_cosmic_hubble_flow(tsnum=8, box="L0200N0360_HYDRO_FIDUCIAL"):
    redshift = db.get_timestep(f"{box}/%{tsnum}.hdf5").redshift
    r_vals, v_vals = cosmic_hubble_flow(redshift, box)
    p.plot(r_vals, v_vals, color='grey', linestyle=':', label="Hubble Flow")

def add_cosmic_mean_flow(tsnum=8, box="L0200N0360_HYDRO_FIDUCIAL", particle=None):
    redshift = db.get_timestep(f"{box}/%{tsnum}.hdf5").redshift
    r_vals, v_vals = cosmic_hubble_flow(redshift, box)
    mean_den = cosmic_density(redshift, particle)  # Msol kpc^-3

    kpc_per_km = 1 / 3.086e16
    yr_per_s = 1 / 3.15576e7
    v_vals *= kpc_per_km / yr_per_s # convert km/s to kpc/yr

    area = 4 * np.pi * ((r_vals * 1e3) ** 2) # kpc^2

    flow = mean_den * v_vals * area # in units Msol kpc^-3 * kpc/yr * kpc^2 = Msol / yr

    flow_min, flow_max = p.gca().get_ylim()
    mask = (flow > flow_min) & (flow < flow_max)
    p.plot(r_vals[mask], flow[mask], color='grey', linestyle=':', label="Cosmic Mean Flow")