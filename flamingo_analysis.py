import matplotlib.pyplot as p
import numpy as np
import tangos as db
import flamingo_tangos as ft
import pandas as pd


class NoHalosInStackError(ValueError):
    pass

def get_xs(ts, property_name, profile):
    if property_name.endswith("()"):
        property_name = property_name[:-2]
    try:
        prop = db.properties.providing_class(property_name, ft.FlamingoInputHandler)(ts.simulation)
        xs = prop.plot_x_values(profile)
        return xs 
    except:
        print(f"Error in get_xs for {property_name}, using default log10 r from 0.01 to 3.0")
        num_bins = len(profile)
        return np.linspace(np.log10(0.01), np.log10(3.0), num_bins)
    
def get_labels(ts, property_name):
    if property_name.endswith("()"):
        property_name = property_name[:-2]
    try:
        prop = db.properties.providing_class(property_name, ft.FlamingoInputHandler)(ts.simulation)
        ylabs = prop.plot_ylabel()
        xlab = prop.plot_xlabel()
        return xlab, ylabs[prop.index_of_name(property_name)]
    except Exception as e:
        print(f"Error in get_labels for {property_name}: {e}")
        return "?", "?"
    
def get_stack(property_name, M_min, M_max, M_name='M200m()', cut=None, earlier=None,
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
        M_and_cutvar = M_name, cut_variable
    else:
        M_and_cutvar = M_name,
    
    if 'r200' in property_name:
        property_name_with_rel = f'({property_name}, 3.0)'
    else:
        property_name_with_rel = f'({property_name}, log10(r200m))'
    
    if earlier is not None:
        if earlier>0:
            property_name_with_rel = f"earlier({earlier}).{property_name_with_rel}"
        elif earlier<0:
            property_name_with_rel = f"later({-earlier}).{property_name_with_rel}"

    if weight_by:
        profiles, r200, weights, *M_and_cutvar = ts.calculate_all(property_name_with_rel, weight_by,  *M_and_cutvar)
        
    else:
        weights = None
        profiles, r200, *M_and_cutvar = ts.calculate_all(property_name_with_rel, *M_and_cutvar)
        

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
        if weights is not None:
            raise NotImplementedError("Weighted percentile not implemented")
        mean_profile = np.nanpercentile([p for p in profiles[mask]], use_percentile, axis=0)
        err_profile = 0.0
    elif 'rho' in property_name:
        if weights is not None:
            raise NotImplementedError("Weighted mean density not implemented")
        # zeros should be counted, otherwise biased mass estimator
        if use_log:
            mean_profile = np.exp(np.nansum([p for p in log_profiles], axis=0)/num_included)
        else:
            mean_profile = np.nansum([p for p in profiles[mask]], axis=0)/num_included
        err_profile = mean_profile/np.sqrt(num_included)
    else:
        # nan bins should not be counted
        mean_profile, err_profile = _get_mean_of_profiles(profiles, weights, use_log, mask)
    
    xs = get_xs(ts, property_name, mean_profile)
    labels = get_labels(ts, property_name)
    return mean_profile, err_profile, xs, labels, r200[mask].mean()

def _get_mean_of_profiles(profiles, weights, use_log, mask):
    log_profiles = []
    for p in profiles[mask]:
        if (p<0).sum() > (p>0).sum():
            p = -p
        ln_p = np.log(p)
        ln_p[ln_p==-np.inf] = np.nan
        log_profiles.append(ln_p)
    if use_log:
        if weights is not None:
            raise NotImplementedError("Weighted log-mean not implemented")
        mean_profile = np.exp(np.nanmean([p for p in log_profiles], axis=0))
    else:
        if weights is not None:
            mean_profile = np.nanmean([p*w for p, w  in zip(profiles[mask], weights[mask])], axis=0)
            mean_weights = np.nanmean([w for w in weights[mask]], axis=0)
            mean_profile /= mean_weights
        else:
            mean_profile = np.nanmean([p for p in profiles[mask]], axis=0)
    num_included = mask.sum()

    err_log_profile = (np.nanstd([p for p in log_profiles], axis=0)/np.sqrt(num_included))
    err_profile = mean_profile * err_log_profile
    return mean_profile, err_profile

def make_flow_ratio_plot(prop_name = 'gas_mdot_inflow', M_min=12.5, M_max=13.0, box1="L0200N0360_HYDRO_STRONGEST_AGN", box2="L0200N0360_HYDRO_WEAK_AGN", tsnum=1):
    try:
        profile1, uncertainty1, xs, labels, _ = get_stack(prop_name, M_min, M_max, timestep_name=f"{box1}/%{tsnum}.hdf5")
        profile2, uncertainty2, _, _, _ = get_stack(prop_name, M_min, M_max, timestep_name=f"{box2}/%{tsnum}.hdf5")
    except NoHalosInStackError:
        print(f"No halos in stack for {(M_min, M_max)}")
        return 
    
    r = 10**xs

    if 'inflow' in prop_name:
        ratio_profile = profile2 - profile1
    else:
        ratio_profile = profile1 - profile2
    ratio_uncertainty = ratio_profile * np.sqrt((uncertainty1/profile1)**2 + (uncertainty2/profile2)**2)
    
    p.plot(r, ratio_profile, label=f"$10^{{{M_min}}} -- 10^{{{M_max}}}$")
    p.fill_between(r, ratio_profile - ratio_uncertainty, ratio_profile + ratio_uncertainty, alpha=0.2)
    p.xlabel(labels[0])
    p.ylabel(labels[1])

def make_flow_ratio_plots(prop_name = 'gas_mdot_inflow', tsnum=1):
    p.figure(figsize=(8, 6))
    for ra in ranges:
        make_flow_ratio_plot(M_min=ra[0], M_max=ra[1], prop_name=prop_name, tsnum=tsnum)
    p.title("Inflow Ratio Profile")
    p.legend()

def make_entropy_radius_stack(M_min=12.5, M_max=13.0, box="L0200N0360_HYDRO_FIDUCIAL", tsnum=8, restriction=None, normed=True):
    property_name = 'gas_entropy_radius_histogram'
    
    if restriction is not None:
        property_name += f"_{restriction}"
    profile, mass = db.get_timestep(f"{box}/%{tsnum}.hdf5").calculate_all(property_name, 'M200m()')
    mask = (mass > 10**M_min) & (mass < 10**M_max)
    if mask.sum() == 0:
        raise NoHalosInStackError("No halos in stack")
    stacked_profile = np.nansum([p for p in profile[mask]], axis=0)

    make_entropy_radius_histogram(stacked_profile, normed)

def make_entropy_radius_histogram(stacked_profile, normed=True):
    histclass = ft.FlamingoEntropyRadiusHistogram
    extent = [np.log10(histclass._min_rad), np.log10(histclass._max_rad), np.log10(histclass._min_entropy), np.log10(histclass._max_entropy)]
    if normed:
        stacked_profile /= np.sum(stacked_profile, axis=1, keepdims=True)

    p.imshow(stacked_profile.T, aspect='auto', origin='lower', extent=extent)
    r_bins = np.logspace(np.log10(histclass._min_rad), np.log10(histclass._max_rad), stacked_profile.shape[0]+1)
    r_bins = 0.5 * (r_bins[1:] + r_bins[:-1])
    entropy_bins = np.logspace(np.log10(histclass._min_entropy), np.log10(histclass._max_entropy), stacked_profile.shape[1]+1)
    entropy_bins = 0.5 * (entropy_bins[1:] + entropy_bins[:-1])
    mean_entropy = (entropy_bins[np.newaxis, :] * stacked_profile).sum(axis=1) / stacked_profile.sum(axis=1)
    p.plot(np.log10(r_bins), np.log10(mean_entropy), color='red', label='Mean Entropy')
    p.xlabel('log10(r)')
    p.ylabel('log10(entropy)')

def make_binned_by_mass_plot(property_name, weight_property_name = None, bin_name='M200m()', num_bins=15, bin_range=(12.5, 14.5),
                             ts_name=r"%FIDUCIAL/%8%", plot_kwargs={},
                             error_range: str | tuple ='std',
                             mask_property_name = None, mask_property_value = None,
                             x_offset=0.0):
    bin_centers, binned_means, binned_range_positive, binned_range_negative = _get_binned_statistics(property_name, weight_property_name, mask_property_name, mask_property_value, bin_name, num_bins, bin_range, ts_name, error_range)

    p.errorbar(bin_centers+x_offset, binned_means, yerr=[binned_range_negative, binned_range_positive], fmt='o', **plot_kwargs)
    p.xlabel(f'log10({bin_name})')
    p.ylabel(property_name)
    p.title(f'{property_name} binned by {bin_name}')

def tabulate_by_mass(property_name, weight_property_name = None, bin_name='M200m()', num_bins=15, bin_range=(12.5, 14.5),
                     mask_property_name = None, mask_property_value = None,
                     ts_name=r"%360%FIDUCIAL/%8%", error_range: str | tuple ='std'):
    bin_centers, binned_means, binned_range_positive, binned_range_negative = _get_binned_statistics(property_name, weight_property_name, mask_property_name, mask_property_value, bin_name, num_bins, bin_range, ts_name, error_range)
    df = pd.DataFrame({
        'bin_centre': bin_centers,
        'mean': binned_means,
        'lower': binned_means - binned_range_negative,
        'upper': binned_means + binned_range_positive
    })
    pd.options.display.float_format = '{:.3g}'.format
    return df

def _calculate_all_or_return_none(timestep, *args):
    args_filtered = [a for a in args if a is not None ]
    results_filtered = timestep.calculate_all(*args_filtered)
    results = []
    j = 0
    for a in args:
        if a is None:
            results.append(None)
        else:
            results.append(results_filtered[j])
            j += 1
    return tuple(results)

def _get_binned_statistics(property_name, weight_property_name, mask_property_name, mask_property_value, bin_name, num_bins, bin_range, ts_name, error_range):

    ts = db.get_timestep(ts_name)
    mass, property, weights, maskvals = _calculate_all_or_return_none(ts, bin_name, property_name, weight_property_name, mask_property_name)
    if weight_property_name is None:
        weights = np.ones_like(property)

    
    bin_edges = np.linspace(bin_range[0], bin_range[1], num_bins+1)
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])

    if mask_property_name is None:
        mask = np.ones(len(property), dtype=np.bool_)
    else:
        # mask_property_value is a string of the following format:
        #  '>val', '<val' -> mask is obtained by thresholding at the value
        #  '>p%', '<p%' -> mask is obtained by thresholding at the given percentile per bin

        if mask_property_value.endswith('%'):
            # Percentile-based masking
            percentile = float(mask_property_value[1:-1])
            is_greater = mask_property_value[0] == '>'
            mask = np.zeros(len(maskvals), dtype=np.bool_)
            # Apply percentile threshold per bin
            for i in range(num_bins):
                bin_mask = (mass > 10**bin_edges[i]) & (mass <= 10**bin_edges[i+1])
                if bin_mask.sum() > 0:
                    threshold = np.nanpercentile(maskvals[bin_mask], percentile)
                    if is_greater:
                       mask[bin_mask] = maskvals[bin_mask] > threshold
                    else:
                        mask[bin_mask] = maskvals[bin_mask] < threshold
        else:
            # Value-based masking
            threshold = float(mask_property_value[1:])
            if mask_property_value[0] == '>':
                mask = maskvals > threshold
            else:
                mask = maskvals < threshold

        
    property = property * weights
    binned_means = []
    binned_range_positive = []
    binned_range_negative = []
    property[~np.isfinite(property)] = np.nan
    for i in range(num_bins):
        bin_mask = mask & (mass > 10**bin_edges[i]) & (mass <= 10**bin_edges[i+1])
        if bin_mask.sum() > 0:
            binned_means.append(np.nanmean(property[bin_mask])/np.nanmean(weights[bin_mask]))
            match error_range:
                case 'std':
                    variance = np.nanmean(property[bin_mask]**2/weights[bin_mask])/np.nanmean(weights[bin_mask]) - (binned_means[-1])**2
                    binned_range_positive.append(np.sqrt(variance))
                    binned_range_negative.append(np.sqrt(variance))
                case 'uncertainty':
                    variance = np.nanstd(property[bin_mask]/weights[bin_mask])**2 / bin_mask.sum()
                    binned_range_positive.append(np.sqrt(variance))
                    binned_range_negative.append(np.sqrt(variance))
                case (lower_percentile, upper_percentile):
                    lower = np.nanpercentile(property[bin_mask]/weights[bin_mask], lower_percentile)
                    upper = np.nanpercentile(property[bin_mask]/weights[bin_mask], upper_percentile)
                    lower = max(binned_means[-1] - lower, 0)
                    upper = max(upper - binned_means[-1], 0)
                    binned_range_negative.append(lower)
                    binned_range_positive.append(upper)
                case _:
                    raise ValueError("error_range must be 'std' or a tuple of (lower_percentile, upper_percentile)")
        else:
            binned_means.append(np.nan)
            binned_range_positive.append(np.nan)
            binned_range_negative.append(np.nan)
    binned_means = np.array(binned_means)
    binned_range_positive = np.array(binned_range_positive)
    binned_range_negative = np.array(binned_range_negative)
    return bin_centers,binned_means,binned_range_positive,binned_range_negative

def make_plot(name='rho', M_min=12.5, M_max=13.0, with_guide=False,
              relative=True, exclusive=False, with_exclusive=False,
              weight_by = None,
              with_alternative_ts=None, particle='gas',
              get_stack_kwargs={}, 
              plot_kwargs={}, norm_guide=False,
              mark_r200=False):
    
    if name.endswith("()"):
        is_function = True
        name = name[:-2]
    else:
        is_function = False
    
    # Determine base property name
    if relative:
        prop_name = f'{name}_r200m_relative'
        if weight_by is not None:
            weight_by += '_r200m_relative'
    else:
        prop_name = name

    if exclusive:
        prop_name += "_exclusive"
        if weight_by is not None:
            weight_by += '_exclusive'

    
    if particle == 'ratio':
        # Create gas and all property names
        gas_prop_name = f'gas_{prop_name}'
        all_prop_name = f'all_{prop_name}'

        try:
            gas_profile, gas_uncertainty, xs, labels, log10_r200 = get_stack(gas_prop_name, M_min, M_max, **(get_stack_kwargs | {'weight_by': weight_by}))
            all_profile, all_uncertainty, _, _, log10_r200 = get_stack(all_prop_name, M_min, M_max, **(get_stack_kwargs | {'weight_by': weight_by}))
        except NoHalosInStackError:
            print(f"No halos in stack for {(M_min, M_max)}")
            return 
        
        # Calculate ratio
        profile = gas_profile / all_profile
        # Propagate uncertainty (assuming independent errors)
        uncertainty = profile * np.sqrt((gas_uncertainty/gas_profile)**2 + (all_uncertainty/all_profile)**2)
        
    else:
        prop_name = f'{particle}_{prop_name}'
        weight_by = None if weight_by is None else f'{particle}_{weight_by}'

        if is_function:
            prop_name += '()'
        
        try:
            profile, uncertainty, xs, labels, log10_r200 = get_stack(prop_name, M_min, M_max, **(get_stack_kwargs | {'weight_by': weight_by}))
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

    plot_kwargs = {'label': f"$10^{{{M_min}}} - 10^{{{M_max}}} \,{{\\rm M}}_{{\\odot}}$"} | plot_kwargs
    if name == 'mdot':
        main_line = p.plot(r, profile, **plot_kwargs)
        p.plot(r, -profile, color=main_line[0].get_color(), 
               **(plot_kwargs | {'alpha': 0.2, 'label': '_nolegend_'}))
    else:
        main_line = p.plot(r, profile, **plot_kwargs)

    if mark_r200:
        # place a dot at (r200, profile at r=r200)
        from scipy.interpolate import interp1d
        interp_func = interp1d(r, profile, bounds_error=True)
        p.plot(10.**(log10_r200-3.0), interp_func(10.**(log10_r200-3.0)), 'o', color=main_line[0].get_color())

    p.fill_between(r, profile-uncertainty, profile+uncertainty, alpha=0.2, color=main_line[0].get_color(), label='_nolegend_')

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
ranges = [(12.5, 13.0), (13.0, 13.5), (13.5, 14.0), (14.0, 15.0)]
#ranges = [(13.4, 13.6)]
#ranges = [(12.6, 12.8), (12.9, 13.1), (13.2, 13.4), (13.5, 13.7)]
#ranges = [(12.5, 13.5)]
mass_name = "M200m()"
vars = ['density', 'entropy', 'temp', 'p']
#plot_guides_for = ['density', 'entropy', 'temp', 'p']
plot_guides_for = []

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
                       newfig=True, with_exclusive=False, norm_guide=False, weight_by=None,
                       particle='gas', plot_kwargs={}, get_stack_kwargs={}, ranges_override=None,
                       panels=('relative','absolute'), with_legend=True,
                       mark_r200=False):
    global ranges, mass_name 
    if ranges_override is None:
        ranges_override = ranges
    
    timestep_name = f"{box}/%{tsnum}.hdf5"
    z = db.get_timestep(timestep_name).redshift
    print(f"Plotting {v} profiles for {timestep_name}")
    n_panels = len(panels)
    if newfig:
        p.figure(figsize=(n_panels*6.9, 5.2))
    
    panel_i = 1
    

    get_stack_kwargs['M_name'] = get_stack_kwargs.get('M_name', mass_name)

    redshift_label = f"$z={z:.1f}$"
    if 'earlier' in get_stack_kwargs:
        earlier = int(get_stack_kwargs['earlier'])
        tsnum_earlier = tsnum - earlier
        z_earlier = db.get_timestep(f"{box}/%{tsnum_earlier}.hdf5").redshift
        redshift_label = f"sel@{redshift_label}, plot@${z_earlier:.1f}$"

    if 'relative' in panels:
        if n_panels>1:
            p.subplot(1, n_panels, panel_i)
        p.title(f"Relative radius profiles ({redshift_label})")
        p.gca().set_prop_cycle(None)
        for i, ra in enumerate(ranges_override):
            with_guide = i == 3 and v in plot_guides_for
            make_plot(v, ra[0], ra[1], with_guide=with_guide, with_exclusive=with_exclusive, relative=True,
                      with_alternative_ts=False, 
                      get_stack_kwargs=get_stack_kwargs | {'timestep_name': timestep_name},
                      norm_guide=norm_guide, particle=particle, plot_kwargs=plot_kwargs, weight_by=weight_by,
                      mark_r200=mark_r200)
        
        if newfig and with_legend:
            p.legend()
        panel_i += 1
    

    if 'absolute' in panels:
        if n_panels>1:
            p.subplot(1, n_panels, panel_i)
        p.gca().set_prop_cycle(None)
        p.title(f"Absolute radius profiles ({redshift_label})")
        for i, ra in enumerate(ranges_override):
            with_guide = i == 3 and v in plot_guides_for
            make_plot(v, ra[0], ra[1], with_guide=with_guide, with_exclusive=with_exclusive, relative=False,
                    with_alternative_ts=False, get_stack_kwargs=get_stack_kwargs|{'timestep_name': timestep_name},
                    norm_guide=norm_guide, particle=particle, plot_kwargs=plot_kwargs, weight_by=weight_by,
                    mark_r200=mark_r200)
        
        if n_panels == 2:
            p.gca().yaxis.tick_right()
            p.gca().yaxis.set_label_position('right')
        p.tight_layout()
        if newfig and with_legend:
            p.legend()

def remove_existing_legend():
    legend = p.gca().get_legend()
    if legend:
        legend.remove()

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
    make_profile_plots(v, **kwargs | {'get_stack_kwargs': get_stack_kwargs, 
                                      'plot_kwargs': kwargs.get('plot_kwargs', {}) | {'linestyle': '--', 'label': '_nolegend_'}, 
                                      'newfig': False})

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