import matplotlib.pyplot as p
import numpy as np
import tangos as db
import flamingo_tangos as ft
import virial_scalings as vs 

import pandas as pd
from scipy.interpolate import RegularGridInterpolator

# For entropy:
# internal units are Msol^{-2/3} kpc^2 km^2 s^{-2}
# multiply by mu mu_e^{2/3} m_p^(5/3) and convert to get keV cm^2 (mu = 0.59)
internal_to_keV_cm2 = 0.570304


class NoHalosInStackError(ValueError):
    pass

def _sample_on_circle(image, radius, n_bins):
    """Interpolate a 2-D image on a circle of given pixel radius about the centre.

    Returns (phi, values) where phi is equally spaced in [0, 2pi).
    """
    ny, nx = image.shape
    cy, cx = (ny - 1) / 2.0, (nx - 1) / 2.0
    interp = RegularGridInterpolator(
        (np.arange(ny, dtype=float), np.arange(nx, dtype=float)),
        image, method='linear', bounds_error=True)
    phi = np.linspace(0, 2 * np.pi, n_bins, endpoint=False)
    pts = np.column_stack((cy + radius * np.sin(phi),
                           cx + radius * np.cos(phi)))
    return phi, interp(pts)

def get_vr_on_circle(vx, vy, radius, n_bins):
    """Return (phi, vr) where phi is equally spaced in [0, 2pi) and vr is the
    radially outward velocity component sampled at the given pixel radius from
    the image centre.

    Parameters
    ----------
    vx, vy : 2-D arrays of shape (ny, nx)
    radius  : float, pixels from image centre
    n_bins  : int, number of equally spaced phi samples
    """
    phi, vx_ring = _sample_on_circle(vx, radius, n_bins)
    _,   vy_ring = _sample_on_circle(vy, radius, n_bins)
    vr = vx_ring * np.cos(phi) + vy_ring * np.sin(phi)
    return phi, vr

def get_xs(ts, property_name, profile):
    """Get x values for plotting a profile of the given property. 
    
    If the property provides its own x values, use those; otherwise, default to log10(r) from 0.01 to 3.0."""
    if property_name.endswith("()"):
        property_name = property_name[:-2]
    try:
        prop = db.properties.providing_class(property_name, ft.FlamingoInputHandler)(ts.simulation)
    except:
        print(f"Warning: getting x values for {property_name} failed, defaulting to FlamingoDensityProfileAbsolute")
        prop = ft.FlamingoDensityProfileAbsolute(ts.simulation)
    xs = prop.plot_x_values(profile)
    return xs 
    
        
    
def get_stack(property_name, M_min, M_max, M_name='M200m()', cut=None, earlier=None,
              use_log=False, timestep_name="L0200%HYDRO%/%8%",
              use_percentile=None,weight_by=None,bootstrap=True):
    
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
        if bootstrap:
            mean_profile, err_profile_min, err_profile_max = _get_mean_of_profiles_with_bootstrap(profiles, weights, mask)
        else:
            mean_profile, err_profile = _get_mean_of_profiles(profiles, weights, use_log, mask)
            err_profile_min = mean_profile - err_profile
            err_profile_max = mean_profile + err_profile

    xs = get_xs(ts, property_name, mean_profile)
    labels = None
    return mean_profile, err_profile_min, err_profile_max, xs, labels, r200[mask].mean()

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

def _get_mean_of_profiles_with_bootstrap(profiles, weights, mask, n_bootstrap=1000):
    if weights is None:
        weights = np.ones_like(profiles)
    p_masked = profiles[mask]
    w_masked = weights[mask]
    n = len(p_masked)

    def weighted_mean(p_arr, w_arr):
        mean_w = np.nanmean(w_arr, axis=0)
        return np.nanmean([p * w for p, w in zip(p_arr, w_arr)], axis=0) / mean_w

    mean_profile = weighted_mean(p_masked, w_masked)

    rng = np.random.default_rng()
    bootstrap_means = np.array([
        weighted_mean(p_masked[idx], w_masked[idx])
        for idx in (rng.integers(0, n, size=n) for _ in range(n_bootstrap))
    ])
    err_profile = np.nanpercentile(bootstrap_means, [16, 84], axis=0)

    return mean_profile, err_profile[0], err_profile[1]


def legend(labels, **kwargs):
    from matplotlib.legend import _get_legend_handles
    ax = p.gca()
    handles = list(_get_legend_handles([ax]))
    selected = [(h, l) for (h, l) in zip(handles, labels) if l is not None]
    p.legend([h for h, l in selected], [l for h, l in selected], **kwargs)

def make_entropy_radius_stack_in_and_out(band_percentiles=(33,67), M_min=12.5, M_max=13.0, 
                                         box="L0200N0360_HYDRO_FIDUCIAL", tsnum=8):
    make_entropy_radius_stack(normed=True, band_percentiles=band_percentiles, 
                              restriction='outflow', M_min=M_min, M_max=M_max, box=box, tsnum=tsnum)
    make_entropy_radius_stack(normed=True, band_percentiles=band_percentiles, 
                              restriction='inflow', M_min=M_min, M_max=M_max, box=box, tsnum=tsnum)
    legend(['$r_{200m}$', None, 'outflow', None, None, 'inflow']) 

def make_entropy_radius_stack(M_min=12.5, M_max=13.0, box="L0200N0360_HYDRO_FIDUCIAL", 
                              tsnum=4, restriction=None, normed=True, band_percentiles=None, 
                              earlier=None):
    property_name = 'gas_entropy_radius_histogram'
    
    if restriction is not None:
        property_name += f"_{restriction}"
    
    if earlier is not None:
        property_name = f"earlier({earlier}).{property_name}"
        
    profile, mass = db.get_timestep(f"{box}/%{tsnum}.hdf5").calculate_all(property_name, 'M200m()')
    mask = (mass > 10**M_min) & (mass < 10**M_max)
    if mask.sum() == 0:
        raise NoHalosInStackError("No halos in stack")
    stacked_profile = np.nansum([p for p in profile[mask]], axis=0)
    p.title(f"$10^{{{M_min}}} < M_{{200m}} / M_{{\\odot}} < 10^{{{M_max}}}$" + (f", {restriction}" if restriction else ""))

    r = vs.radius(10**((M_min + M_max)/2))
    p.axvline(np.log10(r), color='red', linestyle='--', label=r"$r_{200m}$")

    make_entropy_radius_histogram(stacked_profile, normed, band_percentiles=band_percentiles)

def make_entropy_radius_histogram(stacked_profile, normed=True, band_percentiles=None):
    histclass = ft.FlamingoEntropyRadiusHistogram
    extent = [np.log10(histclass._min_rad), np.log10(histclass._max_rad), np.log10(histclass._min_entropy), np.log10(histclass._max_entropy)]
    if normed:
        stacked_profile /= np.sum(stacked_profile, axis=1, keepdims=True)

    if band_percentiles is not None:
        make_entropy_radius_percentile_band(stacked_profile, lower_percentile=band_percentiles[0], upper_percentile=band_percentiles[1], median_percentile=50)
    else:
        p.imshow(stacked_profile.T, aspect='auto', origin='lower', extent=extent, cmap='gray')
        p.xlabel('log10(r/Mpc)')
        p.ylabel('log10(K/simulation unit)')
        p.colorbar().set_label(r'$p(\log_{10} K | r)$')

def _get_percentile_from_histogram(stacked_profile, percentile, normed=True):
    """Given a 2D histogram (r_bins x K_bins), compute the K value at a given percentile
    for each radial bin, assuming constant probability within each K bin."""
    histclass = ft.FlamingoEntropyRadiusHistogram
    entropy_edges = np.logspace(np.log10(histclass._min_entropy), np.log10(histclass._max_entropy), stacked_profile.shape[1] + 1)
    log_entropy_edges = np.log10(entropy_edges)

    # Normalize each radial bin to get a proper conditional PDF
    row_sums = np.sum(stacked_profile, axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    pdf = stacked_profile / row_sums

    # Compute cumulative distribution along the K axis for each r bin
    cdf = np.cumsum(pdf, axis=1)

    n_r = stacked_profile.shape[0]
    percentile_values = np.full(n_r, np.nan)
    frac = percentile / 100.0

    for i in range(n_r):
        cdf_row = cdf[i]
        if cdf_row[-1] == 0:
            continue
        # Find the bin where the CDF crosses the desired percentile
        idx = np.searchsorted(cdf_row, frac)
        if idx == 0:
            # Interpolate within the first bin
            t = frac / cdf_row[0] if cdf_row[0] > 0 else 0.0
            percentile_values[i] = log_entropy_edges[0] + t * (log_entropy_edges[1] - log_entropy_edges[0])
        elif idx >= len(cdf_row):
            percentile_values[i] = log_entropy_edges[-1]
        else:
            # Linear interpolation within the bin in log-K space
            cdf_lo = cdf_row[idx - 1]
            cdf_hi = cdf_row[idx]
            if cdf_hi == cdf_lo:
                t = 0.5
            else:
                t = (frac - cdf_lo) / (cdf_hi - cdf_lo)
            percentile_values[i] = log_entropy_edges[idx] + t * (log_entropy_edges[idx + 1] - log_entropy_edges[idx])

    return percentile_values


def _get_radial_bin_centers():
    """Return log10 of radial bin centers for the entropy-radius histogram."""
    histclass = ft.FlamingoEntropyRadiusHistogram
    r_edges = np.logspace(np.log10(histclass._min_rad), np.log10(histclass._max_rad), -1)  # placeholder
    # We don't know n_r yet, so this is deferred; use the profile shape instead.
    return histclass


def make_entropy_radius_percentile_band(stacked_profile, lower_percentile=16, upper_percentile=84,
                                            median_percentile=50, **plot_kwargs):
    """Plot a band between specified percentiles of the conditional entropy PDF at each radius."""
    histclass = ft.FlamingoEntropyRadiusHistogram
    n_r = stacked_profile.shape[0]
    r_edges = np.logspace(np.log10(histclass._min_rad), np.log10(histclass._max_rad), n_r + 1)
    log_r_centers = 0.5 * (np.log10(r_edges[1:]) + np.log10(r_edges[:-1]))

    log_k_lower = _get_percentile_from_histogram(stacked_profile, lower_percentile)
    log_k_upper = _get_percentile_from_histogram(stacked_profile, upper_percentile)
    log_k_median = _get_percentile_from_histogram(stacked_profile, median_percentile)

    valid = np.isfinite(log_k_lower) & np.isfinite(log_k_upper) & np.isfinite(log_k_median)

    default_kwargs = {'alpha': 0.3}
    band_kwargs = {**default_kwargs, **plot_kwargs}
    label = band_kwargs.pop('label', f'{lower_percentile}–{upper_percentile}th percentile')

    p.fill_between(log_r_centers[valid], log_k_lower[valid], log_k_upper[valid], label=label, **band_kwargs)
    line_kwargs = {k: v for k, v in band_kwargs.items() if k not in ('alpha',)}
    p.plot(log_r_centers[valid], log_k_median[valid], label=f'{median_percentile}th percentile', **line_kwargs)

    p.xlabel('log10(r/Mpc)')
    p.ylabel('log10(K/simulation unit)')

def plot_entropy_guide():
    mass_ar = np.linspace(12.5, 14.5, 50)
    entrop_in = vs.entropy(10**mass_ar, z=0.4)
    p.plot(mass_ar, internal_to_keV_cm2 * entrop_in, color='orange', linestyle=':', label="Virial")

def plot_temp_guide():
    mass_ar = np.linspace(12.5, 14.5, 50)
    temp_in = vs.temperature(10**mass_ar, z=0.4)
    p.plot(mass_ar, temp_in, color='orange', linestyle=':', label="Virial")

def make_binned_by_mass_plot(property_name, weight_property_name = None, 
                             bin_name='M200m()', num_bins=15, bin_range=(12.5, 14.5),
                             ts_name=r"%FIDUCIAL/%8%", plot_kwargs={},
                             error_range: str | tuple ='uncertainty-bootstrap',
                             mask_property_name = None, mask_property_value = None,
                             use_band = True, with_fit=False, fit_range=None,
                             x_offset=0.0, readoff_values_at=None):
    bin_centers, binned_means, binned_range_positive, binned_range_negative = _get_binned_statistics(property_name, weight_property_name, mask_property_name, mask_property_value, bin_name, num_bins, bin_range, ts_name, error_range)

    if use_band:
        p.plot(bin_centers + x_offset, binned_means, **plot_kwargs)
        plot_kwargs_no_label = {**plot_kwargs, 'label': None}
        p.fill_between(bin_centers + x_offset, binned_means - binned_range_negative, binned_means + binned_range_positive, alpha=0.2, **plot_kwargs_no_label)
        
    else:
        p.errorbar(bin_centers+x_offset, binned_means, yerr=[binned_range_negative, binned_range_positive], fmt='o', **plot_kwargs)

    if readoff_values_at is not None:
        readoff_values = np.interp(readoff_values_at, bin_centers, binned_means)
        readoff_range_positive = np.interp(readoff_values_at, bin_centers, binned_range_positive)
        readoff_range_negative = np.interp(readoff_values_at, bin_centers, binned_range_negative)
        for x, y, yerr_pos, yerr_neg in zip(readoff_values_at, readoff_values, readoff_range_positive, readoff_range_negative):
            print(f"At {bin_name} = 10^{x:.2f}, value = {y:.3e} (+{yerr_pos:.3e}/-{yerr_neg:.3e})")

    if with_fit:
        def power_law_model(log_mass, offset, alpha):
            return offset + alpha * log_mass
    
        from scipy.optimize import curve_fit
        log_binned_means = np.log10(binned_means)
        valid = np.isfinite(log_binned_means) & np.isfinite(bin_centers)
        if fit_range:
            valid *= bin_centers > fit_range[0]
            valid *= bin_centers < fit_range[1]
        popt, _ = curve_fit(power_law_model, bin_centers[valid], log_binned_means[valid])
        fit_line = 10**power_law_model(bin_centers, *popt)
        p.plot(bin_centers, fit_line, linestyle='--', color=plot_kwargs.get('color', 'black'), label=plot_kwargs.get('label', None) + ' fit' if plot_kwargs.get('label', None) else None)
        print(f"Fitted power-law: log_10 {property_name} = {popt[0]:.3f} + {popt[1]:.3f} * log_10 {bin_name}")

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
    mass, property, weights, maskvals = _calculate_all_or_return_none(ts, bin_name, property_name, 
                                                                      weight_property_name if weight_property_name!='median' else None, mask_property_name)
    if weight_property_name is None or weight_property_name == 'median':
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
            if weight_property_name == 'median':
                binned_means.append(np.nanmedian(property[bin_mask]/weights[bin_mask]))
            else:       
                binned_means.append(np.nanmean(property[bin_mask])/np.nanmean(weights[bin_mask]))
            # binned_means.append(np.nanmedian(property[bin_mask]/weights[bin_mask]))
            match error_range:
                case 'std':
                    variance = np.nanmean(property[bin_mask]**2/weights[bin_mask])/np.nanmean(weights[bin_mask]) - (binned_means[-1])**2
                    binned_range_positive.append(np.sqrt(variance))
                    binned_range_negative.append(np.sqrt(variance))
                case 'uncertainty':
                    variance = np.nanstd(property[bin_mask]/weights[bin_mask])**2 / bin_mask.sum()
                    binned_range_positive.append(np.sqrt(variance))
                    binned_range_negative.append(np.sqrt(variance))
                case 'uncertainty-bootstrap':
                    _, err_min, err_max = _get_mean_of_profiles_with_bootstrap(property/weights, weights, bin_mask)
                    binned_range_positive.append(err_max - binned_means[-1])
                    binned_range_negative.append(binned_means[-1] - err_min)
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

def make_plot(name='rho', M_min=12.5, M_max=13.0, 
              relative=True,
              weight_by = None, rescale=1.0,
              particle='gas',
              get_stack_kwargs={}, 
              plot_kwargs={}, 
              mark_r200=False, mark_radius=None):
    
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

    if is_function:
        prop_name += '()'

    if particle == 'ratio':
        prop_name = f'gas_{prop_name}/all_{prop_name}'
    else:
        prop_name = f'{particle}_{prop_name}'

    weight_by = None if weight_by is None else f'{particle}_{weight_by}'
   
    
    try:
        profile, profile_min, profile_max, xs, labels, log10_r200 = get_stack(prop_name, M_min, M_max, **(get_stack_kwargs | {'weight_by': weight_by}))
    except NoHalosInStackError:
        print(f"No halos in stack for {(M_min, M_max)}")
        return 

    profile*=rescale
    profile_min*=rescale
    profile_max*=rescale

    r = 10**xs

    if (profile<=0).all():
        profile = -profile
        profile_min = -profile_min
        profile_max = -profile_max

    if M_max < 100:
        label = fr"$10^{{{M_min}}} - 10^{{{M_max}}}\,{{\rm M}}_{{\odot}}$"
    else:
        label = fr"$>10^{{{M_min}}}\,{{\rm M}}_{{\odot}}$"
    plot_kwargs = {'label': label} | plot_kwargs
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
        if 'r200' in prop_name:
            mark_is_at = 1.0
        else:
            mark_is_at = 10.**(log10_r200-3.0)
        p.plot(mark_is_at, interp_func(mark_is_at), 'o', color=main_line[0].get_color())

    if mark_radius:
        # place a cross at mark_radius kpc
        from scipy.interpolate import interp1d
        interp_func = interp1d(r, profile, bounds_error=True)
        mark_is_at = mark_radius/10.**(log10_r200)
        p.plot(mark_is_at, interp_func(mark_is_at), 'x', color=main_line[0].get_color())
        
    p.fill_between(r, profile_min, profile_max, alpha=0.2, color=main_line[0].get_color(), label='_nolegend_')

    if name == 'vr':
        p.semilogx()
    else:
        p.loglog()

    if name == 'mdot':
        p.ylim(bottom=10.0)


#ranges = [(11.8, 12.2), (12.6, 13.0), (13.0, 13.5), (13.5, 14.0), (14.0, 15.0)]
ranges = [(12.5, 13.0), (13.0, 13.5), (13.5, 14.0), (14.0, np.inf)][::-1]
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

def tabulate_profile_plot_data(v, tsnum=8, box="L0200N0720_HYDRO_FIDUCIAL",
                                particle='gas', weight_by=None, rescale=1.0,
                                ranges_override=None, panels=('relative', 'absolute'),
                                get_stack_kwargs=None, mass_name=None):
    """Recompute the numeric data underlying make_profile_plots(...) and return it as a
    tidy (long-format) pandas DataFrame, i.e. one row per (panel, mass bin, radius) point.

    This reproduces exactly the same mass bins/panels/lines that make_profile_plots would
    draw, but returns the underlying numbers instead of plotting them. Only the mean
    profile values that are actually plotted as lines are included -- no bootstrap error
    ranges/bands, and no reference/comparison curves (e.g. Arnaud profiles) are output.

    Parameters mirror make_profile_plots. Columns of the returned DataFrame:
        panel, mass_bin, M_min_log10Msol, M_max_log10Msol, x_units, x, <v>
    where <v> is named after the requested property (e.g. 'p'), x is r/r200m for the
    'relative' panel or r in Mpc for the 'absolute' panel.
    """
    if ranges_override is None:
        ranges_override = ranges
    if get_stack_kwargs is None:
        get_stack_kwargs = {}
    if mass_name is None:
        mass_name = globals()['mass_name']

    timestep_name = f"{box}/%{tsnum}.hdf5"
    is_function = v.endswith('()')
    name = v[:-2] if is_function else v

    rows = []
    for panel in panels:
        relative = (panel == 'relative')
        prop_name = f'{name}_r200m_relative' if relative else name
        if is_function:
            prop_name += '()'
        if particle == 'ratio':
            prop_name = f'gas_{prop_name}/all_{prop_name}'
        else:
            prop_name = f'{particle}_{prop_name}'

        panel_weight_by = weight_by
        if panel_weight_by is not None and relative:
            panel_weight_by += '_r200m_relative'
        panel_weight_by = None if panel_weight_by is None else f'{particle}_{panel_weight_by}'

        x_units = 'r/r200m' if relative else 'Mpc'

        for M_min, M_max in ranges_override:
            kwargs = get_stack_kwargs | {'M_name': mass_name, 'timestep_name': timestep_name,
                                          'weight_by': panel_weight_by}
            try:
                profile, _profile_min, _profile_max, xs, _labels, _log10_r200 = get_stack(
                    prop_name, M_min, M_max, **kwargs)
            except NoHalosInStackError:
                continue

            profile = profile * rescale
            if (profile <= 0).all():
                profile = -profile
            x = 10.0 ** xs

            mass_bin = f">10^{M_min}" if M_max >= 100 else f"10^{M_min}-10^{M_max}"

            for x_val, y_val in zip(x, profile):
                rows.append({
                    'panel': panel,
                    'mass_bin': mass_bin,
                    'M_min_log10Msol': M_min,
                    'M_max_log10Msol': M_max,
                    'x_units': x_units,
                    'x': x_val,
                    v: y_val,
                })

    return pd.DataFrame(rows)


def make_profile_plots(v, tsnum=8, box="L0200N0720_HYDRO_FIDUCIAL", 
                       newfig=True, weight_by=None,
                       particle='gas', plot_kwargs={}, get_stack_kwargs={}, ranges_override=None,
                       panels=('relative','absolute'), with_legend=True, rescale=1.0,
                       mark_r200=False, mark_radius=None):
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
            make_plot(v, ra[0], ra[1], relative=True,
                      get_stack_kwargs=get_stack_kwargs | {'timestep_name': timestep_name},
                      particle=particle, plot_kwargs=plot_kwargs, weight_by=weight_by,
                      mark_r200=mark_r200, mark_radius=mark_radius, 
                      rescale=rescale)
        
        if newfig and with_legend:
            p.legend()
        panel_i += 1
    

    if 'absolute' in panels:
        if n_panels>1:
            p.subplot(1, n_panels, panel_i)
        p.title(f"Absolute radius profiles ({redshift_label})")
        p.gca().set_prop_cycle(None)
        for i, ra in enumerate(ranges_override):
            make_plot(v, ra[0], ra[1], relative=False,
                    get_stack_kwargs=get_stack_kwargs|{'timestep_name': timestep_name},
                    particle=particle, plot_kwargs=plot_kwargs, weight_by=weight_by,
                    mark_r200=mark_r200, mark_radius=mark_radius, rescale=rescale)
        
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


def make_stacked_entropy_image_plot(timestep_name, axis='13', M_min=12.8, M_max=13.2,
                                    vmin=1.7, vmax=3.0, cmap='RdYlBu_r', 
                                    with_colorbar=False, with_quiverkey=False, panel_label=None):
    mean, _, _, _, _ = get_stack(f'aligned_{axis}_entropy_image', timestep_name=timestep_name, M_min=M_min, M_max=M_max, bootstrap=False)
    
    mean*=internal_to_keV_cm2

    p.imshow(np.log10(mean), origin='lower', extent=(-2,2,-2,2), vmin=vmin, vmax=vmax, cmap=cmap)

    mean_vx, _, _, _, _ = get_stack(f'aligned_{axis}_vx_image', timestep_name=timestep_name, M_min=M_min, M_max=M_max, bootstrap=False)
    mean_vy, _, _, _, _ = get_stack(f'aligned_{axis}_vy_image', timestep_name=timestep_name, M_min=M_min, M_max=M_max, bootstrap=False)
    axis_names = {'1': 'z', '2': 'y', '3': 'x'}
    axis_name_x = axis_names[axis[1]]
    axis_name_y = axis_names[axis[0]]
    p.xlabel(f"${axis_name_x} / r_{{200m}}$")
    p.ylabel(f"${axis_name_y} / r_{{200m}}$")

    if panel_label:
        p.gca().text(0.02, 0.98, f"{panel_label} ${axis_name_x}-{axis_name_y}$", transform=p.gca().transAxes,
            color='black', va='top', ha='left',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='black'))


    ax = p.gca()
    for scale in [1.0, 2.0]:
        ax.add_patch(p.Circle((0, 0), scale, fill=False, color='black', 
                                linewidth=1))

    p.xticks([-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5], 
             ["$-1.5$", "$-1.0$", "$-0.5$", "$0.0$", "$0.5$", "$1.0$", "$1.5$"])
    p.yticks([-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5], 
             ["$-1.5$", "$-1.0$", "$-0.5$", "$0.0$", "$0.5$", "$1.0$", "$1.5$"])

    mean_vx = mean_vx[1::2, 1::2]
    mean_vy = mean_vy[1::2, 1::2]
    vector_resolution = len(mean_vx)
    width = 4.0
    pixel_size = width/vector_resolution

    X, Y = np.meshgrid(np.linspace(-width / 2 + pixel_size/2, width / 2 - pixel_size/2, vector_resolution),
                       np.linspace(-width / 2 + pixel_size/2, width / 2 - pixel_size/2, vector_resolution))

    # Invert: dark background → light arrow, light background → dark arrow
    Q = ax.quiver(X, Y, mean_vx, mean_vy, scale=2500., color='k')

    if with_quiverkey:
        from pynbody.plot import util
        qk = util.PynbodyQuiverKey(Q, 0.8, 0.1,
                                    500, "500 km/s",
                                    color='k', labelcolor='k',
                                    boxedgecolor='k', boxfacecolor='w')
        qk.set_zorder(6)
        p.gca().add_artist(qk)

    if with_colorbar:
        p.colorbar().set_label(r"$\log_{10} \langle K \rangle / {\rm kpc^2\,km^2\,M_{\odot}^{-2/3}\,s^{-2}}$")

def make_stacked_vr_plot(timestep_name, axis='13', M_min=12.8, M_max=13.2,
                         r_frac=1.0, n_bins=64, plot_kwargs={}, with_quadratic_reference=False,
                         with_mean_reference=False, stack_mode='mean'):
    """Plot the stacked radial velocity as a function of azimuthal angle at a
    given fraction of r200m.

    Parameters
    ----------
    r_frac      : float, radius in units of r200m at which to evaluate vr
    n_bins      : int, number of equally spaced phi bins
    stack_mode  : str, how to combine per-halo vr profiles (currently 'mean')
    """
    ts = db.get_timestep(timestep_name)
    vx_images, _, mass = ts.calculate_all(
        f'(aligned_{axis}_vx_image, log10(r200m))', 'M200m()')
    vy_images, _, _ = ts.calculate_all(
        f'(aligned_{axis}_vy_image, log10(r200m))', 'M200m()')

    mask = (mass > 10**M_min) & (mass < 10**M_max)
    if mask.sum() == 0:
        raise NoHalosInStackError("No halos in stack")

    # Images span -2 to +2 r200m; convert r_frac to pixels
    n_pixels = vx_images[mask][0].shape[0]
    radius_pixels = r_frac * n_pixels / 4.0

    vr_all = []
    for vx, vy in zip(vx_images[mask], vy_images[mask]):
        phi, vr_i = get_vr_on_circle(vx, vy, radius_pixels, n_bins)
        vr_all.append(vr_i)

    vr_array = np.array(vr_all)
    

    axis_names = {'1': 'z', '2': 'y', '3': 'x'}
    axis_name_x = axis_names[axis[1]]
    axis_name_y = axis_names[axis[0]]

    if stack_mode == 'mean':
        vr = np.nanmean(vr_array, axis=0)
        main_line = p.plot(phi, vr, **plot_kwargs)
    else:
        lower_p, upper_p = stack_mode
        vr_lower = np.nanpercentile(vr_array, lower_p, axis=0)
        vr_upper = np.nanpercentile(vr_array, upper_p, axis=0)
        vr = np.nanmean(vr_array, axis=0)
        print(f"vr mean range: {vr.max():.1f} to {vr.min():.1f} km/s")
        main_line = p.plot(phi, vr, **plot_kwargs)
        p.fill_between(phi, vr_lower, vr_upper,
                       alpha=0.2, color=main_line[0].get_color(), label='_nolegend_')

    color = main_line[0].get_color()

    if with_mean_reference:
        p.axhline(np.mean(vr), color=color, linestyle=':')

    if with_quadratic_reference:
        # Reference curve from stacked flow_alignment_eigvals.
        # Eigenvalues are stored reversed: index 0 = axis 3, 1 = axis 2, 2 = axis 1.
        # axis[1] is the image x-direction, axis[0] is the image y-direction.
        ts = db.get_timestep(timestep_name)
        eigvals, mass = ts.calculate_all('flow_alignment_eigvals', 'M200m()')
        mask = (mass > 10**M_min) & (mass < 10**M_max)
        if mask.sum() > 0:
            mean_eigvals = np.nanmean(np.stack(eigvals[mask]), axis=0)
            if axis=='13':
                lambda_a = mean_eigvals[0]
                lambda_b = mean_eigvals[2]
            elif axis=='12':
                lambda_a = mean_eigvals[1]
                lambda_b = mean_eigvals[2]
            phi_ref = np.linspace(0, 2 * np.pi, 256, endpoint=False)
            vr_ref = lambda_a * np.cos(phi_ref)**2 + lambda_b * np.sin(phi_ref)**2
            p.plot(phi_ref, vr_ref, ':', color=color)

    p.axhline(0, color='grey', linestyle=':')
    p.xlabel(r"$\phi$")
    p.ylabel(r"$v_r$ [km s$^{-1}$]")
    p.title(f"$v_r$ at $r = {r_frac:.2f}\\,r_{{200m}}$, "
            f"$10^{{{M_min}}} < M_{{200m}}/M_\\odot < 10^{{{M_max}}}$, "
            f"({axis_name_x}–{axis_name_y} plane)")
    p.xticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi],
             ["$0$", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"])

def make_stacked_entropy_angle_plot(timestep_name, axis='13', M_min=12.8, M_max=13.2,
                                    r_frac=1.0, n_bins=64, plot_kwargs={},
                                    stack_mode='mean', show_reference_values=False):
    """Plot the stacked log10 entropy as a function of azimuthal angle at a
    given fraction of r200m.

    Parameters
    ----------
    r_frac     : float, radius in units of r200m at which to evaluate the entropy
    n_bins     : int, number of equally spaced phi bins
    stack_mode : 'mean' or a tuple (lower_p, upper_p) of percentiles
    """
    ts = db.get_timestep(timestep_name)
    images, _, mass = ts.calculate_all(
        f'(aligned_{axis}_entropy_image, log10(r200m))', 'M200m()')

    mask = (mass > 10**M_min) & (mass < 10**M_max)
    if mask.sum() == 0:
        raise NoHalosInStackError("No halos in stack")

    n_pixels = images[mask][0].shape[0]
    radius_pixels = r_frac * n_pixels / 4.0

    entropy_all = []
    for img in images[mask]:
        phi, ring = _sample_on_circle(img * internal_to_keV_cm2, radius_pixels, n_bins)
        entropy_all.append(np.log10(ring))

    entropy_array = np.array(entropy_all)

    axis_names = {'1': 'z', '2': 'y', '3': 'x'}
    axis_name_x = axis_names[axis[1]]
    axis_name_y = axis_names[axis[0]]

    if stack_mode == 'mean':
        entropy_ring = np.nanmean(entropy_array, axis=0)
        main_line = p.plot(phi, entropy_ring, **plot_kwargs)
    else:
        lower_p, upper_p = stack_mode
        entropy_lower = np.nanpercentile(entropy_array, lower_p, axis=0)
        entropy_upper = np.nanpercentile(entropy_array, upper_p, axis=0)
        entropy_ring = np.nanmedian(entropy_array, axis=0)
        main_line = p.plot(phi, entropy_ring, **plot_kwargs)
        p.fill_between(phi, entropy_lower, entropy_upper,
                       alpha=0.2, color=main_line[0].get_color(), label='_nolegend_')

    color = main_line[0].get_color()

    if show_reference_values:
        # Stacked inflow / outflow reference values at this radius
        ts = db.get_timestep(timestep_name)
        log10_r = np.log10(r_frac)
        for restriction, linestyle, ref_label in [('inflow',  '--', 'inflow'),
                                                ('outflow', ':',  'outflow')]:
            prop = f'at({log10_r}, gas_entropy_{restriction}_r200m_relative)'
            vals, mass = ts.calculate_all(prop, 'M200m()')
            mask = (mass > 10**M_min) & (mass < 10**M_max)
            if mask.sum() == 0:
                continue
            mean_val = np.nanmean(vals[mask]) * internal_to_keV_cm2
            p.axhline(np.log10(mean_val), color=color, linestyle=linestyle,
                    label=ref_label)

    p.xlabel(r"$\phi$")
    p.ylabel(r"$\log_{10}(K\,/\,{\rm keV\,cm^2})$")
    p.title(f"Entropy at $r = {r_frac:.2f}\\,r_{{200m}}$, "
            f"$10^{{{M_min}}} < M_{{200m}}/M_\\odot < 10^{{{M_max}}}$, "
            f"({axis_name_x}–{axis_name_y} plane)")
    p.xticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi],
             ["$0$", r"$\pi/2$", r"$\pi$", r"$3\pi/2$", r"$2\pi$"])
