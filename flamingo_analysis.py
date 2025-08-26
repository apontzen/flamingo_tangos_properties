import matplotlib.pyplot as p
import numpy as np
import tangos as db


class NoHalosInStackError(ValueError):
    pass

def get_xs(ts, property_name, profile):
    prop = ts.halos[2].get_description(property_name)
    return prop.plot_x_values(profile)
        
def get_labels(ts, property_name):
    prop = ts.halos[2].get_description(property_name)
    ylabs = prop.plot_ylabel()
    xlab = prop.plot_xlabel()
    return xlab, ylabs[prop.index_of_name(property_name)]

def get_stack(property_name, M_min, M_max, use_log=False, timestep_name="L0200%HYDRO%/%8%"):
    ts = db.get_timestep(timestep_name)
    M200m, profiles = ts.calculate_all('M200m()', property_name)
    mask = (M200m>10**M_min)*(M200m<10**M_max)
    num_included = mask.sum()
    if num_included == 0 :
        raise NoHalosInStackError("No halos in stack")
    
    log_profiles = []
    for p in profiles[mask]:
        ln_p = np.log(p)
        ln_p[ln_p==-np.inf] = np.nan
        log_profiles.append(ln_p)
    if 'rho' in property_name:
        # zeros should be counted, otherwise biased mass estimator
        if use_log:
            mean_profile = np.exp(np.nansum([p for p in log_profiles], axis=0)/num_included)
        else:
            mean_profile = np.nansum([p for p in profiles[mask]], axis=0)/num_included
    else:
        # nan bins should not be counted
        if use_log:
            mean_profile = np.exp(np.nanmean([p for p in log_profiles], axis=0))
        else:
            mean_profile = np.nanmean([p for p in profiles[mask]], axis=0)
    err_log_profile = (np.nanstd([p for p in log_profiles], axis=0)/np.sqrt(num_included))

    xs = get_xs(ts, property_name, mean_profile)
    labels = get_labels(ts, property_name)
    return mean_profile, mean_profile * err_log_profile, xs, labels

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
        main_line = p.plot(r, -profile, **plot_kwargs)
        # plot +profile as dashed:
        p.plot(r, profile, linestyle='--', color=main_line[0].get_color())
    else:
        main_line = p.plot(r, profile, **plot_kwargs)
    p.fill_between(r, profile-uncertainty, profile+uncertainty, alpha=0.2)

    if name == 'vr':
        p.semilogx()
    else:
        p.loglog()

    
    
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
ranges = [(12.0, 12.5), (13.0, 13.5), (14.0, 14.5)]
vars = ['density', 'entropy', 'temp', 'p']
plot_guides_for = ['density', 'entropy', 'temp', 'p']

def make_profile_plots(v, tsnum=8, box="L0200N0720_HYDRO_FIDUCIAL", 
                       newfig=True, with_exclusive=False, norm_guide=False, particle='gas', plot_kwargs={}):
    timestep_name = f"{box}/%{tsnum}.hdf5"
    z = db.get_timestep(timestep_name).redshift
    print(f"Plotting {v} profiles for {timestep_name}")
    if newfig:
        p.figure(figsize=(12, 5))
    p.subplot(121)
    p.title(f"Relative radius profiles ($z={z:.1f}$)")
    p.gca().set_prop_cycle(None)
    for i, ra in enumerate(ranges):
        with_guide = i == 3 and v in plot_guides_for
        make_plot(v, ra[0], ra[1], with_guide=with_guide, with_exclusive=with_exclusive, relative=True,
                  with_alternative_ts=False, get_stack_kwargs={'timestep_name': timestep_name},
                  norm_guide=norm_guide, particle=particle, plot_kwargs=plot_kwargs)
    if newfig:
        p.legend()
    p.subplot(122)
    p.gca().set_prop_cycle(None)
    p.title(f"Absolute radius profiles ($z={z:.1f}$)")
    for i, ra in enumerate(ranges):
        with_guide = i == 3 and v in plot_guides_for
        make_plot(v, ra[0], ra[1], with_guide=with_guide, with_exclusive=with_exclusive, relative=False,
                  with_alternative_ts=False, get_stack_kwargs={'timestep_name': timestep_name},
                  norm_guide=norm_guide, particle=particle, plot_kwargs=plot_kwargs)

    if newfig:
        p.legend()

