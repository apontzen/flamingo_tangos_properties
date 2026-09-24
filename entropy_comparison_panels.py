import flamingo_analysis as fa
import tangos as db
import virial_scalings as vs
import numpy as np
import pylab as p
import plot_niceties as pn

comparison_lines = [
    {'folder': '%720%FID%/%4.%', 
     'plot_kwargs': {'color': 'grey', 'linestyle': '--', 'label': 'fid m8'},
     'dT_AGN': 10**8.07},
    {'folder': '%360%FID%/%4.%', 'plot_kwargs': {'color': 'black', 'label': 'fid'}, 
     'dT_AGN': 10**7.95},
    {'folder': '%360%STRONGEST_AGN/%4.%', 'plot_kwargs': {'color': 'darkblue', 'label': r'fgas$-8\sigma$'}, 
     'dT_AGN': 10**8.31},
    {'folder': '%360%WEAK_AGN/%4.%', 'plot_kwargs': {'color': 'lightblue', 'label': r'fid$+2\sigma$'}, 
     'dT_AGN': 10**7.71},
    {'folder': '%360%JETS_STRONGER_AGN/%4.%', 'plot_kwargs': {'color': 'purple', 'linestyle': '-.', 'label': r'Jet_fgas$-4\sigma$'}, 
     'v_AGN': 1995.0},
    {'folder': '%360%JETS/%4.%', 'plot_kwargs': {'color': 'pink', 'linestyle': '-.', 'label': r'Jet', 'zorder': -1,},
     'v_AGN': 836.0}
]

def make_redshift_lines():
    redshift_lines = []
        
    cmap = p.get_cmap('berlin_r')
    colors = iter(cmap(np.linspace(0.05, 0.95, 4)))
    steps = [0, 1, 4, 8]
    for tsnum, c in zip(steps, colors):
        ts = f"%720%/%{tsnum}.hdf5"
        redshift_lines.append({'folder': ts, 'plot_kwargs': {'color': c, 'label': f'fid m8 $z={db.get_timestep(ts).redshift:.1f}$'}, 
                               'dT_AGN': 10**8.07, })

    return redshift_lines


def find_critical_mass(ts = '%360%FID%/%4.%', factor = 1.6):
    """Find the critical log mass at which the outflow entropy becomes less than factor x the virial value"""
    pd = fa.tabulate_by_mass(lambda: at(0.0, gas_entropy_outflow_r200m_relative), ts_name=ts)
    log10M = pd['bin_centre'].to_numpy()
    entropy = pd['mean'].to_numpy()

    z = db.get_timestep(ts).redshift
    virial_entropy = vs.entropy(10**log10M, z=z)
    entropy_offset = entropy - factor * virial_entropy

    if entropy_offset[0] < 0:
        return log10M[0]
    if entropy_offset[-1] > 0:
        return log10M[-1]
    
    crossing = np.where(np.diff(np.sign(entropy_offset)) < 0)[0][0]
    x0, x1 = log10M[crossing], log10M[crossing + 1]
    y0, y1 = entropy_offset[crossing], entropy_offset[crossing + 1]
    return x0 + (x1 - x0) * (-y0) / (y1 - y0)


def add_derived_information(comparisons_list):
    """Convert dT_AGN to v_AGN using eq 11 in paper (with zeta=1)"""
    for item in comparisons_list:
        if 'dT_AGN' in item:
                # convert to v_AGN using eq 11 in paper (with zeta=1)
                item['v_AGN'] = 1672 * (item['dT_AGN']/10**8.07)**0.5

    for item in comparisons_list:
        ts = item['folder']
        crit_mass = find_critical_mass(ts=ts)
        item['log10_mcrit'] = crit_mass




redshift_lines = make_redshift_lines()

add_derived_information(comparison_lines)
add_derived_information(redshift_lines)








def entropy_comparison_plot(ts = '%360%FID%/%4.%', plot_kwargs={}, readoff_values_at=[]):
    fa.make_binned_by_mass_plot(f'{fa.internal_to_keV_cm2} * at(0.0, gas_entropy_outflow_r200m_relative)', 
                                    plot_kwargs=plot_kwargs,
                                    ts_name=ts, use_band=True,
                                    weight_property_name=f'at(0.0, gas_mdot_outflow_r200m_relative)',
                                    readoff_values_at=readoff_values_at
                                    )
    
def temp_comparison_plot(ts = '%360%FID%/%4.%', plot_kwargs={}):
    fa.make_binned_by_mass_plot(f'at(0.0, gas_temp_outflow_r200m_relative)', 
                                    plot_kwargs=plot_kwargs,
                                    ts_name=ts, use_band=True,
                                    weight_property_name=f'at(0.0, gas_mdot_outflow_r200m_relative)',
                                    )


def fgas_comparison_plot(ts = '%360%FID%/%4.%', plot_kwargs={}, radius=1.0):
    log_radius = np.log10(radius)
    fa.make_binned_by_mass_plot(f'at({log_radius}, gas_mass_enclosed_r200m_relative) / at({log_radius}, all_mass_enclosed_r200m_relative)', 
                                    plot_kwargs=plot_kwargs,
                                    ts_name=ts, use_band=True,
                                    num_bins=20
                                    )
    
def energy_comparison_plot(ts = '%360%FID%/%4.%', plot_kwargs={}):
    fa.make_binned_by_mass_plot(f'at(0.0, gas_energy_outflow_r200m_relative)', 
                                    plot_kwargs=plot_kwargs,
                                    ts_name=ts, use_band=True,
                                    num_bins=20
                                    )

def zeta_comparison_plot(ts = None, plot_kwargs=None, v_AGN=None, readoff_values_at=[], 
                         inflow_velocity='measured', shock_density='measured', inflow_entropy = 'measured'):
    match shock_density:
        case 'virial':
            den_expr = (200 * ft._cosmic_baryon_density(db.get_timestep(ts).redshift))**-0.666666
        case 'measured':
            den_expr = lambda: entropy_production_rate_weighted_density_m23
        case _:
            raise ValueError(f"shock_density must be 'virial' or 'measured', got {shock_density}")
        
    match inflow_entropy:
        case 'virial':
            entropy_scale = vs.entropy(1e10, z=db.get_timestep(ts).redshift)
            inflow_entrop_expr = lambda: entropy_scale*(M200m()/1e10)**0.66666666
        case 'measured':
            inflow_entrop_expr = lambda: at(0.0, gas_entropy_inflow_r200m_relative)
        case 'none':
            inflow_entrop_expr = 0.0
        case _:
            raise ValueError(f"inflow_entropy must be 'virial' or 'measured', got {inflow_entropy}")

    entrop_expr = lambda: at(0.0, gas_entropy_outflow_r200m_relative) - 0.84 * inflow_entrop_expr() # at(0.0, gas_entropy_inflow_r200m_relative)

    match inflow_velocity:
        case 'virial':
            G = pynbody.units.Unit("G").in_units("km^2 s^-2 Msol^-1 kpc")
            abs_v_in_expr = lambda: sqrt(G * M200m()/r200m)
        case 'measured':
            abs_v_in_expr = lambda: abs(at(0.0, gas_vr_inflow_r200m_relative))
        case 'none':
            abs_v_in_expr = 0.0
        case _:
            raise ValueError(f"inflow_velocity must be 'virial', 'measured', or 'none', got {inflow_velocity}")
        
    v_jump_expr = lambda: sqrt(entrop_expr/(0.13 * den_expr)) - abs_v_in_expr
   
    zeta = lambda: v_jump_expr / v_AGN
    fa.make_binned_by_mass_plot(zeta, ts_name=ts, plot_kwargs = plot_kwargs, 
                                readoff_values_at=readoff_values_at, 
                                split_at=readoff_values_at[0] if len(readoff_values_at)>0 else None)


def multipanel_entropy_comparison_plot(comparison_lines, right_axis=False):
    p.figure(figsize=(6.9, 5.2*2.5))

    p.subplot(311)
    for x in comparison_lines:
        fgas_comparison_plot(x['folder'], x['plot_kwargs'])
    p.text(0.05, 0.95, "Gas fraction at $r_{\\rm 200m}$", transform=p.gca().transAxes, verticalalignment='top', horizontalalignment='left')
    p.xlabel(r"$\log_{10}(M_{200m}/M_\odot)$")
    p.ylabel(r"$f_{\rm gas}(r_{\rm 200m})$")
    p.title("")
    p.xlim(12.45, 14.55)
    pn.upper_mass_axis()
    if right_axis:
        pn.right_axis()
    else:
        pn.left_axis()



    p.subplot(312)
    for x in comparison_lines:
        entropy_comparison_plot(x['folder'], x['plot_kwargs'], readoff_values_at=[x['log10_mcrit']])
    p.semilogy()
    p.text(0.05, 0.95, "Outflow entropy at $r_{\\rm 200m}$", transform=p.gca().transAxes, verticalalignment='top', horizontalalignment='left')
    p.xlabel(r"$\log_{10}(M_{200m}/M_\odot)$")
    p.ylabel(r"$K_{\rm outflow}(r_{\rm 200m}) / \mathrm{keV\,cm^2}$")
    p.title("")
    p.xlim(12.45, 14.55)
    redshifts = {db.get_timestep(x['folder']).redshift for x in comparison_lines}
    if len(redshifts) == 1:
        fa.plot_entropy_guide(z=redshifts.pop(), scale_factor=1.0, color='orange')
        
    pn.upper_mass_axis()
    p.gca().tick_params(axis='x', which='both', labelbottom=False, labeltop=False)
    p.gca().set_xlabel('')
    if right_axis:
        pn.right_axis()
    else:
        pn.left_axis()
    p.ylim(45.0, 2.5e3)
    


    p.subplot(313)
    p.text(0.05, 0.95, "AGN thermalisation parameter", transform=p.gca().transAxes, verticalalignment='top', horizontalalignment='left')
    for x in comparison_lines:
        zeta_comparison_plot(x['folder'], x['plot_kwargs'], v_AGN=x['v_AGN'], readoff_values_at=[x['log10_mcrit']])
    p.ylabel(r"$\zeta_{\rm eff}$")
    p.ylim(0,1.05)
    p.xlabel(r"$\log_{10} M_{200m}/M_\odot$")
    p.title("")
    pn.lower_mass_axis()
    p.legend(
            loc='upper left',
            bbox_to_anchor=(12.52, 0.91),
            bbox_transform=p.gca().transData,
            ncol=2,
            columnspacing=0.8,
            handletextpad=0.2,
            handlelength=1.5,
            borderpad=0.0
        )
    if right_axis:
        pn.right_axis() 
    else:
        pn.left_axis()

    p.tight_layout()
    p.subplots_adjust(hspace=0)