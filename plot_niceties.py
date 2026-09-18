import pylab as p
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

n_col = 4
cmap = LinearSegmentedColormap.from_list("red_blue", ["red", "blue"], N=n_col)
colors = [cmap(i/n_col) for i in range(n_col)]

p.rc('axes', prop_cycle=p.cycler('color', [
    '#4477AA',  # steel blue
    '#EE6677',  # rose
    '#228833',  # forest green
    '#AA3377',  # plum
]))


def outflow_inflow_legend(mgas=None):
    custom_lines = [Line2D([0], [0], color='black', linewidth=1),
                    Line2D([0], [0], color='black', linewidth=1, linestyle='--')]
    labels = ['Outflow', 'Inflow']

    if mgas is not None:
        import flow_orientation # just for _sci_latex
        labels.append(rf"$10^{{{np.log10(mgas):.1f}}} \, {{\rm M}}_\odot$")
        print(labels)
        custom_lines += [Line2D([0], [0], color='black', marker='x', linestyle='None')]
    ax = p.gca()

    existing_legend = ax.get_legend()
    if existing_legend is not None:
        ax.add_artist(existing_legend)
    
    if existing_legend is not None:
        ax.figure.canvas.draw()
        bbox_axes = existing_legend.get_window_extent().transformed(ax.transAxes.inverted())
        p.legend(
            custom_lines, labels,
            loc='lower right',
            bbox_to_anchor=(bbox_axes.x0, bbox_axes.y0),
            borderaxespad=0.0
        )
    else:
        p.legend(
            custom_lines, labels,
            loc='lower right',
            frameon=True,
            facecolor='white',
            framealpha=1.0,
            edgecolor='none'
        )

def radial_ticks(relative=True):
    if relative:
        p.xlabel(r"$r / r_{\rm 200m}$")
    else:
        p.xlabel(r"$r/ \rm Mpc$")
    p.xlim(0.49,5.1)
    p.xticks([0.05, 0.1, 0.3, 1.0, 3.0], ['0.05', '0.1', '0.3', '1.0', '3.0'])

    if relative:
        p.axvline(1.0, color='lightgray', linestyle='-', zorder=0)
        p.axvline(2.0, color='#ddd', linestyle='-', zorder=0)
        

def mass_ticks():
    p.xticks([12.5, 13.0, 13.5, 14.0, 14.5], ['$10^{12.5}$', '$10^{13.0}$', '$10^{13.5}$', '$10^{14.0}$', '$10^{14.5}$'])
    p.xlabel(r"$M_{200\rm m} / {\rm M_\odot}$")

def upper_axis():
    p.gca().xaxis.set_ticks_position('top')
    p.gca().xaxis.set_label_position('top')
    p.gca().xaxis.set_ticks_position('both')
    p.gca().set_xlabel(
        p.gca().get_xlabel(),
        labelpad=4
    )

def lower_axis():
    p.gca().xaxis.set_ticks_position('bottom')
    p.gca().xaxis.set_label_position('bottom')
    p.gca().xaxis.set_ticks_position('both')

def right_axis():
    p.gca().yaxis.set_ticks_position('right')
    p.gca().yaxis.set_label_position('right')
    p.gca().yaxis.set_ticks_position('both')

def left_axis():
    p.gca().yaxis.set_ticks_position('left')
    p.gca().yaxis.set_label_position('left')
    p.gca().yaxis.set_ticks_position('both')

def no_x_labels():
    p.gca().xaxis.set_ticks_position('none')
    p.gca().set_xlabel('')

def upper_radial_axis(relative=True):
    upper_axis()
    radial_ticks(relative=relative)

def lower_radial_axis(relative=True):
    lower_axis()
    radial_ticks(relative=relative)

def lower_mass_axis():
    lower_axis()
    mass_ticks()

def upper_mass_axis():
    upper_axis()
    mass_ticks()