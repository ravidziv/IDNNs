import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

def adjustAxes(axes,axis_font=20,title_str='', x_ticks=[], y_ticks=[], x_lim=None, y_lim=None,
               set_xlabel=True, set_ylabel=True, x_label='', y_label='',set_xlim=True,set_ylim=True, set_ticks=True,label_size=20, set_yscale=False,
               set_xscale = False, yscale=None, xscale=None,ytick_labels = '',genreal_scaling=False):
    """Organize the axes of the given figure"""
    if set_xscale:
        axes.set_xscale(xscale)
    if set_yscale:
        axes.set_yscale(yscale)
    if genreal_scaling:
        axes.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
        axes.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
        axes.xaxis.major.formatter._useMathText = True
        axes.set_yticklabels(ytick_labels)
        axes.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter(useMathText=True))
    if set_xlim:
        axes.set_xlim(x_lim)
    if set_ylim:
        axes.set_ylim(y_lim)
    axes.set_title(title_str, fontsize=axis_font + 2)
    axes.tick_params(axis='y', labelsize=axis_font)
    axes.tick_params(axis='x', labelsize=axis_font)
    if set_ticks:
        axes.set_xticks(x_ticks)
        axes.set_yticks(y_ticks)
    if set_xlabel:
        axes.set_xlabel(x_label, fontsize=label_size)
    if set_ylabel:
        axes.set_ylabel(y_label,fontsize=label_size)

def create_color_bar(f, cmap, colorbar_axis, bar_font, epochsInds, title):
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
    sm._A = []
    cbar_ax = f.add_axes(colorbar_axis)
    cbar = f.colorbar(sm, ticks=[], cax=cbar_ax)
    cbar.ax.tick_params(labelsize=bar_font)
    cbar.set_label(title, size=bar_font)
    cbar.ax.text(0.5, -0.01, epochsInds[0], transform=cbar.ax.transAxes,
                 va='top', ha='center', size=bar_font)
    cbar.ax.text(0.5, 1.0, str(epochsInds[-1]), transform=cbar.ax.transAxes,
                 va='bottom', ha='center', size=bar_font)