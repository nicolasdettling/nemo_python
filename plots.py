import matplotlib.pyplot as plt
import xarray as xr
import socket
import numpy as np
from .utils import polar_stereo, extend_grid_edges
from .plot_utils import set_colours
from .constants import line_colours, region_names

# If a figure name is defined, save the figure to that file. Otherwise, display the figure on screen.
def finished_plot (fig, fig_name=None, dpi=None):

    if fig_name is not None:
        print(('Saving ' + fig_name))
        fig.savefig(fig_name, dpi=dpi)
    else:
        fig.show()
        

# Plot a 2D field (lat-lon) on a polar stereographic projection of Antarctica. Assumes it's a periodic grid covering all longitudes.
# Arguments:
# data: an xarray DataArray of a 2D field (lat-lon)
# grid: an xarray Dataset containing the fields (option 1:) nav_lon_grid_T, nav_lat_grid_T, bounds_nav_lon_grid_T, bounds_nav_lat_grid_T or (option 2): glamt, gphit, glamf, gphif
# Optional keyword arguments:
# ax: handle to Axes object to make the plot in
# make_cbar: whether to make a colourbar (default True)
# masked: whether data is already masked; if False (default) it will be masked wherever it's identically zero
# title: optional string for the title, if None, will use the DataArray name
# titlesize: fontsize for the title (default 16)
# fig_name: optional filename for figure (otherwise show interactively)
# return_fig: if True, return the figure and axes handles
# vmin, vmax: optional bounds on colour map
# ctype: colourmap type (see set_colours in plot_utils.py)
# change_points: arguments to ismr colourmap (see above)

# TODO colour maps, contour ice front, shade land in grey
def circumpolar_plot (data, grid, ax=None, make_cbar=True, masked=False, title=None, titlesize=16, fig_name=None, return_fig=False, vmin=None, vmax=None, ctype='viridis', change_points=None, periodic=True, lat_max=None):

    new_fig = ax is None
    if title is None:
        title = data.name

    if not masked:
        # Mask where identically zero
        data = data.where(data!=0)

    if 'nav_lat_grid_T' in grid:
        lat_name = 'nav_lat_grid_T'
    elif 'gphit' in grid:
        lat_name = 'gphit'
    # Enforce northern boundary
    if lat_max is None:
        if grid[lat_name].max() > 0:
            print('Warning: this grid includes the northern hemisphere. Can cause weirdness in plotting')
        # Manually find northern boundary - careful with -1 used as missing values
        lat_max = grid[lat_name].where(grid[lat_name]!=-1).max().item()

    # Get cell edges in polar stereographic coordinates
    if lat_name == 'nav_lat_grid_T':
        import cf_xarray as cfxr
        lon_edges = cfxr.bounds_to_vertices(grid['bounds_nav_lon_grid_T'], 'nvertex_grid_T')
        lat_edges = cfxr.bounds_to_vertices(grid['bounds_nav_lat_grid_T'], 'nvertex_grid_T')
    elif lat_name == 'gphit':
        lon_edges = extend_grid_edges(grid['glamf'], 'f', periodic=True)
        lat_edges = extend_grid_edges(grid['gphif'], 'f', periodic=True)
    x_edges, y_edges = polar_stereo(lon_edges, lat_edges)

    # Get axes bounds
    x_bounds, y_bounds = polar_stereo(np.array([0, 90, 180, -90]), np.array([lat_max]*4))
    xlim = [x_bounds[3], x_bounds[1]]
    ylim = [y_bounds[2], y_bounds[0]]

    # Set up colour map
    cmap, vmin, vmax = set_colours(data, ctype=ctype, vmin=vmin, vmax=vmax, change_points=change_points)

    if new_fig:
        fig, ax = plt.subplots()
    img = ax.pcolormesh(x_edges, y_edges, data, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.axis('off')
    ax.set_title(title, fontsize=titlesize)
    if make_cbar:
        plt.colorbar(img)
    if return_fig:
        return fig, ax
    elif fig_name is not None:
        fig.savefig(fig_name)
    elif new_fig:
        fig.show()
    else:
        return img


# Plot one or more timeseries on the same axis, with different colours and a legend if needed.
def timeseries_plot (datas, labels=None, colours=None, title='', units='', ax=None, fig_name=None):

    new_ax = ax is None
    
    # Check if multiple lines to plot
    multi_data = isinstance(datas, list)
    if multi_data:
        if labels is None or colours is None:
            raise Exception('Need to set labels and colours')
        if len(labels) != len(datas) or len(colours) != len(datas):
            raise Exception('Wrong length of labels or colours')
    else:
        datas = [datas]
        labels = [None]
        colours = [None]

    if new_ax:
        if multi_data:
            figsize = (11,6)
        else:
            figsize = (6,4)
        fig, ax = plt.subplots(figsize=figsize)
    for data, label, colour in zip(datas, labels, colours):
        ax.plot_date(data.time_centered, data, '-', color=colour, label=label)
    ax.grid(linestyle='dotted')
    ax.set_title(title, fontsize=16)
    ax.set_ylabel(units, fontsize=16)
    if multi_data and new_ax:
        # Make legend
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width*0.8, box.height])
        ax.legend(loc='center left', bbox_to_anchor=(1,0.5))
    if new_ax:
        finished_plot(fig, fig_name=fig_name)


# Plot timeseries of the same variable in different regions. Can either do for a single simulation (sim_dir is a string) or an initial conditions ensemble (sim_dir is a list of strings). 
def timeseries_by_region (var_name, sim_dir, regions=['all', 'amundsen_sea', 'bellingshausen_sea', 'larsen', 'filchner_ronne', 'east_antarctica', 'amery', 'ross'], colours=None, timeseries_file='timeseries.nc', fig_name=None):

    ensemble = isinstance(sim_dir, list) and len(sim_dir)>1
    if not ensemble:
        sim_dir = [sim_dir]
    if colours is None:
        if len(regions) <= len(line_colours):
            colours = line_colours[:len(regions)]
        else:
            raise Exception('Too many regions to use default line_colours: set colours instead')

    if ensemble:
        all_ds = []
        for d in sim_dir:
            all_ds.append(xr.open_dataset(d+'/'+timeseries_file))
    else:
        all_ds = [xr.open_dataset(sim_dir+'/'+timeseries_file)]
    num_ens = len(all_ds)

    datas = []
    labels = []
    colours_plot = []
    title = None
    units = None
    for region, colour in zip(regions, colours):
        labels += [region_names[region]]*num_ens
        colours_plot += [colour]*num_ens
        var_full = region+'_'+var_name
        for ds in all_ds:
            datas.append(ds[var_full])
            if title is None:
                long_name = ds[var_full].long_name
                title = long_name[:long_name.index(' on ')]
                units = ds[var_full].units

    timeseries_plot(datas, labels=labels, colours=colours_plot, title=title, units=units, fig_name=fig_name)
        

def timeseries_by_expt (var_name, sim_dir, sim_names=None, colours=None, timeseries_file='timeseries.nc', fig_name=None):

    pass
        

    
    

    # Single simulation, by_region: sim_dir is a string or list of length 1, var_name needs to be preceded by regions, sim_names unused, choose list of colours corresponding to regions
    # Single ensemble, by_region: now same colour for each ensemble member
    # by_expt, multiple simulations: sim_dir is a list of length>1, choose colours up to some max number
    # by_expt, multiple ensembles: sim_dir is a list of lists, same colour for each ensemble member
    
            

    

    
