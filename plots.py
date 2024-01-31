import matplotlib.pyplot as plt
import matplotlib.colors as cl
import xarray as xr
import socket
import numpy as np
from .utils import polar_stereo, extend_grid_edges, moving_average, build_ocean_mask
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
# grid: an xarray Dataset containing the fields (option 1:) nav_lon_grid_T, nav_lat_grid_T, bounds_nav_lon_grid_T, bounds_nav_lat_grid_T or (option 2): glamt, gphit, glamf, gphif or (option3): nav_lon, nav_lat, bounds_lon, bounds_lat
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
# contour: list of levels to contour in black
# shade_land: whether to shade the land mask in grey

# TODO contour ice front
def circumpolar_plot (data, grid, ax=None, make_cbar=True, masked=False, title=None, titlesize=16, fig_name=None, return_fig=False, vmin=None, vmax=None, ctype='viridis', change_points=None, periodic=True, lat_max=None, contour=None, shade_land=True):

    new_fig = ax is None
    if title is None:
        title = data.name

    if not masked:
        # Mask where identically zero
        data = data.where(data!=0)

    if 'nav_lat_grid_T' in grid:
        lat_name = 'nav_lat_grid_T'
        lon_name = 'nav_lon_grid_T'
    elif 'gphit' in grid:
        lat_name = 'gphit'
        lon_name = 'glamt'
    elif 'nav_lat' in grid:
        lat_name = 'nav_lat'
        lon_name = 'nav_lon'
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
    elif lat_name == 'nav_lat':
        import cf_xarray as cfxr
        lon_edges = cfxr.bounds_to_vertices(grid['bounds_lon'], 'nvertex')
        lat_edges = cfxr.bounds_to_vertices(grid['bounds_lat'], 'nvertex')
    x_edges, y_edges = polar_stereo(lon_edges, lat_edges)

    # Get axes bounds
    x_bounds, y_bounds = polar_stereo(np.array([0, 90, 180, -90]), np.array([lat_max]*4))
    xlim = [x_bounds[3], x_bounds[1]]
    ylim = [y_bounds[2], y_bounds[0]]

    # Set up colour map
    cmap, vmin, vmax = set_colours(data, ctype=ctype, vmin=vmin, vmax=vmax, change_points=change_points)

    if shade_land:
        ocean_mask = build_ocean_mask(grid)[0]
        mask_plot = np.invert(ocean_mask)
        mask_plot = mask_plot.where(mask_plot)

    if new_fig:
        fig, ax = plt.subplots()
    img = ax.pcolormesh(x_edges, y_edges, data, cmap=cmap, vmin=vmin, vmax=vmax)
    if shade_land:
        ax.pcolormesh(x_edges, y_edges, mask_plot, cmap=cl.ListedColormap(['DarkGrey']), linewidth=0)
    if contour is not None:
        x, y = polar_stereo(grid[lon_name], grid[lat_name])
        ax.contour(x, y, data, levels=contour, colors=('black'), linewidths=1, linestyles='solid')
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
def timeseries_plot (datas, labels=None, colours=None, title='', units='', ax=None, fig_name=None, linewidth=None):

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
        ax.plot_date(data.time_centered, data, '-', color=colour, label=label, linewidth=linewidth)
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
def timeseries_by_region (var_name, sim_dir, regions=['all', 'amundsen_sea', 'bellingshausen_sea', 'larsen', 'filchner_ronne', 'east_antarctica', 'amery', 'ross'], colours=None, timeseries_file='timeseries.nc', smooth=0, fig_name=None, linewidth=None):

    if isinstance(sim_dir, str):
        sim_dir = [sim_dir]
    if colours is None:
        if len(regions) <= len(line_colours):
            colours = line_colours[:len(regions)]
        else:
            raise Exception('Too many regions to use default line_colours: set colours instead')

    all_ds = []
    for d in sim_dir:
        all_ds.append(xr.open_dataset(d+'/'+timeseries_file))
    num_ens = len(all_ds)

    datas = []
    labels = []
    colours_plot = []
    title = None
    units = None
    for region, colour in zip(regions, colours):
        labels.append(region_names[region])
        labels += [None]*(num_ens-1)
        colours_plot += [colour]*num_ens
        var_full = region+'_'+var_name
        for ds in all_ds:
            datas.append(moving_average(ds[var_full], smooth))
            if title is None:
                long_name = ds[var_full].long_name
                title = long_name.replace(region_names[region]+' ','')
                units = ds[var_full].units

    timeseries_plot(datas, labels=labels, colours=colours_plot, title=title, units=units, fig_name=fig_name, linewidth=linewidth)
        

# Plot timeseries of the same variable in different experiments. Each experiment can be a single simulation (sim_dirs=list of strings) or an ensemble (sim_dirs=list of lists of strings).
# For ensembles, if sim_names is set, it can be the name of every member (list of lists of strings) or one name for every ensemble (list of strings).
# If sim_names is not set, lines will be labelled with the suite IDs (extracted from sim_dirs)
def timeseries_by_expt (var_name, sim_dirs, sim_names=None, colours=None, timeseries_file='timeseries.nc', smooth=0, fig_name=None, linewidth=None):

    num_expt = len(sim_dirs)
    if colours is None:
        if num_expt <= len(line_colours):
            colours = line_colours[:num_expt]
        else:
            raise Exception('Too many experiments to use default line_colours: set colours instead')

    # Inner function to extract the name of the current directory (with trailing slashes and parent directories stripped out)
    def make_sim_name (dir_name):
        # First strip out trailing slash(es), if it's there
        while True:
            if dir_name[-1] != '/':
                break
            dir_name = dir_name[:-1]
        # Now strip out parent directories
        if '/' in dir_name:
            dir_name = dir_name[dir_name.rfind('/')+1:]
        return dir_name
        
    if sim_names is None:
        # Generate simulation names from directory names (hopefully suite IDs), with same structure as sim_dirs
        sim_names = []
        for sim_dir in sim_dirs:
            if isinstance(sim_dir, str):
                # Single simulation
                sim_names.append(make_sim_name(sim_dir))
            elif isinstance(sim_dir, list):
                # Ensemble
                ens_names = []
                for d in sim_dir:
                    ens_names.append(make_sim_name(d))
                sim_names.append(ens_names)

    datas = []
    labels = []
    colours_plot = []
    title = None
    units = None
    for sim_dir, sim_name, colour in zip(sim_dirs, sim_names, colours):
        if isinstance(sim_dir, str):
            # Generalise to ensemble of 1
            sim_dir = [sim_dir]
            sim_name = [sim_name]
        elif isinstance(sim_dir, list) and isinstance(sim_name, str):
            # Just one name for the whole ensemble; only label the first member
            sim_name = [sim_name] + [None]*(len(sim_dir)-1)
        num_ens = len(sim_dir)
        colours_plot += [colour]*num_ens
        for d, n in zip(sim_dir, sim_name):
            ds = xr.open_dataset(d+'/'+timeseries_file)
            datas.append(moving_average(ds[var_name],smooth))
            labels.append(n)
            if title is None:
                title = ds[var_name].long_name
                units = ds[var_name].units

    timeseries_plot(datas, labels=labels, colours=colours_plot, title=title, units=units, fig_name=fig_name, linewidth=linewidth)
        

    
