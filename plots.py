import matplotlib.pyplot as plt
import xarray as xr
import socket
import numpy as np
from .utils import polar_stereo, extend_grid_edges
from .plot_utils import set_colours

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

    

    
