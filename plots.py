import matplotlib.pyplot as plt
import xarray as xr
import cf_xarray as cfxr
import socket
import numpy as np
from .utils import polar_stereo
from .plot_utils import set_colours

# Plot a 2D field (lat-lon) on a polar stereographic projection of Antarctica. Assumes it's a periodic grid covering all longitudes.
# Arguments:
# data: an xarray DataArray of a 2D field (lat-lon)
# grid: an xarray Dataset containing the fields nav_lon_grid_T, nav_lat_grid_T, bounds_nav_lon_grid_T, bounds_nav_lat_grid_T
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
def circumpolar_plot (data, grid, ax=None, make_cbar=True, masked=False, title=None, titlesize=16, fig_name=None, return_fig=False, vmin=None, vmax=None, ctype='viridis', change_points=None):

    new_fig = ax is None
    if title is None:
        title = data.name

    if not masked:
        # Mask where identically zero
        data = data.where(data!=0)
        
    lon_edges = cfxr.bounds_to_vertices(grid['bounds_nav_lon_grid_T'], 'nvertex_grid_T')
    lat_edges = cfxr.bounds_to_vertices(grid['bounds_nav_lat_grid_T'], 'nvertex_grid_T')
    x_edges, y_edges = polar_stereo(lon_edges, lat_edges)

    # Manually find northern boundary - careful with -1 used as missing values
    lat_max = grid['nav_lat_grid_T'].where(grid['nav_lat_grid_T']!=-1).max().item()
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

    

    
