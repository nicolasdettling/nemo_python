import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

from ..utils import region_mask
from ..plots import circumpolar_plot, finished_plot

def find_cgrid_issues (grid_file='/gws/nopw/j04/terrafirma/kaight/input_data/grids/domcfg_eORCA025_v3.nc'):

    from shapely.geometry import Point, Polygon

    ds = xr.open_dataset(grid_file)

    tlon = np.squeeze(ds['glamt'].values)
    tlat = np.squeeze(ds['gphit'].values)
    flon = np.squeeze(ds['glamf'].values)
    flat = np.squeeze(ds['gphif'].values)
    land = np.isnan(ds['closea_mask'].values)

    aligned = np.ones(land.shape)
    for j in range(1, ds.sizes['y']):
        for i in range(1, ds.sizes['x']-1):
            lon_corners = np.array([flon[j,i], flon[j-1,i], flon[j-1,i-1], flon[j,i-1]])
            lat_corners = np.array([flat[j,i], flat[j-1,i], flat[j-1,i-1], flat[j,i-1]])
            if tlon[j,i] < -179 and np.amax(lon_corners) > 179:
                index = lon_corners > 0
                lon_corners[index] = lon_corners[index] - 360
            elif tlon[j,i] > 179 and np.amin(lon_corners) < -179:
                index = lon_corners < 0
                lon_corners[index] = lon_corners[index] + 360
            tpoint = Point(tlon[j,i], tlat[j,i])
            grid_cell = Polygon([(lon_corners[n], lat_corners[n]) for n in range(4)])
            aligned[j,i] = tpoint.within(grid_cell)
    aligned = aligned.astype(bool)
    ocean_good = np.invert(land)*aligned
    ocean_bad = np.invert(land)*np.invert(aligned)
    land_bad = land*np.invert(aligned)

    fig, ax = plt.subplots()
    ax.plot(tlon[ocean_good], tlat[ocean_good], 'o', markersize=1, color='blue')
    ax.plot(tlon[ocean_bad], tlat[ocean_bad], 'o', markersize=1, color='red')
    ax.plot(tlon[land_bad], tlat[land_bad], 'o', markersize=1, color='green')
    ax.set_title('Misaligned cells in ocean (red) and land (green)')
    fig.savefig('misaligned_cells.png')


def plot_region_map (mesh_mask='/gws/nopw/j04/terrafirma/kaight/input_data/grids/mesh_mask_UKESM1.1_ice.nc', option='all', fig_name=None):

    regions = ['amundsen_sea', 'bellingshausen_sea', 'west_antarctic_peninsula', 'larsen', 'filchner_ronne', 'east_antarctica', 'amery', 'ross']
    colours = ['IndianRed', 'SandyBrown', 'LemonChiffon', 'LightGreen', 'MediumTurquoise', 'SteelBlue', 'Plum', 'Pink']
    grid = xr.open_dataset(mesh_mask).squeeze()

    for n in range(len(regions)):
        mask = region_mask(regions[n], mesh_mask, option=option)
        if n==0:
            fig, ax = circumpolar_plot(data, grid, make_cbar=False, return_fig=True, ctype=colours[n])
        else:
            circumpolar_plot(data, grid, ax=ax, make_cbar=False, ctype=colours[n])

    finished_plot(fig, fig_name=fig_name)
