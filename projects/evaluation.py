import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from ..utils import select_bottom
from ..constants import deg_string, gkg_string
from ..plots import circumpolar_plot, finished_plot
from ..interpolation import interp_latlon_cf
from ..file_io import read_schmidtko, read_woa

# Compare the bottom temperature and salinity in NEMO (time-averaged over the given xarray Dataset) to observations: Schmidtko on the continental shelf, World Ocean Atlas 2018 in the deep ocean.
def bottom_TS_vs_obs (nemo, schmidtko_file='/gws/nopw/j04/terrafirma/kaight/input_data/schmidtko_TS.txt', woa_files='/gws/nopw/j04/terrafirma/kaight/input_data/WOA18/woa18_decav_*00_04.nc', fig_name=None):

    obs = read_schmidtko(schmidtko_file=schmidtko_file, eos='teos10')
    woa = read_woa(woa_files=woa_files, eos='teos10')

    # Regrid to the NEMO grid
    obs_interp = interp_latlon_cf(obs, nemo, method='bilinear')
    woa_interp = interp_latlon_cf(woa, nemo, method='bilinear')
    # Now combine them, giving precedence to the Schmidtko obs where both datasets exist
    obs_plot = xr.where(obs_interp.isnull(), woa_interp, obs_interp)

    # Select the NEMO variables we need and time-average
    nemo_plot = xr.Dataset({'temp':nemo['sbt'], 'salt':nemo['sbs']}).mean(dim='time_counter')
    nemo_plot = nemo_plot.rename({'x_grid_T_inner':'x', 'y_grid_T_inner':'y'})
    # Apply NEMO land mask to both
    nemo_plot = nemo_plot.where(nemo_plot['temp']!=0)
    obs_plot = obs_plot.where(nemo_plot['temp'].notnull()*obs_plot.notnull())
    obs_plot = obs_plot.where(nemo_plot['temp']!=0)
    nemo_plot = nemo_plot.where(nemo_plot['temp']!=0)
    # Get difference from obs
    bias = nemo_plot - obs_plot

    # Make the plot
    fig = plt.figure(figsize=(10,7))
    gs = plt.GridSpec(2,3)
    gs.update(left=0.1, right=0.9, bottom=0.05, top=0.95, hspace=0.2, wspace=0.1)
    data_plot = [nemo_plot, obs_plot, bias]
    var_plot = ['temp', 'salt']
    var_titles = ['Bottom temperature ('+deg_string+'C)', 'Bottom salinity ('+gkg_string+')']
    alt_titles = [None, 'Observations', 'Model bias']
    vmin = [-2, 34.5]
    vmax = [2, 35]
    ctype = ['RdBu_r', 'RdBu_r', 'plusminus']
    for v in range(2):
        for n in range(3):
            ax = plt.subplot(gs[v,n])
            ax.axis('equal')
            img = circumpolar_plot(data_plot[n][var_plot[v]], nemo, ax=ax, masked=True, make_cbar=False, title=(var_titles[v] if n==0 else alt_titles[n]), vmin=(vmin[v] if n<2 else None), vmax=(vmax[v] if n<2 else None), ctype=ctype[n])
            if n != 1:
                cax = fig.add_axes([0.01+0.46*n, 0.58-0.48*v, 0.02, 0.3])
                plt.colorbar(img, cax=cax, extend='both' if n==0 else 'neither')
    finished_plot(fig, fig_name=fig_name)
                
            
    
    
    
    
    
    
