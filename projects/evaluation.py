import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from ..utils import select_bottom
from ..constants import deg_string, gkg_string
from ..plots import circumpolar_plot, finished_plot
from ..interpolation import interp_latlon_cf

# Compare the bottom temperature and salinity in NEMO (time-averaged over the given xarray Dataset) to observations: Schmidtko on the continental shelf, World Ocean Atlas 2018 in the deep ocean.
# Everything uses TEOS-10 (conservative temperature and absolute salinity) so we're golden.
def bottom_TS_vs_obs (nemo, schmidtko_file='/gws/nopw/j04/terrafirma/kaight/input_data/schmidtko_TS.txt', woa_files='/gws/nopw/j04/terrafirma/kaight/input_data/WOA18/woa18_decav_*00_04.nc', fig_name=None):

    import gsw

    # Read Schmidtko data on continental shelf
    obs = np.loadtxt(schmidtko_file, dtype=np.str)[1:,:]
    obs_lon_vals = obs[:,0].astype(float)
    obs_lat_vals = obs[:,1].astype(float)
    obs_depth_vals = obs[:,2].astype(float)
    obs_temp_vals = obs[:,3].astype(float)
    obs_salt_vals = obs[:,5].astype(float)
    num_obs = obs_temp_vals.size
    # Grid it
    obs_lon = np.unique(obs_lon_vals)
    obs_lat = np.unique(obs_lat_vals)
    obs_temp = np.zeros([obs_lat.size, obs_lon.size]) - 999
    obs_salt = np.zeros([obs_lat.size, obs_lon.size]) - 999
    for n in range(num_obs):
        j = np.argwhere(obs_lat==obs_lat_vals[n])[0][0]
        i = np.argwhere(obs_lon==obs_lon_vals[n])[0][0]
        if obs_temp[j,i] != -999:
            raise Exception('Multiple values at same point')
        obs_temp[j,i] = obs_temp_vals[n]
        obs_salt[j,i] = obs_salt_vals[n]
    # Wrap up into an xarray Dataset for later interpolation
    obs_temp = xr.DataArray(obs_temp, coords=[obs_lat, obs_lon], dims=['lat', 'lon'])
    obs_temp = obs_temp.where(obs_temp!=-999)
    obs_salt = xr.DataArray(obs_salt, coords=[obs_lat, obs_lon], dims=['lat', 'lon'])
    obs_salt = obs_salt.where(obs_salt!=-999)
    obs = xr.Dataset({'temp':obs_temp, 'salt':obs_salt})

    # Read WOA data for deep ocean
    woa = xr.open_mfdataset(woa_files, decode_times=False)
    # Find seafloor depth
    woa_bathy = -1*woa['t_an'].coords['depth'].where(woa['t_an'].notnull()).max(dim='depth')
    # Pressure in dbar is approx depth in m
    woa_press = np.abs(woa_bathy)
    # Mask shallow regions in the Amundsen and Bellingshausen Seas where weird things happen
    mask = (woa['lon'] >= -130)*(woa['lon'] <= -60)*(woa_bathy >= -500)
    # Now get bottom temperature and salinity    
    woa_temp = select_bottom(woa['t_an'], 'depth').where(~mask)
    woa_salt = select_bottom(woa['s_an'], 'depth').where(~mask)
    # Have to convert to conservative temperature and absolute salinity to match NEMO; pretty sure the source data is in-situ temperature and practical salinity
    woa_salt = gsw.SA_from_SP(woa_salt, woa_press, woa['lon'], woa['lat'])
    woa_temp = gsw.CT_from_t(woa_salt, woa_temp, woa_press)
    # Now wrap up into a new Dataset
    woa = xr.Dataset({'temp':woa_temp, 'salt':woa_salt}).drop_vars('depth').squeeze()

    # Regrid to the NEMO grid
    # Rename a few bits in the NEMO dataset for interpolation
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
                
            
    
    
    
    
    
    
