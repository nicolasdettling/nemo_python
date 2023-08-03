import xarray as xr
import numpy as np
from .utils import select_bottom

# Compare the bottom temperature and salinity in NEMO (time-averaged over the given xarray Dataset) to observations: Schmidtko on the continental shelf, World Ocean Atlas 2018 in the deep ocean.
# Everything uses TEOS-10 (conservative temperature and absolute salinity) so we're golden.
def bottom_TS_vs_obs (nemo, schmidtko_file='/gws/nopw/j04/terrafirma/kaight/input_data/schmidtko_TS.txt', woa_files='/gws/nopw/j04/terrafirma/kaight/input_data/woa18_decav_*00_04.nc', fig_name=None):

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
    # Mask shallow regions in the Amundsen and Bellingshausen Seas where weird things happen
    mask = (woa['lon'] >= -130)*(woa['lon'] <= -60)*(woa_bathy >= -500)
    # Now get bottom temperature and salinity    
    woa_temp = select_bottom(woa['t_an'], 'depth').where(~mask)
    woa_salt = select_bottom(woa['s_an'], 'depth').where(~mask)
    # Now wrap up into a new Dataset
    woa = xr.Dataset({'temp':woa_temp, 'salt':woa_salt})

    # Regrid to the NEMO grid
    # Rename a few bits in the NEMO dataset for interpolation
    obs_interp = interp_latlon_cf(obs, nemo, method='bilinear')
    woa_interp = interp_latlon_cf(woa, nemo, method='bilinear')
    
    
    
    
