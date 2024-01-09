import netCDF4 as nc
import numpy as np
import xarray as xr

from ..utils import select_bottom
from ..interpolation import interp_latlon_cf

# NEMO 4.2 mesh_mask files are written with 2D variables x and y instead of nav_lon and nav_lat - at the same time as 1D dimensions x and y. This causes all manner of problems with xarray so the best thing is just to create a new file from scratch and copy over the variables one at a time, renaming as needed.
def fix_mesh_mask (file_in, file_out):

    id_in = nc.Dataset(file_in, 'r')
    id_out = nc.Dataset(file_out, 'w')

    print('Setting up dimensions')
    for dim in ['x', 'y', 'nav_lev']:
        id_out.createDimension(dim, id_in.dimensions[dim].size)
        id_out.createVariable(dim, 'f8', (dim))
        id_out.variables[dim][:] = np.arange(id_in.dimensions[dim].size)
    id_out.createDimension('time_counter', None)

    for var in id_in.variables:
        if var in ['nav_lev', 'time_counter']:
            continue
        print('Writing '+var)
        if var == 'x':
            var_new = 'nav_lon'
        elif var == 'y':
            var_new = 'nav_lat'
        else:
            var_new = var
        id_out.createVariable(var_new, id_in.variables[var].dtype, id_in.variables[var].dimensions)
        id_out.variables[var_new][:] = id_in.variables[var][:]

    id_in.close()
    id_out.close()


# Read bottom temperature and salinity from the Schmidtko dataset.
def read_schmidtko (schmidtko_file='/gws/nopw/j04/terrafirma/kaight/input_data/schmidtko_TS.txt', eos='teos10'):

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
    obs_depth = np.zeros([obs_lat.size, obs_lon.size]) - 999
    for n in range(num_obs):
        j = np.argwhere(obs_lat==obs_lat_vals[n])[0][0]
        i = np.argwhere(obs_lon==obs_lon_vals[n])[0][0]
        if obs_temp[j,i] != -999:
            raise Exception('Multiple values at same point')
        obs_temp[j,i] = obs_temp_vals[n]
        obs_salt[j,i] = obs_salt_vals[n]
        obs_depth[j,i] = obs_depth_vals[n]
    if eos == 'eos80':
        # Convert from TEOS10 to EOS80
        # Have conservative temperature and absolute salinity; want potential temperature and practical salinity
        # Pressure in dbar is approx depth in m
        obs_press = np.abs(obs_depth)
        obs_temp = gsw.pt_from_CT(obs_salt, obs_temp)
        obs_salt = gsw.SP_from_SA(obs_salt, obs_press, obs_lon, obs_lat)
    # Wrap up into an xarray Dataset for later interpolation
    obs_temp = xr.DataArray(obs_temp, coords=[obs_lat, obs_lon], dims=['lat', 'lon'])
    obs_temp = obs_temp.where(obs_temp!=-999)
    obs_salt = xr.DataArray(obs_salt, coords=[obs_lat, obs_lon], dims=['lat', 'lon'])
    obs_salt = obs_salt.where(obs_salt!=-999)
    obs = xr.Dataset({'temp':obs_temp, 'salt':obs_salt})
    return obs


# Read World Ocean Atlas 2018 data for the deep ocean.
def read_woa (woa_files='/gws/nopw/j04/terrafirma/kaight/input_data/WOA18/woa18_decav_*00_04.nc', eos='teos10'):

    import gsw
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
    # Regardless of EOS will need absolute salinity at least temporarily
    abs_salt = gsw.SA_from_SP(woa_salt, woa_press, woa['lon'], woa['lat'])
    if eos == 'eos80':
        # Convert to EOS80
        # Have in-situ temperature and practical salinity; want potential temperature        
        woa_temp = gsw.pt0_from_t(abs_salt, woa_temp, woa_press)
    elif eos == 'teos10':
        # Convert to TEOS80
        # Want conservative temperature and absolute salinity
        woa_salt = abs_salt
        woa_temp = gsw.CT_from_t(woa_salt, woa_temp, woa_press)
    # Now wrap up into a new Dataset
    woa = xr.Dataset({'temp':woa_temp, 'salt':woa_salt}).drop_vars('depth').squeeze()
    return woa


