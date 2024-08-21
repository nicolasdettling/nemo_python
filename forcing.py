###########################################################
# Generate forcing including atmospheric, runoff etc.
###########################################################
import xarray as xr
import numpy as np
from .utils import distance_btw_points, closest_point, convert_to_teos10, fix_lon_range
from .grid import get_coast_mask, get_icefront_mask
from .ics_obcs import fill_ocean
from .file_io import find_cesm2_file
from .constants import temp_C2K, rho_fw

# Function subsets global forcing files from the same grid to the new domain, and fills any NaN values with connected 
# nearest neighbour and then fill_val.
# Inputs:
# file_path: string of location of forcing file
# nemo_mask: xarray Dataset of nemo meshmask (must contain tmask)
# fill_ocn: boolean to turn on (or off) the connected nearest neighbour fill
# fill_val: float or NaN value to fill any remaining NaN values with
# Returns an xarray dataset of the original forcing file subset to the new domain and filled
def subset_global(file_path, nemo_mask, fill_ocn=False, fill_val=0):

    # this subset is not generalized for other domains; can fix later
    ds = xr.open_dataset(f'{file_path}').isel(y=slice(0,453)) 

    if fill_ocn:
        for var in list(ds.keys()):
            # Check for values that are NaN and in the ocean and fill with nearest neighbour
            ds_filled = fill_ocean(ds.isel(time_counter=0), var, nemo_mask, dim='2D', fill_val=fill_val)
            ds[var]   = ds[var].dims, ds_filled[var].values[np.newaxis, ...]   

    for var in list(ds.keys()):
        # Then fill any NaN values with fill_val
        ds[var] = xr.where(np.isnan(ds[var]), fill_val, ds[var])
    
    new_file_path = file_path.replace(file_path.split('/')[-1], f"AntArc_{file_path.split('/')[-1]}") 

    # save file with time_counter as unlimited dimension, if time_counter is present
    if 'time_counter' in ds.dims:
        ds.to_netcdf(f"{new_file_path}", unlimited_dims='time_counter')
    else:
        ds.to_netcdf(f"{new_file_path}")
    
    return ds

# Function identifies locations where calving does not occur at the ocean edge of the iceshelf front 
# Inputs:
# calving: xarray variable of calving proportions (2D)
# nemo_mask: xarray Dataset of nemo meshmask (must contain tmask and tmaskutil)
# Returns three 2D arrays of calving that occurs not at the icefront, but further in the ocean (calving_ocn), 
# on land (calving_land), or on an ice shelf grid point (calving_ice)
def calving_at_coastline(calving, nemo_mask):
    
    calving   = xr.where(calving > 0, calving, np.nan)

    # Boolean arrays to identify regions:
    ocean     = (nemo_mask.tmask.isel(time_counter=0, nav_lev=0) == 1)
    iceshelf  = (nemo_mask.tmaskutil.isel(time_counter=0) - nemo_mask.tmask.isel(time_counter=0, nav_lev=0)).astype(bool);
    land      = (nemo_mask.tmaskutil.isel(time_counter=0) == 0)

    # Cases where calving does not occur at the ocean edge of the icefront:
    icefront_mask_ocn = get_icefront_mask(nemo_mask, side='ocean')
    calving_ocn  = xr.where((~icefront_mask_ocn & ocean), calving, np.nan)  # calving occurs in the ocean but not right by the icefront
    calving_land = xr.where(land      , calving, np.nan)  # calving occurs on land
    calving_ice  = xr.where(iceshelf  , calving, np.nan)  # calving occurs on ice shelf

    return calving_ocn, calving_land, calving_ice

# Function shifts the x,y location of a calving point to the nearest iceshelf front ocean point 
# Inputs:
# calving: xarray dataset containing calving proportions 
# mask: locations of cells that need to be shifted
# nemo_mask: xarray dataset of nemo mesh mask file (must contain nav_lon, nav_lat)
# icefront_mask_ocn: mask of ocean points nearest to iceshelf, produced by get_icefront_mask
# max_distance: float of maximum distance (in meters) that an ocean calving point will get moved
# calving_var: string of the calving variable name
def shift_calving(calving, mask, nemo_mask, icefront_mask_ocn, max_distance=11000, calving_var='soicbclv'):
    # NEMO domain grid points
    x, y        = np.meshgrid(nemo_mask.nav_lon.x, nemo_mask.nav_lon.y)
    calving_x   = x[(~np.isnan(mask))]
    calving_y   = y[(~np.isnan(mask))]
    calving_new = np.copy(calving[calving_var].values);

    # Coordinates of ocean points closest to iceshelf front 
    icefront_x     = x[icefront_mask_ocn]
    icefront_y     = y[icefront_mask_ocn]
    icefront_coord = (nemo_mask.nav_lon.values[icefront_mask_ocn], nemo_mask.nav_lat.values[icefront_mask_ocn])

    # For each land iceberg calving point, check distance to nearest iceshelf front and move closer if possible:
    for index in list(zip(calving_y, calving_x)):
    
        calving_coord = (nemo_mask.nav_lon.values[index], nemo_mask.nav_lat.values[index])
        distances     = distance_btw_points(calving_coord, icefront_coord)

        # only move cell if it is within a certain distance
        if np.min(np.abs(distances)) < max_distance:     
            new_x         = icefront_x[np.argmin(np.abs(distances))]
            new_y         = icefront_y[np.argmin(np.abs(distances))]
            # Move calving to the nearest icefront point and add to any pre-existing calving at that point
            calving_new[new_y, new_x] = calving_new[new_y, new_x] + calving[calving_var].values[index]    
            calving_new[index]        = 0 # remove calving from originating point

    # Write new locations to xarray dataset
    calving_ds = calving.copy()
    calving_ds[calving_var] = calving[calving_var].dims, calving_new

    return calving_ds
    
# Main function to move pre-existing calving dataset to a new coastline (on the same underlying grid, but subset)
# Inputs:
# nemo_mask: xarray dataset of nemo meshmask file (must contain nav_lon, nav_lat, tmask, tmaskutil)
# calving: xarray dataset of old calving forcing NetCDF file
# calving_var: string of the calving variable name
# new_file_path: string of name and location of new calving forcing file
# Returns: xarray dataset with calving at new locations
def create_calving(calving, nemo_mask, calving_var='soicbclv', new_file_path='./new-calving.nc'):

    # Identify locations with calving that need to be moved
    calving_ocn, calving_land, calving_ice = calving_at_coastline(calving[calving_var], nemo_mask)

    # Mask of ocean grid points nearest to icefront, to move calving to
    icefront_mask_ocn = get_icefront_mask(nemo_mask, side='ocean')

    # shift calving points to the iceshelf edge
    calv_ocn_new  = shift_calving(calving      , calving_ocn , nemo_mask, icefront_mask_ocn) # from ocean
    calv_land_new = shift_calving(calv_ocn_new , calving_land, nemo_mask, icefront_mask_ocn) # from land
    calv_ice_new  = shift_calving(calv_land_new, calving_ice , nemo_mask, icefront_mask_ocn) # from ice

    # Write the calving dataset to a netcdf file:
    calv_ice_new[calving_var] = ('time_counter',) + calving[calving_var].dims, calv_ice_new[calving_var].values[np.newaxis, ...] 
    calv_ice_new.to_netcdf(f"{new_file_path}", unlimited_dims='time_counter')

    # Check that the amount of calving that occurs in the new file is approximately equal to the original file:
    # allow for a tolerance of 0.1% (although it should be essentially equal within machine precision)
    tolerance = 0.001*(np.sum(calv_ice_new[calving_var].values)) 
    if np.abs(np.sum(calv_ice_new[calving_var].values) - np.sum(calving[calving_var]).values) >= tolerance:
        raise Exception('The total amount of calving in the new file is not equal to the ')
    
    return calv_ice_new

# Process ocean conditions from CESM2 scenarios for a single variable and single ensemble member (for initial and boundary conditions).
def cesm2_ocn_forcing (expt, var, ens, out_dir, start_year=1850, end_year=2100):

    if expt not in ['LE2']:
        raise Exception(f'Invalid experiment {expt}')

    # cesm2 ocean files are already land masked and will be exteneded into cavities later so don't need to landmask here

    freq = 'monthly'
    for year in range(start_year, end_year+1):
        # read in the data and subset to the specified year
        if var in ['aice','sithick','sisnthick']:
            file_path = find_cesm2_file(expt, var, 'ice', freq, ens, year)
        else:
            file_path = find_cesm2_file(expt, var, 'ocn', freq, ens, year)
        ds   = xr.open_dataset(file_path)
        data = ds[var].isel(time=(ds.time.dt.year == year))

        # Unit conversions ### need to check that these are still consistent between CESM1 and CESM2
        if var in ['SSH','UVEL','VVEL']:
            # convert length units from cm to m
            data *= 0.01 
        # Convert from practical salinity to absolute salinity??
        elif var == 'TEMP':
            # Convert from potential temperature to conservative temperature
            # need to load absolute salinity from file
            salt_path = find_cesm2_file(expt, 'SALT', 'ocn', freq, ens, year)
            salt_ds   = xr.open_dataset(salt_path)
            salt_data = salt_ds['SALT'].isel(time=(salt_ds.time.dt.year == year))

            dsc  = xr.Dataset({'PotTemp':data, 'AbsSal':salt_data, 'depth':data.z_t, 'lon':data.TLONG, 'lat':data.TLAT})
            data = convert_to_teos10(dsc, var='PotTemp')

        # Convert calendar to Gregorian
        data = data.convert_calendar('gregorian')
        # Convert longitude range from 0-360 to degrees east
        if var in ['SSH','SALT','TEMP']:
            lon_name = 'TLONG'
        elif var in ['aice','sithick','sisnthick']:
            lon_name = 'TLON'
        elif var in ['UVEL','VVEL']:
            lon_name = 'ULONG'
        data[lon_name] = fix_lon_range(data[lon_name])
        
        # Convert depth (z_t) from cm to m
        if var in ['SALT','TEMP','UVEL','VVEL']:
            data['z_t'] = data['z_t']*0.01
            data['z_t'].attrs['units'] = 'meters'
            data['z_t'].attrs['long_name'] = 'depth from surface to midpoint of layer'

        # Change variable names and units in the dataset:
        if var=='TEMP':
            varname = 'ConsTemp'
            data.attrs['long_name'] ='Conservative Temperature'
        elif var=='SALT':
            varname = 'AbsSal'
            data = data.rename(varname)
            data.attrs['long_name'] ='Absolute Salinity'
        else:
            varname=var
            if var=='SSH':
                data.attrs['units'] = 'meters'
            elif var in ['UVEL','VVEL']:
                data.attrs['units'] = 'm/s'

        # Write data
        out_file_name = f'{out_dir}CESM2-{expt}_ens{ens}_{varname}_y{year}.nc'
        data.to_netcdf(out_file_name, unlimited_dims='time')

    return


# Process atmospheric forcing from CESM2 scenarios (PPACE, LENS, MENS, LW1.5, LW2.0) for a single variable and single ensemble member.
# expt='LE2', var='PRECT', ens='1011.001' etc.
def cesm2_atm_forcing (expt, var, ens, out_dir, start_year=1850, end_year=2100, 
                       land_mask='/gws/nopw/j04/anthrofail/birgal/NEMO_AIS/climate-forcing/CESM2/LE2/b.e21.BHISTsmbb.f09_g17.LE2-1011.001.cam.h0.LANDFRAC.185001-185912.nc'):

    if expt not in ['LE2']:
        raise Exception('Invalid experiment {expt}')

    # load cesm2 land-ocean mask
    cesm2_mask = xr.open_dataset(land_mask).LANDFRAC

    freq     = 'daily'
    for year in range(start_year, end_year+1):
        # read in the data and subset to the specified year
        if var=='PRECS': # snowfall
            file_pathc = find_cesm2_file(expt, 'PRECSC', 'atm', freq, ens, year)
            file_pathl = find_cesm2_file(expt, 'PRECSL', 'atm', freq, ens, year)
            ds_conv    = xr.open_dataset(file_pathc) # convective snow rate
            ds_large   = xr.open_dataset(file_pathl) # large-scale snow rate
            data_conv  = ds_conv['PRECSC'].isel(time=(ds_conv.time.dt.year == year))
            data_large = ds_large['PRECSL'].isel(time=(ds_large.time.dt.year == year))
        else: 
            file_path = find_cesm2_file(expt, var, 'atm', freq, ens, year)
            ds        = xr.open_dataset(file_path)
            data      = ds[var].isel(time=(ds.time.dt.year == year))

        # Unit conversions #
        # notes: don't think I need to swap FSDS,FLDS signs like Kaitlin did for CESM1, qrefht is specific 
        #        humidity so don't need to convert, but will need to change read in option in namelist_cfg
        if var=='PRECT': # total precipitation
            # Convert from m/s to kg/m2/s
            data *= rho_fw
        elif var=='PRECS': # snowfall
            # Combine convective and large scale snowfall rates and convert from m of water equivalent to kg/m2/s
            data  = (data_conv + data_large) * rho_fw        
            data  = data.rename('PRECS')

        # Mask atmospheric forcing over land based on cesm2 land mask (since land values might not be representative for the ocean areas)
        data = data.where(cesm2_mask.isel(time=0) == 0)
        # And then fill masked areas with nearest non-NaN latitude neighbour
        data = data.interpolate_na(dim='lat', method='nearest', fill_value="extrapolate")
        
        # Convert calendar to Gregorian
        data = data.convert_calendar('gregorian')

        # Change variable names and units in the dataset:
        if var=='PRECS':
            data[var].attrs['long_name'] ='Total snowfall (convective + large-scale)'
            data[var].attrs['units'] = 'kg/m2/s'
        elif var=='PRECT':
            data[var].attrs['units'] = 'kg/m2/s'

        # Write data
        out_file_name = f'{out_dir}CESM2-{expt}_ens{ens}_{var}_y{year}.nc'
        data.to_netcdf(out_file_name, unlimited_dims='time')
    return


# Create CESM atmospheric forcing for the given scenario, for all variables and ensemble members.
# ens_strs : list of strings of ensemble member names
def cesm2_expt_all_atm_forcing (expt, ens_strs=None, out_dir=None, start_year=1850, end_year=2100):
    
    if out_dir is None:
        raise Exception('Please specify an output directory via optional argument out_dir')

    var_names = ['UBOT','VBOT','FSDS','FLDS','TREFHT','QREFHT','PRECT','PSL','PRECS']
    for ens in ens_strs:
        print(f'Processing ensemble member {ens}')
        for var in var_names:
            print(f'Processing {var}')
            cesm2_atm_forcing(expt, var, ens, out_dir, start_year=start_year, end_year=end_year)

    return

def cesm2_expt_all_ocn_forcing(expt, ens_strs=None, out_dir=None, start_year=1850, end_year=2100):

    if out_dir is None:
        raise Exception('Please specify an output directory via optional argument out_dir')

    ocn_var_names = ['SSH'] # ['TEMP','SALT','UVEL','VVEL','SSH']
    ice_var_names = ['aice','sithick','sisnthick']
    var_names = ocn_var_names + ice_var_names
 
    for ens in ens_strs:
        print(f'Processing ensemble member {ens}')
        for var in var_names:
            print(f'Processing {var}')
            cesm2_ocn_forcing(expt, var, ens, out_dir, start_year=start_year, end_year=end_year)

    return
