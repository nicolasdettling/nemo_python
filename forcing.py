###########################################################
# Generate forcing including atmospheric, runoff etc.
###########################################################
import xarray as xr
import numpy as np
import glob
from .utils import distance_btw_points, closest_point, convert_to_teos10, fix_lon_range, dewpoint_to_specific_humidity
from .grid import get_coast_mask, get_icefront_mask
from .ics_obcs import fill_ocean
from .interpolation import regrid_era5_to_cesm2, extend_into_mask, regrid_to_NEMO, neighbours
from .file_io import find_cesm2_file, find_processed_cesm2_file
from .constants import temp_C2K, rho_fw, cesm2_ensemble_members, sec_per_day, sec_per_hour

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
def shift_calving(calving, mask, nemo_mask, icefront_mask_ocn, max_distance=2e6, calving_var='soicbclv', ocn=False):
    # NEMO domain grid points
    x, y        = np.meshgrid(nemo_mask.nav_lon.x, nemo_mask.nav_lon.y)
    calving_x   = x[(~np.isnan(mask))]
    calving_y   = y[(~np.isnan(mask))]
    calving_new = np.copy(calving[calving_var].values);

    ice_shelf  = (nemo_mask.tmaskutil.isel(time_counter=0) - nemo_mask.tmask.isel(time_counter=0, nav_lev=0)).astype(bool)
    open_ocean = (nemo_mask.tmask.isel(time_counter=0, nav_lev=0) == 1)
    land       = ~open_ocean
    # Return ocean points with at least 3 ice shelf neighbour
    num_ice_shelf_neighbours = neighbours(ice_shelf, missing_val=0)[-1]
    num_land_neighbours      = neighbours(land, missing_val=0)[-1] # also includes the iceshelf points
    confined_points          = (open_ocean*(num_land_neighbours > 2)).astype(bool)

    # Coordinates of ocean points closest to iceshelf front 
    # increase number of possible points by shifting the icefront mask:
    ocean_neighbours       = neighbours(icefront_mask_ocn, missing_val=0)[-1]
    icefront_mask_extended = (((icefront_mask_ocn)+(ocean_neighbours))*open_ocean).astype(bool)
    icefront_x             = x[icefront_mask_extended]
    icefront_y             = y[icefront_mask_extended]
    icefront_coord         = (nemo_mask.nav_lon.values[icefront_mask_extended], nemo_mask.nav_lat.values[icefront_mask_extended])

    # For each land iceberg calving point, check distance to nearest iceshelf front and move closer if possible:
    for index in list(zip(calving_y, calving_x)):
     
        calving_coord = (nemo_mask.nav_lon.values[index], nemo_mask.nav_lat.values[index])
        distances     = distance_btw_points(calving_coord, icefront_coord)
        distances[distances < 2e4] = np.nan
        # only move cell if it is within a certain distance
        if np.nanmin(np.abs(distances)) < max_distance:     
            new_x         = icefront_x[np.nanargmin(np.abs(distances))]
            new_y         = icefront_y[np.nanargmin(np.abs(distances))]
            # Move calving to the nearest icefront point and add to any pre-existing calving at that point
            calving_new[(new_y, new_x)] = calving_new[(new_y, new_x)] + calving_new[index]    
            calving_new[index]          = 0 # remove calving from originating point

    # For each ocean iceberg calving point, check that it isn't surrounded on three sides by iceshelf points or land points
    # to prevent accumulation of icebergs in small coastal regions
    if ocn:
        # Points that are ocean that have at least three iceshelf neighbour points, these are the ones that I'll need to move
        confined_x = x[confined_points*icefront_mask_ocn]
        confined_y = y[confined_points*icefront_mask_ocn]
        unconfined_x = x[~confined_points*open_ocean]
        unconfined_y = y[~confined_points*open_ocean]
        unconfined_coord = (nemo_mask.nav_lon.values[~confined_points*open_ocean], nemo_mask.nav_lat.values[~confined_points*open_ocean])
        
        for ind in list(zip(confined_y, confined_x)):
            confined_coord = (nemo_mask.nav_lon.values[ind], nemo_mask.nav_lat.values[ind])
            distances      = distance_btw_points(confined_coord, unconfined_coord)#icefront_coord)
            distances[distances < 2e4] = np.nan

            new_x = unconfined_x[np.nanargmin(np.abs(distances))]
            new_y = unconfined_y[np.nanargmin(np.abs(distances))]
            # Move calving to the nearest icefront point and add to any pre-existing calving at that point
            calving_new[new_y, new_x] = calving_new[new_y, new_x] + calving_new[ind]    
            calving_new[ind]        = 0 # remove calving from originating point

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
    icefront_mask_ice = get_icefront_mask(nemo_mask, side='ice')

    # Shift calving points to the iceshelf edge
    calv_ocn_new  = shift_calving(calving       , calving_ocn , nemo_mask, icefront_mask_ocn, ocn=True) # from ocean
    calv_land_new = shift_calving(calv_ocn_new  , calving_land, nemo_mask, icefront_mask_ocn, ocn=False) # from land
    calv_ice_new  = shift_calving(calv_land_new , calving_ice , nemo_mask, icefront_mask_ocn, ocn=False) # from ice

    # Check if the icebergs calve in regions shallower than the minimum initial iceberg thickness (40 m)
    calving_depth    = nemo_mask.bathy_metry.squeeze().values*(calv_ice_new[calving_var].squeeze().values.astype(bool))
    if np.sum((calving_depth < 40)*(calving_depth > 0)) != 0:
        print('Warning: number of cells with calving shallower than minimum iceberg thickness is, ', np.sum((calving_depth < 40)*(calving_depth > 0)))

    # Write the calving dataset to a netcdf file:
    calv_ice_new[calving_var] = ('time_counter',) + calving[calving_var].dims, calv_ice_new[calving_var].values[np.newaxis, ...] 
    calv_ice_new.to_netcdf(f"{new_file_path}", unlimited_dims='time_counter')

    # Check that the amount of calving that occurs in the new file is approximately equal to the original file:
    # allow for a tolerance of 0.1% (although it should be essentially equal within machine precision)
    tolerance = 0.001*(np.sum(calv_ice_new[calving_var].values)) 
    if np.abs(np.sum(calv_ice_new[calving_var].values) - np.sum(calving[calving_var].values)) >= tolerance:
        raise Exception('The total amount of calving in the new file is not equal to the original total', \
                np.sum(calv_ice_new[calving_var].values), np.sum(calving[calving_var].values))
    
    return calv_ice_new

# Process ocean conditions from CESM2 scenarios for a single variable and single ensemble member (for initial and boundary conditions).
def cesm2_ocn_forcing (expt, var, ens, out_dir, start_year=1850, end_year=2100):

    if expt not in ['LE2']:
        raise Exception(f'Invalid experiment {expt}')

    freq = 'monthly'
    for year in range(start_year, end_year+1):
        # read in the data and subset to the specified year
        if var in ['aice','sithick','sisnthick']:
            if year > 1850:
                file_path_prev = find_cesm2_file(expt, var, 'ice', freq, ens, year-1)
            file_path = find_cesm2_file(expt, var, 'ice', freq, ens, year)
        else:
            if year > 1850:
                file_path_prev = find_cesm2_file(expt, var, 'ocn', freq, ens, year-1) # load last month from previous file
            file_path = find_cesm2_file(expt, var, 'ocn', freq, ens, year)
        
        if year==1850:
            ds = xr.open_dataset(file_path)
        else:
            if file_path_prev != file_path:
                ds = xr.open_mfdataset([file_path_prev, file_path])
            else:
                ds = xr.open_dataset(file_path)
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
  
        # Mask sea ice conditions based on tmask (so that land is NaN and no ice areas are zero)
        if var in ['sithick','sisnthick']:
            data = data.fillna(0)
            data = data.where((ds.isel(time=(ds.time.dt.year == year)).tmask.fillna(0).values) != 0)

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

# Convert wind velocities from the lowest atmospheric model level grid cell to the 10 m wind level 
# (CESM2 output UBOT and VBOT is at 992 hPa) by using the wind speed magnitude from the variable U10 and the wind directions from UBOT and VBOT
def UBOT_to_U10_wind(UBOT, VBOT, U10):

    # calculate the angle between the UBOT and VBOT wind vectors
    theta = np.arctan2(VBOT, UBOT) 
    # then use this angle to create the x and y wind vector components that sum to the magnitude of the U10 wind speed
    U10x = U10*np.cos(theta)
    U10y = U10*np.sin(theta)

    return U10x, U10y

# Get CESM2 wind velocities from the wind speed at 10 meters and the direction from the surface stress 
# (because wind speed vectors are not available for the CESM2 single forcing experiments at daily frequency)
def TAU_to_U10xy(TAUX, TAUY, U10):

    # calculate the angle between the surface stress vectors (need to take opposite direction because of definition difference)
    theta = np.arctan2(-1*TAUY, -1*TAUX)
    # then use this angle to create the x and y wind vector components that sum to the magnitude of the U10 wind speed
    U10x = U10*np.cos(theta)
    U10y = U10*np.sin(theta)

    return U10x, U10y

# Process atmospheric forcing from CESM2 scenarios (LE2, etc.) for a single variable and single ensemble member.
# expt='LE2', var='PRECT', ens='1011.001' etc.
def cesm2_atm_forcing (expt, var, ens, out_dir, start_year=1850, end_year=2100, year_ens_start=1750, shift_wind=False):

    if expt not in ['LE2', 'piControl', 'SF-xAER', 'SF-AAER', 'SF-BMB', 'SF-GHG', 'SF-EE']:
        raise Exception('Invalid experiment {expt}')

    freq     = 'daily'
    for year in range(start_year, end_year+1):
        # read in the data and subset to the specified year

        # load cesm2 land-ocean mask
        land_mask  = find_cesm2_file(expt, 'LANDFRAC', 'atm', 'monthly', ens, year)
        cesm2_mask = xr.open_dataset(land_mask).LANDFRAC.isel(time=0)
        if var=='PRECS': # snowfall
            if expt=='piControl': freq='monthly' # only monthly files available
            file_pathc = find_cesm2_file(expt, 'PRECSC', 'atm', freq, ens, year)
            file_pathl = find_cesm2_file(expt, 'PRECSL', 'atm', freq, ens, year)
            ds_conv    = xr.open_dataset(file_pathc) # convective snow rate
            ds_large   = xr.open_dataset(file_pathl) # large-scale snow rate
            data_conv  = ds_conv['PRECSC'].isel(time=(ds_conv.time.dt.year == year))
            data_large = ds_large['PRECSL'].isel(time=(ds_large.time.dt.year == year))
        elif var=='wind':
            if expt=='piControl': var_U = 'U'; var_V='V';
            elif expt=='LE2': var_U='UBOT'; var_V='VBOT';
            elif expt in ['SF-xAER', 'SF-AAER', 'SF-BMB', 'SF-GHG', 'SF-EE']: var_U='TAUX'; var_V='TAUY'; # single forcing expts don't have UBOT, VBOT so need to calc from stress

            file_path_U10  = find_cesm2_file(expt, 'U10', 'atm', freq, ens, year)
            file_path_UBOT = find_cesm2_file(expt, var_U, 'atm', freq, ens, year)
            file_path_VBOT = find_cesm2_file(expt, var_V, 'atm', freq, ens, year)
            ds_U10         = xr.open_dataset(file_path_U10)
            ds_UBOT        = xr.open_dataset(file_path_UBOT)
            ds_VBOT        = xr.open_dataset(file_path_VBOT)
            data_U10       = ds_U10['U10'].isel(time=(ds_U10.time.dt.year == year))
            data_UBOT      = ds_UBOT[var_U].isel(time=(ds_UBOT.time.dt.year == year))
            data_VBOT      = ds_VBOT[var_V].isel(time=(ds_VBOT.time.dt.year == year))
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

        # Convert the wind to the 10 m wind
        if var=='wind':
            # For the piControl experiment, only full column winds are available, so select the bottom wind
            if expt=='piControl':
                data_UBOT = data_UBOT.isel(lev=-1) # bottom wind is the last entry (992 hPa)
                data_VBOT = data_VBOT.isel(lev=-1)
            elif expt in ['SF-xAER', 'SF-AAER', 'SF-BMB', 'SF-GHG', 'SF-EE']:
                Ux, Uy    = TAU_to_U10xy(data_UBOT, data_VBOT, data_U10)
                data_UBOT = Ux.rename('U10x')
                data_VBOT = Uy.rename('U10y')
                shift_wind=False # since wind speed vectors are already calculated from the U10 speed

            # Convert the wind to the corrected height
            if shift_wind:
                Ux, Uy    = UBOT_to_U10_wind(data_UBOT, data_VBOT, data_U10)
                data_UBOT = Ux.rename('U10x')
                data_VBOT = Uy.rename('U10y')

        if var=='wind':
            data_arrays = [data_UBOT, data_VBOT]
        else:
            data_arrays = [data]       

        for arr in data_arrays:
            if var in ['QREFHT','TREFHT','FSDS','FLDS','PSL','PRECT','PRECS']: 
                # Mask atmospheric forcing over land based on cesm2 land mask (since land values might not be representative for the ocean areas)
                arr = xr.where(cesm2_mask.values != 0, -9999, arr)
                # And then fill masked areas with nearest non-NaN latitude neighbour
                print(f'Filling land for variable {var} in year {year}')
                var_filled_array = np.empty(arr.shape)
                for tind, t in enumerate(arr.time):
                    var_filled_array[tind,:,:] = extend_into_mask(arr.isel(time=tind).values, missing_val=-9999, fill_val=np.nan, use_2d=True, use_3d=False, num_iters=100)
                   
                arr.data = var_filled_array 
     
            # Convert longitude range from (0,360) to (-180,180) degrees east
            arr['lon'] = fix_lon_range(arr['lon'])
            # CESM2 does not do leap years, but NEMO does, so fill 02-29 with 02-28        
            # Also convert calendar to Gregorian
            fill_value = arr.isel(time=((arr.time.dt.month==2)*(arr.time.dt.day==28)))
            arr = arr.convert_calendar('gregorian', dim='time', missing=fill_value)

            # Change variable names and units in the dataset:
            varname = arr.name 
            if var=='PRECS':
                arr.attrs['long_name'] ='Total snowfall (convective + large-scale)'
                arr.attrs['units'] = 'kg/m2/s'
            elif var=='PRECT':
                arr.attrs['units'] = 'kg/m2/s'
            elif var=='wind':
                if varname=='U10x':
                    arr.attrs['long_name'] = 'zonal wind at 10 m'
                elif varname=='U10y':
                    arr.attrs['long_name'] = 'meridional wind at 10 m'

            # Write data
            out_file_name = f'{out_dir}CESM2-{expt}_ens{ens}_{varname}_y{year}.nc'
            #if expt=='piControl': # split files into ensemble chunks
                # for now just keep the smae:
                #out_file_name = f'{out_dir}CESM2-{expt}_ens{ens}_{varname}_y{1850+(year-year_ens_start)}.nc'
            arr.to_netcdf(out_file_name, unlimited_dims='time')
    return


# Create CESM2 atmospheric forcing for the given scenario, for all variables and ensemble members.
# ens_strs : list of strings of ensemble member names
def cesm2_expt_all_atm_forcing (expt, ens_strs=None, out_dir=None, start_year=1850, end_year=2100, year_ens_start=1750, shift_wind=False):
    
    if out_dir is None:
        raise Exception('Please specify an output directory via optional argument out_dir')

    var_names = ['wind','FSDS','FLDS','TREFHT','QREFHT','PRECT','PSL','PRECS'] 
    for ens in ens_strs:
        print(f'Processing ensemble member {ens}')
        for var in var_names:
            print(f'Processing {var}')
            cesm2_atm_forcing(expt, var, ens, out_dir, start_year=start_year, end_year=end_year, year_ens_start=year_ens_start, shift_wind=shift_wind)

    return

# Create CESM2 ocean forcing for the given scenario, for all variables and ensemble members.
# ens_strs : list of strings of ensemble member names
def cesm2_expt_all_ocn_forcing(expt, ens_strs=None, out_dir=None, start_year=1850, end_year=2100):

    if out_dir is None:
        raise Exception('Please specify an output directory via optional argument out_dir')

    ocn_var_names = ['TEMP','SALT','UVEL','VVEL','SSH']
    ice_var_names = ['aice','sithick','sisnthick']
    var_names = ocn_var_names + ice_var_names
 
    for ens in ens_strs:
        print(f'Processing ensemble member {ens}')
        for var in var_names:
            print(f'Processing {var}')
            cesm2_ocn_forcing(expt, var, ens, out_dir, start_year=start_year, end_year=end_year)

    return


# Helper function calculates the time-mean over specified year range for ERA5 output (for bias correction)
# Input: 
# - variable : string of forcing variable name (in ERA5 naming convention)
# - (optional) year_start : start year for time averaging
# - (optional) end_year   : end year for time averaging
# - (optional) out_file   : path to file to write time mean to NetCDF in case you want to store it
def era5_time_mean_forcing(variable, year_start=1979, year_end=2015, out_file=None, monthly=False,
                           era5_folder='/gws/nopw/j04/anthrofail/birgal/NEMO_AIS/ERA5-forcing/daily/files/processed/',
                           land_mask='/gws/nopw/j04/anthrofail/birgal/NEMO_AIS/ERA5-forcing/daily/files/land_sea_mask.nc'):

    ERA5_ds   = xr.open_mfdataset(f'{era5_folder}{variable}_*.nc')
    ERA5_ds   = ERA5_ds.isel(time=((ERA5_ds.time.dt.year <= year_end)*(ERA5_ds.time.dt.year >= year_start)))
    if monthly:
        time_mean = ERA5_ds.groupby('time.month').mean(dim='time')
    else:
        time_mean = ERA5_ds.mean(dim='time')

    # mask areas that are land:
    #era5_mask = xr.open_dataset(land_mask).lsm.isel(valid_time=0)
    #era5_mask['longitude'] = fix_lon_range(era5_mask['longitude'])
    time_mean['longitude'] = fix_lon_range(time_mean['longitude'])
    #time_mean = xr.where(era5_mask != 0, np.nan, time_mean)

    if out_file:
        time_mean.to_netcdf(out_file)
    return time_mean


# Function calculates the time-mean over specified year range for mean of all CESM2 ensemble members in the specified experiment (for bias correction)
# Input:
# - expt : string of CESM2 experiment name (e.g. 'LE2')
# - variable : string of forcing variable name
# - (optional) year_start : start year for time averaging
# - (optional) end_year   : end year for time averaging
# - (optional) out_file   : path to file to write time mean to NetCDF in case you want to store it
# - (optional) ensemble_members : list of strings of ensemble members to average (defaults to all the ones that have been downloaded)
def cesm2_ensemble_time_mean_forcing(expt, variable, year_start=1979, year_end=2015, out_file=None, ensemble_members=cesm2_ensemble_members, monthly=False,
                             land_mask='/gws/nopw/j04/anthrofail/birgal/NEMO_AIS/climate-forcing/CESM2/LE2/b.e21.BHISTsmbb.f09_g17.LE2-1011.001.cam.h0.LANDFRAC.185001-185912.nc'):

    # calculate ensemble mean for each year
    year_mean = xr.Dataset()
    print('Ensemble members:', ensemble_members)
    for year in range(year_start, year_end+1):
        print(year)
        files_to_open = []
        for ens in ensemble_members:
            file_path     = find_processed_cesm2_file(expt, variable, ens, year)
            files_to_open += [file_path]
        # calculate ensemble mean    
        ens_files = xr.open_mfdataset(files_to_open, concat_dim='ens', combine='nested')
        ens_year  = ens_files.isel(time=(ens_files.time.dt.year==year))
        if monthly:
            ens_mean = ens_year.groupby('time.month').mean(dim=['time','ens']) # dimensions should be x,y
        else:
            ens_mean = ens_year.mean(dim=['time','ens'])
        # save ensemble mean to xarray dataset
        if year == year_start:
            year_mean = ens_mean
        else:
            year_mean = xr.concat([year_mean, ens_mean], dim='year')

            
    # and then calculate time-mean of all ensemble means:
    time_mean = year_mean.copy().mean(dim='year')

    # mask areas that are land:
    #cesm2_mask = xr.open_dataset(land_mask).LANDFRAC
    #cesm2_mask['lon'] = fix_lon_range(cesm2_mask['lon'])  
    #time_mean  = xr.where(cesm2_mask.isel(time=0) != 0, np.nan, time_mean)

    if out_file:
        time_mean.to_netcdf(out_file)
    
    return time_mean

# Function calculate the bias correction for the atmospheric variable from the specified source type based on 
# the difference between its mean state and the ERA5 mean state.
# Input:
# - source : string of source type (currently only set up for 'CESM2') 
# - variable : string of the variable from the source dataset to be corrected
# - (optional) expt : 
# - (optional) year_start : start year for time averaging
# - (optional) end_year   : end year for time averaging
# - (optional) fill_land  : boolean whether or not to fill grid cells that are land in CESM2 with ocean values along lines of latitudes
# - (optional) ensemble_mean_file, era5_mean_file : string paths to files in case you've already ran the averaging separately
# - (optional) nemo_grid, nemo_mask : string of path to NEMO domain_cfg and mesh_mask files
# - (optional) out_folder : string to location to save the bias correction file
def atm_bias_correction(source, variable, expt='LE2', year_start=1979, year_end=2015, 
                        ensemble_mean_file=None, era5_mean_file=None, fill_land=False, monthly=False, method='conservative',
                        era5_folder='/gws/nopw/j04/anthrofail/birgal/NEMO_AIS/ERA5-forcing/daily/files/processed/',
                        out_folder='/gws/nopw/j04/anthrofail/birgal/NEMO_AIS/climate-forcing/CESM2/LE2/processed/'):

    # process_forcing_for_correction(source, variable)
    if source=='CESM2':
        # Read in ensemble time mean (or calculate it)
        if ensemble_mean_file:
            CESM2_time_mean = xr.open_dataset(ensemble_mean_file)
        else:
            if monthly:
                ensemble_mean_file = f'{out_folder}{variable}_mean_{year_start}-{year_end}_monthly.nc'
            else:
                ensemble_mean_file = f'{out_folder}{variable}_mean_{year_start}-{year_end}.nc'
            print('Calculating CESM2 ensemble mean')
            CESM2_time_mean = cesm2_ensemble_time_mean_forcing(expt, variable, out_file=ensemble_mean_file, year_start=year_start, year_end=year_end, monthly=monthly)
        # Regrid CESM2 to NEMO:
        print('Regridding CESM2 to NEMO')
        if monthly:
            for m in CESM2_time_mean.month:
                CESM2_month_regridded = regrid_to_NEMO(CESM2_time_mean.sel(month=m), variable, calc_regrid_operator=False, method=method,
                                                       ro_filename=f'/gws/nopw/j04/anthrofail/birgal/NEMO_AIS/cf-regridding/CESM2-to-NEMO-{method}.pickle')
                if m > 1:
                    CESM2_regridded = xr.concat([CESM2_regridded, CESM2_month_regridded], dim='month')
                else:
                    CESM2_regridded = CESM2_month_regridded
        else:
            CESM2_regridded = regrid_to_NEMO(CESM2_time_mean, variable, calc_regrid_operator=True, method=method,
                                             ro_filename=f'/gws/nopw/j04/anthrofail/birgal/NEMO_AIS/cf-regridding/CESM2-to-NEMO-{method}.pickle')


        # Read in time mean of ERA5 files (or calculate it)
        CESM2_to_ERA5_varnames = {'TREFHT':'t2m','FSDS':'msdwswrf','FLDS':'msdwlwrf','QREFHT':'sph2m', 'PRECS':'msr', 'PRECT':'mtpr', 'PSL':'msl'} # I calculated specific humidity
        varname = CESM2_to_ERA5_varnames[variable]
        if era5_mean_file:
            ERA5_time_mean = xr.open_dataset(era5_mean_file)
        else:
            if monthly:
                era5_mean_file = f'{era5_folder}{variable}_mean_{year_start}-{year_end}_monthly.nc'
            else:
                era5_mean_file = f'{era5_folder}{variable}_mean_{year_start}-{year_end}.nc'
            print('Calculating ERA5 mean')
            ERA5_time_mean = era5_time_mean_forcing(varname, year_start=year_start, year_end=year_end, out_file=era5_mean_file, era5_folder=era5_folder, monthly=monthly)
        if variable=='QREFHT':
           # convert dewpoint temperature to specific humidity
           varname='specific_humidity'
        
        ERA5_time_mean = ERA5_time_mean.rename({varname:variable, 'longitude':'lon', 'latitude':'lat'})
        
        # Adjust the longitude and regrid time means to NEMO configuration grid, so that they can be used to bias correct
        print('Regridding ERA5 to NEMO')
        if monthly:
            for m in ERA5_time_mean.month:
                #ERA5_month_regridded = regrid_era5_to_cesm2(CESM2_time_mean.sel(month=m), ERA5_time_mean.sel(month=m), variable)
                ERA5_month_regridded = regrid_to_NEMO(ERA5_time_mean.sel(month=m), variable, calc_regrid_operator=False, method=method,
                                                      ro_filename=f'/gws/nopw/j04/anthrofail/birgal/NEMO_AIS/cf-regridding/ERA5-to-NEMO-{method}.pickle')
                if m > 1:
                    ERA5_regridded = xr.concat([ERA5_regridded, ERA5_month_regridded], dim='month')
                else:
                    ERA5_regridded = ERA5_month_regridded
        else:
            ERA5_regridded = regrid_to_NEMO(ERA5_time_mean, variable, calc_regrid_operator=True, method=method,
                                            ro_filename=f'/gws/nopw/j04/anthrofail/birgal/NEMO_AIS/cf-regridding/ERA5-to-NEMO-{method}.pickle')
            #ERA5_regridded = regrid_era5_to_cesm2(CESM2_time_mean, ERA5_time_mean, variable)
 
        # thermodynamic correction
        if variable in ['TREFHT','QREFHT','FLDS','FSDS','PRECS','PRECT']:
            print('Correcting thermodynamics')
            if monthly:
                out_file = f'{out_folder}{source}-{expt}_{variable}_bias_corr_monthly.nc'
            else:
                out_file = f'{out_folder}{source}-{expt}_{variable}_bias_corr_highres.nc'
            CESM2_regridded.to_netcdf(f'{out_folder}CESM2_{variable}_monthly_mean_regridded.nc')
            ERA5_regridded.to_netcdf(f'{out_folder}ERA5_{variable}_monthly_mean_regridded.nc')
            thermo_correction(CESM2_regridded, ERA5_regridded, variable, out_file, fill_land=fill_land, monthly=monthly)
        else:
            raise Exception(f'Variable {variable} does not need bias correction. Check that this is true.')
    else:
        raise Exception("Bias correction currently only set up to correct CESM2. Sorry you'll need to write some more code!")

    return

# Function to calculate the bias and associated thermodynamic correction between source (CESM2) and ERA5 mean fields
# Inputs:
# - source_mean : xarray Dataset containing the ensemble and time mean of the source dataset variable (currently CESM2)
# - ERA5_mean  : xarray Dataset containing the time mean of the ERA5 variable
# - out_file   : string to path to write NetCDF file to
# - fill_land (optional) : boolean indicating whether to fill areas that are land in the source mask with nearest values or whether to just leave it as is
def thermo_correction(source_mean, ERA5_mean, variable, out_file, fill_land=True, monthly=False):
    
    # Calculate difference:
    bias = ERA5_mean - source_mean
    # Fill land regions along latitudes
    if fill_land:
       # bias = bias.interpolate_na(dim='lat', method='nearest', fill_value="extrapolate")
       src_to_fill = xr.where(np.isnan(bias), -9999, bias) # which cells need to be filled
       var_filled  = extend_into_mask(src_to_fill[variable].values, missing_val=-9999, fill_val=np.nan, use_2d=True, use_3d=False, num_iters=100)
       if monthly:
           bias[variable] = (('time','lat','lon'), var_filled)
       else:
           bias[variable] = (('lat','lon'), var_filled)

    # write to file
    if monthly:
        bias.to_netcdf(out_file, unlimited_dims='month')
    else:
        bias.to_netcdf(out_file)
    
    return


def process_era5_forcing(variable, year_start=1979, year_end=2023, era5_folder='/gws/nopw/j04/anthrofail/birgal/NEMO_AIS/ERA5-forcing/daily/files/'):

    for year in range(year_start,year_end+1):
        if variable=='d2m':
            print('Convert dewpoint temperature')
            # convert dewpoint temperature to specific humidity and write to file (for bias correction calc)
            dewpoint_to_specific_humidity(file_dew=f'd2m_y{year}.nc', variable_dew='d2m',
                                          file_slp=f'msl_y{year}.nc', variable_slp='msl',
                                          dataset='ERA5', folder=era5_folder)

        # fill land with nearest connected point and
        landmask = xr.open_dataset(f'{era5_folder}land_sea_mask.nc').isel(valid_time=0).lsm
        # convert time dimension to unlimited so that NEMO reads in the calendar correctly
        for filename in glob.glob(f'{era5_folder}{variable}*y{year}.nc'):
            with xr.open_dataset(filename, mode='a') as data:
                print('Processing', filename)
                try:
                    data = data.rename({'valid_time':'time'})
                except:
                    pass
                
                variable = filename.split('files/')[1].split('_')[0]
                if variable in ['msdwlwrf','msdwswrf','t2m','sph2m','d2m','msl']: 
                    print(f'Filling land for variable {variable} year {year}')
                    if variable=='sph2m':
                        varname='specific_humidity'
                    else:
                        varname=variable
                    src_to_fill = xr.where(landmask!=0, -9999, data[varname]) # which cells need to be filled
                    var_filled_array = np.empty(src_to_fill.shape)
                    for tind, t in enumerate(src_to_fill.time):
                        var_filled_array[:,:,tind] = extend_into_mask(src_to_fill.isel(time=tind).values, missing_val=-9999, fill_val=np.nan, 
                                                                      use_2d=True, use_3d=False, num_iters=200)
                    data[varname] = (('latitude','longitude','time'), var_filled_array)
                    data = data.transpose('time','latitude','longitude')
               
                data.to_netcdf(f'{era5_folder}processed/{variable}_time_y{year}.nc', unlimited_dims={'time':True})

    return
