###########################################################
# Generate forcing including atmospheric, runoff etc.
###########################################################
import xarray as xr
import numpy as np
from .utils import distance_btw_points, closest_point, convert_to_teos10, fix_lon_range
from .grid import get_coast_mask, get_icefront_mask
from .ics_obcs import fill_ocean
from .interpolation import regrid_era5_to_cesm2
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

    # check that the differences between U10 speed and recreated U10 speed are small:
    U10_recreate = np.sqrt(U10x**2 + U10y**2)
    if np.max(np.abs(U10 - U10_recreate)) > 0.1: 
        raise Exception('The maximum difference between the provided U10 wind speed and the recreated speed' + \
                        'is greater than 0.1 m/s. Double check that wind velocities were recreated correctly.')

    return U10x, U10y

# Process atmospheric forcing from CESM2 scenarios (LE2, etc.) for a single variable and single ensemble member.
# expt='LE2', var='PRECT', ens='1011.001' etc.
def cesm2_atm_forcing (expt, var, ens, out_dir, start_year=1850, end_year=2100, year_ens_start=1750, shift_wind=False,
                       land_mask='/gws/nopw/j04/anthrofail/birgal/NEMO_AIS/climate-forcing/CESM2/LE2/b.e21.BHISTsmbb.f09_g17.LE2-1011.001.cam.h0.LANDFRAC.185001-185912.nc'):

    if expt not in ['LE2', 'piControl']:
        raise Exception('Invalid experiment {expt}')

    # load cesm2 land-ocean mask
    cesm2_mask = xr.open_dataset(land_mask).LANDFRAC

    freq     = 'daily'
    for year in range(start_year, end_year+1):
        # read in the data and subset to the specified year
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
            else: var_U='UBOT'; var_V='VBOT';
            file_path_UBOT = find_cesm2_file(expt, var_U, 'atm', freq, ens, year)
            file_path_VBOT = find_cesm2_file(expt, var_V, 'atm', freq, ens, year)
            file_path_U10  = find_cesm2_file(expt, 'U10', 'atm', freq, ens, year)

            ds_UBOT = xr.open_dataset(file_path_UBOT)
            ds_VBOT = xr.open_dataset(file_path_VBOT)
            ds_U10  = xr.open_dataset(file_path_U10)
            data_UBOT = ds_UBOT[var_U].isel(time=(ds_UBOT.time.dt.year == year))
            data_VBOT = ds_VBOT[var_V].isel(time=(ds_VBOT.time.dt.year == year))
            data_U10  = ds_U10['U10'].isel(time=(ds_U10.time.dt.year == year))
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
            # Convert the wind to the corrected height
            if shift_wind:
                Ux, Uy = UBOT_to_U10_wind(data_UBOT, data_VBOT, data_U10)
                data_UBOT = Ux.rename('U10x')
                data_VBOT = Uy.rename('U10y')

        if var=='wind':
            data_arrays = [data_UBOT, data_VBOT]
        else:
            data_arrays = [data]       

        for arr in data_arrays:
            if var in ['QREFHT','TREFHT','FSDS','FLDS','PSL']: 
                # Mask atmospheric forcing over land based on cesm2 land mask (since land values might not be representative for the ocean areas)
                arr = xr.where(cesm2_mask.isel(time=0).values != 0, np.nan, arr)
                # And then fill masked areas with nearest non-NaN latitude neighbour
                arr = arr.interpolate_na(dim='lat', method='nearest', fill_value="extrapolate")
     
            # CESM2 does not do leap years, but NEMO does, so fill 02-29 with 02-28        
            # Also convert calendar to Gregorian
            fill_value = arr.isel(time=((arr.time.dt.month==2)*(arr.time.dt.day==28)))
            arr = arr.convert_calendar('gregorian', missing=fill_value)

            # Convert longitude range from (0,360) to (-180,180) degrees east
            arr['lon'] = fix_lon_range(arr['lon'])

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
def cesm2_expt_all_atm_forcing (expt, ens_strs=None, out_dir=None, start_year=1850, end_year=2100, year_ens_start=1750):
    
    if out_dir is None:
        raise Exception('Please specify an output directory via optional argument out_dir')

    var_names = ['wind','FSDS','FLDS','TREFHT','QREFHT','PRECT','PSL','PRECS']
    for ens in ens_strs:
        print(f'Processing ensemble member {ens}')
        for var in var_names:
            print(f'Processing {var}')
            cesm2_atm_forcing(expt, var, ens, out_dir, start_year=start_year, end_year=end_year, year_ens_start=year_ens_start)

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
def era5_time_mean_forcing(variable, year_start=1979, year_end=2015, out_file=None, 
                           era5_folder='/gws/nopw/j04/anthrofail/birgal/NEMO_AIS/ERA5-forcing/files/',
                           land_mask='/gws/nopw/j04/anthrofail/birgal/NEMO_AIS/ERA5-forcing/ERA5-landmask.nc'):

    ERA5_ds   = xr.open_mfdataset(f'{era5_folder}era5_{variable}_*_daily_averages.nc')
    ERA5_ds   = ERA5_ds.isel(time=((ERA5_ds.time.dt.year <= year_end)*(ERA5_ds.time.dt.year >= year_start)))
    if variable in ['sf','tp']:
        # should never be negative (but has few tiny negative values, so zero those)
        ERA5_ds[variable] = xr.where(ERA5_ds[variable] < 0, 0, ERA5_ds[variable]) 
    time_mean = ERA5_ds.mean(dim='time') 

    if variable in ['sf','tp']:
        time_mean *= rho_fw/sec_per_day # convert to match units
    elif variable in ['ssrd','strd']:
        time_mean /= sec_per_hour # convert to match units
    
    # mask areas that are land:
    era5_mask = xr.open_dataset(land_mask).lsm.isel(time=0) 
    era5_mask['longitude'] = fix_lon_range(era5_mask['longitude'])  
    time_mean['longitude'] = fix_lon_range(time_mean['longitude'])  
    time_mean = xr.where(era5_mask != 0, np.nan, time_mean)

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
def cesm2_ensemble_time_mean_forcing(expt, variable, year_start=1979, year_end=2015, out_file=None, ensemble_members=cesm2_ensemble_members,
                             land_mask='/gws/nopw/j04/anthrofail/birgal/NEMO_AIS/climate-forcing/CESM2/LE2/b.e21.BHISTsmbb.f09_g17.LE2-1011.001.cam.h0.LANDFRAC.185001-185912.nc'):

    # calculate ensemble mean for each year
    year_mean = xr.Dataset()
    for year in range(year_start, year_end+1):
        files_to_open = []
        for ens in ensemble_members:
            file_path     = find_processed_cesm2_file(expt, variable, ens, year)
            files_to_open += [file_path]
        # calculate ensemble mean    
        ens_files = xr.open_mfdataset(files_to_open, concat_dim='ens', combine='nested')
        ens_year  = ens_files.isel(time=(ens_files.time.dt.year==year))
        ens_mean  = ens_year.mean(dim=['time','ens']) # dimensions should be x,y
        # save ensemble mean to xarray dataset
        if year == year_start:
            year_mean = ens_mean
        else:
            year_mean = xr.concat([year_mean, ens_mean], dim='year')
            
    # and then calculate time-mean of all ensemble means:
    time_mean = year_mean.mean(dim='year')

    # mask areas that are land:
    cesm2_mask = xr.open_dataset(land_mask).LANDFRAC
    cesm2_mask['lon'] = fix_lon_range(cesm2_mask['lon'])  
    time_mean  = xr.where(cesm2_mask.isel(time=0) != 0, np.nan, time_mean)

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
                        ensemble_mean_file=None, era5_mean_file=None, fill_land=False,
                        era5_folder='/gws/nopw/j04/anthrofail/birgal/NEMO_AIS/ERA5-forcing/files/',
                        out_folder='/gws/nopw/j04/anthrofail/birgal/NEMO_AIS/climate-forcing/CESM2/LE2/processed/'):

    # process_forcing_for_correction(source, variable)
    if source=='CESM2':
        # Read in ensemble time mean (or calculate it)
        if ensemble_mean_file:
            CESM2_time_mean = xr.open_dataset(ensemble_mean_file)
        else:
            ensemble_mean_file = f'{out_folder}{variable}_mean_{year_start}-{year_end}.nc'
            CESM2_time_mean = cesm2_ensemble_time_mean_forcing(expt, variable, out_file=ensemble_mean_file, year_start=year_start, year_end=year_end)

        # Read in time mean of ERA5 files (or calculate it)
        if era5_mean_file:
            ERA5_time_mean = xr.open_dataset(era5_mean_file)
        else:
            CESM2_to_ERA5_varnames = {'TREFHT':'t2m','FSDS':'ssrd','FLDS':'strd','QREFHT':'sph2m', 'PRECS':'sf', 'PRECT':'tp'} # I calculated specific humidity
            varname = CESM2_to_ERA5_varnames[variable]
            era5_mean_file = f'{era5_folder}{variable}_mean_{year_start}-{year_end}.nc'
            ERA5_time_mean = era5_time_mean_forcing(varname, year_start=year_start, year_end=year_end, out_file=era5_mean_file, era5_folder=era5_folder)
            if variable=='QREFHT':
                varname='specific_humidity'
            ERA5_time_mean = ERA5_time_mean.rename({varname:variable, 'longitude':'lon', 'latitude':'lat'})
        
        # Adjust the longitude and regrid time means to NEMO configuration grid, so that they can be used to bias correct
        ERA5_regridded = regrid_era5_to_cesm2(CESM2_time_mean, ERA5_time_mean, variable)
 
        # thermodynamic correction
        if variable in ['TREFHT','QREFHT','FLDS','FSDS','PRECS','PRECT']:
            print('Correcting thermodynamics')
            out_file = f'{out_folder}{source}-{expt}_{variable}_bias_corr.nc'
            thermo_correction(CESM2_time_mean, ERA5_regridded, out_file, fill_land=fill_land)
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
def thermo_correction(source_mean, ERA5_mean, out_file, fill_land=True):
    
    # Calculate difference:
    bias = ERA5_mean - source_mean
    # Fill land regions along latitudes
    if fill_land:
        bias = bias.interpolate_na(dim='lat', method='nearest', fill_value="extrapolate")

    # write to file
    bias.to_netcdf(out_file)
    
    return
