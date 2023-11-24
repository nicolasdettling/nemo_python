import xarray as xr
import os

from .constants import region_points, region_names, rho_fw, rho_ice, sec_per_year, deg_string, gkg_string
from .utils import single_cavity_mask, region_mask, add_months

# Calculate a timeseries of the given preset variable from an xarray Dataset of NEMO output. Returns DataArrays of the timeseries data, the associated time values, and the variable title. Specify whether there is a halo (true for periodic boundaries in NEMO 3.6).
# Preset variables include:
# <region>_massloss: basal mass loss from the given ice shelf or region of multiple ice shelves (eg brunt, amundsen_sea)
# <region>_bwtemp, <region>_bwsalt: area-averaged bottom water temperature or salinity from the given region or cavity (eg ross_cavity, ross_shelf, ross)
def calc_timeseries (var, ds_nemo, halo=True):
    
    # Parse variable name
    factor = 1
    region_type = None
    if var.endswith('_massloss'):
        option = 'area_int'
        region = var[:var.index('_massloss')]
        region_type = 'cavity'
        nemo_var = 'sowflisf'
        # Convert from kg/s to Gt/y and swap sign
        factor = -rho_ice/rho_fw*1e-12*sec_per_year
        units = 'Gt/y'
        title = 'Basal mass loss'
    elif var.endswith('_bwtemp'):
        option = 'area_avg'
        region = var[:var.index('_bwtemp')]
        nemo_var = 'tob'
        units = deg_string+'C'
        title = 'Bottom temperature'
    elif var.endswith('_bwsalt'):
        option = 'area_avg'
        region = var[:var.index('_bwsalt')]
        nemo_var = 'sob'
        units = gkg_string
        title = 'Bottom salinity'
    elif var.endswith('_temp'):
        option = 'volume_avg'
        region = var[:var.index('_temp')]
        nemo_var = 'thetao'
        units = deg_string+'C'
        title = 'Volume-averaged temperature'
    elif var.endswith('_salt'):
        option = 'volume_avg'
        region = var[:var.index('_salt')]
        nemo_var = 'so'
        units = gkg_string
        title = 'Volume-averaged salinity'

    # Select region
    if region_type is None:
        if region.endswith('cavity'):
            region = region[:region.index('_cavity')]
            region_type = 'cavity'
        elif region.endswith('shelf'):
            region = region[:region.index('_shelf')]
            region_type = 'shelf'
        else:
            region_type = 'all'
    if region in region_points and region_type == 'cavity':
        # Single ice shelf
        mask, region_name = single_cavity_mask(region, ds_nemo, return_name=True)
    else:
        mask, region_name = region_mask(region, ds_nemo, option=region_type, return_name=True)
    title += ' on '+region_name    

    # Trim datasets as needed
    if ds_nemo.sizes['y'] < mask.sizes['y']:
        # The NEMO dataset was trimmed (eg by MOOSE for UKESM) to the southernmost latitudes. Do the same for the mask.
        mask = mask.isel(y=slice(0, ds_nemo.sizes['y']))
    if halo:
        # Remove the halo
        ds_nemo = ds_nemo.copy().isel(x=slice(1,-1))
        mask = mask.isel(x=slice(1,-1))
        
    if option == 'area_int':
        # Area integral
        dA = ds_nemo['area']*mask
        data = (ds_nemo[nemo_var]*dA).sum(dim=['x','y'])
    elif option == 'area_avg':
        # Area average
        dA = ds_nemo['area']*mask
        data = (ds_nemo[nemo_var]*dA).sum(dim=['x','y'])/dA.sum(dim=['x','y'])
    elif option == 'volume_avg':
        # Volume average
        # First need a 3D mask
        mask_3d = xr.where(ds_nemo[nemo_var]==0, 0, mask)
        dV = ds_nemo['area']*ds_nemo['thkcello']*mask_3d
        data = (ds_nemo[nemo_var]*dV).sum(dim=['x','y','deptht'])/dV.sum(dim=['x','y','deptht'])
    data *= factor
    data = data.assign_attrs(long_name=title, units=units)

    return data


# Precompute the given list of timeseries from the given xarray Dataset of NEMO output. Save in a NetCDF file which concatenates after each call to the function.
def precompute_timeseries (ds_nemo, timeseries_types, timeseries_file, halo=True):

    # Calculate each timeseries and save to a Dataset
    ds_new = None
    for var in timeseries_types:
        data = calc_timeseries(var, ds_nemo, halo=halo)
        if ds_new is None:            
            ds_new = xr.Dataset({var:data})
        else:
            ds_new = ds_new.assign({var:data})
    # Use time_centered as the dimension as it includes real times - time_counter is reset to 0 every output file
    ds_new = ds_new.swap_dims({'time_counter':'time_centered'})

    if os.path.isfile(timeseries_file):
        # File already exists; read it
        ds_old = xr.open_dataset(timeseries_file)
        # Concatenate new data
        ds_new.load()
        ds_new = xr.concat([ds_old, ds_new], dim='time_centered')
        ds_old.close()

    # Save to file, overwriting if needed
    ds_new.to_netcdf(timeseries_file, mode='w')


# Precompute timeseries from the given simulation, either from the beginning (timeseries_file does not exist) or picking up where it left off (timeseries_file does exist). Considers all NEMO output files stamped with suite_id in the given directory sim_dir, and assumes the timeseries file is in that directory too.
def update_simulation_timeseries (suite_id, timeseries_types, timeseries_file='timeseries.nc', sim_dir='./', halo=True):

    update = os.path.isfile(sim_dir+timeseries_file)
    if update:
        # Timeseries file already exists
        # Get last time index
        ds_ts = xr.open_dataset(sim_dir+timeseries_file)
        time_last = ds_ts['time_centered'].data[-1]
        year_last = time_last.year
        month_last = time_last.month

    # Identify NEMO output files in the given directory, constructed as wildcard strings for each date code
    nemo_files = []
    for f in os.listdir(sim_dir):
        if os.path.isdir(sim_dir+'/'+f):
            # Skip directories
            continue
        if f.startswith('nemo_'+suite_id+'o'):
            # UKESM file naming conventions
            file_head = 'nemo_'+suite_id+'o'
        elif f.startswith(suite_id):
            # Standalone NEMO file naming conventions
            file_head = suite_id
        else:
            # Not a NEMO output file; skip it
            continue
        if '_1m_' not in f:
            raise Exception('update_simulation_timeseries can only handle monthly NEMO output files. Need to code other options or move the non-monthly files elsewhere.')
        file_head += '_1m_'
        # Extract date code (yyyymmdd_yyyymmdd)
        date_code = f[len(file_head):len(file_head)+17]
        if update:
            # Need to check if date code has already been processed
            year = int(date_code[:4])
            month = int(date_code[4:6])
            if year < year_last or (year==year_last and month<=month_last):
                # Skip it
                continue
        # Now construct wildcard string and add to list if it's not already there
        file_pattern = file_head + date_code + '*'
        if file_pattern not in nemo_files:
            nemo_files.append(file_pattern)        
    # Now sort alphabetically - i.e. by ascending date code
    nemo_files.sort()

    # Loop through each date code and process
    for file_pattern in nemo_files:
        print('Processing '+file_pattern)
        ds_nemo = xr.open_mfdataset(sim_dir+'/'+file_pattern)
        ds_nemo.load()
        precompute_timeseries(ds_nemo, timeseries_types, sim_dir+'/'+timeseries_file, halo=halo)
                    
                
        

    
                                

    
        
        
