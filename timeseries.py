import xarray as xr
import os

from .constants import region_points, region_names, rho_fw, rho_ice, sec_per_year, deg_string, gkg_string, drake_passage_lon0, drake_passage_lat_bounds
from .utils import single_cavity_mask, region_mask, add_months, closest_point


# Calculate zonal or meridional transport across the given section. The code will choose a constant slice in x or y corresponding to a constant value of latitude or longitude - so maybe not appropriate in highly rotated regions.
# For zonal transport, set lon0 and lat_bounds, and make sure the dataset includes uo, thkcello, and e2u (can get from domain_cfg).
# For meridional transport, set lat0 and lon_bounds, and make sure the dataset includes vo, thkcello, and e1v.
# Returns value in Sv.
def transport (ds, lon0=None, lat0=None, lon_bounds=None, lat_bounds=None):

    if lon0 is not None and lat_bounds is not None and lat0 is None and lon_bounds is None:
        # Zonal transport across line of constant longitude
        [j_start, i_start] = closest_point(ds, [lon0, lat_bounds[0]])
        [j_end, i_end] = closest_point(ds, [lon0, lat_bounds[1]])
        # Want a single value for i
        if i_start == i_end:
            # Perfect
            i0 = i_start
        else:
            # Choose the midpoint
            print('Warning (transport): grid is rotated; compromising on constant x-coordinate')
            i0 = int(round(0.5*(i_start+i_end)))
        # Assume velocity is already masked to 0 in land mask
        integrand = (ds['uo']*ds['thkcello']*ds['e2u']).isel(x=i0, y=slice(j_start, j_end+1))
        return integrand.sum(dim={'depthu', 'y'})*1e-6
    elif lat0 is not None and lon_bounds is not None and lon0 is None and lat_bounds is None:
        # Meridional transport across line of constant latitude
        [j_start, i_start] = closest_point(ds, [lon_bounds[0], lat0])
        [j_end, i_end] = closest_point(ds, [lon_bounds[1], lat0])
        if j_start == j_end:
            j0 = j_start
        else:
            print('Warning (transport): grid is rotated; compromising on constant y-coordinate')
            j0 = int(round(0.5*(j_start+j_end)))
        integrand = (ds['vo']*ds['thkcello']*ds['e1v']).isel(x=slice(i_start, i_end+1), y=j0)
        return integrand.sum(dim={'depthv', 'x'})*1e-6


# Calculate a timeseries of the given preset variable from an xarray Dataset of NEMO output (must have halo removed). Returns DataArrays of the timeseries data, the associated time values, and the variable title. Specify whether there is a halo (true for periodic boundaries in NEMO 3.6).
# Preset variables include:
# <region>_massloss: basal mass loss from the given ice shelf or region of multiple ice shelves (eg brunt, amundsen_sea)
# <region>_bwtemp, <region>_bwsalt: area-averaged bottom water temperature or salinity from the given region or cavity (eg ross_cavity, ross_shelf, ross)
# <region>_temp, <region>_salt: volume-averaged temperature or salinity from the given region or cavity
# drake_passage_transport: zonal transport across Drake Passage (need to pass path to domain_cfg)
def calc_timeseries (var, ds_nemo, domain_cfg='/gws/nopw/j04/terrafirma/kaight/input_data/grids/domcfg_eORCA1v2.2x.nc'):
    
    # Parse variable name
    factor = 1
    region_type = None
    region = None
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
    elif var == 'drake_passage_transport':
        lon0 = drake_passage_lon0
        lat_bounds = drake_passage_lat_bounds
        lat0 = None
        lon_bounds = None
        option = 'transport'
        units = 'Sv'
        title = 'Drake Passage Transport'
        # Need to add e2u from domain_cfg
        ds_domcfg = xr.open_dataset(domain_cfg).squeeze()
        if ds_nemo.sizes['y'] < ds_domcfg.sizes['y']:
            # The NEMO dataset was trimmed (eg by MOOSE for UKESM) to the southernmost latitudes. Do the same for domain_cfg.
            ds_domcfg = ds_domcfg.isel(y=slice(0, ds_nemo.sizes['y']))
        ds_nemo = ds_nemo.assign({'e2u':ds_domcfg['e2u']})

    # Select region
    if region is not None:
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
            mask, ds_nemo, region_name = single_cavity_mask(region, ds_nemo, return_name=True)
        else:
            mask, ds_nemo, region_name = region_mask(region, ds_nemo, option=region_type, return_name=True)
        title += ' for '+region_name    
        
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
    elif option == 'transport':
        # Calculate zonal or meridional transport
        data = transport(ds_nemo, lon0=lon0, lat0=lat0, lon_bounds=lon_bounds, lat_bounds=lat_bounds)
        
    data *= factor
    data = data.assign_attrs(long_name=title, units=units)

    return data, ds_nemo


# Precompute the given list of timeseries from the given xarray Dataset of NEMO output. Save in a NetCDF file which concatenates after each call to the function.
def precompute_timeseries (ds_nemo, timeseries_types, timeseries_file, halo=True):

    if halo:
        # Remove the halo
        ds_nemo = ds_nemo.isel(x=slice(1,-1))

    # Calculate each timeseries and save to a Dataset
    ds_new = None
    for var in timeseries_types:
        data, ds_nemo = calc_timeseries(var, ds_nemo)
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


# Precompute timeseries from the given simulation, either from the beginning (timeseries_file does not exist) or picking up where it left off (timeseries_file does exist). Considers all NEMO output files stamped with suite_id in the given directory sim_dir on the given grid (gtype='T', 'U', etc), and assumes the timeseries file is in that directory too.
def update_simulation_timeseries (suite_id, timeseries_types, timeseries_file='timeseries.nc', sim_dir='./', freq='m', halo=True, gtype='T'):

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
        if not f.endswith('-'+gtype+'.nc'):
            # Not a NEMO output file on this grid; skip it
            continue
        if f.startswith('nemo_'+suite_id+'o'):
            # UKESM file naming conventions
            file_head = 'nemo_'+suite_id+'o'
        elif f.startswith(suite_id):
            # Standalone NEMO file naming conventions
            file_head = suite_id
        else:
            # Something else; skip it
            continue
        if '_1'+freq+'_' not in f:
            raise Exception('update_simulation_timeseries can only handle one frequency of NEMO output files. Need to code other options or move the other frequency files elsewhere.')
        file_head += '_1'+freq+'_'
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
                    
                
        

    
                                

    
        
        
