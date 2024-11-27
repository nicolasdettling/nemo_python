import xarray as xr
import numpy as np
import os
import glob
from .constants import region_points, region_names, rho_fw, rho_ice, sec_per_year, deg_string, gkg_string, drake_passage_lon0, drake_passage_lat_bounds
from .utils import add_months, closest_point, month_convert
from .grid import single_cavity_mask, region_mask, calc_geometry
from .diagnostics import transport

# Calculate a timeseries of the given preset variable from an xarray Dataset of NEMO output (must have halo removed). Returns DataArrays of the timeseries data, the associated time values, and the variable title. Specify whether there is a halo (true for periodic boundaries in NEMO 3.6).
# Preset variables include:
# <region>_massloss: basal mass loss from the given ice shelf or region of multiple ice shelves (eg brunt, amundsen_sea)
# <region>_draft: area-averaged ice draft from the given ice shelf or region of multiple ice shelves - only useful if there's a coupled ice sheet
# <region>_bwtemp, <region>_bwsalt: area-averaged bottom water temperature or salinity from the given region or cavity (eg ross_cavity, ross_shelf, ross)
# <region>_temp, <region>_salt: volume-averaged temperature or salinity from the given region or cavity
# <region>_temp_btw_xxx_yyy_m, <region>_salt_btw_xxx_yyy_m: volume-averaged temperature or salinity from the given region or cavity, between xxx and yyy metres (positive integers, shallowest first)
# drake_passage_transport: zonal transport across Drake Passage (need to pass path to domain_cfg)
# Inputs:
# name_remapping: optional dictionary of dimensions and variable names that need to be remapped to match the code below (depends on the runset)
# nemo_mesh: optional string of the location of a bathymetry meshmask file for calculating the region masks (otherwise calculates it from ds_nemo)
def calc_timeseries (var, ds_nemo, name_remapping='', nemo_mesh='', 
                     domain_cfg='/gws/nopw/j04/terrafirma/kaight/input_data/grids/domcfg_eORCA1v2.2x.nc', halo=True):

    # Remap NetCDF variable names to match the generalized case:
    if name_remapping:
        try:
            ds_nemo = ds_nemo.rename(name_remapping)
        except: # if it doesn't seem to need to be renamed, continue looping through
            pass
    
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
    elif var.endswith('_draft'):
        option = 'area_avg'
        region = var[:var.index('_draft')]
        region_type = 'cavity'
        nemo_var = 'draft'  # Will trigger a special case later
        units = 'm'
        title = 'Mean ice draft'
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
    elif var.endswith('_ssh'):
        option = 'area_avg'
        region = var[:var.index('_ssh')]
        nemo_var = 'zos'
        units = 'm'
        title = 'Sea surface height'
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
    elif '_temp_btw_' in var:
        option = 'avg_btw_depths'
        region = var[:var.index('_temp_btw_')]
        z_vals = var[len(region+'_temp_btw_'):-1]
        z_shallow = int(z_vals[:z_vals.index('_')])
        z_deep = int(z_vals[z_vals.index('_')+1:])
        nemo_var = 'thetao'
        title = 'Average temperature between '+str(z_shallow)+'-'+str(z_deep)+'m'
        units = deg_string+'C'
    elif '_salt_btw_' in var:
        option = 'avg_btw_depths'
        region = var[:var.index('_salt_btw_')]
        z_vals = var[len(region+'_salt_btw_'):-1]
        z_shallow = int(z_vals[:z_vals.index('_')])
        z_deep = int(z_vals[z_vals.index('_')+1:])
        nemo_var = 'so'
        title = 'Average salinity between '+str(z_shallow)+'-'+str(z_deep)+'m'
        units = gkg_string
    elif var == 'drake_passage_transport':
        lon0 = drake_passage_lon0
        lat_bounds = drake_passage_lat_bounds
        lat0 = None
        lon_bounds = None
        option = 'transport'
        units = 'Sv'
        title = 'Drake Passage Transport'

    if var == 'drake_passage_transport' and 'e2u' not in ds_nemo:
        # Need to add e2u from domain_cfg
        ds_domcfg = xr.open_dataset(domain_cfg).squeeze()
        if ds_nemo.sizes['y'] < ds_domcfg.sizes['y']:
            # The NEMO dataset was trimmed (eg by MOOSE for UKESM) to the southernmost latitudes. Do the same for domain_cfg.
            ds_domcfg = ds_domcfg.isel(y=slice(0, ds_nemo.sizes['y']))
        if halo:
            ds_domcfg = ds_domcfg.isel(x=slice(1,-1))
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
            if nemo_mesh:
                nemo_file = xr.open_dataset(nemo_mesh)
                mask, _, region_name = single_cavity_mask(region, nemo_file, return_name=True)
            else:
                mask, ds_nemo, region_name = single_cavity_mask(region, ds_nemo, return_name=True)
        else:
            if nemo_mesh:
                nemo_file = xr.open_dataset(nemo_mesh)
                mask, _, region_name = region_mask(region, nemo_file, option=region_type, return_name=True)
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
        if nemo_var == 'draft':
            data_xy = calc_geometry(ds_nemo, keep_time_dim=True)[1]
        else:
            data_xy = ds_nemo[nemo_var]
        data = (data_xy*dA).sum(dim=['x','y'])/dA.sum(dim=['x','y'])
    elif option == 'volume_avg':
        # Volume average
        # First need a 3D mask
        mask_3d = xr.where(ds_nemo[nemo_var]==0, 0, mask)
        dV = ds_nemo['area']*ds_nemo['thkcello']*mask_3d
        data = (ds_nemo[nemo_var]*dV).sum(dim=['x','y','deptht'])/dV.sum(dim=['x','y','deptht'])
    elif option == 'avg_btw_depths':
        # Volume average between two depths
        # Create an extra mask to multiply dV with, which is 1 between the two depths and 0 otherwise
        depth_below = ds_nemo['thkcello'].cumsum(dim='deptht')
        depth_above = depth_below.shift(deptht=1, fill_value=0)
        depth_centres = 0.5*(depth_above + depth_below)
        mask_depth = xr.where((depth_centres >= z_shallow)*(depth_centres <= z_deep), 1, 0)
        mask_3d = xr.where(ds_nemo[nemo_var]==0, 0, mask)
        dV = ds_nemo['area']*ds_nemo['thkcello']*mask_3d*mask_depth
        data = (ds_nemo[nemo_var]*dV).sum(dim=['x','y','deptht'])/dV.sum(dim=['x','y','deptht'])        
    elif option == 'transport':
        # Calculate zonal or meridional transport
        data = transport(ds_nemo, lon0=lon0, lat0=lat0, lon_bounds=lon_bounds, lat_bounds=lat_bounds)
        
    data *= factor
    data = data.assign_attrs(long_name=title, units=units)

    return data, ds_nemo


# As above, but for PP output files from the UM atmosphere. 
def calc_timeseries_um (var, file_path):

    import iris
    import warnings

    # Parse variable name
    if var == 'global_mean_sat':
        option = 'area_avg'
        um_var = 'air_temperature'
        units = 'K'
        title = 'Global mean near-surface air temperature'

    # Read the correct variable from the file
    # Suppress warnings (year_zero kwarg ignored for idealised calendars)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        cube = iris.load_cube(file_path, um_var)

    if option == 'area_avg':
        # Following code by Jane Mulcahy and Catherine Hardacre
        if cube.coord('latitude').bounds is None:
            cube.coord('latitude').guess_bounds()
        if cube.coord('longitude').bounds is None:
            cube.coord('longitude').guess_bounds()
        grid_areas = iris.analysis.cartography.area_weights(cube)
        data_iris = cube.collapsed(['longitude', 'latitude'], iris.analysis.MEAN, weights=grid_areas)

    # Now convert to a DataArray and get the time dimension to match NEMO conventions
    data = xr.DataArray.from_iris(data_iris)
    data = data.expand_dims(dim='time')
    data = data.rename({'time':'time_centered'})
    data = data.assign_attrs(long_name=title, units=units)
    data.load()

    return data


# Precompute the given list of timeseries from the given xarray Dataset of NEMO output (or PP file if pp=True). Save in a NetCDF file which concatenates after each call to the function.
def precompute_timeseries (ds_nemo, timeseries_types, timeseries_file, halo=True, 
                           domain_cfg='/gws/nopw/j04/terrafirma/kaight/input_data/grids/domcfg_eORCA1v2.2x.nc',
                           name_remapping='', nemo_mesh='', pp=False):

    if halo and not pp:
        # Remove the halo
        ds_nemo = ds_nemo.isel(x=slice(1,-1))

    # Calculate each timeseries and save to a Dataset
    ds_new = None
    for var in timeseries_types:
        if pp:
            data = calc_timeseries_um(var, ds_nemo)
        else:
            data, ds_nemo = calc_timeseries(var, ds_nemo, domain_cfg=domain_cfg, halo=halo, 
                                            name_remapping=name_remapping, nemo_mesh=nemo_mesh)
        if ds_new is None:            
            ds_new = xr.Dataset({var:data})
        else:
            ds_new = ds_new.assign({var:data})
    if not pp:
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
    ds_new.close()


# Precompute timeseries from the given simulation, either from the beginning (timeseries_file does not exist) or picking up where it left off (timeseries_file does exist). Considers all NEMO output files stamped with suite_id in the given directory sim_dir on the given grid (gtype='T', 'U', etc), and assumes the timeseries file is in that directory too.
def update_simulation_timeseries (suite_id, timeseries_types, timeseries_file='timeseries.nc', config='', 
                                  sim_dir='./', freq='m', halo=True, gtype='T', name_remapping='', nemo_mesh='',
                                  domain_cfg='/gws/nopw/j04/terrafirma/kaight/input_data/grids/domcfg_eORCA1v2.2x.nc'):

    update = os.path.isfile(sim_dir+timeseries_file)
    if update:
        # Timeseries file already exists
        # Get last time index
        ds_ts = xr.open_dataset(sim_dir+timeseries_file)
        time_last  = ds_ts['time_centered'][-1].dt
        year_last  = time_last.year
        month_last = time_last.month
        ds_ts.close()

    # Identify NEMO output files in the given directory, constructed as wildcard strings for each date code
    nemo_files = []
    if config=='eANT025':
        file_tail = f'_{gtype}.nc'
    else:
        file_tail = f'-{gtype}.nc'
        
    for f in os.listdir(sim_dir):
        if os.path.isdir(f'{sim_dir}/{f}'): continue # skip directories
        if not f.endswith(file_tail): continue    # Not a NEMO output file on this grid; skip it

        if config=='eANT025':
            if f.startswith(f'{config}.{suite_id}_1{freq}_'):
                # file naming conventions
                file_head = f'{config}.{suite_id}_1{freq}_' 
            else:
                # Something else; skip it
                continue
        else:
            if f.startswith('nemo_'+suite_id+'o_1'+freq+'_'):
                # UKESM file naming conventions
                file_head = 'nemo_'+suite_id+'o_1'+freq+'_'
            elif f.startswith(suite_id+'_1'+freq+'_'):
                # Standalone NEMO file naming conventions
                file_head = suite_id+'_1'+freq+'_'
            else:
                # Something else; skip it
                continue
        # Extract date code (yyyymmdd_yyyymmdd or yyyymmdd-yyyymmdd)
        if config == 'eANT025':
            date_code = f"{(f.split(f'{file_head}')[1]).split('_')[0]}_{(f.split(f'{file_head}')[1]).split('_')[1]}"
        else:
            date_code = f"{(f.split(f'{file_head}')[1]).split('_')[0]}"
        if update:
            # Need to check if date code has already been processed
            year = int(date_code[:4])
            month = int(date_code[4:6])
            if year < year_last or (year==year_last and month<=month_last):
                # Skip it
                continue
        # Now construct wildcard string and add to list if it's not already there
        file_pattern = f'{file_head}{date_code}*{file_tail}'
        if file_pattern not in nemo_files:
            nemo_files.append(file_pattern)        
    # Now sort alphabetically - i.e. by ascending date code
    nemo_files.sort()

    # Loop through each date code and process
    for file_pattern in nemo_files:
        print('Processing '+file_pattern)
        if os.path.isfile(f"{sim_dir}/{file_pattern.replace('*','_isf')}") and not os.path.isfile(f"{sim_dir}/{file_pattern.replace('*','_grid')}"):
            print('Warning: isf-T file exists with no matching grid-T file. Probably reached the end of complete months pulled from MASS. Stopping')
            break
        if os.path.isfile(f"{sim_dir}/{file_pattern.replace('*','_grid')}") and not os.path.isfile(f"{sim_dir}/{file_pattern.replace('*','_isf')}"):
            print('Warning: grid-T file exists with no matching isf-T file. It may have been skipped in call to MASS. Stopping')
            break
        ds_nemo = xr.open_mfdataset(f'{sim_dir}/{file_pattern}')
        ds_nemo.load()
        precompute_timeseries(ds_nemo, timeseries_types, f'{sim_dir}/{timeseries_file}', halo=halo, domain_cfg=domain_cfg,
                              name_remapping=name_remapping, nemo_mesh=nemo_mesh)
        ds_nemo.close()


# As above, but for PP output files from the UM atmosphere. 
def update_simulation_timeseries_um (suite_id, timeseries_types, timeseries_file='timeseries_um.nc', sim_dir='./', stream='p5'):

    update = os.path.isfile(sim_dir+timeseries_file)
    if update:
        # Timeseries file already exists
        # Get last time index
        ds_ts = xr.open_dataset(sim_dir+timeseries_file)
        time_last = ds_ts['time_centered'].data[-1]
        year_last = time_last.year
        month_last = time_last.month
        ds_ts.close()

    # Identify all the PP files for the given stream
    date_codes = []
    file_head = suite_id+'a.'+stream
    file_tail = '.pp'
    for f in os.listdir(sim_dir):
        if os.path.isdir(sim_dir+'/'+f):
            # Skip directories
            continue
        if not (f.startswith(file_head)) or not (f.endswith(file_tail)):
            # Not a UM output file for this stream
            continue
        # Extract date code (yyyymmm)
        date_code = f[len(file_head):len(file_head)+7]
        # Replace mmm abbreviation (eg jan) with numbers (eg 01) so we can sort and compare
        date_code = date_code[:4] + month_convert(date_code[4:])
        if update:
            # Need to check if date code has already been processed
            year = int(date_code[:4])
            month = int(date_code[4:6])
            if year < year_last or (year==year_last and month<=month_last):
                # Skip it
                continue
        date_codes.append(date_code)
    # Now sort alphabetically - i.e. by ascending date code
    date_codes.sort()
    
    # Loop through each date code, reconstruct the filename, and process
    for date_code in date_codes:
        fname = sim_dir + '/' + file_head + date_code[:4] + month_convert(date_code[4:]) + file_tail
        print('Processing '+fname)
        precompute_timeseries(fname, timeseries_types, sim_dir+'/'+timeseries_file, pp=True)
        
def calc_hovmoeller_region(var, region,
                           run_folder='/gws/nopw/j04/anthrofail/birgal/NEMO_AIS/output/reference-4.2.2/',
                           nemo_mesh='/gws/nopw/j04/anthrofail/birgal/NEMO_AIS/bathymetry/mesh_mask-20240305.nc'):
    
    # Load gridT files into dataset:
    gridT_files = glob.glob(f'{run_folder}files/*grid_T*')
    nemo_ds     = xr.open_mfdataset(gridT_files).isel(x_grid_T=region['x'], y_grid_T=region['y']) # load all the gridT files in the run folder

    nemo_mesh_ds     = xr.open_dataset(f'{nemo_mesh}')
    nemo_mesh_subset = nemo_mesh_ds.rename({'x':'x_grid_T','y':'y_grid_T','nav_lev':'deptht'}).isel(x_grid_T=region['x'], y_grid_T=region['y'], time_counter=0)
    
    var_ocean  = xr.where(nemo_mesh_subset.tmask==0, np.nan, nemo_ds[var]) 
    area_ocean = xr.where(nemo_mesh_subset.tmask==0, np.nan, nemo_ds['area_grid_T']) 
    region_var = (var_ocean*area_ocean).sum(dim=['x_grid_T','y_grid_T'])/(area_ocean.sum(dim=['x_grid_T','y_grid_T']))

    return region_var
    
                                

    
        
        
