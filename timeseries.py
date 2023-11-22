import xarray as xr

from .constants import region_points, region_names, rho_fw, rho_ice, sec_per_year
from .utils import cavity_mask, region_mask

def calc_timeseries (var, ds_nemo, grid_file, halo=True):
    
    # Parse variable name
    # Start with default values
    factor = 1
    if var.endswith('massloss'):
        option = 'area_int'
        region = var[:var.index('_massloss')]
        region_type = 'cavity'
        if region in region_points:
            # Single ice shelf
            mask, region_name = cavity_mask(region, grid_file, return_name=True)
        else:
            # Region with multiple ice shelves
            mask, region_name = region_mask(region, grid_file, option='cavity', return_name=True)
        nemo_var = 'sowflisf'
        # Convert from kg/s to Gt/y and swap sign
        factor = -rho_ice/rho_fw*1e-12*sec_per_year
        units = 'Gt/y'
        title = 'Basal mass loss from '+region_name

    # Trim datasets as needed
    if ds_nemo.sizes['y'] < mask.sizes['y']:
        # The NEMO dataset was trimmed (eg by MOOSE for UKESM) to the southernmost latitudes. Do the same for the mask.
        mask = mask.isel(y=slice(0, ds_nemo.sizes['y']))
    if halo:
        # Remove the halo
        ds_nemo = ds_nemo.isel(x=slice(1,-1))
        mask = mask.isel(x=slice(1,-1))
        
    if option == 'area_int':
        # Area integral
        data = (ds_nemo[nemo_var]*ds_nemo['area']*mask).sum(dim=['x','y'])

    data *= factor

    return ds_nemo['time_centered'], data, title
        
        
