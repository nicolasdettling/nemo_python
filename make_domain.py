import xarray as xr
import numpy as np
from .utils import polar_stereo

# Given a pre-existing global domain (eg eORCA025), slice out a regional domain.
# Inputs:
# global_file = path to a NetCDF file for the global domain. The domain_cfg file is ideal, but any file that includes all of these variables will work: nav_lon, nav_lat, e2f, e2v, e2u, e2t, e1f, e1v, e1u, e1t, gphif, gphiv, gphiu, gphit, glamf, glamv, glamu, glamt
# out_file: path to desired output NetCDF file
# imin, imax, jmin, jmax: optional bounds on the i and j indicies to slice. Uses the python convention of 0-indexing, and selecting the indices up to but not including the last value. So, jmin=0, jmax=10 will select the southernmost 10 rows.
# nbdry: optional latitude of the desired northern boundary. The code will generate the value of jmax that corresponds to this on the zonal mean.
def coordinates_from_global (global_file='/gws/nopw/j04/terrafirma/kaight/input_data/grids/domcfg_eORCA025_v3.nc', out_file='coordinates.nc', imin=None, imax=None, jmin=None, jmax=None, nbdry=None):

    ds = xr.open_dataset(global_file)

    if nbdry is not None:
        # Choose the value of jmax based on the supplied latitude
        if jmax is not None:
            raise Exception('nbdry and jmax are both defined')
        # Average over longitude to get a 1D field of "typical" latitude
        lat_1d = ds['nav_lat'].mean(dim='x')
        # Find the first index north of the given boundary
        jmax = np.where(lat_1d > nbdry)[0][0]

    # Now set default values for i and j slicing
    if imin is None:
        imin = 0
    if imax is None:
        imax = ds.sizes['x']
    if jmin is None:
        jmin = 0
    if jmax is None:
        jmax = ds.sizes['y']

    # Now slice to these bounds
    ds_regional = ds.isel(x=slice(imin,imax), y=slice(jmin, jmax))

    # Select only the variables we need and write to file
    var_names = ['nav_lon', 'nav_lat', 'e2f', 'e2v', 'e2u', 'e2t', 'e1f', 'e1v', 'e1u', 'e1t', 'gphif', 'gphiv', 'gphiu', 'gphit', 'glamf', 'glamv', 'glamu', 'glamt']
    ds_regional[var_names].to_netcdf(out_file)


def interp_topo (source='BedMachine3', topo_file='/gws/nopw/j04/terrafirma/kaight/input_data/topo/BedMachineAntarctica-v3.nc', coordinates_file='coordinates.nc', out_file='topo.nc', periodic=True):

    print('Processing input data')
    if source == 'BedMachine3':        
        ds_b = xr.open_dataset(topo_file)
        # x and y coordinates are ints which can overflow later; cast to floats
        x = ds_b['x'].astype('float32')
        y = ds_b['y'].astype('float32')
        # Bathymetry is the variable "bed"
        bathy = ds_b['bed']
        print('...calculating ice draft')
        # Ice draft is the surface minus thickness
        draft = ds_b['surface'] - ds_b['thickness']
        print('...combining masks')
        # Ocean mask includes open ocean (0) and floating ice (3)
        omask = xr.where((ds_b['mask']==0)+(ds_b['mask']==3), 1, 0)
        # Ice sheet mask includes everything except open ocean (1=rock, 2=grounded ice, 3=floating ice, 4=subglacial lake)
        imask = xr.where(ds_b['mask']!=0, 1, 0)
        # Now make a new Dataset containing only the variables we need
        ds_source = xr.Dataset({'x':x, 'y':y, 'bathy':bathy, 'draft':draft, 'omask':omask, 'imask':imask})
    else:
        raise Exception('source dataset not yet supported')

    print('Reading NEMO coordinates')
    ds_n = xr.open_dataset(coordinates_file).squeeze()    
    # Convert t-grids and f-grids to polar stereographic
    print('...converting to polar stereographic projection')
    x_t, y_t = polar_stereo(ds_n['glamt'], ds_n['gphit'])
    x_f, y_f = polar_stereo(ds_n['glamf'], ds_n['gphif'])
    ds_target = xr.Dataset({'x_t':x_t, 'y_t':y_t, 'x_f':x_f, 'y_f':y_f})

    ds_interp = interp_cell_binning(ds_source, ds_target, plot=True, pster=True, periodic=periodic)
