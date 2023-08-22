import xarray as xr
import numpy as np
from .interpolation import interp_cell_binning, interp_latlon_cf
from .utils import polar_stereo
from .plots import circumpolar_plot

# Given a pre-existing global domain (eg eORCA025), slice out a regional domain.
# Inputs:
# global_file = path to a NetCDF file for the global domain. The domain_cfg file is ideal, but any file that includes all of these variables will work: nav_lon, nav_lat, e2f, e2v, e2u, e2t, e1f, e1v, e1u, e1t, gphif, gphiv, gphiu, gphit, glamf, glamv, glamu, glamt
# out_file: path to desired output NetCDF file
# imin, imax, jmin, jmax: optional bounds on the i and j indicies to slice. Uses the python convention of 0-indexing, and selecting the indices up to but not including the last value. So, jmin=0, jmax=10 will select the southernmost 10 rows.
# nbdry: optional latitude of the desired northern boundary. The code will generate the value of jmax that corresponds to this on the zonal mean.
def coordinates_from_global (global_file='/gws/nopw/j04/terrafirma/kaight/input_data/grids/domcfg_eORCA025_v3.nc', out_file='coordinates.nc', imin=None, imax=None, jmin=None, jmax=None, nbdry=None, remove_halo=True):

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
        if remove_halo:
            # Remove the periodic halo (for NEMO 4.2): slice off first index
            imin = 1
        else:
            imin = 0
    if imax is None:
        if remove_halo:
            # also last index
            imax = ds.sizes['x'] - 1
        else:
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


# Interpolate topography from a dataset (default BedMachine3) to the given NEMO coordinates.
# Warning, this is currently VERY SLOW due to memory constraints meaning a double loop is needed but I am hoping CF will come to my rescue soon!
# Inputs:
# dataset: name of source dataset, only 'BedMachine3' supported now
# topo_file: path to file containing dataset
# coordinates_file: path to file containing NEMO coordinates (could be created by coordinates_from_global above)
# out_file: desired path to output file for interpolated dataset
# tmp_file: optional path to temporary output file which the interpolation routine will write to every latitude row. This is useful if the job dies in the middle.
# periodic: whether the NEMO grid is periodic in longitude
def interp_topo (dataset='BedMachine3', topo_file='/gws/nopw/j04/terrafirma/kaight/input_data/topo/BedMachineAntarctica-v3.nc', coordinates_file='coordinates.nc', out_file='topo.nc', tmp_file=None, periodic=True):

    print('Processing input data')
    if dataset == 'BedMachine3':        
        source = xr.open_dataset(topo_file)
        # x and y coordinates are ints which can overflow later; cast to floats
        x = source['x'].astype('float32')
        y = source['y'].astype('float32')
        # Bathymetry is the variable "bed"
        bathy = source['bed']
        print('...calculating ice draft')
        # Ice draft is the surface minus thickness
        draft = source['surface'] - source['thickness']
        print('...combining masks')
        # Ocean mask includes open ocean (0) and floating ice (3)
        omask = xr.where((source['mask']==0)+(source['mask']==3), 1, 0)
        # Ice sheet mask includes everything except open ocean (1=rock, 2=grounded ice, 3=floating ice, 4=subglacial lake)
        imask = xr.where(source['mask']!=0, 1, 0)
        pster_src = True
        periodic_src = False
        # Now make a new Dataset containing only the variables we need
        source = xr.Dataset({'x':x, 'y':y, 'bathy':bathy, 'draft':draft, 'omask':omask, 'imask':imask})
    else:
        raise Exception('source dataset not yet supported')

    print('Reading NEMO coordinates')
    nemo = xr.open_dataset(coordinates_file).squeeze()
    
    print('Interpolating')
    data_interp = interp_cell_binning(source, nemo, pster=pster_src, periodic=periodic, tmp_file=tmp_file)
    data_interp.to_netcdf(out_file)
    #data_interp = interp_latlon_cf(source, nemo, pster_src=pster_src, periodic_src=periodic_src, periodic_nemo=periodic, method='conservative')


# Following interpolation of topography, get everything in the right format for the NEMO tool DOMAINcfg, make diagnostic plots (optional), and save to a file.
# Inputs:
# in_file: path to file created by interp_topo above
# coordinates_file: path to file containing NEMO coordinates (could be created by coordinates_from_global above)
# out_file: desired path to output file
# will_splice: will this new topography be spliced into an existing domain (like eORCA1 for UKESM)? If so, turn errors about missing points into warnings.
# plot: whether to make diagnostic plots
# pster_src: whether the source dataset (eg BedMachine3) was polar stereographic (and hence the x2d, y2d variables in in_file); only matters if plot=True
def process_topo (in_file='topo.nc', coordinates_file='coordinates.nc', out_file='topo_processed.nc', will_splice=False, plot=True, pster_src=True):

    if plot:
        import matplotlib.pyplot as plt

    topo = xr.open_dataset(in_file).transpose('y', 'x')
    if np.count_nonzero(topo['bathy'].isnull()):
        # There are missing points
        if will_splice:
            print('Warning: there are missing points. Hopefully this will be dealt with by your splicing later - check to make sure.')
        else:
            raise Exception('Missing points')
    nemo = xr.open_dataset(coordinates_file).squeeze()

    # Set masks where they fall between 0 and 1
    topo['omask'] = np.round(topo['omask'])
    topo['imask'] = np.round(topo['imask'])
    # Make bathymetry and ice draft positive, and mask with zeros
    topo['bathy'] = xr.where(topo['omask']==1, -topo['bathy'], 0)
    topo['draft'] = xr.where(topo['imask']*topo['omask']==1, -topo['draft'], 0)
    # Now split bathymetry into cavity and non-cavity
    topo['Bathymetry'] = xr.where(topo['imask']==0, topo['bathy'], 0)
    topo['Bathymetry_isf'] = xr.where(topo['imask']==1, topo['bathy'], 0)
    # Make a new dataset with all the variables we need
    output = xr.Dataset({'nav_lon':nemo['nav_lon'], 'nav_lat':nemo['nav_lat'], 'tmaskutil':topo['omask'], 'Bathymetry':topo['Bathymetry'], 'Bathymetry_isf':topo['Bathymetry_isf'], 'isf_draft':topo['draft']})
    output.to_netcdf(out_file)

    if plot:
        if pster_src:
            # Convert nemo lat and lon into polar stereographic for direct comparison with x2d and y2d
            x, y = polar_stereo(nemo['nav_lon'], nemo['nav_lat'])
        else:
            x = nemo['nav_lon']
            y = nemo['nav_lat']
        # Make a bunch of plots
        circumpolar_plot(x-topo['x2d'], nemo, title='Error in x-coordinate (m)', ctype='plusminus', masked=True)
        circumpolar_plot(y-topo['y2d'], nemo, title='Error in y-coordinate (m)', ctype='plusminus', masked=True)
        circumpolar_plot(topo['num_points'], nemo, title='num_points', masked=True)
        for var in ['Bathymetry', 'Bathymetry_isf', 'isf_draft', 'tmaskutil']:
            circumpolar_plot(output[var], nemo, title=var, masked=True)


def interp_ics_TS (dataset='WOA18', source_files='/gws/nopw/j04/terrafirma/kaight/input_data/WOA18/woa18_decav_*01_04.nc', nemo_dom='/gws/nopw/j04/terrafirma/kaight/input_data/grids/domain_cfg_eANT025.L121.nc'):

    import gsw

    if dataset == 'WOA':
        # Read temperature and salinity from January as a single dataset
        woa = xr.open_mfdataset(in_files, decode_times=False).squeeze()
        # Convert to TEOS10
        # Need pressure in dbar at every 3D point: approx depth in m
        woa_press = np.abs(xr.broadcast(woa['depth'], woa['t_an'])[0])
        # Also need 3D lat and lon
        woa_lon = xr.broadcast(woa['lon'], woa['t_an'])[0]
        woa_lat = xr.broadcast(woa['lat'], woa['t_an'])[0]
        # Get absolute salinity
        woa_salt = gsw.SA_from_SP(woa['s_an'], woa_press, woa_lon, woa_lat)
        # Get conservative temperature
        woa_temp = gsw.CT_from_t(woa_salt, woa['t_an'], woa_press)
        # Now wrap up into a new Dataset
        source = xr.Dataset({'temp':woa_temp, 'salt':woa_salt})
        
        
    
        

