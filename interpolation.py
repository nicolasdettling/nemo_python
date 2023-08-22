import xarray as xr
import numpy as np
import os
from .utils import polar_stereo, fix_lon_range, extend_grid_edges, polar_stereo_inv

# Interpolate the source dataset to the NEMO coordinates using a binning approach. This assumes the source dataset is much finer resolution, and will simply average over all source points in each NEMO grid cell.
# Warning, this is VERY SLOW but hopefully it's a temporary measure while CF vectorises their weighting. It loops over every grid cell - this is necessary because the BedMachine source dataset is just too massive to vectorise without overflowing memory.
# Inputs:
# source: xarray Dataset containing the coordinates 'x' and 'y' (could be either lat/lon or polar stereographic), and any data variables you want
# nemo: xarray Dataset containing the NEMO grid: must contain at least glamt, gphit, glamf, gphif
# pster: whether the source dataset is polar stereographic
# periodic: whether the NEMO grid is periodic in longitude
# tmp_file: optional path to temporary output file which this routine will write to every latitude row. This is useful if the job dies in the middle. If it already exists, this function will pick up where it left off. If it doesn't exist yet, it will create the file.
# Returns:
# interp: xarray Dataset containing all data variables from source on the nemo grid, as well as extra variables x2d, y2d (source coordinates interpolated to new grid) and num_points (number of source points used for each grid cell; all data variables will be masked where num_points=0).
def interp_cell_binning (source, nemo, pster=True, periodic=True, tmp_file=None):

    from shapely.geometry import Point, Polygon
    import geopandas as gpd
    from tqdm import trange

    if pster:
        # Antarctic polar stereographic
        crs = 'epsg:3031'
        print('Converting to polar stereographic projection')
        x_t, y_t = polar_stereo(nemo['glamt'], nemo['gphit'])
        x_f, y_f = polar_stereo(nemo['glamf'], nemo['gphif'])
    else:
        # Ordinary lat-lon
        crs = 'epsg:4326'
        x_t = fix_lon_range(nemo['glamt'])
        y_t = nemo['gphit']
        x_f = fix_lon_range(nemo['glamf'])
        y_f = nemo['gphif']
    nx = nemo.sizes['x']
    ny = nemo.sizes['y']

    # Pad x_f, y_f with an extra row and column on the west and south so we have all the grid cell edges. Note this changes the indexing convention relative to the t-grid.
    x_f = extend_grid_edges(x_f, 'f', periodic=periodic)
    y_f = extend_grid_edges(y_f, 'f', periodic=periodic)

    if tmp_file is not None and os.path.isfile(tmp_file):
        # Read the partially completed dataset
        interp = xr.open_dataset(tmp_file)
        # Figure out the first latitude index with no interpolated data
        points_in_row = interp['num_points'].sum(dim='x').values
        if points_in_row[-1] == 0:
            j_start = np.argwhere(points_in_row==0)[0][0]
        else:
            raise Exception('The interpolation appears to be finished. If you want to start again, delete the temporary file.')
    else:
        # Start from scratch
        # Make a copy of the source dataset trimmed to the dimensions of nemo, to fill in later.
        interp = source.isel(x=slice(0,nx), y=slice(0,ny))
        # Make sure x and y are dummy axes, with 2D versions ready to fill in with real values
        if len(interp['x'].shape) == 1:
            y2d, x2d = xr.broadcast(interp['y'], interp['x'])
            interp['x2d'] = x2d
            interp['y2d'] = y2d
        else:
            interp['x2d'] = interp['x']
            interp['y2d'] = interp['y']
        interp = interp.assign_coords(x=np.arange(nx), y=np.arange(ny))
        # Also add extra array showing the number of data points in each grid cell
        interp = interp.assign({'num_points':interp['y2d']*0})
        j_start = 0

    print('Interpolating')
    # Loop over grid cells in NEMO...sorry.
    # Any vectorised way of doing this (using existing packages like cf or xesmf) overflows when you throw all of BedMachine3 at it. This could be solved using parallelisation, but a slower approach using less memory is preferable to asking users to sort out a SLURM job especially for something that won't be done very often (domain generation). Also NEMO is not CF compliant so even the existing tools are a headache when they require cell boundaries...
    for j in trange(j_start, ny):
        # First pass at narrowing down the source points to search: throw away anything that's too far south or north of the extent of this latitude row (based on grid corners). This will drastically speed up the search later.
        source_search1 = source.where((source['y'] >= np.amin(y_f[j:j+2,:]))*(source['y'] <= np.amax(y_f[j:j+2,:])), drop=True)        
        for i in range(nx):
            # Get the cell boundaries
            x_corners = np.array([x_f[j,i], x_f[j,i+1], x_f[j+1,i+1], x_f[j+1,i]])
            y_corners = np.array([y_f[j,i], y_f[j,i+1], y_f[j+1,i+1], y_f[j+1,i]])
            if not pster:
                # Potential for jump in longitude
                if x_t[j,i] < -170 and np.amax(x_corners) > 170:
                    x_corners[x_corners > 0] -= 360
                elif x_t[j,i] > 170 and np.amin(x_corners) < -170:
                    x_corners[x_corners < 0] += 360
            grid_cell = Polygon([(x_corners[n], y_corners[n]) for n in range(4)])
            # Narrow down the source points to search again. Since quadrilaterals are convex, we don't need to consider any points which are west of the westernmost grid cell corner, and so on.
            source_search2 = source_search1.where((source_search1['x'] >= np.amin(x_corners))*(source_search1['x'] <= np.amax(x_corners))*(source_search1['y'] >= np.amin(y_corners))*(source_search1['y'] <= np.amax(y_corners)), drop=True)
            # Use GeoPandas to find which points are in cell, then convert the list of points back to an xarray Dataset
            gdf_cell = gpd.GeoDataFrame(index=[0], geometry=[grid_cell], crs=crs) 
            df = source_search2.to_dataframe().reset_index()
            gdf_points = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df['x'], df['y']), crs=crs)
            points_within = gpd.tools.sjoin(gdf_points, gdf_cell, predicate='within')
            if points_within.size == 0:
                # There are no points within this grid cell
                continue            
            source_within = xr.Dataset.from_dataframe(points_within).drop(['index_right','geometry'])
            if 'time_counter' in source_within:
                source_within = source_within.drop('time_counter')
            source_mean = source_within.mean()
            source_mean = source_mean.rename({'x':'x2d', 'y':'y2d'})
            source_mean = source_mean.assign({'num_points':xr.DataArray(source_within.sizes['index'])})
            # Now fill in this point in the interpolated dataset
            interp = xr.where((interp.coords['x']==i)*(interp.coords['y']==j), source_mean, interp)
        if tmp_file is not None:
            # Save after each latitude row in case the job dies
            interp.to_netcdf(tmp_file)
    # Mask all variables where there are no points
    interp = interp.where(interp['num_points']>0)

    return interp

    
# Helper function to construct a minimal CF field so cf-python can do regridding.
# Mostly following Robin Smith's Unicicles coupling code in UKESM.
# If the coordinate axes (1D) x and y are not lat-lon, pass in auxiliary lat-lon values (2D).
def construct_cf (data, x, y, lon=None, lat=None, lon_bounds=None, lat_bounds=None):

    import cf
    native_latlon = lon is None and lat is None
    
    field = cf.Field()
    if native_latlon:
        dim_x = cf.DimensionCoordinate(data=cf.Data(x, 'degrees_east'), properties={'axis':'X', 'standard_name':'longitude'})
        dim_y = cf.DimensionCoordinate(data=cf.Data(y, 'degrees_north'), properties={'axis':'Y', 'standard_name':'latitude'})
        dim_lon = dim_x
        dim_lat = dim_y
    else:
        dim_x = cf.DimensionCoordinate(data=cf.Data(x, 'm'))
        dim_y = cf.DimensionCoordinate(data=cf.Data(y, 'm'))
    field.set_construct(cf.DomainAxis(size=x.size), key='X')
    field.set_construct(dim_x, axes='X')
    field.set_construct(cf.DomainAxis(size=y.size), key='Y')
    field.set_construct(dim_y, axes='Y')
    if not native_latlon:
        dim_lon = cf.AuxiliaryCoordinate(data=cf.Data(lon, 'degrees_east'), properties={'standard_name':'longitude'})
        field.set_construct(dim_lon, axes=('Y','X'))
        dim_lat = cf.AuxiliaryCoordinate(data=cf.Data(lat, 'degrees_north'), properties={'standard_name':'latitude'})
        field.set_construct(dim_lat, axes=('Y','X'))
    if lon_bounds is not None:
        dim_lon.set_bounds(lon_bounds)
    if lat_bounds is not None:
        dim_lat.set_bounds(lat_bounds)
    field.set_data(cf.Data(data), axes=('Y', 'X'))
    return field


# Interpolate the source dataset to the NEMO coordinates using CF. This is good for smaller interpolation jobs (i.e. not BedMachine3) and hopefully will be good for big interpolation jobs once CF is next updated.
# Inputs:
# source: xarray Dataset containing the coordinates 'x' and 'y' (could be either lat/lon or polar stereographic), and any data variables you want
# nemo: xarray Dataset containing the NEMO grid: must contain at least (option 1:) glamt, gphit, glamf, gphif, and dimensions x and y, or (option 2): nav_lon_grid_T, nav_lat_grid_T, bounds_nav_lon_grid_T, bounds_nav_lat_grid_T, and dimensions x_grid_T and y_grid_T
# pster_src: whether the source dataset is polar stereographic
# periodic_src: whether the source dataset is periodic in the x dimension
# periodic_nemo: whether the NEMO grid is periodic in longitude
# method: CF interpolation method (bilinear or conservative both tested)
def interp_latlon_cf (source, nemo, pster_src=False, periodic_src=False, periodic_nemo=True, method='conservative'):

    # Helper function to get an xarray DataArray of edges (size N+1 by M+1) into a Numpy array of bounds for CF (size 4 x N x M)
    def edges_to_bounds (edges):
        bounds = np.empty([4, edges.shape[1]-1, edges.shape[0]-1])
        bounds[0,:] = edges.values[:-1,:-1]  # SW corner
        bounds[1,:] = edges.values[:-1,1:] # SE
        bounds[2,:] = edges.values[1:,1:] # NE
        bounds[3,:] = edges.values[1:,:-1] # NW
        return bounds

    # Get source grid and data in CF format
    if pster_src:
        print('Converting to lat-lon projection')
        x_src = source['x']
        y_src = source['y']
        lon_src, lat_src = polar_stereo_inv(source['x'], source['y'])
    else:
        x_src = source['lon']
        y_src = source['lat']
        lon_src = None
        lat_src = None
    if method == 'conservative':
        if pster_src and not periodic_src and len(source['x'].shape)==1:
            # Regular grid in x-y, not periodic
            # Get grid cell edges for x and y
            def construct_edges (array, dim):
                centres = 0.5*(array[:-1] + array[1:])
                first_edge = 2*array[0] - array[1]
                last_edge = 2*array[-1] - array[-2]
                edges = np.concatenate(([first_edge], centres, [last_edge]))
                return xr.DataArray(edges, coords={dim:edges})
            x_edges = construct_edges(source['x'].values, 'x')
            y_edges = construct_edges(source['y'].values, 'y')
            # Now convert to lat-lon
            print('Converting edges to lat-lon projection')
            lon_edges, lat_edges = polar_stereo_inv(x_bounds, y_bounds)
            print('Creating bounds')
            lon_bounds_src = edges_to_bounds(lon_edges)
            lat_bounds_src = edges_to_bounds(lat_edges)
        else:
            raise Exception('Need to code definition of bounds for this type of input dataset')
    else:
        lon_bounds_src = None
        lat_bounds_src = None
    # Loop over data fields and convert each to CF
    data_cf = []
    for var in source:
        data_cf.append(construct_cf(source[var], x_src, y_src, lon=lon_src, lat=lat_src, lon_bounds=lon_bounds_src, lat_bounds=lat_bounds_src))

    # Get NEMO grid in CF format
    # Figure out some dimension and coordinate names
    if 'glamt' in nemo:
        # domain_cfg type
        x_name = 'x'
        y_name = 'y'
        lon_name = 'glamt'
        lat_name = 'gphit'
    elif 'nav_lon_grid_T' in nemo:
        # model output type
        x_name = 'x_grid_T'
        y_name = 'y_grid_T'
        lon_name = 'nav_lon_grid_T'
        lat_name = 'nav_lat_grid_T'
    else:
        raise Exception('Unknown type of NEMO dataset.')
        
    dummy_data = np.zeros([nemo.sizes[y_name], nemo.sizes[x_name]])
    if method == 'conservative':
        def construct_nemo_bounds (array):
            edges = extend_grid_edges(array, 'f', periodic=periodic_nemo)
            return edges_to_bounds(edges)
        if lon_name == 'glamt':
            lon_bounds_nemo = construct_nemo_bounds(nemo['glamf'])
            lat_bounds_nemo = construct_nemo_bounds(nemo['gphif'])
        else:
            lon_bounds_nemo = nemo['bounds_nav_lon_grid_T']
            lat_bounds_nemo = nemo['bounds_nav_lat_grid_T']
    else:
        lon_bounds_nemo = None
        lat_bounds_nemo = None
    target_cf = construct_cf(dummy_data, nemo[x_name], nemo[y_name], lon=nemo[lon_name], lat=nemo[lat_name], lon_bounds=lon_bounds_nemo, lat_bounds=lat_bounds_nemo)
    
    # Get weights with CF, using the first data field
    regrid_operator = data_cf[0].regrids(target_cf, src_cyclic=periodic_src, dst_cyclic=periodic_nemo, dst_axes={'X':'X', 'Y':'Y'}, method=method, return_operator=True)

    # Now interpolate each field, re-using the weights each time, and add it to a new Dataset
    interp = xr.Dataset()
    for var, data_cf0 in zip(source, data_cf):
        data_interp = data_cf0.regrids(regrid_operator).array
        data_interp = xr.DataArray(data_interp, dims=['y', 'x'])
        interp = interp.assign({var:data_interp})     

    return interp

    
    
            
        
