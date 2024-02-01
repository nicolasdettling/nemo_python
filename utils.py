import numpy as np
import xarray as xr
import gsw
from .constants import deg2rad, shelf_lat, shelf_depth, shelf_point0, rho_fw, sec_per_hour, temp_C2K, Rdry, Rvap, vap_pres_c1, vap_pres_c3, vap_pres_c4, region_edges, region_edges_flag, region_names, region_points, months_per_year, rEarth

# Given an array containing longitude, make sure it's in the range (max_lon-360, max_lon). Default is (-180, 180). If max_lon is None, nothing will be done to the array.
def fix_lon_range (lon, max_lon=180):

    if isinstance(lon, xr.DataArray):
        lon = xr.where(lon >= max_lon, lon-360, lon)
        lon = xr.where(lon < max_lon-360, lon+360, lon)
    elif isinstance(lon, np.ndarray):
        index = lon >= max_lon
        lon[index] = lon[index] - 360
        index = lon < max_lon-360
        lon[index] = lon[index] + 360
    elif np.isscalar(lon):
        lon = fix_lon_range(np.array([lon]), max_lon=max_lon)[0]
    else:
        raise Exception('unsupported data type')
    return lon


# Convert longitude and latitude to Antarctic polar stereographic projection. Adapted from polarstereo_fwd.m in the MITgcm Matlab toolbox for Bedmap.
def polar_stereo (lon, lat, a=6378137., e=0.08181919, lat_c=-71, lon0=0):

    if lat_c < 0:
        # Southern hemisphere
        pm = -1
    else:
        # Northern hemisphere
        pm = 1

    # Prepare input
    lon_rad = lon*pm*deg2rad
    lat_rad = lat*pm*deg2rad
    lat_c = lat_c*pm*deg2rad
    lon0 = lon0*pm*deg2rad

    # Calculations
    t = np.tan(np.pi/4 - lat_rad/2)/((1 - e*np.sin(lat_rad))/(1 + e*np.sin(lat_rad)))**(e/2)
    t_c = np.tan(np.pi/4 - lat_c/2)/((1 - e*np.sin(lat_c))/(1 + e*np.sin(lat_c)))**(e/2)
    m_c = np.cos(lat_c)/np.sqrt(1 - (e*np.sin(lat_c))**2)
    rho = a*m_c*t/t_c
    x = pm*rho*np.sin(lon_rad - lon0)
    y = -pm*rho*np.cos(lon_rad - lon0)

    if isinstance(x, xr.DataArray) and len(lon.shape)==1:
        # Case that input arrays were 1D: default casting is to have x as the first coordinate; this is not what we want
        lon = lon.transpose()
        lat = lat.transpose()

    return x, y


# Convert from polar stereographic coordinates to lat-lon. Adapated from the function psxy2ll.m used by Ua (with credits to Craig Stewart, Adrian Jenkins, Pierre Dutrieux) and made more consistent with naming convections of function above.
# This is about twice as fast as the pyproj Transformer function (for BedMachine v3 at least), but it is limited to this specific case so could consider changing in the future if I end up using more projections than just these two.
def polar_stereo_inv (x, y, a=6378137., e=0.08181919, lat_c=-71, lon0=0):

    if not isinstance(x, xr.DataArray) and len(x.shape)==1:
        # Need to broadcast dimensions.
        x, y = np.meshgrid(x, y)

    if lat_c < 0:
        pm = -1
    else:
        pm = 1
    lat_c = lat_c*pm*deg2rad
    lon0 = lon0*deg2rad
    epsilon = 1e-12

    tc = np.tan(np.pi/4 - lat_c/2)/((1 - e*np.sin(lat_c))/(1 + e*np.sin(lat_c)))**(e/2)
    mc = np.cos(lat_c)/np.sqrt(1 - e**2*(np.sin(lat_c))**2)
    rho = np.sqrt(x**2 + y**2)
    t = rho*tc/(a*mc)
    lon = lon0 + np.arctan2(x,y)

    lat_new = np.pi/2 - 2*np.arctan(t)
    dlat = 2*epsilon
    while dlat > epsilon:
        lat_old = lat_new
        lat_new = np.pi/2 - 2*np.arctan(t*((1 - e*np.sin(lat_old))/(1 + e*np.sin(lat_old)))**(e/2))
        dlat = np.amax(lat_new - lat_old)
    lat = lat_new

    lat = lat*pm/deg2rad
    lon = fix_lon_range(lon/deg2rad)

    if isinstance(lon, xr.DataArray) and len(x.shape)==1:
        # Case that input arrays were 1D: default casting is to have x as the first coordinate; this is not what we want
        lon = lon.transpose()
        lat = lat.transpose()

    return lon, lat


# Given an array of grid values on the edges (gtype=u, v) or corners (gtype=f) of the grid, extend by one column to the west and/or row to the south so that all of the tracer points have edges defined on both sides.
# Note that the index convention of the resulting array will change relative to the tracer grid. A t-point (j,i) at the centre of the cell originally has the corresponding f-point (j,i) to the northeast corner of the cell, but after this padding of the f-grid, the corresponding f-point (j,i) will be at the southwest corne of the cell.
# This should also work if "array" is a Dataset instead of a DataArray.
def extend_grid_edges (array, gtype, periodic=True, halo=False):

    if gtype in ['u', 'f']:
        # New column to the west
        if periodic:
            # The western edge already exists on the other side
            if halo:
                edge_W = array.isel(x=-3)
            else:
                edge_W = array.isel(x=-1)
        else:
            # Extrapolate
            edge_W = 2*array.isel(x=0) - array.isel(x=1)
        array = xr.concat([edge_W, array], dim='x')
    if gtype in ['v', 'f']:
        # New column to the south: extrapolate
        edge_S = 2*array.isel(y=0) - array.isel(y=1)
        array = xr.concat([edge_S, array], dim='y')
    return array.transpose('y', 'x')


# Return the deepest unmasked values along the named z-dimension of the given xarray DataArray.
# Following https://stackoverflow.com/questions/74172428/calculate-the-first-instance-of-a-value-in-axis-xarray
def select_bottom (array, zdim):

    bottom_depth = array.coords[zdim].where(array.notnull()).max(dim=zdim)
    return array.sel({zdim:bottom_depth.fillna(0).astype(int)}).where(bottom_depth.notnull())


# Given a mask (numpy array, 1='land', 0='ocean') and point0 (j,i) on the "mainland", remove any disconnected "islands" from the mask and return.
def remove_disconnected (mask, point0):

    if not mask[point0]:
        raise Exception('point0 is not on the mainland')

    connected = np.zeros(mask.shape)
    connected[point0] = 1
    ny = mask.shape[0]
    nx = mask.shape[1]

    queue = [point0]
    while len(queue) > 0:
        (j,i) = queue.pop(0)
        neighbours = []
        if j > 0:
            neighbours.append((j-1,i))
        if j < ny-1:
            neighbours.append((j+1,i))
        if i > 0:
            neighbours.append((j,i-1))
        if i < nx-1:
            neighbours.append((j,i+1))
        for point in neighbours:
            if connected[point]:
                continue
            if mask[point]:
                connected[point] = True
                queue.append(point)

    return connected


# Find the (y,x) coordinates of the closest model point to the given (lon, lat) coordinates. Pass an xarray Dataset containing nav_lon, nav_lat, and a target point (lon0, lat0).
def closest_point (ds, target):

    lon = ds['nav_lon'].squeeze()
    lat = ds['nav_lat'].squeeze()
    [lon0, lat0] = target
    # Calculate distance of every model point to the target
    dist = np.sqrt((lon-lon0)**2 + (lat-lat0)**2)
    # Find the indices of the minimum distance
    point0 = dist.argmin(dim=('y','x'))
    return (int(point0['y'].data), int(point0['x'].data))

# Helper function to calculate the Cartesian distance between two longitude and latitude points
# This also works if one of point0, point1 is a 2D array.
def distance_btw_points (point0, point1):
    
    [lon0, lat0] = point0
    [lon1, lat1] = point1
    
    dx = rEarth*np.cos((lat0+lat1)/2*deg2rad)*(lon1-lon0)*deg2rad
    dy = rEarth*(lat1-lat0)*deg2rad
    
    return np.sqrt(dx**2 + dy**2)
    

# Helper function to calculate a bunch of grid variables (bathymetry, draft, ocean mask, ice shelf mask) from a NEMO output file, only using thkcello and the mask on a 3D data variable (current options are to look for thetao and so).
# This varies a little if the sea surface height changes, so not perfect, but it does take partial cells into account.
# If keep_time_dim, will preserve any time dimension even if it's of size 1 (useful for timeseries)
def calc_geometry (ds, keep_time_dim=False):

    mask_3d = None
    for var in ['thetao', 'so']:
        if var in ds:
            mask_3d = xr.where(ds[var]==0, 0, 1).squeeze()
            break
    if mask_3d is None:
        raise Exception('No known 3D masked variable is present. Add another variable to the code?')
    # 2D ocean cells are unmasked at some depth
    ocean_mask = mask_3d.sum(dim='deptht')>0
    # 2D ice shelf cells are ocean cells which are masked at the surface
    ice_mask = ocean_mask*(mask_3d.isel(deptht=0)==0)
    # Water column thickness is sum of thkcello in unmasked cells
    wct = (ds['thkcello']*mask_3d).sum(dim='deptht')
    # Now identify the 3D ice shelf cells using cumulative sum of mask
    ice_mask_3d = (mask_3d.cumsum(dim='deptht')==0)*ocean_mask
    # Ice draft is sum of thkcello in ice shelf cells
    draft = (ds['thkcello']*ice_mask_3d).sum(dim='deptht')
    # Bathymetry is ice draft plus water column thickness
    bathy = draft + wct
    if not keep_time_dim:
        bathy = bathy.squeeze()
        draft = draft.squeeze()
    return bathy, draft, ocean_mask, ice_mask
    

# Select ice shelf cavities. Pass it the path to an xarray Dataset which contains either 'maskisf' (NEMO3.6 mesh_mask), 'top_level' (NEMO4.2 domain_cfg), or 'thkcello' plus a 3D data variable with a zero-mask applied (NEMO output file - see calc_geometry)
def build_ice_mask (ds):

    if 'ice_mask' in ds:
        # Previously computed
        return ds['ice_mask'], ds
    if 'maskisf' in ds:
        ice_mask = ds['maskisf'].squeeze()
    elif 'top_level' in ds:
        ice_mask = xr.where(ds['top_level']>1, 1, 0).squeeze()
    else:
        ice_mask = calc_geometry(ds)[3]
    # Save to the Dataset in case it's useful later
    ds = ds.assign({'ice_mask':ice_mask})
    return ice_mask, ds


# As above, select the ocean mask
def build_ocean_mask (ds):

    if 'ocean_mask' in ds:
        return ds['ocean_mask'], ds
    if 'tmaskutil' in ds:
        ocean_mask = ds['tmaskutil'].squeeze()
    elif 'bottom_level' in ds:
        ocean_mask = xr.where(ds['bottom_level']>0, 1, 0).squeeze()
    else:
        ocean_mask = calc_geometry(ds)[2]
    ds = ds.assign({'ocean_mask':ocean_mask})
    return ocean_mask, ds


# Select the continental shelf and ice shelf cavities. Pass it the path to an xarray Dataset which contains one of the following combinations:
# 1. nav_lon, nav_lat, bathy, tmaskutil (NEMO3.6 mesh_mask)
# 2. nav_lon, nav_lat, bathy_metry, bottom_level (NEMO4.2 domain_cfg)
# 3. nav_lon, nav_lat, thkcello, a 3D data variable with a zero-mask applied (current options are thetao or so) (NEMO output file) 
def build_shelf_mask (ds):

    if 'shelf_mask' in ds:
        # Previously computed
        return ds['shelf_mask'], ds
    
    if 'bathy' in ds and 'tmaskutil' in ds:
        bathy = ds['bathy'].squeeze()
        ocean_mask = ds['tmaskutil'].squeeze()
    elif 'bathy_metry' in ds and 'bottom_level' in ds:
        bathy = ds['bathy_metry'].squeeze()
        ocean_mask = xr.where(ds['bottom_level']>0, 1, 0).squeeze()
    elif 'thkcello' in ds:
        bathy, draft, ocean_mask, ice_mask = calc_geometry(ds)
        # Make sure ice shelves are included in the final mask, by setting bathy to 0 here
        bathy = xr.where(ice_mask, 0, bathy)
    else:
        raise Exception('invalid Dataset for build_shelf_mask')        
    # Apply lat-lon bounds and bathymetry bound to ocean mask
    mask = ocean_mask*(ds['nav_lat'] <= shelf_lat)*(bathy <= shelf_depth)
    # Remove disconnected seamounts
    point0 = closest_point(ds, shelf_point0)
    mask.data = remove_disconnected(mask, point0)
    # Save to the Dataset in case it's useful later
    ds = ds.assign({'shelf_mask':mask})

    return mask, ds


# Select a mask for a single cavity. Pass it an xarray Dataset as for build_shelf_mask.
def single_cavity_mask (cavity, ds, return_name=False):

    if return_name:
        title = region_names[region]

    if cavity+'_single_cavity_mask' in ds:
        # Previously computed
        if return_name:
            return ds[cavity+'_single_cavity_mask'], ds, title
        else:
            return ds[cavity+'_single_cavity_mask'], ds

    ds = ds.load()

    # Get mask for all cavities
    ice_mask, ds = build_ice_mask(ds)
    ice_mask = ice_mask.copy()

    # Select one point in this cavity
    point0 = closest_point(ds, region_points[cavity])
    # Disconnect the other cavities
    mask = remove_disconnected(ice_mask, point0)
    ice_mask.data = mask

    # Save to the Dataset in case it's useful later
    ds = ds.assign({cavity+'_single_cavity_mask':ice_mask})

    if return_name:
        return ice_mask, ds, title
    else:
        return ice_mask, ds


# Select a mask for the given region, either continental shelf only ('shelf'), cavities only ('cavity'), or continental shelf with cavities ('all'). Pass it an xarray Dataset as for build_shelf_mask.
def region_mask (region, ds, option='all', return_name=False):

    if return_name:
        # Construct the title
        title = region_names[region]
        if option in ['shelf', 'all']:
            title += ' continental shelf'
            if option == 'all':
                title += ' and'
        if option in ['cavity', 'all']:
            if region in ['filchner_ronne', 'amery', 'ross']:
                title += ' Ice Shelf cavity'
            else:
                title += ' cavities'

    if region+'_'+option+'_mask' in ds:
        # Previously computed
        if return_name:
            return ds[region+'_'+option+'_mask'], ds, title
        else:
            return ds[region+'_'+option+'_mask'], ds

    # Get mask for entire continental shelf and cavities
    mask, ds = build_shelf_mask(ds)
    mask = mask.copy()

    if region != 'all':
        # Restrict to a specific region of the coast
        # Select one point each on western and eastern boundaries
        [coord_W, coord_E] = region_edges[region]
        point0_W = closest_point(ds, coord_W)
        [j_W, i_W] = point0_W
        point0_E = closest_point(ds, coord_E)
        [j_E, i_E] = point0_E

        # Make two cuts to disconnect the region
        # Inner function to cut the mask in the given direction: remove the given point and all of its connected neighbours to the N/S or E/W
        def cut_mask (point0, direction):
            if direction == 'NS':
                i = point0[1]
                # Travel north until disconnected
                for j in range(point0[0], ds.sizes['y']):
                    if mask[j,i] == 0:
                        break
                    mask[j,i] = 0
                # Travel south until disconnected
                for j in range(point0[0]-1, -1, -1):
                    if mask[j,i] == 0:
                        break
                    mask[j,i] = 0
            elif direction == 'EW':
                j = point0[0]
                # Travel east until disconnected
                for i in range(point0[1], ds.sizes['x']):
                    if mask[j,i] == 0:
                        break
                    mask[j,i] = 0
                # Travel west until disconnected
                for i in range(point0[1]-1, -1, -1):
                    if mask[j,i] == 0:
                        break
                    mask[j,i] = 0
        # Inner function to select one cell "west" of the given point - this might not actually be properly west if the cut is made in the east/west direction, in this case you have to choose one cell north or south depending on the direction of travel.
        def cell_to_west (point0, direction):
            (j,i) = point0
            if direction == 'NS':
                # Cell to the west
                return (j, i-1)
            elif direction == 'EW':
                if j_E > j_W:
                    # Travelling north: cell to the south
                    return (j-1, i)
                elif j_E < j_W:
                    # Travelling south: cell to the north
                    return (j+1, i)
                else:
                    raise Exception('Something is wrong with region_edges')

        [flag_W, flag_E] = region_edges_flag[region]
        # Western boundary is inclusive: cut at cell to "west"
        cut_mask(cell_to_west(point0_W, flag_W), flag_W)
        # Eastern boundary is exclusive: cut at that cell
        cut_mask(point0_E, flag_E)

        # Run remove_disconnected on western point to disconnect the rest of the continental shelf
        mask_region = remove_disconnected(mask, point0_W)
        # Check if it wraps around the periodic boundary
        if i_E < i_W:
            # Make a second region by running remove_disconnected on one cell "west" from eastern point
            mask_region2 = remove_disconnected(mask, cell_to_west(point0_E, flag_E))
            mask_region += mask_region2
        mask.data = mask_region

    # Special cases (where common boundaries didn't agree for eORCA1 and eORCA025)
    if region == 'amundsen_sea':
        # Remove bits of Abbot
        mask_excl, ds = single_cavity_mask('abbot', ds)
    elif region == 'filchner_ronne':
        # Remove bits of Brunt
        mask_excl, ds = single_cavity_mask('brunt', ds)
    else:
        mask_excl = None
    if region == 'bellingshausen_sea':
        # Add back in bits of Abbot
        mask_incl, ds = single_cavity_mask('abbot', ds)
    elif region == 'east_antarctica':
        # Add back in bits of Brunt
        mask_incl, ds = single_cavity_mask('brunt', ds)
    else:
        mask_incl = None
    if mask_excl is not None:
        mask *= 1-mask_excl
    if mask_incl is not None:
        mask = xr.where(mask_incl, 1, mask)

    # Now select cavities, shelf, or both
    ice_mask, ds = build_ice_mask(ds)
    if option == 'cavity':
        mask *= ice_mask
    elif option == 'shelf':
        mask *= 1-ice_mask

    # Save to the Dataset in case it's useful later
    ds = ds.assign({region+'_'+option+'_mask':mask})

    if return_name:
        return mask, ds, title
    else:
        return mask, ds

        
# Function to convert the units of shortwave and longwave radiation to the units expected by NEMO (W m-2)
# Reads the specified variable from the NetCDF file and writes the converted variable to a new file in the same folder
# with the file name starting with "converted_"
# Input: 
# file_rad: string name of the atmospheric forcing NetCDF file 
# variable: string name of the radiation variable within the file specified by file_rad
# dataset: string specifying type of atmospheric forcing dataset (ERA5, JRA etc.)
# folder: string of location that contains the atmospheric forcing files
def convert_radiation(file_rad='era5_strd_1979_daily_averages.nc', variable='strd', 
                      dataset='ERA5', folder='/gws/nopw/j04/terrafirma/birgal/NEMO_AIS/ERA5-forcing/'):
    if dataset=='ERA5':
        # ERA5 is in J m-2, convert to Watt m-2 = J m-2 s-1, so divide by the accumulation period in seconds
        # In this case, the files are daily averages of the original hourly files. So, the J/m-2 is actually the accumulation over an hour. 
        ds = xr.open_dataset(f'{folder}{file_rad}') # shortwave or longwave radiation
        ds[variable] = ds[variable] / sec_per_hour
        ds.to_netcdf(f'{folder}converted_{file_rad}')
        
        return 
    else:
        raise Exception('Only currently set up to convert ERA5 units to nemo units')

# Function to convert the units of precipitation from m of water equivalent to the units expected by NEMO (kg m-2 s-1)
# Reads the specified variable from the NetCDF file and writes the converted variable to a new file in the same folder
# with the file name starting with "converted_"
# Input: 
# file_precip: string name of the atmospheric forcing NetCDF file 
# variable: string name of the precipitation variable within the file specified by file_precip
# dataset: string specifying type of atmospheric forcing dataset (ERA5, JRA etc.)
# folder: string of location that contains the atmospheric forcing files
def convert_precip(file_precip='era5_tp_1979_daily_averages.nc', variable='tp', 
                   dataset='ERA5', folder='/gws/nopw/j04/terrafirma/birgal/NEMO_AIS/ERA5-forcing/'):
    if dataset=='ERA5':
        # ERA5 is in m of water equivalent, convert to kg m-2 s-1, so need to divide by the accumulation period, and convert density
        ds = xr.open_dataset(f'{folder}{file_precip}')
        # m --> m/s --> kg/m2/s
        ds[variable] = (ds[variable] / sec_per_hour) * rho_fw # total precip is in meters of water equivalent
        ds.to_netcdf(f'{folder}converted_{file_precip}')

        return        
    else:
        raise Exception('Only currently set up to convert ERA5 units to nemo units')

# Function to calculate specific humidity from dewpoint temperature and atmospheric pressure
# Reads the specified variable from the NetCDF file and writes the converted variable to a new file in the same folder
# with the file name starting with "converted_"
# Input: 
# file_dew: string name of the dewpoint temperature NetCDF file 
# file_slp: string name of the sea level pressure NetCDF file 
# variable_dew: string name of the dewpoint temperature variable within the file specified by file_dew
# variable_slp: string name of the sea level pressure variable within the file specified by file_slp
# dataset: string specifying type of atmospheric forcing dataset (ERA5, JRA etc.)
# folder: string of location that contains the atmospheric forcing files
def calculate_specific_humidity(file_dew='era5_d2m_1979_daily_averages.nc', variable_dew='d2m',
                                file_slp='era5_msl_1979_daily_averages.nc', variable_slp='msl',
                                dataset='ERA5', folder='/gws/nopw/j04/terrafirma/birgal/NEMO_AIS/ERA5-forcing/'):
    if dataset=='ERA5':
        # ERA5 does not provide specific humidity, but gives the 2 m dewpoint temperature in K
        # Conversion assumes temperature is in K and pressure in Pa.
        # Based off: https://confluence.ecmwf.int/pages/viewpage.action?pageId=171411214

        ds               = xr.open_dataset(f'{folder}{file_dew}')
        surface_pressure = xr.open_dataset(f'{folder}{file_slp}')[variable_slp]

        dewpoint = ds[variable_dew]
        # calculation:
        vapor_pressure = vap_pres_c1*np.exp(vap_pres_c3*(dewpoint.values - temp_C2K)/(dewpoint.values - vap_pres_c4)) # E saturation water vapour from Teten's formula
        spec_humidity  = (Rdry / Rvap) * vapor_pressure / (surface_pressure - ((1-Rdry/Rvap)*vapor_pressure)) # saturation specific humidity

        ds[variable_dew] = spec_humidity
        ds = ds.rename_vars({variable_dew:'specific_humidity'})
        ds.to_netcdf(f'{folder}converted_{file_dew}')
        
        return
    else:
        raise Exception('Only currently set up to convert ERA5 units to nemo units')     

# Function to ensure the reference time is consistent between atmospheric data sources
# JRA uses days since 1900-01-01 00:00:00 on a Gregorian calendar
# ERA uses days since start of that particular year in proleptic gregorian calendar
# Input:
# ds : xarray dataset containing variable 'time'
# dataset : name of atmospheric forcing dataset
def convert_time_units(ds, dataset='ERA5'):

    if dataset=='ERA5':
        ds['time'] = ds.time.values
        ds['time'].encoding['units'] = "days since 1900-01-01"
        ds['time'].encoding['calendar'] = 'gregorian'
        return ds
    else:
        raise Exception('Only currently set up to convert ERA5 reference period')


# Advance the given date (year and month, both ints) by num_months
def add_months (year, month, num_months):
    
    month += num_months
    while month > months_per_year:
        month -= months_per_year
        year += 1
    return year, month


# Smooth the given DataArray with a moving average of the given window, over the given dimension (default time_centered).
def moving_average (data, window, dim='time_centered'):

    if window == 0:
        return data

    # Find axis number of dimension
    dim_axis = 0
    for var in data.sizes:
        if var == dim:
            break
        dim_axis += 1

    centered = window%2==1
    if centered:
        radius = (window-1)//2
    else:
        radius = window//2
    t_first = radius
    t_last = data.sizes[dim] - radius  # First one not selected, as per python convention
    # Array of zeros of the same shape as a single time index of data
    zero_base = (data.isel({dim:slice(0,1)})*0).data
    # Do the smoothing in two steps, in numpy world
    data_cumsum = np.ma.concatenate((zero_base.data, np.ma.cumsum(data.data, axis=dim_axis)), axis=dim_axis)
    if centered:
        data_smoothed = (data_cumsum[t_first+radius+1:t_last+radius+1,...] - data_cumsum[t_first-radius:t_last-radius,...])/(2*radius+1)
    else:
        data_smoothed = (data_cumsum[t_first+radius:t_last+radius,...] - data_cumsum[t_first-radius:t_last-radius,...])/(2*radius)
    # Now trim the original array
    data_trimmed = data.isel({dim:slice(radius, -radius)})
    if not centered:
        # Shift time dimension half an index forward
        time1 = data[dim].isel({dim:slice(radius-1, -radius-1)})
        time2 = data[dim].isel({dim:slice(radius, -radius)})
        time_trimmed = time1.data + (time2.data-time1.data)/2
        data_trimmed[dim] = time_trimmed
    data_trimmed.data = data_smoothed
    return data_trimmed


# Given a string representing a month, convert between string representations (3-letter lowercase, eg jan) and int representations (2-digit string, eg 01).
def month_convert (date_code):

    month_str = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
    month_int = [str(n+1).zfill(2) for n in range(months_per_year)]

    if isinstance(date_code, int):
        date_code = str(date_code).zfill(2)
        
    if date_code in month_str:
        n = month_str.index(date_code)
        return month_int[n]
    elif date_code in month_int:
        n = month_int.index(date_code)
        return month_str[n]
    else:
        raise Exception('Invalid month')


# Rotate a vector from the local x-y space to geographic space (with true zonal and meridional components). Follows subroutines "angle" and "rot_rep" in nemo/src/OCE/SBC/geo2ocean.F90.
# Warning, this might not handle the north pole correctly, so use with caution if you are plotting a global grid and care about the north pole.
# Inputs:
# u, v: xarray DataArrays containing the u and v components of the vector in local x-y space (eg velocities from NEMO output).
# domcfg: either the path to the domain_cfg file, or an xarray Dataset containing glamt, gphit, glamu, etc.
# gtype: grid type: 'T', 'U', 'V', or 'F'. In practice you will interpolate both velocities to the T-grid and then call this with gtype='T' (default).
# periodic: whether the grid is periodic
# halo: whether the halo is included in the arrays (generally true for NEMO3.6, false for NEMO4.2). Only matters if periodic=True.
# return_angles: whether to return cos_grid and sin_grid
# Outputs:
# ug, vg: xarray DataArrays containing the zonal and meridional components of the vector in geographic space
# cos_grid, sin_grid (only if return_angles=True): cos and sin of the angle between the grid and east
def rotate_vector (u, v, domcfg, gtype='T', periodic=True, halo=True, return_angles=False):

    # Repeat all necessary import statements within here so the function is self-contained (someone can just copy and paste the whole thing if wanted)
    import xarray as xr
    import numpy as np
    deg2rad = np.pi/180.

    if isinstance(domcfg, str):
        domcfg = xr.open_dataset(domcfg)
    if domcfg.sizes['y'] != u.sizes['y']:
        # The TerraFIRMA overshoot output was trimmed to only cover the Southern Ocean when I pulled it from MASS, while domain_cfg remains global. Assume this is the reason for the mismatch but print a warning.
        print('Warning (rotate_vector): trimming domain_cfg to select only the southernmost part, to align with input vectors - is this what you want?')
        domcfg = domcfg.isel(y=slice(0, u.sizes['y']))
    domcfg = domcfg.squeeze()

    if u.dims != v.dims:
        # Dimensions don't match. Usually this is because 3D velocities have retained 'depthu' and 'depthv' dimensions even though they've been interpolated to the T-grid.
        if gtype in ['T', 't'] and 'depthu' in u.dims and 'depthv' in v.dims:
            u = u.rename({'depthu':'deptht'})
            v = v.rename({'depthv':'deptht'})
        # Check again
        if u.dims != v.dims:
            raise Exception('Mismatch in dimensions')

    # Get lon and lat on this grid
    lon = domcfg['glam'+gtype.lower()]
    lat = domcfg['gphi'+gtype.lower()]

    # Calculate, at each point, the x and y components and the squared-norm of the vector between the point and the North Pole
    vec_NP_x = -2*np.cos(lon*deg2rad)*np.tan(np.pi/4 - lat*deg2rad/2)
    vec_NP_y = -2*np.sin(lon*deg2rad)*np.tan(np.pi/4 - lat*deg2rad/2)
    vec_NP_norm2 = vec_NP_x**2 + vec_NP_y**2

    # Inner function to get adjacent points on an alternate grid.
    def grid_edges (var_name, shift):
        edge1 = domcfg[var_name]
        if shift == 'j-1':
            edge2 = edge1.shift(y=1)
            # Extrapolate southern boundary
            edge2.isel(y=0).data = 2*edge1.isel(y=1).data - edge1.isel(y=0).data
        elif shift == 'j+1':
            edge2 = edge1.shift(y=-1)
            # Extrapolate northern boundary
            edge2.isel(y=-1).data = 2*edge1.isel(y=-2).data - edge1.isel(y=-1).data
        elif shift == 'i-1':
            edge2 = edge1.shift(x=1)
            if periodic:
                # Western boundary already exists on the other side
                if halo:
                    edge2.isel(x=0).data = edge1.isel(x=-3).data
                else:
                    edge2.isel(x=0).data = edge1.isel(x=-1).data
            else:
                # Extrapolate western boundary
                edge2.isel(x=0).data = 2*edge1.isel(x=1).data - edge1.isel(x=0).data        
        return edge1, edge2
    # Call this function for both lon and lat on the given grid.
    def lonlat_edges (gtype2, shift):
        lon_edge1, lon_edge2 = grid_edges('glam'+gtype2.lower(), shift)
        lat_edge1, lat_edge2 = grid_edges('gphi'+gtype2.lower(), shift)
        return lon_edge1, lat_edge1, lon_edge2, lat_edge2            

    # Calculate, at each point, the x and y components and the norm of the vector between adjacent points on an alternate grid.
    if gtype in ['T', 't']:
        # v-points above and below the given t-point
        lon_edge1, lat_edge1, lon_edge2, lat_edge2 = lonlat_edges('v', 'j-1')
    elif gtype in ['U', 'u']:
        # f-points above and below the given u-point
        lon_edge1, lat_edge1, lon_edge2, lat_edge2 = lonlat_edges('f', 'j-1')
    elif gtype in ['V', 'v']:
        # f-points left and right of the given v-point
        lon_edge1, lat_edge1, lon_edge2, lat_edge2 = lonlat_edges('f', 'i-1')
    elif gtype in ['F', 'f']:
        # u-points above and below the given f-point
        # Note reversed order of how we save the outputs
        lon_edge2, lat_edge2, lon_edge1, lat_edge1 = lonlat_edges('u', 'j+1')
    vec_pts_x = 2*np.cos(lon_edge1*deg2rad)*np.tan(np.pi/4 - lat_edge1*deg2rad/2) - 2*np.cos(lon_edge2*deg2rad)*np.tan(np.pi/4 - lat_edge2*deg2rad/2)
    vec_pts_y = 2*np.sin(lon_edge1*deg2rad)*np.tan(np.pi/4 - lat_edge1*deg2rad/2) - 2*np.sin(lon_edge2*deg2rad)*np.tan(np.pi/4 - lat_edge2*deg2rad/2)
    vec_pts_norm = np.maximum(np.sqrt(vec_NP_norm2*(vec_pts_x**2 + vec_pts_y**2)), 1e-14)

    # Now get sin and cos of the angles of the given grid
    if gtype in ['V', 'v']:
        sin_grid = (vec_NP_x*vec_pts_x + vec_NP_y*vec_pts_y)/vec_pts_norm
        cos_grid = -(vec_NP_x*vec_pts_y - vec_NP_y*vec_pts_x)/vec_pts_norm
    else:
        sin_grid = (vec_NP_x*vec_pts_y - vec_NP_y*vec_pts_x)/vec_pts_norm
        cos_grid = (vec_NP_x*vec_pts_x + vec_NP_y*vec_pts_y)/vec_pts_norm

    # Identify places where the adjacent grid cells are essentially equal (can happen with weird patched-together grids etc filling parts of Antarctic land mask with constant values) - no rotation needed here
    eps = 1e-8
    if gtype in ['T', 't']:
        lon_edge1, lon_edge2 = grid_edges('glamv', 'j-1')        
    elif gtype in ['U', 'u']:
        lon_edge1, lon_edge2 = grid_edges('glamf', 'j-1')
    elif gtype in ['V', 'v']:
        lat_edge1, lat_edge2 = grid_edges('gphif', 'i-1')
    elif gtype in ['F', 'f']:
        lon_edge1, lon_edge2 = grid_edges('glamu', 'j+1')
    if gtype in ['V', 'v']:
        index = np.abs(lat_edge1-lat_edge2) < eps
    else:
        index = np.abs(np.mod(lon_edge1-lon_edge2, 360)) < eps
    sin_grid = xr.where(index, 0, sin_grid)
    cos_grid = xr.where(index, 1, cos_grid)

    # Finally, rotate!
    ug = u*cos_grid - v*sin_grid
    vg = v*cos_grid + u*sin_grid

    if return_angles:
        return ug, vg, cos_grid, sin_grid
    else:
        return ug, vg


# Helper function to convert an xarray dataset with 3D T and S to TEOS10 (absolute salinity and conservative temperature)
# Inputs: 
# dataset: xarray dataset containing variables lon, lat, depth, and THETA (potential temperature) or SALT (practical salinity)
# var:     string of variable name to convert: THETA or SALT
def convert_to_teos10(dataset, var='SALT'):
    # Convert to TEOS10
    # Need 3D lat, lon, pressure at every point, so if 1D or 2D, broadcast to 3D
    if dataset.lon.values.ndim <= 2:
        lon   = xr.broadcast(dataset['lon'], dataset[var])[0]
    if dataset.lat.values.ndim <= 2:
        lat   = xr.broadcast(dataset['lat'], dataset[var])[0]
    if dataset.depth.values.ndim <= 2:
        # Need pressure in dbar at every 3D point: approx depth in m
        press = np.abs(xr.broadcast(dataset['depth'], dataset[var])[0])
    else:
        press = np.abs(dataset['depth'])
    
    if var=='SALT':
        # Get absolute salinity from practical salinity
        absS  = gsw.SA_from_SP(dataset[var], press, lon, lat)

        return absS
    elif var=='THETA':    
        if 'SALT' in list(dataset.keys()):
            # Get absolute salinity from practical salinity
            absS  = gsw.SA_from_SP(dataset['SALT'], press, lon, lat)
            # Get conservative temperature from potential temperature
            consT  = gsw.CT_from_t(absS, dataset[var], press)
        else:
            raise Exception('Must include practical salinity (SALT) variable in dataset when converting potential temperature')
        
        return consT
    else:
        raise Exception('Variable options are SALT or THETA')    
    
            

    

    




    
            
        
        
        

    


    

    
    
