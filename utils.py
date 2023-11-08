import numpy as np
import xarray as xr
from .constants import deg2rad, shelf_lat, shelf_depth, shelf_point0

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
def remove_islands (mask, point0):

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


# Select the continental shelf and ice shelf cavities. Pass it the path to the mesh_mask.nc file.
def shelf_mask (mesh_mask):

    ds = xr.open_dataset(mesh_mask).squeeze()
    # Apply lat-lon bounds and bathymetry bound to ocean mask
    mask = ds['tmaskutil']*(ds['nav_lat'] <= shelf_lat)*(ds['bathy'] <= shelf_depth)
    # Remove disconnected seamounts
    point0 = closest_point(ds, shelf_point0)
    mask.data = remove_islands(mask, point0)   

    return mask

        
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
        ds[variable] = ds[variable] / 3600
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
        rho_water = 1000 # kg/m3
        ds[variable] = (ds[variable] / 3600) * rho_water # total precip is in meters of water equivalent
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
        # constants: # note that these constants could be different over ice
        a1 = 611.21; a3 = 17.502; a4=32.19; T0=273.16;
        Rdry = 287.0597; Rvap=461.5250; 
        # calculation:
        vapor_pressure = a1*np.exp(a3*(dewpoint.values - T0)/(dewpoint.values - a4)) # E saturation water vapour from Teten's formula
        spec_humidity  = (Rdry / Rvap) * vapor_pressure / (surface_pressure - ((1-Rdry/Rvap)*vapor_pressure)) # saturation specific humidity

        ds[variable_dew] = spec_humidity
        ds = ds.rename_vars({variable_dew:'specific_humidity'})
        ds.to_netcdf(f'{folder}converted_{file_dew}')
        
        return
    else:
        raise Exception('Only currently set up to convert ERA5 units to nemo units')     
        
        
        

    


    

    
    
