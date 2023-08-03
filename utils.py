import numpy as np
import xarray as xr
from .constants import deg2rad

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
def extend_grid_edges (array, gtype, periodic=True):

    if gtype in ['u', 'f']:
        # New column to the west
        if periodic:
            # With the 2-cell periodic halo, the western edge already exists on the other side.
            edge_W = array.isel(x=-3)
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


    

    
    
