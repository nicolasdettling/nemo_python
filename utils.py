import numpy as np
from .constants import deg2rad

# Given an array containing longitude, make sure it's in the range (max_lon-360, max_lon). Default is (-180, 180). If max_lon is None, nothing will be done to the array.
def fix_lon_range (lon, max_lon=180):

    if max_lon is not None:
        index = lon >= max_lon
        try:
            lon[index] = lon[index] - 360
        except(TypeError):
            # lon could be a single value
            return fix_lon_range(np.array([lon]), max_lon=max_lon)[0]
        index = lon < max_lon-360
        lon[index] = lon[index] + 360
    return lon

# Convert longitude and latitude to polar stereographic projection used by BEDMAP2. Adapted from polarstereo_fwd.m in the MITgcm Matlab toolbox for Bedmap.
def polar_stereo (lon, lat, a=6378137., e=0.08181919, lat_c=-71, lon0=0):

    # Deep copies of arrays in case they are reused
    lon = np.copy(lon)
    lat = np.copy(lat)

    if lat_c < 0:
        # Southern hemisphere
        pm = -1
    else:
        # Northern hemisphere
        pm = 1

    # Prepare input
    lon = lon*pm*deg2rad
    lat = lat*pm*deg2rad
    lat_c = lat_c*pm*deg2rad
    lon0 = lon0*pm*deg2rad

    # Calculations
    t = np.tan(np.pi/4 - lat/2)/((1 - e*np.sin(lat))/(1 + e*np.sin(lat)))**(e/2)
    t_c = np.tan(np.pi/4 - lat_c/2)/((1 - e*np.sin(lat_c))/(1 + e*np.sin(lat_c)))**(e/2)
    m_c = np.cos(lat_c)/np.sqrt(1 - (e*np.sin(lat_c))**2)
    rho = a*m_c*t/t_c
    x = pm*rho*np.sin(lon - lon0)
    y = -pm*rho*np.cos(lon - lon0)

    return x, y


# Convert from polar stereographic coordinates to lat-lon. Adapated from the function psxy2ll.m used by Ua (with credits to Craig Stewart, Adrian Jenkins, Pierre Dutrieux) and made more consistent with naming convections of function above.
def polar_stereo_inv (x, y, a=6378137., e=0.08181919, lat_c=-71, lon0=0):

    x = np.copy(x)
    y = np.copy(y)
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

    return lon, lat

