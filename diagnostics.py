import xarray as xr

from .utils import closest_point, remove_disconnected
from .constants import ross_gyre_point0

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


# Calculate the barotropic streamfunction. The dataset ds must include the variables uo, thkcello, and e2u (grab it from domain_cfg).
# WARNING, this is in x-y space, working out how to rotate to proper zonal velocities.
def barotropic_streamfunction (ds):

    # Definite integral over depth (thkcello is dz)
    udz = (ds['uo']*ds['thkcello']).sum(dim='depthu')
    # Indefinite integral from south to north (e2u is dy)
    return (udz*ds['e2u']).cumsum(dim='y')


# Calculate the easternmost extent of the Ross Gyre: first find the 0 Sv contour of barotropic streamfunction which contains the point (160E, 70S), and then find the easternmost point within this contour.
# The dataset ds must include the variables uo, thkcello, e2u, nav_lon, nav_lat.
def ross_gyre_eastern_extent (ds):

    # Find all points where the barotropic streamfunction is negative
    strf = barotropic_streamfunction(ds)
    gyre_mask = strf < 0
    # Now only keep the ones connected to the known Ross Gyre point
    gyre_mask.data = remove_disconnected(gyre_mask, closest_point(ds, ross_gyre_point0))
    # Find all longitudes within this mask which are also in the western hemisphere
    gyre_lon = ds['nav_lon'].where((gyre_mask==1)*(ds['nav_lon']<0))
    # Return the easternmost point
    return gyre_lon.max()

