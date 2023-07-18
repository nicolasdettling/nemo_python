from .utils import polar_stereo_inv

# Helper function to construct a minimal CF field so cf-python can do regridding.
# Mostly following Robin Smith's Unicicles coupling code in UKESM.
# If the coordinate axes (1D) x and y are not lat-lon, pass in auxiliary lat-lon values (2D).
def construct_cf (data, x, y, lon=None, lat=None):

    import cf
    native_latlon = lon is not None and lat is not None
    
    field = cf.Field()
    if native_latlon:
        dim_x = cf.DimensionCoordinate(data=cf.Data(x, 'degrees_east'), properties={'axis':'X', 'standard_name':'longitude'})
        dim_y = cf.DimensionCoordinate(data=cf.Data(y, 'degrees_north'), properties={'axis':'Y', 'standard_name':'latitude'})
    else:
        dim_x = cf.DimensionCoordinate(data=cf.Data(x, 'm'))
        dim_y = cf.DimensionCoordinate(data=cf.Data(y, 'm'))
    field.set_construct(cf.DomainAxis(size=x.size, key='X'))
    field.set_construct(dim_x, axes='X')
    field.set_construct(cf.DomainAxis(size=y.size, key='Y'))
    field.set_construct(dim_y, axes='Y')
    if not native_latlon:
        dim_lon = cf.AuxiliaryCoordinate(data=cf.Data(lon, 'degrees_east'), properties={'standard_name':'longitude'})
        field.set_construct(dim_lon, axes=('Y','X'))
        dim_lat = cf.AuxiliaryCoordinate(data=cf.Data(lat, 'degrees_north'), properties={'standard_name':'latitude'})
        field.set_construct(dim_lat, axes=('Y','X'))
    field.set_data(cf.Data(data), axes=('Y', 'X'))
    return field


def interp_latlon_cf (data, x, y, target_lon, target_lat, pster_source=False, periodic_target=True, method='conservative'):

    if not isinstance(data, list):
        data = [data]

    # Get source grid and data in CF format
    if pster_source:
        x_2d, y_2d = np.meshgrid(x, y)
        print('Converting to lat-lon projection')
        lon_in, lat_in = polar_stereo_inv(x_2d, y_2d)
    else:
        lon_in = None
        lat_in = None
    # Loop over data fields and convert each to CF
    data_cf = []
    for data0 in data:
        data_cf.append(construct_cf(data0, x, y, lon=lon_in, lat=lat_in))

    # Get target grid in CF format
    if len(target_lon.shape) == 2:
        # Target grid is not regular lat-lon
        # Need to get coordinate axes
        if isinstance(target_lon, xr.DataArray):
            # Get them from xarray data structures
            target_x = target_lon['x']
            target_y = target_lon['y']
        else:
            # Index them naively
            target_x = np.arange(target_lon.shape[1])
            target_y = np.arange(target_lon.shape[0])
        dummy_data = np.zeros(target_lon.shape)
    else:
        target_x = target_lon
        target_y = target_lat
        target_lon = None
        target_lat = None
        dummy_data = np.zeros((target_y.size, target_x.size))
    target_cf = construct_cf(dummy_data, target_x, target_y, lon=target_lon, lat=target_lat)
    # Get weights with CF, using the first data field
    regrid_operator = data_cf[0].regrids(target_cf, dst_cyclic=periodic_target, method=method, return_operator=True)

    # Now interpolate each field, re-using the weights each time
    data_interp = []
    for data_cf0 in data_cf:
        data_interp.append(data_cf0.regrids(regrid_operator).array)

    return data_interp    
    
