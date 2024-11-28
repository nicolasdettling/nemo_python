# Interface for reading BISICLES output
import xarray as xr

def read_bisicles_var (file_path, var, level=0, order=0):

    from amrfile import io as amrio
    amrID = amrio.load(file_path)
    lo, hi = amrio.queryDomainCorners(amrID, level)
    x, y, data = amrio.readBox2D(amrID, level, lo, hi, var, order)
    da = xr.DataArray(data, coords={'x':x, 'y':y})
    return da
