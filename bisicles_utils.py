# Interface for reading BISICLES output
import xarray as xr


# Read a single 2D variable from a single time-index file and return as a minimal DataArray.
def read_bisicles_var (file_path, var, level=0, order=0):

    from amrfile import io as amrio
    import re
    from datetime import datetime
    
    amrID = amrio.load(file_path)
    lo, hi = amrio.queryDomainCorners(amrID, level)
    x, y, data = amrio.readBox2D(amrID, level, lo, hi, var, order)
    da = xr.DataArray(data, coords={'x':x, 'y':y})

    # Extract year from file name
    date_code = re.findall(r'\d{4}\d{2}\d{2}', file_path)[0]
    date = datetime.strptime(date_code, '%Y%m%d').date()
    da = da.expand_dims(dim={'time':[date]})
    
    da.load()
    amrio.free(amrID)
    return da


# Call read_bisicles_var for every time index file in the given directory.
def read_bisicles_var_all (sim_dir, file_head, file_tail, var, level=0, order=0):

    all_files = []
    for f in os.listdir(sim_dir):
        if os.path.isdir(f'{sim_dir}/{f}'): continue
        if f.startswith(file_head) and f.endswith(file_tail):
            all_files.append(f)
    all_files.sort()
    da = None
    for f in all_files:
        da_tmp = read_bisicles_var(sim_dir+'/'+f, var, level=level, order=order)
        if da is None:
            da = da_tmp
        else:
            da = xr.concat([da, da_tmp], dim='time')
    return da

