# Interface for reading BISICLES output
import xarray as xr


# Read a list of 2D variables from a single time-index file and return as a minimal xarray Dataset.
def read_bisicles (file_path, var_names, level=0, order=0):

    from amrfile import io as amrio
    import re
    from datetime import datetime

    # Extract year from file name
    date_code = re.findall(r'\d{4}\d{2}\d{2}', file_path)[0]
    date = datetime.strptime(date_code, '%Y%m%d').date()

    # Load data and domain corners
    amrID = amrio.load(file_path)
    lo, hi = amrio.queryDomainCorners(amrID, level)
    # Loop over variables and add to dataset
    ds = None
    for var in var_names:
        x, y, data = amrio.readBox2D(amrID, level, lo, hi, var, order)
        da = xr.DataArray(data, coords={'x':x, 'y':y})
        da = da.expand_dims(dim={'time':[date]})
        if ds is None:
            ds = xr.Dataset({var:da})
        else:
            ds = ds.assign({var:da})
    
    ds.load()
    amrio.free(amrID)
    return ds


# Call read_bisicles for every time index file in the given directory.
def read_bisicles_all (sim_dir, file_head, file_tail, var_names, level=0, order=0):

    import os

    all_files = []
    for f in os.listdir(sim_dir):
        if os.path.isdir(f'{sim_dir}/{f}'): continue
        if f.startswith(file_head) and f.endswith(file_tail):
            all_files.append(f)
    if len(all_files)==0:
        raise Exception('No files matching '+sim_dir+file_head+'*'+file_tail)
    all_files.sort()
    ds = None
    for f in all_files:
        print('Reading '+f)
        ds_tmp = read_bisicles(sim_dir+'/'+f, var_names, level=level, order=order)
        if ds is None:
            ds = ds_tmp
        else:
            ds = xr.concat([ds, ds_tmp], dim='time')
    return ds

