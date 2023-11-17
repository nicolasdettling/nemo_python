import netCDF4 as nc
import numpy as np

# NEMO 4.2 mesh_mask files are written with 2D variables x and y instead of nav_lon and nav_lat - at the same time as 1D dimensions x and y. This causes all manner of problems with xarray so the best thing is just to create a new file from scratch and copy over the variables one at a time, renaming as needed.
def fix_mesh_mask (file_in, file_out):

    id_in = nc.Dataset(file_in, 'r')
    id_out = nc.Dataset(file_out, 'w')

    print('Setting up dimensions')
    for dim in ['x', 'y', 'nav_lev']:
        id_out.createDimension(dim, id_in.dimensions[dim].size)
        id_out.createVariable(dim, 'f8', (dim))
        id_out.variables[dim][:] = np.arange(id_in.dimensions[dim].size)
    id_out.createDimension('time_counter', None)

    for var in id_in.variables:
        if var in ['nav_lev', 'time_counter']:
            continue
        print('Writing '+var)
        if var == 'x':
            var_new = 'nav_lon'
        elif var == 'y':
            var_new = 'nav_lat'
        else:
            var_new = var
        id_out.createVariable(var_new, id_in.variables[var].dtype, id_in.variables[var].dimensions)
        id_out.variables[var_new][:] = id_in.variables[var][:]

    id_in.close()
    id_out.close()
