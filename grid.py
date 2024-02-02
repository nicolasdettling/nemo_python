#######################################################
# Everything to do with reading the grid
# You can build this using NetCDF output files 
#######################################################

import numpy as np
import xarray as xr
from .interpolation import neighbours
from .constants import region_edges, region_edges_flag, region_names, region_points, shelf_lat, shelf_depth, shelf_point0


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


# Build and return a T grid mask for coastal points: open-ocean points with at least one neighbour that is land or ice shelf.
def get_coast_mask(mask):
    
    open_ocean = (mask.tmask.isel(time_counter=0) == 1)
    land_ice   = ~open_ocean
    
    num_coast_neighbours = neighbours(land_ice, missing_val=0)[-1]
    coast_mask           = (open_ocean*(num_coast_neighbours > 0)).astype(bool)
    
    return coast_mask


# Build and return a 2D mask for the ice shelf front points of the given ice shelf.
def get_icefront_mask(mask, side='ice'):

    # mask of iceshelves (true for iceshelf, false otherwise)
    # isf  = (mesh_new.misf.isel(time_counter=0) > 0) & (mesh_new.tmask.isel(time_counter=0, nav_lev=0) == 0)
    ice_shelf  = (mask.tmaskutil.isel(time_counter=0) - mask.tmask.isel(time_counter=0, nav_lev=0)).astype(bool)
    open_ocean = (mask.tmask.isel(time_counter=0, nav_lev=0) == 1)

    if side == 'ice':
        # Return ice shelf points with at least 1 open-ocean neighbour
        num_open_ocean_neighbours = neighbours(open_ocean, missing_val=0)[-1]
        
        return (ice_shelf*(num_open_ocean_neighbours > 0)).astype(bool)
    elif side == 'ocean':
        # Return ocean points with at least 1 ice shelf neighbour
        num_ice_shelf_neighbours = neighbours(ice_shelf, missing_val=0)[-1]
        
        return (open_ocean*(num_ice_shelf_neighbours > 0)).astype(bool)
