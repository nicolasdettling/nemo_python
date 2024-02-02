#######################################################
# Everything to do with reading the grid
# You can build this using NetCDF output files 
#######################################################

import numpy as np

from .interpolation import neighbours


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
