import xarray as xr
import numpy as np

def interp_cell_binning (ds_source, ds_target, plot=False, pster=True, periodic=True):

    import geopandas as gpd
    from shapely.geometry import Point, Polygon
    if plot:
        import matplotlib.pylot as plt

    # Determine coordinate reference system - do we actually need this?
    if pster:
        # Antarctic polar stereographic
        crs = 'epsg:3031'
    else:
        # Ordinary lat-lon
        crs = 'epsg:4326'

    # Loop over grid cells in NEMO    
    # Get the cell boundaries
    # Narrow down the source points to search: since quadrilaterals are convex, don't need to consider anything with x < min(x_corners) etc

    # Time two options: make a Polygon and loop over Points, or use GeoPandas spatial join
    # Subset ds_source 
    #df = ds_source.to_dataframe().reset_index()
    #gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df['x'], df['y']))

    # Loop over such points in source dataset
    # Check if point is in polygon; if so, add to list
    # Average over these points
    # Mask anywhere with no points
    # Plot average as well as number of points in each cell
