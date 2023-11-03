import numpy as np

# Degrees to radians conversion factor
deg2rad = np.pi/180.0
# Degrees formatted nicely in a string
deg_string = r'$^{\circ}$'
# 10^-3 formatted nicely in a string (for absolute salinity)
gkg_string = r'10$^{-3}$'

# Dictionary of bounds on different regions
# lon_min, lon_max, lat_min, lat_max
region_bounds = {
    'AIS': [None, None, None, -58],
    'amundsen_sea': [],
    'bellingshausen_sea': [],
    'peninsula': [],
    'ross': [],
    'FRIS': [],
    'EAIS': [],
    'bear_ridge_S': [-110.35, -110.35, -74.35, -73.8791],
    'bear_ridge_N': [-112.05, -109.05, -73.8791, -72.7314],
}
# Names corresponding to regions (for plotting)
region_names = {
    'AIS': 'Antarctic',
    'amundsen_sea': 'Amundsen Sea',
    'bellingshausen_sea': 'Bellingshausen Sea',
    'ross': 'Ross',
    'FRIS': 'Filchner-Ronne',
    'EAIS': 'East Antarctic'
}
# Depth of continental shelf
shelf_depth = 2000
# Lon and lat of a point which is definitely on the given continental shelf (so we can isolate seamounts disconnected from this)
region_point = {
    'AIS': [-51.5, -74.5],
    'amundsen_sea': [-107.5, -73],
    'bellingshausen_sea': [-82.5, -71.5],
    'ross': [176, -75],
    'FRIS': [-51.5, -74.5],
    'EAIS': [97.5, -65]
}
# Isobaths restricting some regions
region_bathy_bounds = {
    'bear_ridge_N': [350, None],
}
