import numpy as np

# Acceleration due to gravity (m/s^2)
gravity = 9.81
# Density of freshwater (kg/m^3)
rho_fw = 1e3
# Seconds per day
sec_per_day = 24*60*60.
# Seconds per hour
sec_per_hour = 60*60.
# Months per year
months_per_year = 12
# Celsius to Kelvins intercept
temp_C2K = 273.15
# Constants for vapor pressure calculation over water (Buck, 1981):
vap_pres_c1 = 611.21; vap_pres_c3 = 17.502; vap_pres_c4=32.19;
# Gas constant for dry air
Rdry = 287.0597
# Gas constant for water vapor
Rvap = 461.5250

# Degrees to radians conversion factor
deg2rad = np.pi/180.0
# Degrees formatted nicely in a string
deg_string = r'$^{\circ}$'
# 10^-3 formatted nicely in a string (for absolute salinity)
gkg_string = r'10$^{-3}$'

# Dictionary of bounds on different regions
# lon_min, lon_max, lat_min, lat_max
region_bounds = {
    'bear_ridge_S': [-110.35, -110.35, -74.35, -73.8791],
    'bear_ridge_N': [-112.05, -109.05, -73.8791, -72.7314],
}
# Isobaths restricting some regions
region_bathy_bounds = {
    'bear_ridge_N': [350, None],
}
# Latitude to bound continental shelf
shelf_lat = -58
# Depth of continental shelf
shelf_depth = 2000
# Lon and lat of a point which is definitely on the given continental shelf (so we can isolate seamounts disconnected from this)
shelf_point0 = [-51.5, -74.5]

# Dictionary of lon-lat points bounding given region. Will be used to "cut" the continental shelf mask (shelf_mask in utils.py) either north-south or east-west depending on the value of region_edges_flag. The first point and its connected N/S (or E/W) neighbours will be included in the mask, but not the second. The direction of travel is west to east around the coastline.
region_edges = {
    'amundsen_sea': [[-150, -76], [-102, -72]],
    'bellingshausen_sea': [[-102, -72], [-70, -68]],  # not including WAP
    'west_antarctic_peninsula': [[-70, -68], [-55, -62]],
    'larsen': [[-55, -62], [-58, -72]],
    'antarctic_peninsula': [[-70, -68], [-58, -72]],  # both sides, i.e. WAP plus Larsen
    'filchner_ronne': [[-58, -72], [-30, -75]],
    'east_antarctica': [[-30, -75], [170, -71]], # includes Amery
    'amery': [[60, -68], [80, -68]],
    'ross': [[170, -71], [-150, -76]],
}
region_edges_flag = {
    'amundsen_sea': ['NS', 'NS'],
    'bellingshausen_sea': ['NS', 'EW'],
    'west_antarctic_peninsula': ['EW', 'NS'],
    'larsen': ['NS', 'EW'],
    'antarctic_peninsula': ['EW', 'EW'],
    'filchner_ronne': ['EW', 'NS'],
    'east_antarctica': ['NS', 'NS'],
    'amery': ['NS', 'NS'],
    'ross': ['NS', 'NS'],
}
# Names of each region
region_names = {
    'amundsen_sea': 'Amundsen Sea',
    'bellingshausen_sea': 'Bellingshausen Sea',
    'west_antarctic_peninsula': 'West Antarctic Peninsula',
    'larsen': 'Larsen',
    'antarctic_peninsula': 'Antarctic Peninsula',
    'filchner_ronne': 'Filchner-Ronne', 
    'east_antarctica': 'East Antarctica',
    'amery': 'Amery',
    'ross': 'Ross',
}
