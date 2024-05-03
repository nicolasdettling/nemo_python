import numpy as np
 
# Acceleration due to gravity (m/s^2)
gravity = 9.81
# Density of freshwater (kg/m^3)
rho_fw = 1e3
# Density of ice
rho_ice = 917.
# Seconds per hour
sec_per_hour = 60*60.
# Seconds per day
sec_per_day = 24*sec_per_hour
# Seconds per year
sec_per_year = 365.25*sec_per_day
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
rad2deg = 1./deg2rad
# Radius of Earth (m)
rEarth = 6.371e6
# Degrees formatted nicely in a string
deg_string = r'$^{\circ}$'
# 10^-3 formatted nicely in a string (for absolute salinity)
gkg_string = r'10$^{-3}$'

# Dictionary of bounds on different regions
# lon_min, lon_max, lat_min, lat_max
region_bounds = {
    'bear_ridge_S': [-110.35, -110.35, -74.35, -73.8791],
    'bear_ridge_N': [-112.05, -109.05, -73.8791, -72.7314],
    'pine_island_bay': [-104, -100.5, -75.2, -74.2],
    'dotson_bay': [-114, -110.5, -74.3, -73.5],
    'amundsen_west_shelf_break': [-115, -112, -72, -71]
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
# Drake Passage transect
drake_passage_lon0 = -68  # Corresponds to i=219 in eORCA1
drake_passage_lat_bounds = [-67.3, -52.6]  # Corresponds to j=79:109 in eORCA1
# Lon and lat of a point which is definitely in the Ross Gyre
ross_gyre_point0 = [-160, -70]

# Dictionary of x,y, coordinates in AntArc configuration delineating roughly the Weddell Sea and Ross sea convection regions 
# (should generalize to lat lons at some point)
weddell_convect = {'x':slice(930,1150), 'y':slice(295,360)}
ross_convect    = {'x':slice(450,580),  'y':slice(220,300)}

# Dictionary of lons and lats describing key waypoints for transect paths.
transect_amundsen = {
    'shelf_west':[[-115.5, -115, -116, -120.6], [-74.3, -73.5, -72.5, -70.78]],
    'shelf_mid' :[[-112.6,-114.6], [-74.38,-70.48]],
    'shelf_east':[[-100.6,-104,-106,-106,-104,-102.6,-101.8], [-75.16,-74.68,-74.2,-73.3,-72.4,-71.2,-69.1]],
    'shelf_edge':[[-137,-135,-132,-131,-127.6,-127,-120.6,-118.6,-114,-102,-97,-93], [-73.7,-73.6,-73.3,-73,-72.7,-72.3,-72.5,-71.7,-71.2,-70.9,-70,-70.3]]
} # transect locations chosen to cross most-sampled regions

# Dictionary of lon-lat points bounding given region. Will be used to "cut" the continental shelf mask (build_shelf_mask in utils.py) either north-south or east-west depending on the value of region_edges_flag. The first point and its connected N/S (or E/W) neighbours will be included in the mask, but not the second. The direction of travel is west to east around the coastline.
region_edges = {
    'amundsen_sea': [[-157.5, -76.5], [-102.75, -72.5]],
    'bellingshausen_sea': [[-102.75, -72.5], [-57.25, -62]],
    'west_antarctica': [[-157.5, -76.5], [-57.25, -62]], # Amundsen and Bellingshausen
    'larsen': [[-57.25, -62], [-57, -71.5]],
    'filchner_ronne': [[-57, -71.5], [-26, -75.5]],
    'east_antarctica': [[-26, -75.5], [169.5, -71]], # includes Amery
    'amery': [[66.5, -67.5], [79.5, -68]],
    'ross': [[169.5, -71], [-157.5, -76.5]],
}
region_edges_flag = {
    'amundsen_sea': ['NS', 'NS'],
    'bellingshausen_sea': ['NS', 'NS'],
    'west_antarctica': ['NS', 'NS'],
    'larsen': ['NS', 'EW'],
    'filchner_ronne': ['EW', 'NS'],
    'east_antarctica': ['NS', 'NS'],
    'amery': ['NS', 'NS'],
    'ross': ['NS', 'NS'],
}
# Dictionary of lon-lat points which are definitely in the given region. The region is then defined by connectivity to that point (eg selecting specific ice shelf cavities in single_cavity_mask in utils.py).
region_points = {
    'abbot': [-95, -73],
    'brunt': [-20, -75],
}
# Names of each region
region_names = {
    'amundsen_sea': 'Amundsen Sea',
    'bellingshausen_sea': 'Bellingshausen Sea',
    'west_antarctica': 'West Antarctica',
    'larsen': 'Larsen',
    'filchner_ronne': 'Filchner-Ronne', 
    'east_antarctica': 'East Antarctica',
    'amery': 'Amery',
    'ross': 'Ross',
    'abbot': 'Abbot Ice Shelf',
    'brunt': 'Brunt and Riiser-Larsen Ice Shelves',
    'pine_island_bay': 'Pine Island Bay',
    'dotson_bay': 'front of Dotson',
    'amundsen_west_shelf_break': 'Western Amundsen Sea shelf break',
    'all': 'Antarctic',
}

# Default colours to use for plotting lines
line_colours = ['black', 'Crimson', 'blue', 'DarkMagenta', 'DimGrey', 'DarkGreen', 'DeepPink', 'DeepSkyBlue']
land_colour     = '#9999a3'
iceshelf_colour = '#d4d5da' 

