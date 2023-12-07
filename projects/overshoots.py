# Analysing TerraFIRMA overshoot simulations with UKESM1.1-ice (NEMO 3.6)

from ..timeseries import update_simulation_timeseries
from ..plots import timeseries_by_region


# Call update_simulation_timeseries for the given suite ID
def update_overshoot_timeseries (suite_id, timeseries_file='timeseries.nc', base_dir='./'):

    # Construct list of timeseries types
    regions = ['all', 'amundsen_sea', 'bellingshausen_sea', 'larsen', 'filchner_ronne', 'east_antarctica', 'amery', 'ross']
    var_names = ['massloss', 'bwtemp', 'bwsalt', 'cavity_temp', 'cavity_salt', 'shelf_temp', 'shelf_salt']
    timeseries_types = []
    # All combinations of region and variable
    for region in regions:
        for var in var_names:
            timeseries_types.append(region+'_'+var)

    update_simulation_timeseries(suite_id, timeseries_types, timeseries_file=timeseries_file, sim_dir=base_dir+'/'+suite_id+'/', freq='y', halo=True)


# Plot timeseries by region for all variables in the given suite ID, and show interactively.
def plot_all_timeseries_by_region (suite_id, regions=['all', 'amundsen_sea', 'bellingshausen_sea', 'larsen', 'filchner_ronne', 'east_antarctica', 'amery', 'ross'], var_names=['massloss', 'bwtemp', 'bwsalt', 'cavity_temp', 'cavity_salt', 'shelf_temp', 'shelf_salt'], colours=None, timeseries_file='timeseries.nc', base_dir='./', smooth=0, fig_dir=None):

    while suite_id.endswith('/'):
        suite_id = suite_id[:-1]

    for var in var_names:
        timeseries_by_region(var, base_dir+'/'+suite_id+'/', colours=colours, timeseries_file=timeseries_file, smooth=smooth, fig_name=None if fig_dir is None else (fig_dir+'/'+var+'_'+suite_id+'.png'))

    
