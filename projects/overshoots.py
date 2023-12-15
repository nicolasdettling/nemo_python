# Analysing TerraFIRMA overshoot simulations with UKESM1.1-ice (NEMO 3.6)

import xarray as xr
import matplotlib.pyplot as plt

from ..timeseries import update_simulation_timeseries, update_simulation_timeseries_um
from ..plots import timeseries_by_region, timeseries_by_expt, finished_plot
from ..utils import moving_average
from ..constants import line_colours


# Call update_simulation_timeseries for the given suite ID
def update_overshoot_timeseries (suite_id, base_dir='./', domain_cfg='/gws/nopw/j04/terrafirma/kaight/input_data/grids/domcfg_eORCA1v2.2x.nc'):

    # Construct list of timeseries types for T-grid
    regions = ['all', 'amundsen_sea', 'bellingshausen_sea', 'larsen', 'filchner_ronne', 'east_antarctica', 'amery', 'ross']
    var_names = ['massloss', 'bwtemp', 'bwsalt', 'cavity_temp', 'cavity_salt', 'shelf_temp', 'shelf_salt']
    # A couple extra things for a few regions
    regions_btw = ['amundsen_sea', 'bellingshausen_sea']
    var_names_btw = ['temp', 'salt']
    depth_btw = '_btw_200_700m'
    timeseries_types = []
    # All combinations of region and variable
    for region in regions:
        for var in var_names:
            timeseries_types.append(region+'_'+var)
        if region in regions_btw:
            for var in var_names_btw:
                timeseries_types.append(region+'_'+var+depth_btw)

    update_simulation_timeseries(suite_id, timeseries_types, timeseries_file='timeseries.nc', sim_dir=base_dir+'/'+suite_id+'/', freq='m', halo=True, gtype='T')

    # Now for u-grid
    update_simulation_timeseries(suite_id, ['drake_passage_transport'], timeseries_file='timeseries_u.nc', sim_dir=base_dir+'/'+suite_id+'/', freq='m', halo=True, gtype='U', domain_cfg=domain_cfg)

    # Now for UM
    update_simulation_timeseries_um(suite_id, ['global_mean_sat'], timeseries_file='timeseries_um.nc', sim_dir=base_dir+'/'+suite_id+'/', stream='p5')


# Call for all simulations (add to the list of suite IDs as needed)
def update_overshoot_timeseries_all (base_dir='./', domain_cfg='/gws/nopw/j04/terrafirma/kaight/input_data/grids/domcfg_eORCA1v2.2x.nc'):

    # To add when ERROR_SINGLE_COPY_UNAVAILABLE is resolved (and these suites are re-pulled from MASS): cz855
    for suite_id in ['cs495', 'cs568', 'cx209', 'cw988', 'cw989', 'cw990', 'cz826', 'cy837', 'cz834', 'da087', 'cy838', 'cz374', 'cz859', 'cz375', 'cz376', 'cz377', 'cz378', 'da697', 'cz944', 'da800']:
        update_overshoot_timeseries(suite_id, base_dir=base_dir, domain_cfg=domain_cfg)


# Plot timeseries by region for all variables in one simulation.
def plot_all_timeseries_by_region (suite_id, regions=['all', 'amundsen_sea', 'bellingshausen_sea', 'larsen', 'filchner_ronne', 'east_antarctica', 'amery', 'ross'], var_names=['massloss', 'bwtemp', 'bwsalt', 'cavity_temp', 'cavity_salt', 'shelf_temp', 'shelf_salt', 'temp_btw_200_700m', 'salt_btw_200_700m'], colours=None, timeseries_file='timeseries.nc', base_dir='./', smooth=24, fig_dir=None):

    while suite_id.endswith('/'):
        suite_id = suite_id[:-1]

    for var in var_names:
        # Special treatment of 200-700m variables as these are only valid for certain regions
        if var.endswith('200_700m'):
            regions_use = []
            for region in ['amundsen_sea', 'bellingshausen_sea']:
                if region in regions:
                    regions_use.append(region)
        else:
            regions_use = regions
        if len(regions) == 0:
            continue                    
        timeseries_by_region(var, base_dir+'/'+suite_id+'/', regions=regions_use, colours=colours, timeseries_file=timeseries_file, smooth=smooth, fig_name=None if fig_dir is None else (fig_dir+'/'+var+'_'+suite_id+'.png'))


# Plot timeseries by experiment for all variables and regions, in all experiments.
def plot_all_timeseries_by_expt (base_dir='./', regions=['all', 'amundsen_sea', 'bellingshausen_sea', 'larsen', 'filchner_ronne', 'east_antarctica', 'amery', 'ross'], var_names=['massloss', 'bwtemp', 'bwsalt', 'cavity_temp', 'cavity_salt', 'shelf_temp', 'shelf_salt', 'temp_btw_200_700m', 'salt_btw_200_700m', 'drake_passage_transport', 'global_mean_sat'], timeseries_file='timeseries.nc', timeseries_file_u='timeseries_u.nc', timeseries_file_um='timeseries_um.nc', smooth=24, fig_dir=None):

    sim_names = ['ramp up', 'ramp up static ice', 'stabilise 1.5 K', 'stabilise 2K', 'stabilise 2.5K', 'stabilise 3K', 'stabilise 4K', 'stabilise 5K', 'stabilise 6K', 'ramp down 1.5K', 'ramp down 2K']
    colours = ['Black', 'DarkGrey', 'DarkMagenta', 'Blue', 'DarkCyan', 'DarkGreen', 'GoldenRod', 'Coral', 'Crimson', 'MediumOrchid', 'CornflowerBlue']
    sim_dirs = [['cx209', 'cw988', 'cw989', 'cw990'],  # ramp up
                'cz826', # ramp up static ice
                ['cy837', 'cz834', 'da087'], # stabilise 1.5K
                'cy838', # stabilise 2K - todo add back in cz855
                ['cz374', 'cz859'], # stabilise 2.5K
                'cz375', # stabilise 3K
                'cz376', # stabilise 4K
                'cz377', # stabilise 5K
                'cz378', # stabilise 6K
                'da697', # ramp down 1.5 K
                ['cz944', 'da800']] # ramp down 2K

    # Now construct master list of variables
    var_names_all = []
    for region in regions:
        for var in var_names:
            if var in ['drake_passage_transport', 'global_mean_sat']:
                # Special cases with no region
                if var not in var_names_all:
                    var_names_all.append(var)
            elif var.endswith('200_700m'):
                # Special cases where only some regions defined
                if region in ['amundsen_sea', 'bellingshausen_sea']:
                    var_names_all.append(region+'_'+var)
            else:
                # Every combination of region and variable
                var_names_all.append(region+'_'+var)

    for var in var_names_all:
        if var == 'drake_passage_transport':
            fname = timeseries_file_u
        elif var == 'global_mean_sat':
            fname = timeseries_file_um
        else:
            fname = timeseries_file
        timeseries_by_expt(var, sim_dirs, sim_names=sim_names, colours=colours, timeseries_file=fname, smooth=smooth, linewidth=1, fig_name=None if fig_dir is None else (fig_dir+'/'+var+'_master.png'))


# Plot the timeseries of one or more experiments/ensembles (expts can be a string, a list of strings, or a list of lists of string) and one variable against global warming level (relative to preindustrial mean in the given PI suite). 
def plot_by_gw_level (expts, var_name, pi_suite='cs568', base_dir='./', fig_name=None, timeseries_file='timeseries.nc', timeseries_file_um='timeseries_um.nc', smooth=24, labels=None, colours=None, linewidth=1):

    if isinstance(expts, str):
        # Just one suite - generalise
        expts = [expts]
    num_expt = len(expts)
    if colours is None:
        if num_expt <= len(line_colours):
            colours = line_colours[:num_expt]
    if labels is None:
        labels = [None]*num_expt            

    # Get baseline global mean SAT
    ds_pi = xr.open_dataset(base_dir+'/'+pi_suite+'/'+timeseries_file_um)
    baseline_temp = ds_pi['global_mean_sat'].mean()
    ds_pi.close()
    
    gw_levels = []
    datas = []
    labels_plot = []
    colours_plot = []
    for expt, label, colour in zip(expts, labels, colours):
        if isinstance(expt, str):
            # Generalise to ensemble of 1
            expt = [expt]
        num_ens = len(expt)
        labels_plot += [label] + [None]*(num_ens-1)
        colours_plot += [colour]*num_ens
        for suite in expt:
            # Read global mean SAT in this suite and convert to GW level
            ds_um = xr.open_dataset(base_dir+'/'+suite+'/'+timeseries_file_um)
            gw_level = ds_um['global_mean_sat'] - baseline_temp
            ds_um.close()
            # Smooth it in time
            gw_level = moving_average(gw_level, smooth)
            # Finally read and smooth the variable
            ds = xr.open_dataset(base_dir+'/'+suite+'/'+timeseries_file)
            data = moving_average(ds[var_name], smooth)
            ds.close()
            # Trim the two timeseries to be the same length, if needed
            new_size = min(gw_level.size, data.size)
            if gw_level.size > new_size:
                gw_level = gw_level.isel(time_centered=slice(0,new_size))
            if data.size > new_size:
                data = data.isel(time_centered=slice(0,new_size))
            gw_levels.append(gw_level)
            datas.append(data)

    # Plot
    if labels is None:
        figsize = (6,4)
    else:
        # Need a bigger plot to make room for a legend
        figsize = (8,5)
    fig, ax = plt.subplots(figsize=figsize)
    for gw_level, data, colour, label in zip(gw_levels, datas, colours_plot, labels_plot):
        ax.plot(gw_level, data, '-', color=colour, label=label, linewidth=linewidth)
    ax.grid(linestyle='dotted')
    ax.set_title(datas[0].long_name, fontsize=16)
    ax.set_ylabel(datas[0].units, fontsize=14)
    ax.set_xlabel('Global warming relative to preindustrial (K)', fontsize=14)
    if labels is not None:
        # Make legend
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width*0.8, box.height])
        ax.legend(loc='center left', bbox_to_anchor=(1,0.5))
    finished_plot(fig, fig_name=fig_name)
    
                    

    
