# Analysing TerraFIRMA overshoot simulations with UKESM1.1-ice (NEMO 3.6)

import xarray as xr
import matplotlib.pyplot as plt

from ..timeseries import update_simulation_timeseries, update_simulation_timeseries_um
from ..plots import timeseries_by_region, timeseries_by_expt, finished_plot
from ..utils import moving_average
from ..constants import line_colours, region_names, deg_string, gkg_string


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
    timeseries_types += ['west_antarctica_bwtemp', 'west_antarctica_bwsalt', 'west_antarctica_massloss']

    update_simulation_timeseries(suite_id, timeseries_types, timeseries_file='timeseries.nc', sim_dir=base_dir+'/'+suite_id+'/', freq='m', halo=True, gtype='T')

    # Now for u-grid
    update_simulation_timeseries(suite_id, ['drake_passage_transport'], timeseries_file='timeseries_u.nc', sim_dir=base_dir+'/'+suite_id+'/', freq='m', halo=True, gtype='U', domain_cfg=domain_cfg)

    # Now for UM
    update_simulation_timeseries_um(suite_id, ['global_mean_sat'], timeseries_file='timeseries_um.nc', sim_dir=base_dir+'/'+suite_id+'/', stream='p5')


# Call for all simulations (add to the list of suite IDs as needed)
def update_overshoot_timeseries_all (base_dir='./', domain_cfg='/gws/nopw/j04/terrafirma/kaight/input_data/grids/domcfg_eORCA1v2.2x.nc'):

    for suite_id in ['cs495', 'cs568', 'cx209', 'cw988', 'cw989', 'cw990', 'cz826', 'cy837', 'cz834', 'da087', 'cy838', 'cz855', 'cz374', 'cz859', 'cz375', 'cz376', 'cz377', 'cz378', 'da697', 'cz944', 'da800', 'db587', 'db723', 'db731', 'da266', 'db597', 'db733', 'dc324', 'da832', 'db223', 'dc051', 'dc052', 'dc248', 'dc249', 'dc251', 'db956', 'dc032', 'dc123', 'dc130', 'dc163']:
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

    sim_names = ['preindustrial', 'ramp up', 'ramp up static ice', 'stabilise 1.5 K', 'stabilise 2K', 'stabilise 2.5K', 'stabilise 3K', 'stabilise 4K', 'stabilise 5K', 'stabilise 6K', 'ramp down 1.5K', 'ramp down 2K', 'ramp down 3K', 'ramp down 4K', 'ramp down 5K', 're-stabilise preindustrial']
    colours = ['Sienna', 'Black', 'DarkGrey', 'DarkMagenta', 'Blue', 'DarkCyan', 'DarkGreen', 'GoldenRod', 'Coral', 'Crimson', 'MediumOrchid', 'CornflowerBlue', 'DarkSeaGreen', 'Gold', 'LightSalmon', 'Pink', 'Peru']
    sim_dirs = ['cs495', # preindustrial
                ['cx209', 'cw988', 'cw989', 'cw990'],  # ramp up
                'cz826', # ramp up static ice
                ['cy837', 'cz834', 'da087'], # stabilise 1.5K
                ['cy838', 'cz855', 'da266'], # stabilise 2K
                ['cz374', 'cz859'], # stabilise 2.5K
                ['cz375', 'db587', 'db597'], # stabilise 3K
                ['cz376', 'db723', 'db733'], # stabilise 4K
                ['cz377', 'db731', 'dc324'], # stabilise 5K
                'cz378', # stabilise 6K
                ['da697', 'dc052', 'dc248', 'db956'], # ramp down 1.5 K
                ['cz944', 'dc051', 'da800'], # ramp down 2K
                ['db223', 'dc032', 'dc249'], # ramp down 3K
                ['da892', 'dc123'], # ramp down 4K
                ['dc251', 'dc130'], # ramp down 5K
                'dc163']  # re-stabilise preindustrial

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
def plot_by_gw_level (expts, var_name, pi_suite='cs568', base_dir='./', fig_name=None, timeseries_file='timeseries.nc', timeseries_file_um='timeseries_um.nc', smooth=24, labels=None, colours=None, linewidth=1, title=None, units=None, ax=None):

    new_ax = ax is None

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
    if new_ax:
        if labels is None:
            figsize = (6,4)
        else:
            # Need a bigger plot to make room for a legend
            figsize = (8,5)
        fig, ax = plt.subplots(figsize=figsize)
    for gw_level, data, colour, label in zip(gw_levels, datas, colours_plot, labels_plot):
        ax.plot(gw_level, data, '-', color=colour, label=label, linewidth=linewidth)
    ax.grid(linestyle='dotted')
    if title is None:
        title = datas[0].long_name
    if units is None:
        units = datas[0].units
    ax.set_title(title, fontsize=16)
    ax.set_ylabel(units, fontsize=14)
    ax.set_xlabel('Global warming relative to preindustrial (K)', fontsize=14)
    if labels is not None and new_ax:
        # Make legend
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width*0.8, box.height])
        ax.legend(loc='center left', bbox_to_anchor=(1,0.5))
    if new_ax:
        finished_plot(fig, fig_name=fig_name)


# Plot timeseries by global warming level for all variables in all experiments.
def plot_all_by_gw_level (base_dir='./', regions=['all', 'amundsen_sea', 'bellingshausen_sea', 'larsen', 'filchner_ronne', 'east_antarctica', 'amery', 'ross'], var_names=['massloss', 'bwtemp', 'bwsalt', 'cavity_temp', 'cavity_salt', 'shelf_temp', 'shelf_salt', 'temp_btw_200_700m', 'salt_btw_200_700m', 'drake_passage_transport'], timeseries_file='timeseries.nc', timeseries_file_u='timeseries_u.nc', timeseries_file_um='timeseries_um.nc', smooth=24, fig_dir=None, pi_suite='cs568'):

    # A bit different to normal timeseries above - plot ramp downs in same colour as stabilised, as they'll be clearly differentiable on the plot.
    sim_names = ['preindustrial', 'ramp up', 'ramp up static ice', '1.5 K stabilise & ramp down', '2K stabilise & ramp down', '2.5K stabilise', '3K stabilise & ramp down', '4K stabilise & ramp down', '5K stabilise & ramp down', '6K stabilise']
    colours = ['Sienna', 'Black', 'DarkGrey', 'DarkMagenta', 'Blue', 'DarkCyan', 'DarkGreen', 'GoldenRod', 'Coral', 'Crimson']
    sim_dirs = ['cs495', # preindustrial
                ['cx209', 'cw988', 'cw989', 'cw990'],  # ramp up
                'cz826', # ramp up static ice
                ['cy837', 'cz834', 'da087', 'da697', 'dc052', 'dc248', 'db956'], # stabilise 1.5K & ramp down
                ['cy838', 'cz855', 'da266', 'cz944', 'dc051', 'da800', 'dc163'], # stabilise 2K & ramp down & re-stabilise preinudstrial
                ['cz374', 'cz859'], # stabilise 2.5K
                ['cz375', 'db587', 'db597', 'db223', 'dc032', 'dc249'], # stabilise 3K & ramp down
                ['cz376', 'db723', 'db733', 'da892', 'dc123'], # stabilise 4K
                ['cz377', 'db731', 'dc324', 'dc251', 'dc130'], # stabilise 5K
                'cz378'] # stabilise 6K

    # Now construct master list of variables as above - modularise this if I do it a third time!
    var_names_all = []
    for region in regions:
        for var in var_names:
            if var == 'drake_passage_transport':
                # Special case with no region
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
        else:
            fname = timeseries_file
        plot_by_gw_level(sim_dirs, var, pi_suite=pi_suite, base_dir=base_dir, timeseries_file=fname, timeseries_file_um=timeseries_file_um, smooth=smooth, labels=sim_names, colours=colours, linewidth=1, fig_name=None if fig_dir is None else (fig_dir+'/'+var+'_gw.png'))


# Synthesise all this into a set of 5-panel plots for 3 different variables showing timeseries by GW level.
def gw_level_panel_plots (base_dir='./', pi_suite='cs568', fig_dir=None):

    regions = ['ross', 'filchner_ronne', 'west_antarctica', 'east_antarctica', 'all']
    var_names = ['bwtemp', 'bwsalt', 'massloss']
    var_titles = ['Bottom temperature on continental shelf and cavities', 'Bottom salinity on continental shelf and cavities', 'Basal mass loss']
    units = [deg_string+'C', gkg_string, 'Gt/y']
    sim_names = ['preindustrial', 'ramp up', 'ramp up static ice', '1.5 K stabilise & ramp down', '2K stabilise & ramp down', '2.5K stabilise', '3K stabilise & ramp down', '4K stabilise & ramp down', '5K stabilise & ramp down', '6K stabilise']
    colours = ['Sienna', 'Black', 'DarkGrey', 'DarkMagenta', 'Blue', 'DarkCyan', 'DarkGreen', 'GoldenRod', 'Coral', 'Crimson']
    sim_dirs = ['cs495', # preindustrial
                ['cx209', 'cw988', 'cw989', 'cw990'],  # ramp up
                'cz826', # ramp up static ice
                ['cy837', 'cz834', 'da087', 'da697', 'dc052', 'dc248', 'db956'], # stabilise 1.5K & ramp down
                ['cy838', 'cz855', 'da266', 'cz944', 'dc051', 'da800', 'dc163'], # stabilise 2K & ramp down & re-stabilise preinudstrial
                ['cz374', 'cz859'], # stabilise 2.5K
                ['cz375', 'db587', 'db597', 'db223', 'dc032', 'dc249'], # stabilise 3K & ramp down
                ['cz376', 'db723', 'db733', 'da892', 'dc123'], # stabilise 4K
                ['cz377', 'db731', 'dc324', 'dc251', 'dc130'], # stabilise 5K
                'cz378'] # stabilise 6K
    timeseries_file = 'timeseries.nc'
    smooth = 24

    for v in range(len(var_names)):
        fig = plt.figure(figsize=(9,10))
        gs = plt.GridSpec(3,2)
        gs.update(left=0.09, right=0.98, bottom=0.07, top=0.9, hspace=0.3, wspace=0.15)
        for n in range(len(regions)):
            ax = plt.subplot(gs[n//2,n%2])
            plot_by_gw_level(sim_dirs, regions[n]+'_'+var_names[v], pi_suite=pi_suite, base_dir=base_dir, timeseries_file=timeseries_file, smooth=smooth, labels=sim_names, colours=colours, linewidth=0.5, ax=ax)
            if n == len(regions)-1:
                title = 'Antarctica mean'
            else:
                title = region_names[regions[n]]
            ax.set_title(title, fontsize=14)
            if n%2 == 0:
                ax.set_ylabel(units[v], fontsize=12)
            else:
                ax.set_ylabel('')
            if n//2 == 2:
                ax.set_xlabel('Global warming relative to preindustrial (K)', fontsize=12)
            else:
                ax.set_xlabel('')
        plt.suptitle(var_titles[v], fontsize=16)
        # Make legend in the last box
        ax.legend(loc='center left', bbox_to_anchor=(1.2,0.5), fontsize=11)
        finished_plot(fig, fig_name=None if fig_dir is None else fig_dir+'/'+var_names[v]+'_gw_panels.png')

    
        



                    

    
