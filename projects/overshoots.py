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

    # To do: add in dc123 once ERROR_SINGLE_COPY_UNAVAILABLE goes away
    for suite_id in ['cs495', 'cs568', 'cx209', 'cw988', 'cw989', 'cw990', 'cz826', 'cy837', 'cz834', 'da087', 'cy838', 'cz855', 'cz374', 'cz859', 'cz375', 'cz376', 'cz377', 'cz378', 'da697', 'cz944', 'da800', 'db587', 'db723', 'db731', 'da266', 'db597', 'db733', 'dc324', 'da832', 'db223', 'dc051', 'dc052', 'dc248', 'dc249', 'dc251', 'db956', 'dc032', 'dc130', 'dc163']:
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


# Set the list of experiments, names for the legend, and colours to use for timeseries etc. If separate_stages=True, the ramp down and re-stabilise stages will be separated out from the initial stabilisations.
def set_expt_list (separate_stages=False):

    sim_names = []
    colours = []
    sim_dirs = []

    sim_names.append('preindustrial')
    colours.append('Sienna')
    sim_dirs.append('cs495')

    sim_names.append('ramp up')
    colours.append('Black')
    sim_dirs.append(['cx209', 'cw988', 'cw989', 'cw990'])

    sim_names.append('ramp up static ice')
    colours.append('DarkGrey')
    sim_dirs.append('cz826')

    # Inner function to process stabilisation targets. 7 arguments:
    # gw: string with the global warming level (eg '2K')
    # stabilise, ramp_down, restabilise: strings or lists of strings containing suite(s) corresponding to stabilisation, ramp-down, and restabilisation (at 0K) respectively. They can be None if no such suites exist.
    # colour1, colour2, colour3: colour names corresponding to stabilise, ramp_down, and restabilise. Can be None if the corresponding suite lists are None.
    def add_gw_level (gw, stabilise, ramp_down, restabilise, colour1, colour2, colour3):
        colours.append(colour1)
        if separate_stages or ramp_down is None:
            sim_names.append(gw+' stabilise')
            sim_dirs.append(stabilise)
        elif separate_stages:
            sim_names.append(gw+' ramp down')
            colours.append(colour2)
            sim_dirs.append(ramp_down)
            if restabilise is not None:
                sim_names.append(gw+' restabilise at 0K')
                colours.append(colour3)
                sim_dirs.append(restabilise)
        else:
            if restabilise is not None:
                sim_names.append(gw+' stabilise & ramp down & restabilise')
                sim_dirs.append(stabilise + ramp_down + restabilise)
            else:
                sim_names.append(gw+' stabilise & ramp down')
                sim_dirs.append(stabilise + ramp_down)

    add_gw_level('1.5K', ['cy837', 'cz834', 'da087'], ['da697', 'dc052', 'dc248', 'db956'], None, 'DarkMagenta', 'MediumOrchid', None)
    add_gw_level('2K', ['cy838', 'cz855', 'da266'], ['cz944', 'dc051', 'da800'], 'dc163', 'Blue', 'CornflowerBlue', 'LightBlue')
    add_gw_level('2.5K', ['cz374', 'cz859'], None, None, 'DarkCyan', None, None)
    add_gw_level('3K', ['cz375', 'db587', 'db597'], ['db223', 'dc032', 'dc249'], None, 'DarkGreen', 'DarkSeaGreen', None)
    add_gw_level('4K', ['cz376', 'db723', 'db733'], 'da892', None, 'GoldenRod', 'Gold', None)  # To do: add in dc123 (ramp down) once ERROR_SINGLE_COPY_UNAVAILABLE fixed
    add_gw_level('5K', ['cz377', 'db731', 'dc324'], ['dc251', 'dc130'], None, 'Coral', 'LightSalmon', None)
    add_gw_level('6K', 'cz378', None, None, 'Crimson', None, None)

    return sim_names, colours, sim_dirs


# Plot timeseries by experiment for all variables and regions, in all experiments.
def plot_all_timeseries_by_expt (base_dir='./', regions=['all', 'amundsen_sea', 'bellingshausen_sea', 'larsen', 'filchner_ronne', 'east_antarctica', 'amery', 'ross'], var_names=['massloss', 'bwtemp', 'bwsalt', 'cavity_temp', 'cavity_salt', 'shelf_temp', 'shelf_salt', 'temp_btw_200_700m', 'salt_btw_200_700m', 'drake_passage_transport', 'global_mean_sat'], timeseries_file='timeseries.nc', timeseries_file_u='timeseries_u.nc', timeseries_file_um='timeseries_um.nc', smooth=24, fig_dir=None):

    sim_names, colours, sim_dirs = set_expt_list(separate_stages=True)

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
            # Trim the two timeseries to line up and be the same length
            # Find most restrictive endpoints
            date_start = max(data.time_centered[0], gw_level.time_centered[0])
            date_end = min(data.time_centered[-1], gw_level.time_centered[-1])
            # Inner function to trim
            def trim_timeseries (A):
                t_start = np.argwhere(A.time_centered.data == date_start)
                t_end = np.argwhere(A.time_centered.data == date_end)
                return A.isel(time_centered=slice(t_start, t_end+1))
            data = trim_timeseries(data)
            gw_level = trim_timeseries(gw_level)
            if data.size != gw_level.size:
                raise Exception('Timeseries do not align')
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

    sim_names, colours, sim_dirs = set_expt_list()
    
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
    sim_names, colours, sim_dirs = set_expt_list()
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


# Plot timeseries of the given variable (massloss or cavity_temp) against bottom salinity in the corresponding region for the three big cold cavities (FRIS, Ross, and Amery).
def cold_cavities_by_bwsalt (var_name, base_dir='./', fig_name=None):

    regions = ['filchner_ronne', 'ross', 'amery']
    var_x = [region+'_bwsalt' for region in regions]  # Update to cavity_bwsalt when timeseries ready
    var_y = [region+'_'+var_name for region in regions]
    sim_names, colours, sim_dirs = set_expt_list()
    timeseries_file = 'timeseries.nc'
    smooth = 24
    if var_name == 'massloss':
        units = 'Gt/y'
    elif 'temp' in var_name:
        units = deg_string+'C'

    fig = plt.figure(figsize=(9,5))
    gs = plt.GridSpec(1,3)
    gs.update(left=0.05, right=0.95, bottom=0.3, top=0.9, wspace=0.1)
    for n in range(len(regions)):
        ax = plt.subplot(gs[0,n])
        # Read timeseries from every experiment
        data_x = []
        data_y = []
        labels_plot = []
        colours_plot = []
        for expt, label, colour in zip(sim_dirs, sim_names, colours):
            if isinstance(expt, str):
                expt = [expt]
            num_ens = len(expt)
            labels_plot += [label] + [None]*(num_ens-1)
            colours_plot += [colour]*num_ens
            for suite in expt:
                ds = xr.open_dataset(base_dir+'/'+suite+'/'+timeseries_file)
                data_x.append(moving_average(ds[var_x[n]], smooth))
                data_y.append(moving_average(ds[var_y[n]], smooth))
        # Plot
        for x, y, label, colour in zip(data_x, data_y, labels_plot, colours_plot):
            ax.plot(x, y, '-', color=colour, label=label, linewidth=1)
        ax.grid(linestyle='dotted')
        ax.set_title(region_names[regions[n]], fontsize=14)
        if n==0:
            ax.set_xlabel('Bottom salinity ('+gkg_string+')', fontsize=12)
            ax.set_ylabel(var_name+' ('+units+')', fontsize=12)
    # Legend at bottom
    ax.legend(loc='lower center', bbox_to_anchor=(-0.5, -0.5), fontsize=10, ncol=4)
    finished_plot(fig, fig_name=fig_name)
        

    
        



                    

    
