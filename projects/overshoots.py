# Analysing TerraFIRMA overshoot simulations with UKESM1.1-ice (NEMO 3.6)

import xarray as xr
import matplotlib.pyplot as plt
import os
import numpy as np

from ..timeseries import update_simulation_timeseries, update_simulation_timeseries_um
from ..plots import timeseries_by_region, timeseries_by_expt, finished_plot, timeseries_plot, circumpolar_plot
from ..utils import moving_average, region_mask
from ..constants import line_colours, region_names, deg_string, gkg_string, months_per_year
from ..file_io import read_schmidtko, read_woa
from ..interpolation import interp_latlon_cf
from ..diagnostics import barotropic_streamfunction

# Global dictionaries of suites - update these as more suites become available!

# Dictionary of which suites correspond to which scenario
suites_by_scenario = {'piControl' : ['cs495'],
                      'piControl_static_ice' : ['cs568'],
                      'ramp_up' : ['cx209', 'cw988', 'cw989', 'cw990'],
                      'ramp_up_static_ice': ['cz826'],
                      '1.5K_stabilise': ['cy837','cz834','da087'],
                      '1.5K_ramp_down': ['da697', 'dc052', 'dc248'],
                      '2K_stabilise': ['cy838','cz855','da266'],
                      '2K_ramp_down': ['cz944', 'dc051', 'da800'],
                      '2K_restabilise' : ['dc163'],
                      '2.5K_stabilise' : ['cz374','cz859'],
                      '3K_stabilise' : ['cz375','db587','db597'],
                      '3K_ramp_down' : ['db223', 'dc032', 'dc249'],
                      '4K_stabilise' : ['cz376','db723','db733'],
                      '4K_ramp_down' : ['da892'], # To do: add dc123 once single_copy_unavailable errors gone
                      '5K_stabilise' : ['cz377','db731','dc324'],
                      '5K_ramp_down' : ['dc251', 'dc130'],
                      '6K_stabilise' : ['cz378']}
# Dictionary of which suites branch from which. None means it's a ramp-up suite (so branched from a piControl run, but we don't care about that for the purposes of integrated GW)
suites_branched = {'cx209':None, 'cw988':None, 'cw989':None, 'cw990':None, 'cz826':None, 'cy837':'cx209', 'cy838':'cx209', 'cz374':'cx209', 'cz375':'cx209', 'cz376':'cx209', 'cz377':'cx209', 'cz378':'cx209', 'cz834':'cw988', 'cz855':'cw988', 'cz859':'cw988', 'db587':'cw988', 'db723':'cw988', 'db731':'cw988', 'da087':'cw989', 'da266':'cw989', 'db597':'cw989', 'db733':'cw989', 'dc324':'cw989', 'cz944':'cy838', 'da800':'cy838', 'da697':'cy837', 'da892':'cz376', 'db223':'cz375', 'dc051':'cy838', 'dc052':'cy837', 'dc248':'cy837', 'dc249':'cz375', 'dc251':'cz377', 'dc032':'cz375', 'dc123':'cz376', 'dc130':'cz377', 'dc163':'cz944'}

# End global vars


# Call update_simulation_timeseries for the given suite ID
def update_overshoot_timeseries (suite_id, base_dir='./', domain_cfg='/gws/nopw/j04/terrafirma/kaight/input_data/grids/domcfg_eORCA1v2.2x.nc'):

    # Construct list of timeseries types for T-grid
    regions = ['all', 'amundsen_sea', 'bellingshausen_sea', 'larsen', 'filchner_ronne', 'east_antarctica', 'amery', 'ross']
    var_names = ['massloss', 'draft', 'bwtemp', 'bwsalt', 'cavity_temp', 'cavity_salt', 'shelf_temp', 'shelf_salt']
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
    timeseries_types += ['west_antarctica_bwtemp', 'west_antarctica_bwsalt', 'west_antarctica_massloss', 'filchner_ronne_shelf_bwsalt', 'ross_shelf_bwsalt', 'amery_bwsalt', 'ross_gyre_eastern_extent']

    update_simulation_timeseries(suite_id, timeseries_types, timeseries_file='timeseries.nc', sim_dir=base_dir+'/'+suite_id+'/', freq='m', halo=True, gtype='T')

    # Now for u-grid
    update_simulation_timeseries(suite_id, ['drake_passage_transport'], timeseries_file='timeseries_u.nc', sim_dir=base_dir+'/'+suite_id+'/', freq='m', halo=True, gtype='U', domain_cfg=domain_cfg)

    # Now for UM
    update_simulation_timeseries_um(suite_id, ['global_mean_sat'], timeseries_file='timeseries_um.nc', sim_dir=base_dir+'/'+suite_id+'/', stream='p5')


# Call for all simulations (add to the list of suite IDs as needed)
def update_overshoot_timeseries_all (base_dir='./', domain_cfg='/gws/nopw/j04/terrafirma/kaight/input_data/grids/domcfg_eORCA1v2.2x.nc'):

    for scenario in suites_by_scenario:
        for suite_id in suites_by_scenario[scenario]:
            update_overshoot_timeseries(suite_id, base_dir=base_dir, domain_cfg=domain_cfg)


# Calculate a new timeseries variable(s) for the given suite, and then concatenate it with the existing corresponding timeseries file. After running this, add the variable(s) to the list in update_overshoot_timeseries.
def new_timeseries_var (suite_id, timeseries_types, timeseries_file_new, timeseries_file='timeseries.nc', base_dir='./', domain_cfg='/gws/nopw/j04/terrafirma/kaight/input_data/grids/domcfg_eORCA1v2.2x.nc'):

    # Calculate a new file
    if timeseries_file == 'timeseries_um.nc':
        # Atmospheric variable
        update_simulation_timeseries_um(suite_id, timeseries_types, timeseries_file=timeseries_file_new, sim_dir=base_dir+'/'+suite_id+'/', stream='p5')
    else:
        # Ocean variable
        if timeseries_file == 'timeseries.nc':
            gtype = 'T'
        elif timeseries_file == 'timeseries_u.nc':
            gtype = 'U'
        else:
            raise Exception('unknown gtype - add another case to the code?')
        update_simulation_timeseries(suite_id, timeseries_types, timeseries_file=timeseries_file_new, sim_dir=base_dir+'/'+suite_id+'/', freq='m', halo=True, gtype=gtype, domain_cfg=domain_cfg)

    # Now concatenate with existing file
    print('Merging with '+timeseries_file)
    os.rename(suite_id+'/'+timeseries_file, suite_id+'/tmp_'+timeseries_file)
    ds = xr.open_mfdataset([suite_id+'/tmp_'+timeseries_file, suite_id+'/'+timeseries_file_new])
    ds.to_netcdf(suite_id+'/'+timeseries_file)
    os.remove(suite_id+'/tmp_'+timeseries_file)
    os.remove(suite_id+'/'+timeseries_file_new)
    ds.close()    


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


# Set the list of experiments, names for the legend, and colours to use for timeseries etc. If separate_stages=True, the ramp down and re-stabilise stages will be separated out from the initial stabilisations. If only_up=True, only the ramp-up and stabilise suites will be considered. If only_down=True, only the ramp-down and re-stabilise suites will be considered.
def set_expt_list (separate_stages=False, only_up=False, only_down=False):

    sim_names = []
    colours = []
    sim_dirs = []

    # Inner function to add a new suite or ensemble to the lists
    def add_ens (name, colour, dirs):
        sim_names.append(name)
        colours.append(colour)
        sim_dirs.append(dirs)
        
    # Inner function to combine two strings and/or lists into lists
    def combine_ens (ens1, ens2):
        if isinstance(ens1, str):
            all_ens = [ens1]
        else:
            all_ens = ens1
        if isinstance(ens2, str):
            all_ens += [ens2]
        else:
            all_ens += ens2
        return all_ens    

    # Inner function to process stabilisation targets. 7 arguments:
    # gw: string with the global warming level (eg '2K')
    # stabilise, ramp_down, restabilise: strings or lists of strings containing suite(s) corresponding to stabilisation, ramp-down, and restabilisation (at 0K) respectively. They can be None if no such suites exist.
    # colour1, colour2, colour3: colour names corresponding to stabilise, ramp_down, and restabilise. Can be None if the corresponding suite lists are None.
    def add_gw_level (gw, stabilise, ramp_down, restabilise, colour1, colour2, colour3):
        if separate_stages:
            # Keep the distinct colours regardless of values of only_up, only_down
            if stabilise is not None and not only_down:
                add_ens(gw+' stabilise', colour1, stabilise)
            if ramp_down is not None and not only_up:
                add_ens(gw+' ramp down', colour2, ramp_down)
            if restabilise is not None and not only_up:
                add_ens(gw+' re-stabilise', colour3, restabilise)
        else:
            # Only use colour1
            if only_up and stabilise is not None:
                add_ens(gw+' stabilise', colour1, stabilise)
            if only_down and ramp_down is not None:
                if restabilise is None:
                    add_ens(gw+' ramp down', colour1, ramp_down)
                else:
                    add_ens(gw+' ramp down & re-stabilise', colour1, combine_ens(ramp_down, restabilise))
            if not only_up and not only_down:
                name = gw+' stabilise'
                ens = stabilise
                if ramp_down is not None:
                    ens = combine_ens(stabilise, ramp_down)
                    name += ' & ramp_down'
                    if restabilise is not None:
                        ens = combine_ens(ens, restabilise)
                        name += ' & re-stabilise'
                add_ens(name, colour1, ens)

    # Now add the suites
    if not only_down:
        add_ens('ramp up', 'Black', suites_by_scenario['ramp_up'])
        add_ens('ramp up static ice', 'DarkGrey', suites_by_scenario['ramp_up_static_ice'])
    add_gw_level('1.5K', suites_by_scenario['1.5K_stabilise'], suites_by_scenario['1.5K_ramp_down'], None, 'DarkMagenta', 'MediumOrchid', None)
    add_gw_level('2K', suites_by_scenario['2K_stabilise'], suites_by_scenario['2K_ramp_down'], suites_by_scenario['2K_restabilise'], 'Blue', 'CornflowerBlue', 'LightBlue')
    add_gw_level('2.5K', suites_by_scenario['2.5K_stabilise'], None, None, 'DarkCyan', None, None)
    add_gw_level('3K', suites_by_scenario['3K_stabilise'], suites_by_scenario['3K_ramp_down'], None, 'DarkGreen', 'DarkSeaGreen', None)
    add_gw_level('4K', suites_by_scenario['4K_stabilise'], suites_by_scenario['4K_ramp_down'], None, 'GoldenRod', 'Gold', None)
    add_gw_level('5K', suites_by_scenario['5K_stabilise'], suites_by_scenario['5K_ramp_down'], None, 'Chocolate', 'LightSalmon', None)
    add_gw_level('6K', suites_by_scenario['6K_stabilise'], None, None, 'Crimson', None, None)

    return sim_names, colours, sim_dirs


# Plot timeseries by experiment for all variables and regions, in all experiments.
def plot_all_timeseries_by_expt (base_dir='./', regions=['all', 'amundsen_sea', 'bellingshausen_sea', 'larsen', 'filchner_ronne', 'east_antarctica', 'amery', 'ross'], var_names=['massloss', 'draft', 'bwtemp', 'bwsalt', 'cavity_temp', 'cavity_salt', 'shelf_temp', 'shelf_salt', 'temp_btw_200_700m', 'salt_btw_200_700m', 'drake_passage_transport', 'global_mean_sat'], timeseries_file='timeseries.nc', timeseries_file_u='timeseries_u.nc', timeseries_file_um='timeseries_um.nc', smooth=24, fig_dir=None):

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


# Calculate the integrated global warming relative to preindustrial mean, in Kelvin-years, for the given suite (starting from the beginning of the relevant ramp-up simulation). Returns a timeseries over the given experiment, with the first value being the sum of all branched-from experiments before then.
def integrated_gw (suite, pi_suite='cs568', timeseries_file_um='timeseries_um.nc', base_dir='./'):

    # Inner function to read global mean SAT
    def global_mean_sat (suite):
        ds = xr.open_dataset(base_dir+'/'+suite+'/'+timeseries_file_um)
        sat = ds['global_mean_sat']
        ds.close()
        return sat

    # Get baseline global mean SAT
    baseline_temp = global_mean_sat(pi_suite).mean()
    # Get timeseries of global warming relative to preindustrial, in this suite
    gw_level = global_mean_sat(suite) - baseline_temp
    # Indefinite integral over time
    integrated_gw = (gw_level/months_per_year).cumsum(dim='time_centered')

    # Now add on definite integrals of all branched-from suites before that
    prev_suite = suites_branched[suite]
    while prev_suite is not None:
        # Find the starting date of the current suite
        start_date = gw_level.time_centered[0]
        # Now calculate global warming timeseries over the previous suite
        gw_level = global_mean_sat(prev_suite) - baseline_temp
        # Find the time index at the branch point
        time_branch = np.argwhere(gw_level.time_centered.data == start_date.data)[0][0]
        # Integrate from the beginning to just before that point
        integrated_gw += (gw_level.isel(time_centered=slice(0,time_branch))/months_per_year).sum(dim='time_centered')
        # Prepare for next iteration of loop
        suite = prev_suite
        prev_suite = suites_branched[suite]

    return integrated_gw


# Plot timeseries of integrated global warming in every experiment.
def plot_integrated_gw (base_dir='./', timeseries_file_um='timeseries_um.nc', pi_suite='cs568', fig_name=None):

    sim_names, colours, sim_dirs = set_expt_list(separate_stages=True)
    datas = []
    labels_plot = []
    colours_plot = []
    for expt, label, colour in zip(sim_dirs, sim_names, colours):
        if isinstance(expt, str):
            expt = [expt]
        num_ens = len(expt)
        for suite in expt:
            datas.append(integrated_gw(suite, pi_suite=pi_suite, timeseries_file_um=timeseries_file_um, base_dir=base_dir))
        labels_plot += [label] + [None]*(num_ens-1)
        colours_plot += [colour]*num_ens
    timeseries_plot(datas, labels=labels_plot, colours=colours_plot, title='Integrated global warming relative to preindustrial', units='Kelvin-years', linewidth=1, fig_name=fig_name)
    

# Plot the timeseries of one or more experiments/ensembles (expts can be a string, a list of strings, or a list of lists of string) and one variable against global warming level (relative to preindustrial mean in the given PI suite).
# If integrate=True, plot as a function of integrated global warming level (i.e. degree-years above preindustrial).
def plot_by_gw_level (expts, var_name, pi_suite='cs568', base_dir='./', fig_name=None, timeseries_file='timeseries.nc', timeseries_file_um='timeseries_um.nc', smooth=24, labels=None, colours=None, linewidth=1, title=None, units=None, ax=None, integrate=False):

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

    if not integrate:
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
        for suite in expt:
            if integrate:
                gw_level = integrated_gw(suite, pi_suite=pi_suite, timeseries_file_um=timeseries_file_um, base_dir=base_dir)
            else:
                # Read global mean SAT in this suite and convert to GW level
                ds_um = xr.open_dataset(base_dir+'/'+suite+'/'+timeseries_file_um)
                gw_level = ds_um['global_mean_sat'] - baseline_temp
                ds_um.close()
            # Read the variable
            ds = xr.open_dataset(base_dir+'/'+suite+'/'+timeseries_file)
            data = ds[var_name]
            ds.close()
            # Trim the two timeseries to line up and be the same length
            # Find most restrictive endpoints
            date_start = max(data.time_centered[0], gw_level.time_centered[0])
            date_end = min(data.time_centered[-1], gw_level.time_centered[-1])
            # Inner function to trim
            def trim_timeseries (A):
                t_start = np.argwhere(A.time_centered.data == date_start.data)[0][0]
                t_end = np.argwhere(A.time_centered.data == date_end.data)[0][0]
                return A.isel(time_centered=slice(t_start, t_end+1))
            data = trim_timeseries(data)
            gw_level = trim_timeseries(gw_level)
            if data.size != gw_level.size:
                print('Warning: timeseries do not align for suite '+suite+'. Removing suite from plot')
                num_ens -= 1
                continue
            # Smooth in time
            gw_level = moving_average(gw_level, smooth)
            data = moving_average(data, smooth)
            gw_levels.append(gw_level)
            datas.append(data)
        if num_ens > 0:
            labels_plot += [label] + [None]*(num_ens-1)
            colours_plot += [colour]*num_ens

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
    if integrate:
        ax.set_xlabel('Time-integrated global warming (K*years above preindustrial)', fontsize=14)
    else:
        ax.set_xlabel('Global warming relative to preindustrial (K)', fontsize=14)
    if labels is not None and new_ax:
        # Make legend
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width*0.8, box.height])
        ax.legend(loc='center left', bbox_to_anchor=(1,0.5))
    if new_ax:
        finished_plot(fig, fig_name=fig_name)


# Plot timeseries by global warming level for all variables in all experiments.
def plot_all_by_gw_level (base_dir='./', regions=['all', 'amundsen_sea', 'bellingshausen_sea', 'larsen', 'filchner_ronne', 'east_antarctica', 'amery', 'ross'], var_names=['massloss', 'draft', 'bwtemp', 'bwsalt', 'cavity_temp', 'cavity_salt', 'shelf_temp', 'shelf_salt', 'temp_btw_200_700m', 'salt_btw_200_700m', 'drake_passage_transport'], timeseries_file='timeseries.nc', timeseries_file_u='timeseries_u.nc', timeseries_file_um='timeseries_um.nc', smooth=24, fig_dir=None, pi_suite='cs568', integrate=False):

    sim_names, colours, sim_dirs = set_expt_list(separate_stages=True)
    
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
        plot_by_gw_level(sim_dirs, var, pi_suite=pi_suite, base_dir=base_dir, timeseries_file=fname, timeseries_file_um=timeseries_file_um, smooth=smooth, labels=sim_names, colours=colours, linewidth=1, fig_name=None if fig_dir is None else (fig_dir+'/'+var+'_gw.png'), integrate=integrate)


# Synthesise all this into a set of 5-panel plots for 3 different variables showing timeseries by GW level.
def gw_level_panel_plots (base_dir='./', pi_suite='cs568', fig_dir=None, integrate=False):

    regions = ['ross', 'filchner_ronne', 'west_antarctica', 'east_antarctica', 'all']
    var_names = ['bwtemp', 'bwsalt', 'massloss']
    var_titles = ['Bottom temperature on continental shelf and cavities', 'Bottom salinity on continental shelf and cavities', 'Basal mass loss']
    units = [deg_string+'C', gkg_string, 'Gt/y']
    sim_names, colours, sim_dirs = set_expt_list(separate_stages=True)
    timeseries_file = 'timeseries.nc'
    smooth = 24

    for v in range(len(var_names)):
        fig = plt.figure(figsize=(9,10))
        gs = plt.GridSpec(3,2)
        gs.update(left=0.09, right=0.98, bottom=0.07, top=0.9, hspace=0.3, wspace=0.15)
        for n in range(len(regions)):
            ax = plt.subplot(gs[n//2,n%2])
            plot_by_gw_level(sim_dirs, regions[n]+'_'+var_names[v], pi_suite=pi_suite, base_dir=base_dir, timeseries_file=timeseries_file, smooth=smooth, labels=sim_names, colours=colours, linewidth=0.5, ax=ax, integrate=integrate)
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
    var_x = [region+'_shelf_bwsalt' for region in regions] 
    var_y = [region+'_'+var_name for region in regions]
    sim_names, colours, sim_dirs = set_expt_list()
    timeseries_file = 'timeseries.nc'
    smooth = 24
    if var_name == 'massloss':
        units = 'Gt/y'
    elif 'temp' in var_name:
        units = deg_string+'C'

    fig = plt.figure(figsize=(10,6))
    gs = plt.GridSpec(1,3)
    gs.update(left=0.08, right=0.98, bottom=0.3, top=0.92, wspace=0.2)
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
                if not os.path.isfile(base_dir+'/'+suite+'/'+timeseries_file):
                    data_x.append(None)
                    data_y.append(None)
                    continue
                ds = xr.open_dataset(base_dir+'/'+suite+'/'+timeseries_file)
                data_x.append(moving_average(ds[var_x[n]], smooth))
                data_y.append(moving_average(ds[var_y[n]], smooth))
                ds.close()
        # Plot
        for x, y, label, colour in zip(data_x, data_y, labels_plot, colours_plot):
            if x is not None and y is not None:
                ax.plot(x, y, '-', color=colour, label=label, linewidth=1)
        ax.grid(linestyle='dotted')
        ax.set_title(region_names[regions[n]], fontsize=14)
        if n==0:
            ax.set_xlabel('Shelf bottom salinity ('+gkg_string+')', fontsize=10)
            ax.set_ylabel(var_name+' ('+units+')', fontsize=10)
    # Legend at bottom
    ax.legend(loc='lower center', bbox_to_anchor=(-0.7, -0.42), fontsize=10, ncol=3)
    finished_plot(fig, fig_name=fig_name)


# Before running this on Jasmin, do "source ~/pyenv/bin/activate" so we can use gsw
def plot_bwsalt_vs_obs (suite='cy691', schmidtko_file='/gws/nopw/j04/terrafirma/kaight/input_data/schmidtko_TS.txt', woa_files='/gws/nopw/j04/terrafirma/kaight/input_data/WOA18/woa18_decav_*00_04.nc', precomputed_file='bwsalt_1995_2014.nc', fig_name=None, base_dir='./'):

    start_year = 1995
    end_year = 2014
    eos = 'eos80'

    sim_dir = base_dir + '/' + suite + '_bwsalt/'
    if os.path.isfile(sim_dir+precomputed_file):
        nemo = xr.open_dataset(sim_dir+precomputed_file)
    else:
        # Identify NEMO output files in the suite directory within the given date range
        file_head = 'nemo_'+suite+'o_1m_'
        file_tail = '_grid-T.nc'
        nemo_files = []
        for f in os.listdir(sim_dir):
            if f.startswith(file_head) and f.endswith(file_tail):
                year = int(f[len(file_head):len(file_head)+4])
                if year >= start_year and year <= end_year:
                    nemo_files.append(sim_dir+f)
        # Read and time-average
        nemo = xr.open_mfdataset(nemo_files, concat_dim='time_counter', combine='nested')
        nemo = nemo.mean(dim='time_counter').squeeze()
        nemo.load()
        # Save to NetCDF for next time
        nemo.to_netcdf(sim_dir+precomputed_file)
    # Trim halo
    nemo = nemo.isel(x=slice(1,-1))

    # Read observations
    schmidtko = read_schmidtko(schmidtko_file=schmidtko_file, eos=eos)
    woa = read_woa(woa_files=woa_files, eos=eos)
    # Regrid to the NEMO grid, giving precedence to Schmidtko where both datasets exist
    schmidtko_interp = interp_latlon_cf(schmidtko, nemo, method='bilinear')
    woa_interp = interp_latlon_cf(woa, nemo, method='bilinear')
    obs = xr.where(schmidtko_interp.isnull(), woa_interp, schmidtko_interp)
    # Apply NEMO land mask to both
    nemo_plot = nemo['sob'].where(nemo['sob']!=0)
    obs_plot = obs['salt'].where(nemo_plot.notnull()*obs['salt'].notnull())
    obs_plot = obs_plot.where(nemo['sob']!=0)

    # Make the plot
    fig = plt.figure(figsize=(8,3))
    gs = plt.GridSpec(1,3)
    gs.update(left=0.1, right=0.9, bottom=0.05, top=0.8, wspace=0.1)
    data_plot = [nemo_plot, obs_plot, nemo_plot-obs_plot]
    titles = ['UKESM', 'Observations', 'Model bias']
    vmin = [34, 34, -0.5]
    vmax = [34.85, 34.85, 0.5]
    ctype = ['RdBu_r', 'RdBu_r', 'plusminus']
    for n in range(3):
        ax = plt.subplot(gs[0,n])
        ax.axis('equal')
        img = circumpolar_plot(data_plot[n], nemo, ax=ax, masked=True, make_cbar=False, title=titles[n], titlesize=14, vmin=vmin[n], vmax=vmax[n], ctype=ctype[n], lat_max=-63)
        if n != 1:
            cax = cax = fig.add_axes([0.01+0.45*n, 0.1, 0.02, 0.6])
            plt.colorbar(img, cax=cax, extend='both')
    plt.suptitle('Bottom salinity (psu), historical ('+str(start_year)+'-'+str(end_year)+')', fontsize=18)
    finished_plot(fig, fig_name=fig_name)


# Time-average each stabilisation scenario (all years and all ensemble members) for the given file type (grid-T, isf-T, grid-U).
def calc_stabilisation_means (base_dir='./', file_type='grid-T', out_dir='time_averaged/'):

    from tqdm import tqdm
    scenarios = ['piControl', '1.5K', '2K', '2.5K', '3K', '4K', '5K', '6K']
    
    for scenario in scenarios:
        print('Processing '+scenario)
        if scenario == 'piControl':
            scenario_full = scenario
        else:
            scenario_full = scenario + '_stabilise'
        out_file = base_dir+'/'+out_dir+'/'+scenario+'_'+file_type+'.nc'
        log_file = base_dir+'/'+out_dir+'/'+scenario+'_'+file_type+'.log'
        update_file = os.path.isfile(out_file)
        if update_file:
            # Read the log file to find out which files have already been processed
            old_files = np.loadtxt(log_file, dtype=str)
            num_old_files = old_files.size
        # Open the log file (may or may not already exist) to append to it
        log = open(log_file, 'a')
        nemo_files = []
        for suite in suites_by_scenario[scenario_full]:
            sim_dir = base_dir+'/'+suite+'/'
            file_head = 'nemo_'+suite+'o_1m_'
            file_tail = '_'+file_type+'.nc'
            for f in os.listdir(sim_dir):
                if f.startswith(file_head) and f.endswith(file_tail):
                    if update_file and f in old_files:
                        # Skip it; already processed
                        continue
                    nemo_files.append(sim_dir+f)
                    log.write(f+'\n')        
        log.close()
        ds_accum = None
        num_files = len(nemo_files)
        if num_files == 0:
            continue
        for n in tqdm(range(num_files), desc=' files'):
            ds = xr.open_dataset(nemo_files[n]).squeeze()
            if ds_accum is None:
                ds_accum = ds
            else:
                ds_accum += ds
            ds.close()
        ds_mean = ds_accum/num_files
        if update_file:
            # Now combine with existing mean as weighted average
            num_files_total = num_files + num_old_files
            ds_mean_old = xr.open_dataset(out_file)
            ds_mean = ds_mean*num_files/num_files_total + ds_mean_old*num_old_files/num_files_total
            ds_mean_old.close()
        ds_mean.to_netcdf(out_file, mode='w')
        ds_mean.close()
        

# Plot maps of the time-mean of the given variable in each stabilisation scenario
def plot_stabilisation_maps (var_name, fig_name=None):

    scenarios = ['piControl', '1.5K', '2K', '2.5K', '3K', '4K', '5K', '6K']
    in_dir = 'time_averaged/' 
    num_scenarios = len(scenarios)
    domain_cfg = '/gws/nopw/j04/terrafirma/kaight/input_data/grids/domcfg_eORCA1v2.2x.nc'
    if var_name == 'barotropic_streamfunction':
        title = 'Barotropic streamfunction (Sv)'
        file_type = 'grid-U'
        contour = [-15, 0]
        vmin = -60
        vmax = 60
        lat_max = None
        ctype = 'plusminus'
    elif var_name == 'tob':
        title = 'Bottom temperature ('+deg_string+'C)'
        file_type = 'grid-T'
        contour = None
        vmin = -2
        vmax = 4.5
        lat_max = -63
        ctype = 'RdBu_r'
    elif var_name == 'sob':
        title = 'Bottom salinity ('+gkg_string+')'
        file_type = 'grid-T'
        contour = None
        vmin = 33.5
        vmax = 34.8
        lat_max = -63
        ctype = 'RdBu_r'
    else:
        raise Exception('invalid var_name')
    
    fig = plt.figure(figsize=(4,8))
    gs = plt.GridSpec(num_scenarios//2, 2)
    gs.update(left=0.05, right=0.95, bottom=0.1, top=0.9, hspace=0.2, wspace=0.2)
    for n in range(num_scenarios):
        ds = xr.open_dataset(in_dir+scenarios[n]+'_'+file_type+'.nc').squeeze()
        if var_name=='barotropic_streamfunction':
            if n==0:
                # Grab e2u from domain_cfg
                ds_domcfg = xr.open_dataset(domain_cfg).squeeze()
                ds_domcfg = ds_domcfg.isel(y=slice(0, ds.sizes['y']))
            ds = ds.assign({'e2u':ds_domcfg['e2u']})
        if var_name == 'barotropic_streamfunction':
            data_plot = barotropic_streamfunction(ds)
        else:
            data_plot = ds[var_name]
            contour = None
        ax = plt.subplot(gs[n//2, n%2])
        ax.axis('equal')
        img = circumpolar_plot(data_plot, ds, ax=ax, make_cbar=False, title=scenarios[n], ctype=ctype, titlesize=14, vmin=vmin, vmax=vmax, contour=contour, lat_max=lat_max)
        if n == num_scenarios-1:
            cax = fig.add_axes([0.2, 0.04, 0.6, 0.02])
            plt.colorbar(img, cax=cax, orientation='horizontal', extend='both')
        ds.close()
    plt.suptitle(title, fontsize=18)
    finished_plot(fig, fig_name=fig_name)


# Plot T and S profiles averaged over the given region, for each stabilisation scenario
def plot_stabilisation_profiles (region='amundsen_sea', fig_name=None):

    scenarios = ['piControl', '1.5K', '2K', '2.5K', '3K', '4K', '5K', '6K']
    colours = ['Black', 'DarkMagenta', 'Blue', 'DarkCyan', 'DarkGreen', 'GoldenRod', 'Chocolate', 'Crimson']
    in_dir = 'time_averaged/'
    file_tail = '_grid-T.nc'
    num_scenarios = len(scenarios)
    var_names = ['thetao', 'so']
    num_var = len(var_names)
    var_titles = ['Temperature', 'Salinity']
    var_units = [deg_string+'C', 'psu']

    fig = plt.figure(figsize=(8,5))
    gs = plt.GridSpec(1,num_var)
    gs.update(left=0.1, right=0.98, bottom=0.2, top=0.85, wspace=0.2)
    ax_all = [plt.subplot(gs[0,v]) for v in range(num_var)]
    for n in range(num_scenarios):
        ds = xr.open_dataset(in_dir+scenarios[n]+file_tail).squeeze()
        if n==0:
            mask, ds, region_name = region_mask(region, ds, option='shelf', return_name=True)
            mask_3d = xr.where(ds[var_names[0]]==0, 0, mask)
            dA = ds['area']*mask_3d
        for v in range(num_var):
            ax = ax_all[v]
            # Area-average the given variable to get a depth profile
            data = (ds[var_names[v]]*dA).sum(dim=['x','y'])/dA.sum(dim=['x','y'])            
            ax.plot(data, ds['deptht'], '-', color=colours[n], label=scenarios[n])
            # Find deepest unmasked depth
            zdeep = ds['deptht'].where(data.notnull()).max()
            ax.set_ylim([zdeep, 0])
            ax.set_title(var_titles[v], fontsize=14)
            ax.set_xlabel(var_units[v], fontsize=12)
            if v==0:
                ax.set_ylabel('Depth (m)', fontsize=12)
            ax.grid(linestyle='dotted')
    ax.legend(loc='lower center', bbox_to_anchor=(-0.1,-0.3), fontsize=10, ncol=num_scenarios)
    plt.suptitle(region_name, fontsize=16)
    finished_plot(fig, fig_name=fig_name)


# Assemble a list of lists of all the possible suite trajectories. eg, one entry is ['cx209', 'cy837', 'da697']: each simulation branches from the previous one.
def all_suite_trajectories (static_ice=False):

    suite_sequences = []
    
    # Recursive inner function to build the list
    def complete_sequence (sequence):
        # Add the (possibly partial) sequence to the list, for example to consider perpetual ramp-up case
        suite_sequences.append(sequence)
        # Find the last suite in the sequence
        suite = sequence[-1]
        # Find all new suites which branched from this original suite
        new_suites = []
        for scenario in suites_by_scenario:
            for s in suites_by_scenario[scenario]:
                if suites_branched[s] == suite:
                    new_suites.append(s)
        if len(new_suites)==0:
            # Exit condition: nothing branches from this suite; sequence is complete
            pass
        else:
            # Loop over each suite that branches, and go another level down in the recursion
            for s in new_suite:
                complete_sequence(sequence+[s])

    # Start from each ramp-up ensemble member and use it as a seed
    if static_ice:
        name0 = 'ramp_up_static_ice'
    else:
        name0 = 'ramp_up'
    for suite in suites_by_scenario[name0]:
        complete_sequence([suite])


# Assemble a list of all possible trajectories of the given timeseries variable (precomputed).
def all_timeseries_trajectories (var_name, base_dir='./', timeseries_file='timeseries.nc', static_ice=False):

    suite_sequences = all_suite_trajectories(static_ice=static_ice)
    timeseries = []
    suite_strings = []
    # Loop over each suite trajectory and build the timeseries
    for suite_list in suite_sequences:
        for suite in suite_list:
            ds = xr.open_dataset(base_dir+'/'+suite+'/'+timeseries_file)
            data = ds[var_name]            
            if suite == suite_list[0]:
                suite_string = suite
            else:
                suite_string += '-'+suite
                # Find the starting date of the current suite
                start_date = data.time_centered[0]
                if data_old.time_centered[-1].data > start_date.data:
                    # Trim the previous timeseries to just before that date
                    time_branch = np.argwhere(data_prev.time_centered.data == start_date.data)[0][0]
                    data_prev = data_prev.isel(time_centered=slice(0,time_branch))
                # Concatenate with the previous timeseries
                data = xr.concat([data_prev, data], dim='time_centered')
            # Prepare for next iteration of loop
            data_prev = data
        suite_strings.append(suite_string)
        timeseries.append(data)
    return timeseries, suite_strings              


def cold_cavity_hysteresis_plots (base_dir='./', fig_name=None):

    # Assemble all possible trajectories of given variable (<region>_cavity_temp) - separate function
    # Do the same for global mean SAT relative to PI; trim timeseries as needed
    # Smooth both - 24 months?
    # Identify the ones that tip (>-1.9C) and save the time index where this happens
    # Get global mean SAT anomalies relative to that time index
    # Print some statistics (which ones tip; mean and std of warming at that point; risk of tipping eventually if GW exceeds given target)
    # Plot cavity temp vs SAT anomalies (should all line up around 0): each trajectory in a different colour, or all in grey with one highlighted on top?
    pass
            

    



        
                
                  
            
    
       

    
        



                    

    
