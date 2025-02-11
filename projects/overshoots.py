# Analysing TerraFIRMA overshoot simulations with UKESM1.1-ice (NEMO 3.6)

import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as cl
import matplotlib.animation as animation
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
import os
import subprocess
import numpy as np
import cf_xarray as cfxr
import re
import datetime
from scipy.stats import ttest_ind, linregress
from tqdm import tqdm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from ..timeseries import update_simulation_timeseries, update_simulation_timeseries_um, check_nans, fix_missing_months
from ..plots import timeseries_by_region, timeseries_by_expt, finished_plot, timeseries_plot, circumpolar_plot
from ..plot_utils import truncate_colourmap
from ..utils import moving_average, add_months, rotate_vector, polar_stereo, convert_ismr
from ..grid import region_mask, calc_geometry
from ..constants import line_colours, region_names, deg_string, gkg_string, months_per_year, rho_fw, rho_ice, sec_per_year, vaf_to_gmslr
from ..file_io import read_schmidtko, read_woa
from ..interpolation import interp_latlon_cf, interp_grid
from ..diagnostics import barotropic_streamfunction
from ..plot_utils import set_colours, latlon_axes

# Global dictionaries of suites - update these as more suites become available!

# Dictionary of which suites correspond to which scenario
suites_by_scenario = {'piControl' : ['cs495'],
                      'piControl_static_ice' : ['cs568'],
                      'ramp_up' : ['cx209', 'cw988', 'cw989', 'cw990'],
                      'ramp_up_static_ice': ['cz826'],
                      '1.5K_stabilise': ['cy837','cz834','da087'],
                      '1.5K_ramp_down': ['da697', 'dc052', 'dc248'],
                      '2K_stabilise': ['cy838','cz855','da266'],
                      '2K_ramp_down': ['di335','dc051', 'da800', 'dc565', 'df025', 'df027'],
                      '2.5K_stabilise' : ['cz374','cz859'],
                      '3K_stabilise' : ['cz375','db587','db597'],
                      '3K_ramp_down' : ['dc032', 'dc249', 'df453', 'df028', 'df023', 'df021'],
                      '4K_stabilise' : ['cz376','db723','db733'],
                      '4K_ramp_down' : ['da892', 'dc123', 'dh859', 'dd210', 'dh541'],
                      '5K_stabilise' : ['cz377','db731','dc324'],
                      '5K_ramp_down' : ['dc251', 'dc130', 'dg095'], #'dg093', 'dg094'],
                      '6K_stabilise' : ['cz378'],
                      '6K_ramp_down' : ['de943', 'de962', 'de963', 'dm357', 'dm358', 'dm359']}
# Choose one ensemble member of each main scenario type for plotting a less-messy timeseries.
suites_by_scenario_1ens = {'ramp_up': 'cx209',  # First ensemble member for ramp-up and all stabilisation
                           '1.5K_stabilise': 'cy837', 
                           '2K_stabilise': 'cy838',
                           '3K_stabilise': 'cz375',
                           '4K_stabilise': 'cz376',
                           '5K_stabilise': 'cz377',
                           '6K_stabilise': 'cz378',
                           '1.5K_ramp_down': 'dc052',  # 50y overshoot, -4 Gt/y for all ramp-downs
                           '2K_ramp_down': 'dc051',
                           '3K_ramp_down': 'df028',
                           '4K_ramp_down': 'dc123',
                           '5K_ramp_down': 'dc130',
                           '6K_ramp_down': 'de962'}
# Dictionary of ramp-down rates
suites_ramp_down_rates = {'8 Gt/y' : ['di335', 'da800', 'da697', 'da892', 'df453', 'dc251', 'de943', 'dg093', 'dm357'],
                          '4 Gt/y' : ['dc051', 'dc052', 'dc248', 'dc249', 'dc565', 'dd210', 'dc032', 'df028', 'dc123', 'dc130', 'de962', 'dg094', 'dm358'],
                          '2 Gt/y' : ['df025', 'df027', 'df021', 'df023', 'dh541', 'dh859', 'de963', 'dg095', 'dm359']}
# Dictionary of overshoot lengths
suites_overshoot_lengths = {'50 years': ['da697', 'dc052', 'di335', 'dc051', 'df025', 'df453', 'df028', 'df021', 'da892', 'dc123', 'dh541', 'dc251', 'dc130', 'de943', 'de962', 'de963'],
                            '200 years': ['dc248', 'da800', 'dc565', 'df027', 'dc249', 'df023', 'dh859', 'dd210', 'dg093', 'dg094', 'dg095', 'dm357', 'dm358', 'dm359'],
                            '30 years': ['dc032']}

# Dictionary of which suites branch from which. None means it's a ramp-up suite (so branched from a piControl run, but we don't care about that for the purposes of integrated GW)
suites_branched = {'cx209':None, 'cw988':None, 'cw989':None, 'cw990':None, 'cz826':None, 'cy837':'cx209', 'cy838':'cx209', 'cz374':'cx209', 'cz375':'cx209', 'cz376':'cx209', 'cz377':'cx209', 'cz378':'cx209', 'cz834':'cw988', 'cz855':'cw988', 'cz859':'cw988', 'db587':'cw988', 'db723':'cw988', 'db731':'cw988', 'da087':'cw989', 'da266':'cw989', 'db597':'cw989', 'db733':'cw989', 'dc324':'cw989', 'da800':'cy838', 'da697':'cy837', 'da892':'cz376', 'dc051':'cy838', 'dc052':'cy837', 'dc248':'cy837', 'dc249':'cz375', 'dc251':'cz377', 'dc032':'cz375', 'dc123':'cz376', 'dc130':'cz377', 'di335':'cy838', 'df453':'cz375', 'dc565':'cy838', 'dd210':'cz376', 'df028':'cz375', 'df025':'cy838', 'df027':'cy838', 'df021':'cz375', 'df023':'cz375', 'dh541':'cz376', 'dh859':'cz376', 'de943':'cz378', 'de962':'cz378', 'de963':'cz378', 'dg093':'cz377', 'dg094':'cz377', 'dg095':'cz377', 'dm357':'cz378', 'dm358':'cz378', 'dm359':'cz378'}

tipping_threshold = -1.9  # If cavity mean temp is warmer than surface freezing point, it's tipped

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
    timeseries_types += ['west_antarctica_bwtemp', 'west_antarctica_bwsalt', 'west_antarctica_massloss', 'filchner_ronne_shelf_bwsalt', 'ross_shelf_bwsalt', 'amery_bwsalt']

    update_simulation_timeseries(suite_id, timeseries_types, timeseries_file='timeseries.nc', sim_dir=base_dir+'/'+suite_id+'/', freq='m', halo=True, gtype='T')

    # Now for u-grid
    #update_simulation_timeseries(suite_id, ['drake_passage_transport'], timeseries_file='timeseries_u.nc', sim_dir=base_dir+'/'+suite_id+'/', freq='m', halo=True, gtype='U', domain_cfg=domain_cfg)

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


# Set the list of experiments, names for the legend, and colours to use for timeseries etc.
# If separate_stages=True, the ramp down and re-stabilise stages will be separated out from the initial stabilisations.
# If only_up=True, only the ramp-up and stabilise suites will be considered.
# If only_down=True, only the ramp-down and re-stabilise suites will be considered.
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


# Set the list of experiments, names for the legend, and timeseries to use - but only separating (by names/colours) the ramp-up, stabilise, and ramp-down scenarios, rather than showing all the different stabilisation targets individually. Also, do not include static ice cases.
def minimal_expt_list (one_ens=False):

    if one_ens:
        suite_list = suites_by_scenario_1ens
    else:
        suite_list = suites_by_scenario

    keys = ['ramp_up', '_stabilise', '_ramp_down']
    sim_names = ['Ramp up', 'Stabilise', 'Ramp down']
    colours = ['Crimson', 'Grey', 'DodgerBlue']
    sim_dirs = []
    for key in keys:
        dirs = []
        for scenario in suite_list:
            if 'static_ice' in scenario:
                continue
            if key in scenario:
                if isinstance(suite_list[scenario], str):
                    dirs += [suite_list[scenario]]
                else:
                    dirs += suite_list[scenario]
        sim_dirs.append(dirs)

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
def integrated_gw (suite, pi_suite='cs495', timeseries_file_um='timeseries_um.nc', base_dir='./'):

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
def plot_integrated_gw (base_dir='./', timeseries_file_um='timeseries_um.nc', pi_suite='cs495', fig_name=None):

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


# Given two timeseries, trim them to line up along the time axis and be the same length.
def align_timeseries (data1, data2, time_coord='time_centered'):

    # Find most restrictive endpoints
    date_start = max(data1[time_coord][0], data2[time_coord][0])
    date_end = min(data1[time_coord][-1], data2[time_coord][-1])
    # Inner function to trim
    def trim_timeseries (A):
        t_start = np.argwhere(A[time_coord].data == date_start.data)[0][0]
        t_end = np.argwhere(A[time_coord].data == date_end.data)[0][0]
        return A.isel({time_coord:slice(t_start, t_end+1)})
    data1 = trim_timeseries(data1)
    data2 = trim_timeseries(data2)
    return data1, data2
    

# Plot the timeseries of one or more experiments/ensembles (expts can be a string, a list of strings, or a list of lists of string) and one variable against global warming level (relative to preindustrial mean in the given PI suite, unless offsets is not None).
# Can also use an ocean variable to plot against instead of GW level (eg cavity temperature) with alternate_var='something' (assumed to be within timeseries_file).
# Can also set offsets as a list of the same shape as expts, with different global warming baselines for each - this is still on top of PI mean, so eg pass 3 for 3K above PI.
def plot_by_gw_level (expts, var_name, pi_suite='cs495', base_dir='./', fig_name=None, timeseries_file='timeseries.nc', timeseries_file_um='timeseries_um.nc', smooth=24, labels=None, colours=None, linewidth=1, title=None, units=None, ax=None, offsets=None, alternate_var=None):

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

    if alternate_var is None:
        # Get baseline global mean SAT
        ds_pi = xr.open_dataset(base_dir+'/'+pi_suite+'/'+timeseries_file_um)
        baseline_temp = ds_pi['global_mean_sat'].mean()
        ds_pi.close()

    if offsets is None:
        # Set up dummy list of 0s, same shape as expts
        offsets = []
        for expt in expts:
            if isinstance(expt, str):
                offsets.append([0])
            else:
                offsets.append([0]*len(expt))
    
    gw_levels = []
    datas = []
    labels_plot = []
    colours_plot = []
    for expt, label, colour, expt_offsets in zip(expts, labels, colours, offsets):
        if isinstance(expt, str):
            # Generalise to ensemble of 1
            expt = [expt]
        num_ens = len(expt)
        for suite, offset in zip(expt, expt_offsets):
            if np.isnan(offset):
                # Flag to skip this suite
                continue            
            # Join with the parent suite so there isn't a gap when simulations branch and smoothing is applied.
            if suites_branched[suite] is not None:
                suite_list = [suites_branched[suite], suite]
            else:
                suite_list = [suite]
            # Read global mean SAT in this suite and convert to GW level
            if alternate_var is None:
                gw_level = build_timeseries_trajectory(suite_list, 'global_mean_sat', base_dir=base_dir, timeseries_file=timeseries_file_um, offset=-baseline_temp-offset)
            else:
                gw_level = build_timeseries_trajectory(suite_list, alternate_var, base_dir=base_dir, timeseries_file=timeseries_file)
            # Read the variable
            data = build_timeseries_trajectory(suite_list, var_name, base_dir=base_dir, timeseries_file=timeseries_file)
            # Trim the two timeseries to line up and be the same length
            data, gw_level = align_timeseries(data, gw_level)
            if data.size != gw_level.size:
                print('Warning: timeseries do not align for suite '+suite+'. Removing suite from plot. Try running fix_missing_months()')
                num_ens -= 1
                continue
            # Figure out the scenario type we're actually trying to plot
            stype = data.scenario_type[-1]
            # Smooth in time            
            gw_level = moving_average(gw_level, smooth)
            data = moving_average(data, smooth)
            # Now trim off the parent suite, so we only keep the bit needed to smooth over the gap
            t_start = np.where(data.scenario_type==stype)[0][0]
            gw_level = gw_level.isel(time_centered=slice(t_start,None))
            data = data.isel(time_centered=slice(t_start,None))
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
    if alternate_var is None:
        ax.set_xlabel('Global warming relative to preindustrial (K)', fontsize=14)
    else:
        ax.set_xlabel(gw_levels[0].long_name+' ('+gw_levels[0].units+')', fontsize=14)
    if labels is not None and new_ax:
        # Make legend
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width*0.8, box.height])
        ax.legend(loc='center left', bbox_to_anchor=(1,0.5))
    if new_ax:
        finished_plot(fig, fig_name=fig_name)


# Plot timeseries by global warming level for all variables in all experiments.
def plot_all_by_gw_level (base_dir='./', regions=['all', 'amundsen_sea', 'bellingshausen_sea', 'larsen', 'filchner_ronne', 'east_antarctica', 'amery', 'ross'], var_names=['massloss', 'draft', 'bwtemp', 'bwsalt', 'cavity_temp', 'cavity_salt', 'shelf_temp', 'shelf_salt', 'temp_btw_200_700m', 'salt_btw_200_700m', 'drake_passage_transport'], timeseries_file='timeseries.nc', timeseries_file_u='timeseries_u.nc', timeseries_file_um='timeseries_um.nc', smooth=24, fig_dir=None, pi_suite='cs495', integrate=False):

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
def gw_level_panel_plots (base_dir='./', pi_suite='cs495', fig_dir=None, integrate=False):

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

    sim_dir = base_dir + '/' + suite + '/'
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
        ds_mean.to_netcdf(out_file+'_tmp', mode='w')
        ds_mean.close()
        if update_file:
            ds_mean_old.close()
        os.rename(out_file+'_tmp',out_file)
        

# Plot maps of the time-mean of the given variable in each stabilisation scenario
def plot_stabilisation_maps (var_name, fig_name=None):

    scenarios = ['piControl', '1.5K', '2K', '2.5K', '3K', '4K', '5K', '6K']
    in_dir = 'time_averaged/' 
    num_scenarios = len(scenarios)
    domain_cfg = '/gws/nopw/j04/terrafirma/kaight/input_data/grids/domcfg_eORCA1v2.2x.nc'
    if var_name == 'barotropic_streamfunction':
        title = 'Barotropic streamfunction (Sv)'
        file_type = 'grid-U'
        file_type_2 = 'grid-V'
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
    elif var_name == 'temp500m':
        title = 'Temperature at 500m ('+deg_string+'C)'
        file_type = 'grid-T'
        contour = None
        vmin = -2
        vmax = 7
        lat_max = None
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
            ds_v = xr.open_dataset(in_dir+scenarios[n]+'_'+file_type_2+'.nc').squeeze()
        if var_name == 'barotropic_streamfunction':
            data_plot = barotropic_streamfunction(ds, ds_v, ds_domcfg, periodic=True, halo=True)
        elif var_name == 'temp500m':
            data_3d = ds['thetao'].where(ds['thetao']!=0)
            data_plot = data_3d.interp(deptht=500)
        else:
            data_plot = ds[var_name]
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
                if s not in suites_branched:
                    # Some suites (eg preindustrial) not considered in branching
                    continue
                if suites_branched[s] == suite:
                    new_suites.append(s)
        if len(new_suites)==0:
            # Exit condition: nothing branches from this suite; sequence is complete
            pass
        else:
            # Loop over each suite that branches, and go another level down in the recursion
            for s in new_suites:
                complete_sequence(sequence+[s])

    # Start from each ramp-up ensemble member and use it as a seed
    if static_ice:
        name0 = 'ramp_up_static_ice'
    else:
        name0 = 'ramp_up'
    for suite in suites_by_scenario[name0]:
        complete_sequence([suite])

    return suite_sequences


# Helper function to assemble a timeseries for the given trajectory and variable (precomputed).
def build_timeseries_trajectory (suite_list, var_name, base_dir='./', timeseries_file='timeseries.nc', offset=0):

    xr.set_options(keep_attrs=True)

    for suite in suite_list:
        # Figure out whether it's a ramp-up (1), stabilisation (0), or ramp-down (-1) simulation
        stype = None
        for scenario in suites_by_scenario:
            if suite in suites_by_scenario[scenario]:
                if 'ramp_up' in scenario:
                    stype = 1
                elif 'restabilise' in scenario:
                    stype = -2
                elif 'stabilise' in scenario:
                    stype = 0
                elif 'ramp_down' in scenario:
                    stype = -1
                else:
                    raise Exception('invalid scenario type')
                break
        if stype is None:
            raise Exception('Simulation type not found')
        ds = xr.open_dataset(base_dir+'/'+suite+'/'+timeseries_file)
        data = ds[var_name] + offset
        data = data.assign_coords(scenario_type=('time_centered', np.ones(data.size)*stype))
        if suite != suite_list[0]:
            # Find the starting date of the current suite
            start_date = data.time_centered[0]
            if data_prev.time_centered[-1].data > start_date.data:
                # Trim the previous timeseries to just before that date
                time_branch = np.argwhere(data_prev.time_centered.data == start_date.data)[0][0]
                data_prev = data_prev.isel(time_centered=slice(0,time_branch))
            # Concatenate with the previous timeseries
            data = xr.concat([data_prev, data], dim='time_centered')
        # Prepare for next iteration of loop
        data_prev = data
        ds.close()
    return data    


# Assemble a list of all possible trajectories of the given timeseries variable (precomputed).
# Can add an offset if needed (eg the negative of the preindustrial baseline temperature).
def all_timeseries_trajectories (var_name, base_dir='./', timeseries_file='timeseries.nc', static_ice=False, offset=0):

    suite_sequences = all_suite_trajectories(static_ice=static_ice)
    timeseries = []
    suite_strings = []
    # Loop over each suite trajectory and build the timeseries
    for suite_list in suite_sequences:
        data = build_timeseries_trajectory(suite_list, var_name, base_dir=base_dir, timeseries_file=timeseries_file, offset=offset)
        suite_strings.append('-'.join(suite_list))
        timeseries.append(data)
    return timeseries, suite_strings


# Helper function to check if the given suite tips.
# Input:
# suite: suite name or trajectory string, eg 'cx209' or 'cx209-cy838'
# region: 'ross' or 'filchner_ronne'
# XOR
# cavity_temp: DataArray of temperature averaged over the cavity
# Optional:
# smoothed: whether cavity_temp is already smoothed
# return_date: whether to return a cftime.datetime object showing when it tips
# return_t: whether to return the time index (integer) showing when cavity_temp tips (only if cavity_temp is set)
def check_tip (suite=None, region=None, cavity_temp=None, smoothed=False, return_date=False, return_t=False, base_dir='./'):

    smooth = 5*months_per_year

    # Error checking input
    if suite is not None and region is None:
        raise Exception('Must set region if suite is defined')
    if suite is not None and cavity_temp is not None:
        raise Exception('Cannot set both suite and cavity_temp')
    if return_t and cavity_temp is None:
        raise Exception('Can only set return_t if cavity_temp is set')

    if suite is not None:
        # Get cavity temp array
        cavity_temp = build_timeseries_trajectory(suite.split('-'), region+'_cavity_temp', base_dir=base_dir)
        smoothed = False
    if not smoothed:
        cavity_temp = moving_average(cavity_temp, smooth)

    tipped = cavity_temp.max() > tipping_threshold
    if tipped:
        t_tip = np.argwhere(cavity_temp.data > tipping_threshold)[0][0]
        date_tip = cavity_temp.time_centered[t_tip]
    else:
        t_tip = None
        date_tip = None
    if return_date and return_t:
        return tipped, date_tip, t_tip
    elif return_date:
        return tipped, date_tip
    elif return_t:
        return tipped, t_tip
    else:
        return tipped


# Helper function to check if the given suite recovers.
# Input as in tips().
def check_recover (suite=None, region=None, cavity_temp=None, smoothed=False, return_date=False, return_t=False, base_dir='./'):

    smooth = 5*months_per_year
    
    if suite is not None and region is None:
        raise Exception('Must set region if suite is defined')
    if suite is not None and cavity_temp is not None:
        raise Exception('Cannot set both suite and cavity_temp')
    if return_t and cavity_temp is None:
        raise Exception('Can only set return_t if cavity_temp is set')

    if suite is not None:
        cavity_temp = build_timeseries_trajectory(suite.split('-'), region+'_cavity_temp', base_dir=base_dir)
        smoothed = False
    if not smoothed:
        cavity_temp = moving_average(cavity_temp, smooth)

    # Check if it even tips
    tipped, t_tip = check_tip(cavity_temp=cavity_temp, smoothed=True, return_t=True)
    recovered = False
    if tipped:        
        # Check every time index after tipping
        for t in range(t_tip, cavity_temp.sizes['time_centered']):
            if cavity_temp.isel(time_centered=slice(t,None)).max() < tipping_threshold:
                recovered = True
                t_recover = t
                date_recover = cavity_temp.time_centered[t_recover]
                break
    if not tipped or not recovered:
        t_recover = None
        date_recover = None
    if return_date and return_t:
        return recovered, date_recover, t_recover
    elif return_date:
        return recovered, date_recover
    elif return_t:
        return recovered, t_recover
    else:
        return recovered    


# Analyse the cavity temperature beneath Ross and FRIS to see which scenarios tip and/or recover, under which global warming levels. Also plot this.
def tipping_stats (base_dir='./'):

    regions = ['ross', 'filchner_ronne']
    temp_correction = [1.0087846842764405, 0.8065649751736049]  # Precomputed by warming_implied_by_salinity_bias()
    bias_print_x = [4.5, 2.5]
    bias_print_y = 1.5
    pi_suite = 'cs495'
    smooth = 5*months_per_year
    timeseries_file = 'timeseries.nc'
    timeseries_file_um = 'timeseries_um.nc'

    # Assemble all possible trajectories of global mean temperature anomalies relative to preindustrial
    ds = xr.open_dataset(base_dir+'/'+pi_suite+'/'+timeseries_file_um)
    baseline_temp = ds['global_mean_sat'].mean()
    ds.close()
    warming_ts, suite_strings = all_timeseries_trajectories('global_mean_sat', base_dir=base_dir, timeseries_file=timeseries_file_um, static_ice=False, offset=-1*baseline_temp)
    num_trajectories = len(warming_ts)
    # Smooth each
    for n in range(num_trajectories):
        warming_ts[n] = moving_average(warming_ts[n], smooth)

    # Loop over regions
    all_temp_tip = []
    all_temp_recover = []
    all_tips = []
    threshold_bounds = []
    for r in range(len(regions)):
        # Assemble all possible trajectories of cavity mean temperature
        cavity_temp_ts = all_timeseries_trajectories(regions[r]+'_cavity_temp', base_dir=base_dir, static_ice=False)[0]
        # Now loop over them and find the ones that have tipped and/or recovered
        suites_tipped = []
        warming_at_tip = []
        suites_recovered = []        
        warming_at_recovery = []
        tips = []
        for n in range(num_trajectories):
            # Smooth and trim/align with warming timeseries
            cavity_temp = moving_average(cavity_temp_ts[n], smooth)
            cavity_temp, warming = align_timeseries(cavity_temp, warming_ts[n])
            traj_tips, t_tip = check_tip(cavity_temp=cavity_temp, smoothed=True, return_t=True, base_dir=base_dir)
            if traj_tips:
                tip_warming = warming.isel(time_centered=t_tip)
                if np.isnan(tip_warming):
                    continue
                suites_tipped.append(suite_strings[n])
                warming_at_tip.append(tip_warming)
                recovers, t_recovers = check_recover(cavity_temp=cavity_temp, smoothed=True, return_t=True, base_dir=base_dir)
                if recovers:
                    recover_warming = warming.isel(time_centered=t_recovers)
                    if np.isnan(recover_warming):
                        continue
                    suites_recovered.append(suite_strings[n])
                    warming_at_recovery.append(recover_warming)
            tips.append(traj_tips)
        tips = np.array(tips)
        # Now throw out duplicates, eg if tipping happened before a suite branched into multiple trajectories, should only count it once for the statistics.
        # Do this by only considering unique values of warming_at_tip and warming_at_recovery.
        # This assumes it's impossible for two distinct suites to tip at exactly the same global warming level, to machine precision. I think I'm happy with this!
        warming_at_tip = np.unique(warming_at_tip)
        warming_at_recovery = np.unique(warming_at_recovery)
        # Find maximum warming in each trajectory
        max_warming = np.array([warming_ts[n].max() for n in range(num_trajectories)])                
        # Print some statistics about which ones tipped and recovered
        print('\n'+regions[r]+':')
        print(str(len(suites_tipped))+' trajectories tip, '+str(len(warming_at_tip))+' unique')
        print('Global warming at time of tipping has mean '+str(np.mean(warming_at_tip)+temp_correction[r])+'K, standard deviation '+str(np.std(warming_at_tip))+'K')
        if len(suites_recovered) == 0:
            print('No tipped trajectories recover')
        else:
            print(str(len(suites_recovered))+' tipped trajectories recover ('+str(len(suites_recovered)/len(suites_tipped)*100)+'%), '+str(len(warming_at_recovery))+' unique')
            print('Global warming at time of recovery has mean '+str(np.mean(warming_at_recovery)+temp_correction[r])+'K, standard deviation '+str(np.std(warming_at_recovery))+'K')
        # Save results for plotting
        all_temp_tip.append(warming_at_tip)
        all_temp_recover.append(warming_at_recovery)
        all_tips.append(tips)

        # Find bounds on threshold
        # Find coolest instance of tipping
        first_tip = np.amin(max_warming[tips])
        # Step down to one simulation cooler than that: lower bound on threshold
        threshold_min = np.amax(max_warming[max_warming < first_tip]) + temp_correction[r]
        # Find warmest instance of non-tipping
        last_notip = np.amax(max_warming[~tips])
        # Step up to one simulation warmer than that: upper bound on threshold
        threshold_max = np.amin(max_warming[max_warming > last_notip]) + temp_correction[r]
        threshold_bounds.append([threshold_min, threshold_max])

    # Plot
    fig = plt.figure(figsize=(6,5))
    gs = plt.GridSpec(2,1)
    gs.update(left=0.25, right=0.95, bottom=0.1, top=0.92, hspace=0.4)
    for r in range(len(regions)):
        ax = plt.subplot(gs[r,0])
        # Violin plots: warming level at time of tipping (red), recovery (blue)
        violin_data = [np.array(all_temp_tip[r])+temp_correction[r]]
        y_pos = [3]
        colours = ['Crimson']
        # Check if there are instances of recovery to show - can streamline this once FRIS recovery starts
        if len(all_temp_recover[r]) > 0:
            violin_data.append(np.array(all_temp_recover[r])+temp_correction[r])
            y_pos.append(2)
            colours.append('DodgerBlue')
        violins = ax.violinplot(violin_data, y_pos, vert=False, showmeans=True)
        # Set colours of violin bodies and lines
        for pc, colour in zip(violins['bodies'], colours):
            pc.set_facecolor(colour)
        for bar in ['cmeans', 'cmins', 'cmaxes', 'cbars']:
            violins[bar].set_colors(colours)
        # Bottom row: peak warming in each trajectory, plotted in red (tips) or grey (doesn't tip)
        # Start with the grey, to make sure the red is visible where they overlap
        ax.plot(max_warming[~all_tips[r]]+temp_correction[r], np.ones(np.count_nonzero(~all_tips[r])), 'o', markersize=4, color='DarkGrey')
        ax.plot(max_warming[all_tips[r]]+temp_correction[r], np.ones(np.count_nonzero(all_tips[r])), 'o', markersize=4, color='Crimson')
        # Plot bounds on threshold: vertical dashed lines with labels
        ax.plot(threshold_bounds[r][0]*np.ones(2), [0, 0.9], color='black', linestyle='dashed', linewidth=1)
        plt.text(threshold_bounds[r][0]-0.05, 0.5, 'never tips', ha='right', va='center', fontsize=9)
        plt.arrow(threshold_bounds[r][0]-0.05, 0.2, -0.3, 0, head_width=0.1, head_length=0.08)
        ax.plot(threshold_bounds[r][1]*np.ones(2), [0, 0.9], color='black', linestyle='dashed', linewidth=1)
        plt.text(threshold_bounds[r][1]+0.05, 0.5, 'always tips', ha='left', va='center', fontsize=9)
        plt.arrow(threshold_bounds[r][1]+0.05, 0.2, 0.3, 0, head_width=0.1, head_length=0.08)
        ax.set_title(region_names[regions[r]], fontsize=14)
        ax.set_xlim([2, 7])
        ax.set_ylim([0, 4])
        ax.set_yticks(np.arange(1,4))
        ax.set_yticklabels(['peak warming', 'at time of recovery', 'at time of tipping'])
        ax.grid(linestyle='dotted')       
    ax.set_xlabel('global warming level ('+deg_string+'C), corrected', fontsize=10)
    # Manual legend
    colours = ['Crimson', 'DarkGrey']
    labels = ['tips', 'does not tip']
    handles = []
    for m in range(len(colours)):
        handles.append(Line2D([0], [0], marker='o', markersize=4, color=colours[m], label=labels[m], linestyle=''))
    plt.legend(handles=handles, loc='center left', bbox_to_anchor=(-0.35, 1.2), fontsize=9)
    finished_plot(fig, fig_name='figures/tipping_stats.png', dpi=300)
    

# Plot: (1) bottom temperature on continental shelf and in cavities, and (2) ice shelf basal mass loss as a function of global warming level, for 2 different regions, showing ramp-up, stabilise, and ramp-down in different colours
def plot_bwtemp_massloss_by_gw_panels (base_dir='./'):

    pi_suite = 'cs495'
    regions = ['ross', 'filchner_ronne']
    title_prefix = [r'$\bf{a}$. ', r'$\bf{b}$. ', r'$\bf{c}$. ', r'$\bf{d}$. ']
    var_names = ['bwtemp', 'massloss']
    var_titles = ['Bottom temperature on continental shelf and in ice shelf cavities', 'Basal mass loss beneath ice shelves']
    var_units = [deg_string+'C', 'Gt/y']
    num_var = len(var_names)
    timeseries_file = 'timeseries.nc'
    smooth = [10*months_per_year, 30*months_per_year]
    sim_names, colours, sim_dirs = minimal_expt_list(one_ens=True)
    sample_file = base_dir+'/time_averaged/piControl_grid-T.nc'  # Just to build region masks
    ds = xr.open_dataset(sample_file).squeeze()

    fig = plt.figure(figsize=(10,7.5))
    gs = plt.GridSpec(2,2)
    gs.update(left=0.07, right=0.98, bottom=0.1, top=0.9, hspace=0.5, wspace=0.16)
    for v in range(num_var):
        for n in range(len(regions)):
            ax = plt.subplot(gs[v,n])
            plot_by_gw_level(sim_dirs, regions[n]+'_'+var_names[v], pi_suite=pi_suite, base_dir=base_dir, timeseries_file=timeseries_file, smooth=smooth[v], labels=sim_names, colours=colours, linewidth=1, ax=ax)
            ax.set_title(title_prefix[v*2+n]+region_names[regions[n]], fontsize=14)
            if n == 0:
                ax.set_ylabel(var_units[v], fontsize=12)
            else:
                ax.set_ylabel('')
            if n==0 and v==0:
                ax.set_xlabel('Global warming relative to preindustrial ('+deg_string+'C)', fontsize=12)
            else:
                ax.set_xlabel('')
            ax.set_xlim([0,8])
            if v==0:
                # Inset panel in top left showing region
                mask = region_mask(regions[n], ds, option='all')[0]
                ax2 = inset_axes(ax, "25%", "40%", loc='upper left')
                ax2.axis('equal')                
                circumpolar_plot(mask, ds, ax=ax2, make_cbar=False, ctype='IndianRed', lat_max=-66, shade_land=True)
                ax2.axis('on')
                ax2.set_xticks([])
                ax2.set_yticks([])
        plt.text(0.5, 0.99-0.5*v, var_titles[v], fontsize=16, ha='center', va='top', transform=fig.transFigure)
    ax.legend(loc='center left', bbox_to_anchor=(-0.6,-0.2), fontsize=11, ncol=3)
    finished_plot(fig, fig_name='figures/temp_massloss_by_gw_panels.png', dpi=300)


# Calculate UKESM's bias in bottom salinity on the continental shelf of Ross and FRIS. To do this, find the global warming level averaged over 1995-2014 of a historical simulation with static cavities (cy691) and identify the corresponding 10-year period in each ramp-up ensemble member. Then, average bottom salinity over those years and ensemble members, compare to observational climatologies interpolated to NEMO grid, and calculate the area-averaged bias.
# Before running this on Jasmin, do "source ~/pyenv/bin/activate" so we can use gsw
def calc_salinity_bias (base_dir='./'):

    regions = ['ross', 'filchner_ronne']
    pi_suite = 'cs495'  # Preindustrial, static cavities
    hist_suite = 'cy691'  # Historical, static cavities: to get UKESM's idea of warming relative to preindustrial
    timeseries_file_um = 'timeseries_um.nc'
    num_years = 10
    schmidtko_file='/gws/nopw/j04/terrafirma/kaight/input_data/schmidtko_TS.txt'
    woa_files='/gws/nopw/j04/terrafirma/kaight/input_data/WOA18/woa18_decav_*00_04.nc'
    eos = 'eos80'

    # Inner function to read global mean SAT from precomputed timeseries
    def global_mean_sat (suite):
        ds = xr.open_dataset(base_dir+'/'+suite+'/'+timeseries_file_um)
        sat = ds['global_mean_sat']
        ds.close()
        return sat
    # Get preindustrial baseline
    baseline_temp = global_mean_sat(pi_suite).mean()
    # Get "present-day" warming according to UKESM
    hist_warming = global_mean_sat(hist_suite).mean() - baseline_temp
    print('UKESM historical 1995-2014 was '+str(hist_warming.data)+'K warmer than preindustrial')

    # Loop over ramp-up suites (no static ice)
    ramp_up_bwsalt = None
    for suite in suites_by_scenario['ramp_up']:
        # Get timeseries of global warming relative to PI
        warming = global_mean_sat(suite) - baseline_temp
        for t in range(warming.size):
            if warming.isel(time_centered=slice(t,t+num_years*months_per_year)).mean() >= hist_warming:
                # Care about the 10-year period beginning at this point
                time_select = warming.time_centered.isel(time_centered=slice(t,t+num_years*months_per_year))
                print(suite+' matches historical warming from '+str(t//months_per_year)+' years')
                break
        for time in time_select.data:
            # Find the corresponding grid-T ocean file
            year_start = time.year
            month_start = time.month
            year_end, month_end = add_months(year_start, month_start, 1)
            file_path = base_dir+'/'+suite+'/nemo_'+suite+'o_1m_'+str(year_start)+str(month_start).zfill(2)+'01-'+str(year_end)+str(month_end).zfill(2)+'01_grid-T.nc'
            ds = xr.open_dataset(file_path)
            bwsalt = ds['sob']
            if ramp_up_bwsalt is None:
                # Initialise
                ramp_up_bwsalt = bwsalt
            else:
                # Accumulate
                ramp_up_bwsalt += bwsalt
    # Convert from integral to average (over months and ensemble members)
    ramp_up_bwsalt /= (num_years*months_per_year*len(suites_by_scenario['ramp_up']))

    # Now read observations of bottom salinity
    schmidtko = read_schmidtko(schmidtko_file=schmidtko_file, eos=eos)
    woa = read_woa(woa_files=woa_files, eos=eos)
    # Regrid to the NEMO grid, giving precedence to Schmidtko where both datasets exist
    schmidtko_interp = interp_latlon_cf(schmidtko, ds, method='bilinear')
    woa_interp = interp_latlon_cf(woa, ds, method='bilinear')
    obs = xr.where(schmidtko_interp.isnull(), woa_interp, schmidtko_interp)
    obs_bwsalt = obs['salt']

    # Make a quick plot
    fig = plt.figure(figsize=(8,3))
    gs = plt.GridSpec(1,3)
    gs.update(left=0.1, right=0.9, bottom=0.05, top=0.8, wspace=0.1)
    ukesm_plot = ramp_up_bwsalt.where(ramp_up_bwsalt!=0).squeeze()
    obs_plot = obs_bwsalt.where(ukesm_plot.notnull()*obs_bwsalt.notnull())
    obs_plot = obs_plot.where(ramp_up_bwsalt!=0).squeeze()
    data_plot = [ukesm_plot, obs_plot, ukesm_plot-obs_plot]
    titles = ['UKESM', 'Observations', 'Model bias']
    vmin = [34, 34, -0.5]
    vmax = [34.85, 34.85, 0.5]
    ctype = ['RdBu_r', 'RdBu_r', 'plusminus']
    for n in range(3):
        ax = plt.subplot(gs[0,n])
        ax.axis('equal')
        img = circumpolar_plot(data_plot[n], ds, ax=ax, masked=True, make_cbar=False, title=titles[n], titlesize=14, vmin=vmin[n], vmax=vmax[n], ctype=ctype[n], lat_max=-63)
        if n != 1:
            cax = cax = fig.add_axes([0.01+0.45*n, 0.1, 0.02, 0.6])
            plt.colorbar(img, cax=cax, extend='both')
    plt.suptitle('Bottom salinity (psu)', fontsize=18)
    finished_plot(fig)

    # Loop over regions and print means and biases
    bias = []
    for region in regions:
        print('\n'+region_names[region])
        mask = region_mask(region, ds, option='shelf')[0]
        dA = ds['area']*mask
        ukesm_mean = (ramp_up_bwsalt*dA).sum(dim=['x','y'])/dA.sum(dim=['x','y'])
        print('UKESM mean '+str(ukesm_mean.data)+' psu')
        # Might have to area-average over a smaller region with missing observational points
        mask_obs = mask.where(obs_bwsalt.notnull())
        dA_obs = ds['area']*mask_obs
        obs_mean = (obs_bwsalt*dA_obs).sum(dim=['x','y'])/dA_obs.sum(dim=['x','y'])
        print('Observational mean '+str(obs_mean.data)+' psu')
        bias.append((ukesm_mean-obs_mean).data)
        print('UKESM bias '+str(bias[-1]))

    return bias


# Calculate the global warming implied by the salinity biases (from above), using a linear regression below 3K for Ross, 5K for FRIS.
def warming_implied_by_salinity_bias (ross_bias=None, fris_bias=None, base_dir='./'):

    pi_suite = 'cs495'
    max_warming_regions = [2, 4.5]
    smooth = 5*months_per_year
    timeseries_file = 'timeseries.nc'
    timeseries_file_um = 'timeseries_um.nc'
    p0 = 0.05

    # Inner function to read global mean SAT from precomputed timeseries
    def global_mean_sat (suite):
        ds = xr.open_dataset(base_dir+'/'+suite+'/'+timeseries_file_um)
        sat = ds['global_mean_sat']
        ds.close()
        return sat
    # Get preindustrial baseline
    baseline_temp = global_mean_sat(pi_suite).mean()

    if ross_bias is None or fris_bias is None:
        # Salinity biases are not precomputed
        [ross_bias, fris_bias] = calc_salinity_bias(base_dir=base_dir)

    for region, bwsalt_bias, max_warming in zip(['ross', 'filchner_ronne'], [ross_bias, fris_bias], max_warming_regions):
        # Assemble timeseries of bwsalt and global warming (below given max) in each ramp-up ensemble member
        warming = None
        bwsalt = None
        for suite in suites_by_scenario['ramp_up']:
            warming_tmp = global_mean_sat(suite) - baseline_temp
            ds = xr.open_dataset(base_dir+'/'+suite+'/'+timeseries_file)
            bwsalt_tmp = ds[region+'_shelf_bwsalt']
            ds.close()
            # Trim and align
            warming_tmp, bwsalt_tmp = align_timeseries(warming_tmp, bwsalt_tmp)
            # Smooth
            warming_tmp = moving_average(warming_tmp, smooth)
            bwsalt_tmp = moving_average(bwsalt_tmp, smooth)
            # Trim to just before the warming threshold is crossed
            t_end = np.argwhere(warming_tmp.data > max_warming)[0][0]
            warming_tmp = warming_tmp.isel(time_centered=slice(0,t_end))
            bwsalt_tmp = bwsalt_tmp.isel(time_centered=slice(0,t_end))
            if warming is None:
                warming = warming_tmp
                bwsalt = bwsalt_tmp
            else:
                warming = xr.concat([warming, warming_tmp], dim='time_centered')
                bwsalt = xr.concat([bwsalt, bwsalt_tmp], dim='time_centered')
        # Now find a linear regression of bwsalt in response to warming
        slope, intercept, r_value, p_value, std_err = linregress(bwsalt, warming)
        if p_value > p0:
            raise Exception('No significant trend')
        implied_warming = slope*bwsalt_bias
        print(region_names[region]+': Salinity bias of '+str(bwsalt_bias)+' psu implies global warming of '+str(implied_warming)+'K')


# Plot cavity-mean temperature beneath Ross and FRIS as a function of shelf-mean bottom water salinity, in all scenarios. Colour the lines based on the global warming level relative to preindustrial, and indicate the magnitude of the salinity bias.
def plot_ross_fris_by_bwsalt (base_dir='./'):

    regions = ['ross', 'filchner_ronne']
    title_prefix = [r'$\bf{a}$. ', r'$\bf{b}$. ']
    bwsalt_bias = [-0.13443893, -0.11137423]  # Precomputed above
    bias_print_x = [34.4, 34.1]
    bias_print_y = -1
    timeseries_file = 'timeseries.nc'
    timeseries_file_um = 'timeseries_um.nc'
    smooth = 5*months_per_year
    pi_suite = 'cs495'
    cmap = ['Reds', 'Blues']
    p0 = 0.05
    tipping_temp = -1.9

    # Inner function to read global mean SAT from precomputed timeseries
    def global_mean_sat (suite):
        ds = xr.open_dataset(base_dir+'/'+suite+'/'+timeseries_file_um)
        sat = ds['global_mean_sat']
        ds.close()
        return sat
    # Get preindustrial baseline
    baseline_temp = global_mean_sat(pi_suite).mean()

    # Read timeseries from every experiment
    all_bwsalt = []
    all_cavity_temp = []
    all_warming = []
    max_warming = 0
    for n in range(len(regions)):
        data_bwsalt = []
        data_cavity_temp = []
        data_warming = []
        direction = []
        bwsalt_tip = []
        bwsalt_recover = []
        for scenario in suites_by_scenario:
            if 'piControl' in scenario or 'static_ice' in scenario:
                continue
            for suite in suites_by_scenario[scenario]:
                # Flag whether temperature is going up (ramp-up or stabilisation) or down (ramp-down)
                if 'ramp_down' in scenario or 'restabilise' in scenario:
                    direction.append(-1)
                elif 'ramp_up' in scenario or 'stabilise' in scenario:
                    direction.append(0)
                ds = xr.open_dataset(base_dir+'/'+suite+'/'+timeseries_file)
                bwsalt = ds[regions[n]+'_shelf_bwsalt']
                cavity_temp = ds[regions[n]+'_cavity_temp']
                ds.close()
                warming = global_mean_sat(suite) - baseline_temp
                # Smooth and align
                if bwsalt.sizes['time_centered'] < smooth:
                    # Simulation hasn't run long enough to include
                    continue
                bwsalt = moving_average(bwsalt, smooth)
                cavity_temp = moving_average(cavity_temp, smooth)
                warming = moving_average(warming, smooth)
                bwsalt = align_timeseries(bwsalt, warming)[0]
                cavity_temp, warming = align_timeseries(cavity_temp, warming)
                max_warming = max(max_warming, warming.max())
                # Throw away any ramp-down data where global temp has overshot preindustrial and gone into negative
                data_bwsalt.append(bwsalt.where(warming>0))
                data_cavity_temp.append(cavity_temp.where(warming>0))
                data_warming.append(warming.where(warming>0))
        all_bwsalt.append(data_bwsalt)
        all_cavity_temp.append(data_cavity_temp)
        all_warming.append(data_warming)

    # Now find the salinity at tipping and recovery points; for this we want trajectories, not suites. It means reading everything again unfortunately - can't find a cleaner way to do this (given some double-tipped trajectories)
    threshold_tip = []
    threshold_recover = []
    suite_lists = all_suite_trajectories()
    for region in regions:
        bwsalt_tip = []
        bwsalt_recover = []
        for suite_list in suite_lists:
            cavity_temp = moving_average(build_timeseries_trajectory(suite_list, region+'_cavity_temp', base_dir=base_dir), smooth)
            bwsalt = moving_average(build_timeseries_trajectory(suite_list, region+'_shelf_bwsalt', base_dir=base_dir), smooth)
            tips, t_tip = check_tip(cavity_temp=cavity_temp, smoothed=True, return_t=True, base_dir=base_dir)
            if tips:
                bwsalt_tip.append(bwsalt.isel(time_centered=t_tip))
                recovers, t_recover = check_recover(cavity_temp=cavity_temp, smoothed=True, return_t=True, base_dir=base_dir)
                if recovers:
                    bwsalt_recover.append(bwsalt.isel(time_centered=t_recover))
        bwsalt_tip = np.unique(bwsalt_tip)
        bwsalt_recover = np.unique(bwsalt_recover)
        # Print some statistics
        print(region)
        print('Shelf salinity at tipping has mean '+str(np.mean(bwsalt_tip))+' psu, std '+str(np.std(bwsalt_tip))+' psu')
        threshold_tip.append(np.mean(bwsalt_tip))
        if np.size(bwsalt_recover) > 0:
            print('Shelf salinity at recovery has mean '+str(np.mean(bwsalt_recover))+' psu, std '+str(np.std(bwsalt_recover))+' psu')
            threshold_recover.append(np.mean(bwsalt_recover))
            # 2-sample t-test to check if they're significantly different
            p_val = ttest_ind(bwsalt_tip, bwsalt_recover, equal_var=False)[1]
            distinct = p_val < p0
            if distinct:
                print('Significant difference (p='+str(p_val)+')')
            else:
                print('No significant difference (p='+str(p_val)+')')
        else:
            threshold_recover.append(None)

    # Set up colour map to vary with global warming level
    norm = plt.Normalize(0, max_warming)
    num_suites = len(all_bwsalt[0])

    # Plot
    fig = plt.figure(figsize=(8,6))
    gs = plt.GridSpec(1,2)
    gs.update(left=0.1, right=0.98, bottom=0.25, top=0.93, wspace=0.2)
    cax1 = fig.add_axes([0.1, 0.12, 0.4, 0.03])
    cax2 = fig.add_axes([0.1, 0.075, 0.4, 0.03]) 
    for n in range(len(regions)):
        ax = plt.subplot(gs[0,n])
        for m in range(num_suites):
            #ax.plot(all_bwsalt[n][m], all_cavity_temp[n][m], '-', linewidth=1)
            # Plot each line with colour varying by global warming level
            points = np.array([all_bwsalt[n][m].data, all_cavity_temp[n][m].data]).T.reshape(-1,1,2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            lc = LineCollection(segments, cmap=truncate_colourmap(cmap[direction[m]], minval=0.3), norm=norm)
            lc.set_array(all_warming[n][m].data)
            lc.set_linewidth(1)
            img = ax.add_collection(lc)
            if direction[m] == 0:
                img_up = img
            else:
                img_down = img
        ax.grid(linestyle='dotted')
        ax.axhline(tipping_temp, color='black', linestyle='dashed')
        # Plot threshold salinity stars
        ax.plot(threshold_tip[n], tipping_temp, marker='*', markersize=15, markerfacecolor='Crimson', markeredgecolor='black')
        if threshold_recover[n] is not None:
            ax.plot(threshold_recover[n], tipping_temp, marker='*', markersize=15, markerfacecolor='DodgerBlue', markeredgecolor='black')
        ax.set_title(title_prefix[n]+region_names[regions[n]], fontsize=16)
        if n==0:
            ax.set_xlabel('Bottom salinity on continental shelf (psu)', fontsize=12)
            ax.set_ylabel('Temperature in ice shelf cavity ('+deg_string+'C)', fontsize=12)
        # Indicate salinity bias
        x_start = bias_print_x[n]
        x_end = bias_print_x[n] + np.abs(bwsalt_bias[n])
        ax.plot([x_start, x_end], [bias_print_y]*2, color='black')
        ax.plot([x_start]*2, [bias_print_y-0.05, bias_print_y+0.05], color='black')
        ax.plot([x_end]*2, [bias_print_y-0.05, bias_print_y+0.05], color='black')
        plt.text(0.5*(x_start+x_end), bias_print_y+0.3, 'Salinity bias of\n'+str(np.round(bwsalt_bias[n],3))+' psu', fontsize=10, color='black', ha='center', va='center')
    # Two colour bars: red on the way up, purple on the way down
    cbar = plt.colorbar(img_up, cax=cax1, orientation='horizontal')
    cbar.set_ticklabels([])
    plt.colorbar(img_down, cax=cax2, orientation='horizontal')
    plt.text(0.3, 0.02, 'Global warming relative to preindustrial ('+deg_string+'C)', ha='center', va='center', fontsize=12, transform=fig.transFigure)
    plt.text(0.51, 0.135, 'ramp-up + stabilise', ha='left', va='center', fontsize=10, transform=fig.transFigure)
    plt.text(0.51, 0.09, 'ramp-down', ha='left', va='center', fontsize=10, transform=fig.transFigure)
    # Manual legend
    colours = ['Crimson', 'DodgerBlue']
    labels = ['mean tipping', 'mean recovery']
    handles = []
    for m in range(len(colours)):
        handles.append(Line2D([0], [0], marker='*', markersize=15, markerfacecolor=colours[m], markeredgecolor='black', label=labels[m], linestyle=''))
    plt.legend(handles=handles, loc='lower right', bbox_to_anchor=(0.85, -0.27))
    finished_plot(fig, fig_name='figures/ross_fris_by_bwsalt.png', dpi=300)


# Plot Amundsen Sea 500m temperature, barotropic velocity, and zero contour of barotropic streamfunction, averaged over 3 scenarios: (1) piControl, (2) 1.5K stabilisation, (3) 6K stabilisation.
def plot_amundsen_temp_velocity (base_dir='./'):

    scenarios = ['piControl', '1.5K', '6K']
    num_scenarios = len(scenarios)
    title_prefix = [r'$\bf{a}$. ', r'$\bf{b}$. ', r'$\bf{c}$. ']
    scenario_titles = ['Preindustrial', '1.5'+deg_string+'C stabilisation', '6'+deg_string+'C stabilisation']
    mean_dir = base_dir + '/time_averaged/'
    depth_temp = 500
    [xmin, xmax] = [-170, -62]
    [ymin, ymax] = [-76, -60]
    vel_scale = 0.4
    domain_cfg = '/gws/nopw/j04/terrafirma/kaight/input_data/grids/domcfg_eORCA1v2.2x.nc'
    [vmin, vmax] = [0, 4.5]

    # Set up grid for plotting
    ds_grid = xr.open_dataset(mean_dir+scenarios[0]+'_grid-T.nc').squeeze()
    lon_edges = cfxr.bounds_to_vertices(ds_grid['bounds_lon'], 'nvertex')
    lat_edges = cfxr.bounds_to_vertices(ds_grid['bounds_lat'], 'nvertex')
    bathy, draft, ocean_mask, ice_mask = calc_geometry(ds_grid)
    ds_domcfg = xr.open_dataset(domain_cfg).squeeze()
    ds_domcfg = ds_domcfg.isel(y=slice(0, ds_grid.sizes['y']))

    # Inner functions
    # Read a variable
    def read_var (var_name, scenario, gtype):
        ds = xr.open_dataset(mean_dir + scenario + '_grid-' + gtype + '.nc')
        data = ds[var_name].squeeze()
        ds.close()
        return data
    # Interpolate a variable to a given depth
    def interp_depth (data_3d, gtype, depth0):
        data_3d = data_3d.where(data_3d!=0)
        dim = 'depth'+gtype.lower()
        data_interp = data_3d.interp({dim:depth0})
        return data_interp
    # Vertically average a velocity component
    def barotropic (vel, gtype):
        dz = read_var('thkcello', scenario, gtype)
        mask_3d = xr.where(vel==0, 0, 1)
        dim = 'depth'+gtype.lower()
        vel_avg = (vel*mask_3d*dz).sum(dim=dim)/(mask_3d*dz).sum(dim=dim)
        return vel_avg
    # Read and vertically average a velocity component and interpolate it to the tracer grid
    def process_vel (scenario, direction):
        data_3d = read_var(direction+'o', scenario, direction.upper())
        data = barotropic(data_3d, direction.upper())
        data_t = interp_grid(data, direction, 't', periodic=True, halo=True)
        return data_t
    # Mask out anything beyond region of interest, plus ice shelf cavities
    def apply_mask (data, mask_shallow=False):
        data = data.where((ds_grid['nav_lon']>=xmin-2)*(ds_grid['nav_lon']<=xmax+2)*(ds_grid['nav_lat']>=ymin-2)*(ds_grid['nav_lat']<=ymax+2))
        data = data.where(np.invert(ice_mask))
        if mask_shallow:
            # Also mask anything shallower than 500m
            data = data.where(bathy >= depth_temp)
        return data

    # Read all the data
    all_temp = []
    all_u = []
    all_v = []
    all_strf = []
    for scenario in scenarios:
        # Temperature, interpolated to 500m
        temp_3d = read_var('thetao', scenario, 'T')
        temp = interp_depth(temp_3d, 'T', depth_temp)
        all_temp.append(apply_mask(temp))
        # Barotropic velocity, interpolated to tracer grid
        u = process_vel(scenario, 'u')
        v = process_vel(scenario, 'v')
        # Rotate to geographic components
        ug, vg = rotate_vector(u, v, domain_cfg, periodic=True, halo=True)
        all_u.append(apply_mask(ug))
        all_v.append(apply_mask(vg))
        # Barotropic streamfunction, interpolated to tracer grid
        ds_u = xr.open_dataset(mean_dir+scenario+'_grid-U.nc').squeeze()
        ds_v = xr.open_dataset(mean_dir+scenario+'_grid-V.nc').squeeze()
        strf = barotropic_streamfunction(ds_u, ds_v, ds_domcfg, periodic=True, halo=True)
        all_strf.append(apply_mask(strf, mask_shallow=True))

    # Plot
    fig = plt.figure(figsize=(6,8))
    gs = plt.GridSpec(num_scenarios, 1)
    gs.update(left=0.1, right=0.95, bottom=0.1, top=0.87, hspace=0.3)
    cax = fig.add_axes([0.2, 0.04, 0.6, 0.02])
    cmap, vmin, vmax = set_colours(all_temp[0], ctype='RdBu_r', vmin=vmin, vmax=vmax)
    mask_plot = np.invert(ocean_mask*np.invert(ice_mask))
    mask_plot = apply_mask(mask_plot)
    mask_plot = mask_plot.where(mask_plot)
    for n in range(num_scenarios):
        ax = plt.subplot(gs[n,0])
        # Shade temperature 
        img = ax.pcolormesh(lon_edges, lat_edges, all_temp[n], cmap=cmap, vmin=vmin, vmax=vmax)
        # Overlay land+ice mask in grey
        ax.pcolormesh(lon_edges, lat_edges, mask_plot, cmap=cl.ListedColormap(['DarkGrey']), linewidth=0)
        # Overlay zero contour of streamfunction
        cs = ax.contour(ds_grid['nav_lon'], ds_grid['nav_lat'], all_strf[n], levels=[0], colors=('magenta'), linewidths=2, linestyles='solid')
        # Overlay every third velocity vector in black
        q = ax.quiver(ds_grid['nav_lon'].data[::4,::4], ds_grid['nav_lat'].data[::4,::4], all_u[n].data[::4,::4], all_v[n].data[::4,::4], scale=vel_scale, color='black')
        ax.set_xlim([xmin, xmax])
        ax.set_ylim([ymin, ymax])
        ax.tick_params(direction='in')
        ax.set_title(title_prefix[n]+scenario_titles[n], fontsize=14)
        if n == 0:
            latlon_axes(ax)
            ax.clabel(cs, cs.levels, inline=True, fmt={0:'0 Sv'}, fontsize=10)
        else:
            ax.set_xticklabels([])
            ax.set_yticklabels([])
        if n == num_scenarios-1:
            plt.colorbar(img, cax=cax, orientation='horizontal', extend='max')
            plt.text(0.18, 0.04, deg_string+'C', fontsize=12, ha='right', va='bottom', transform=fig.transFigure)
            ax.quiverkey(q, 0.9, 0.05, 0.01, '1 cm/s', labelpos='E', coordinates='figure')            
    plt.suptitle('West Antarctic ocean temperature (500m)\nand barotropic velocity', fontsize=15)
    finished_plot(fig, fig_name='figures/amundsen_temp_velocity.png', dpi=300)


# Helper function to construct a nice suite title describing the given trajectory
def trajectory_title (suites):

    if not isinstance(suites, list):
        suites = suites.split('-')
    title = ''
    for suite in suites:
        # Figure out what sort of scenario it is
        for scenario in suites_by_scenario:
            if suite in suites_by_scenario[scenario]:
                if 'ramp_up' in scenario:
                    title += 'Ramp up 8 GtC/y'
                elif 'stabilise' in scenario:
                    title += ', stabilise '+scenario[:scenario.index('K_')]+deg_string+'C'
                elif 'ramp_down' in scenario:
                    # Figure out length of overshoot
                    for length in suites_overshoot_lengths:
                        if suite in suites_overshoot_lengths[length]:
                            title += ' for '+length
                    # Figure out ramp-down rate
                    for rate in suites_ramp_down_rates:
                        if suite in suites_ramp_down_rates[rate]:
                            title += ', ramp down -'+rate
    return title


# Make an animation showing all sorts of things for the given trajectory.
def dashboard_animation (suite_string, region, base_dir='./', out_dir='animations/', only_precompute=False):

    pi_suite = 'cs495'
    timeseries_file = 'timeseries.nc'
    timeseries_file_um = 'timeseries_um.nc'
    smooth = 5*months_per_year
    suite_list = suite_string.split('-')
    if region not in ['ross', 'filchner_ronne']:
        raise Exception('Invalid region '+region)

    # Assemble main title of plot, to describe the trajectory
    sim_title = trajectory_title(suite_list)

    print('Reading timeseries')
    # Assemble timeseries, 5-year smoothed
    # First get preindustrial baseline temp
    ds = xr.open_dataset(base_dir+'/'+pi_suite+'/'+timeseries_file_um)
    baseline_temp = ds['global_mean_sat'].mean()
    ds.close()
    # Global mean warming
    warming = build_timeseries_trajectory(suite_list, 'global_mean_sat', base_dir=base_dir, timeseries_file=timeseries_file_um, offset=-1*baseline_temp)
    warming = moving_average(warming, smooth)
    # Inner function for oceanographic timeseries
    def process_timeseries (var_name):
        data = build_timeseries_trajectory(suite_list, var_name, base_dir=base_dir, timeseries_file = timeseries_file)
        data = moving_average(data, smooth)
        # Align with warming timeseries
        data, warming_tmp = align_timeseries(data, warming)
        return data, warming_tmp
    shelf_bwsalt = process_timeseries(region+'_shelf_bwsalt')[0]
    bwtemp = process_timeseries(region+'_bwtemp')[0]
    cavity_temp = process_timeseries(region+'_cavity_temp')[0]
    # For the last one, overwrite warming timeseries (now aligned)
    massloss, warming = process_timeseries(region+'_massloss')

    # Identify if/when cavity tips or recovers
    tips, date_tip, t_tip = check_tip(cavity_temp=cavity_temp, smoothed=True, return_date=True, return_t=True, base_dir=base_dir)
    if tips:
        tip_string = 'Tips in '+str(date_tip.dt.year.item())+' ('+str(round(warming[t_tip].item(),2))+deg_string+'C)'
        recovers, date_recover, t_recover = check_recover(cavity_temp=cavity_temp, smoothed=True, return_date=True, return_t=True, base_dir=base_dir)
        if recovers:
            recover_string = 'Recovers in '+str(date_recover.dt.year.item())+' ('+str(round(warming[t_recover].item(),2))+deg_string+'C)'
        else:
            recover_string = 'Does not recover'
            recover_time = None                
    else:
        tip_string = 'Does not tip'
        recover_string = ''
        tip_time = None
        recover_time = None

    # Set up maps
    # Open one file just to get coordinates
    # Use the file with the most retreated GL - end of ramp up
    file_list = []
    for f in os.listdir(base_dir+'/cx209'):
        if f.startswith('nemo_cx209o_1m_') and f.endswith('_grid-T.nc'):
            file_list.append(f)
    file_list.sort()
    f = file_list[-1]
    grid = xr.open_dataset(base_dir+'/cx209/'+f)
    # Choose bounds of map to show
    mask = region_mask(region, grid)[0]
    x, y = polar_stereo(grid['nav_lon'], grid['nav_lat'])
    xmin = x.where(mask).min()
    xmax = x.where(mask).max()
    ymin = y.where(mask).min()
    ymax = y.where(mask).max()    
    plot_mask = (x >= xmin)*(x <= xmax)*(y >= ymin)*(y <= ymax)
    # Trim in latitude direction to save memory: find bounds
    # (trimming in longitude is harder with periodic boundary)
    lat_min = grid['nav_lat'].where(plot_mask).min()
    lat_max = grid['nav_lat'].where(plot_mask).max()
    jmin = np.argwhere(grid['nav_lat'].min(dim='x').data > lat_min.item())[0][0] - 1
    jmax = np.argwhere(grid['nav_lat'].max(dim='x').data > lat_max.item())[0][0]
    grid = grid.isel(y=slice(jmin,jmax))
    plot_mask = plot_mask.isel(y=slice(jmin,jmax))
    # Set up plotting coordinates
    lon_edges = cfxr.bounds_to_vertices(grid['bounds_lon'], 'nvertex')
    lat_edges = cfxr.bounds_to_vertices(grid['bounds_lat'], 'nvertex')
    x_edges, y_edges = polar_stereo(lon_edges, lat_edges)
    x_bg, y_bg = np.meshgrid(np.linspace(x_edges.min(), x_edges.max()), np.linspace(y_edges.min(), y_edges.max()))
    mask_bg = np.ones(x_bg.shape)

    # Calculate annual means of bwtemp, bwsalt, ismr within the plotting region.
    print('Reading 2D data')
    precomputed_file = out_dir+'/precomputed/'+suite_string+'_'+region+'.nc'
    if os.path.isfile(precomputed_file):
        ds_2D = xr.open_dataset(precomputed_file)
        num_years = ds_2D.sizes['time_centered']
    else:
        ds_2D = None
        num_years = 0
    # Select the middle month of each 12 year chunk of timeseries data
    for t in range(12*num_years+6, massloss.sizes['time_centered'], 12):
        # What year is it?
        year = massloss.coords['time_centered'][t].dt.year.item()
        print('...'+str(year))
        # What stage in the simulation is it?
        stype = int(massloss.coords['scenario_type'][t].item())
        stype_mapping = [1, 0, 3, 2]
        suite = suite_list[stype_mapping[stype]]
        sim_dir = base_dir+'/'+suite
        # Build list of the file patterns we're going to read for this year
        nemo_files = []
        for f in os.listdir(sim_dir):
            if os.path.isdir(f'{sim_dir}/{f}'): continue
            if f.startswith('nemo_'+suite+'o_1m_'+str(year)) and f.endswith('-T.nc'):
                # Extract date codes
                date_codes = re.findall(r'\d{4}\d{2}\d{2}', f)
                file_pattern = sim_dir+'/nemo_'+suite+'o_1m_'+date_codes[0]+'-'+date_codes[1]+'*-T.nc'
                if file_pattern not in nemo_files:
                    nemo_files.append(file_pattern)
        nemo_files.sort()
        # Expect 12 file patterns (one for each month), each containing 2 files (grid-T and isf-T)
        if len(nemo_files) > months_per_year:
            raise Exception('Too many NEMO files for '+suite+', '+str(year))
        if len(nemo_files) < months_per_year:
            print('Warning: incomplete year for '+suite+', '+str(year))
        # Now read each month
        bwsalt_accum = None
        bwtemp_accum = None
        ismr_accum = None
        for file_pattern in nemo_files:
            ds = xr.open_mfdataset(file_pattern).isel(y=slice(jmin,jmax))
            ds.load()
            ds = ds.swap_dims({'time_counter':'time_centered'}).drop_vars(['time_counter'])
            if os.path.isfile(file_pattern.replace('*','_grid')):
                bwsalt_tmp = ds['sob'].where(ds['sob']>0).where(plot_mask)
                bwtemp_tmp = ds['tob'].where(ds['sob']>0).where(plot_mask)
                if bwsalt_accum is None:
                    bwsalt_accum = bwsalt_tmp
                    bwtemp_accum = bwtemp_tmp
                else:
                    bwsalt_accum = xr.concat([bwsalt_accum, bwsalt_tmp], dim='time_centered')
                    bwtemp_accum = xr.concat([bwtemp_accum, bwtemp_tmp], dim='time_centered')
            else:
                print('Warning: missing '+file_pattern.replace('*','_grid'))
            if os.path.isfile(file_pattern.replace('*','_isf')):
                ismr_tmp = convert_ismr(ds['sowflisf'].where(ds['sowflisf']!=0)).where(plot_mask).drop_vars({'time_counter'})
                if ismr_accum is None:
                    ismr_accum = ismr_tmp
                else:
                    ismr_accum = xr.concat([ismr_accum, ismr_tmp], dim='time_centered')
            else:
                print('Warning: missing '+file_pattern.replace('*','_isf'))
            ds.close()
        # Calculate (annual) means and save
        ds_2D_tmp = xr.Dataset({'bwsalt':bwsalt_accum, 'bwtemp':bwtemp_accum, 'ismr':ismr_accum}).mean(dim='time_centered')
        ds_2D_tmp = ds_2D_tmp.assign_coords(time_centered=bwsalt_accum.coords['time_centered'][6]).expand_dims('time_centered')
        if ds_2D is None:
            ds_2D = ds_2D_tmp
        else:
            ds_2D = xr.concat([ds_2D, ds_2D_tmp], dim='time_centered')
    # Save data for precomputing next time - 2-stage overwrite in case of errors
    print('Writing '+precomputed_file)
    ds_2D.to_netcdf(precomputed_file+'_tmp')
    os.rename(precomputed_file+'_tmp', precomputed_file)
    if only_precompute:
        return
        
    var_plot_2D = ['bwsalt', 'bwtemp', 'ismr']
    titles_2D = ['Bottom salinity (psu)', 'Bottom temperature ('+deg_string+'C)', 'Ice shelf melt rate (m/y)']
    # Calculate variable min/max over all years
    vmin = [ds_2D[var].min() for var in var_plot_2D]
    vmax = [ds_2D[var].max() for var in var_plot_2D]
    ctype = ['viridis', 'viridis', 'ismr']
    cmap = [set_colours(ds_2D[var_plot_2D[n]].isel(time_centered=0), ctype=ctype[n], vmin=vmin[n], vmax=vmax[n])[0] for n in range(3)]
    num_years = ds_2D.sizes['time_centered']

    print('Setting up plot')

    # Initialise the plot
    fig = plt.figure(figsize=(9,7))
    gs = plt.GridSpec(2,3)
    gs.update(left=0.07, right=0.98, bottom=0.08, top=0.86, wspace=0.22, hspace=0.4)
    cax = [fig.add_axes([0.095+0.33*n, 0.04, 0.2, 0.02]) for n in range(3)]
    ax = []
    for n in range(2):
        for m in range(3):
            ax.append(plt.subplot(gs[n,m]))

    # Inner function to plot a frame
    def plot_one_frame (t):
        if t == 0:
            # Trajectory information at the top
            plt.suptitle(suite_string+'\n'+sim_title, fontsize=14)
        # Top row: timeseries: plot 12 months at a time, colour-coded by simulation type
        t_start = t*months_per_year
        t_end = t_start + months_per_year
        if t != 0:
            # Make sure to connect to lines from before
            t_start -= 1
        stype = massloss.coords['scenario_type'][t_start+6].item()
        if stype == 1:
            colour = 'Crimson'
        elif stype in [0, -2]:
            colour = 'Grey'
        elif stype == -1:
            colour = 'DodgerBlue'
        # Top left panel: temperature in cavity vs bottom salinity on continental shelf
        ax[0].plot(shelf_bwsalt[t_start:t_end], cavity_temp[t_start:t_end], color=colour, linewidth=1)
        ax[0].set_xlim([shelf_bwsalt.min()-0.02, shelf_bwsalt.max()+0.02])
        ax[0].set_ylim([cavity_temp.min()-0.02, max(tipping_threshold, cavity_temp.max())+0.02])
        if t == 0:
            ax[0].grid(linestyle='dotted')
            ax[0].axhline(tipping_threshold, color='black', linestyle='dashed')
            ax[0].set_xlabel('Bottom salinity on shelf (psu)', fontsize=10)
            ax[0].set_title('Temperature in cavity ('+deg_string+'C)', fontsize=12)
        # Other two top panels: bottom temperature (shelf+cavity) and massloss vs global warming level
        for n, ts_data, title, offset in zip(range(2), [bwtemp, massloss], ['Bottom temp, shelf+cavity ('+deg_string+'C)', 'Basal mass loss (Gt/y)'], [0.02, 5]):
            ax[n+1].plot(warming[t_start:t_end], ts_data[t_start:t_end], color=colour, linewidth=1)
            ax[n+1].set_xlim([warming.min()-0.02, warming.max()+0.02])
            ax[n+1].set_ylim([ts_data.min()-offset, ts_data.max()+offset])
            if t == 0:
                ax[n+1].grid(linestyle='dotted')
                ax[n+1].set_xlabel('Global warming ('+deg_string+'C)', fontsize=10)
                ax[n+1].set_title(title, fontsize=12)
        # Bottom three panels: maps of 2D data
        for n in range(3):
            # This time clear all previous plotting data - we're not adding to it
            ax[n+3].cla()
            # Shade land in grey
            ocean_mask = xr.where(ds_2D['bwsalt'].isel(time_centered=t)>0, 1, 0)
            ocean_mask = ocean_mask.where(ocean_mask)
            ax[n+3].pcolormesh(x_bg, y_bg, mask_bg, cmap=cl.ListedColormap(['DarkGrey']))
            ax[n+3].pcolormesh(x_edges, y_edges, ocean_mask, cmap=cl.ListedColormap(['white']))
            # Plot the data
            img = ax[n+3].pcolormesh(x_edges, y_edges, ds_2D[var_plot_2D[n]].isel(time_centered=t), cmap=cmap[n], vmin=vmin[n], vmax=vmax[n])
            ax[n+3].set_xlim([xmin, xmax])
            ax[n+3].set_ylim([ymin, ymax])
            ax[n+3].set_xticks([])
            ax[n+3].set_yticks([])
            ax[n+3].set_title(titles_2D[n], fontsize=12)
            if t == 0:
                cbar = plt.colorbar(img, cax=cax[n], orientation='horizontal')
        # Print tipping information
        plt.text(0.99, 0.99, tip_string, fontsize=14, ha='right', va='top', transform=fig.transFigure)
        plt.text(0.99, 0.95, recover_string, fontsize=14, ha='right', va='top', transform=fig.transFigure)
        # Print the year and global warming level
        year_string = str(massloss.coords['time_centered'][t_start+6].dt.year.item())
        temp_string = str(round(warming[t_start+6].item(),2))+deg_string+'C'        
        year_text = plt.text(0.01, 0.99, year_string, fontsize=14, ha='left', va='top', transform=fig.transFigure)
        temp_text = plt.text(0.01, 0.95, temp_string, fontsize=14, ha='left', va='top', transform=fig.transFigure)
        
    # First frame
    plot_one_frame(0)

    # Function to update figure with the given frame
    def animate(t):
        plot_one_frame(t)

    # Call this for each frame
    anim = animation.FuncAnimation(fig, func=animate, frames=list(range(num_years)))
    writer = animation.FFMpegWriter(bitrate=5000, fps=5)
    print('Saving animation')
    anim.save(out_dir+'/'+suite_string+'_'+region+'.mp4', writer=writer)


# Call the batch script precompute_animations.sh for every trajectory. Assumes base_dir and out_dir are default values.
def precompute_all_animations ():

    for suite_list in all_suite_trajectories():
        suite_string = '-'.join(suite_list)
        for region in ['ross', 'filchner_ronne']:
            command = 'sbatch --export=SUITES='+suite_string+',CAVITY='+region+' precompute_animation.sh'
            print(command)
            sbatch_id = subprocess.check_output(command, shell=True, text=True)
            print(sbatch_id)


# Call dashboard_animation for every precomputed file.
# To do: only the ones which have actually changed since last time - look at timestamps on precomputed files vs MP4s?
def animate_all (out_dir='animations/'):

    for f in os.listdir(out_dir+'/precomputed/'):
        if f.endswith('.nc'):
            # Extract suite string and region name
            suite_string = f[:f.index('_')]
            region = f[f.index('_')+1:f.index('.nc')]
            print('Processing '+suite_string+' '+region)
            dashboard_animation(suite_string, region, out_dir=out_dir)
            plt.close('all')


# Plot a histogram of how long after stabilisation tipping happens, for each region. Also print some statistics.
def tipping_time_histogram (base_dir='./', fig_name=None):

    regions = ['ross', 'filchner_ronne']
    smooth = 5*months_per_year
    timeseries_file = 'timeseries.nc'
    num_bins = 10

    all_times = []
    for region in regions:
        times = []
        # Find all trajectories of cavity temperature
        cavity_temp_ts = all_timeseries_trajectories(region+'_cavity_temp', base_dir=base_dir)[0]
        # Loop over them
        for n in range(len(cavity_temp_ts)):
            cavity_temp = moving_average(cavity_temp_ts[n], smooth)
            # Find time index of when stabilisation starts
            if cavity_temp.scenario_type.min() == 1:
                # Perpetual ramp-up; skip it
                continue
            stab_time = np.argwhere(cavity_temp.scenario_type.data==0)[0][0]
            tips, tip_time = check_tip(cavity_temp=cavity_temp, smoothed=True, return_t=True, base_dir=base_dir)
            if tips and  tip_time >= stab_time:
                # Tips after stabilisation
                # Calculate years since emissions stabilised
                times.append((tip_time-stab_time)/months_per_year)
        # Throw away duplicates
        times = np.unique(times)
        all_times.append(times)
        # Print some statistics
        print(region+':')
        print(str(np.size(times))+' tips happen after stabilisation')
        print('Range: '+str(np.amin(times))+' to '+str(np.amax(times))+' years')
        print('Mean: '+str(np.mean(times))+' years')
        print('Standard deviation: '+str(np.std(times))+' years')
    tmax = np.amax([np.amax(times) for times in all_times])
    bins = np.linspace(0, tmax, num=num_bins)

    # Plot
    fig = plt.figure(figsize=(5,5))
    gs = plt.GridSpec(2,1)
    gs.update(left=0.05, right=0.99, bottom=0.1, top=0.85, hspace=0.4)
    for r in range(len(regions)):
        ax = plt.subplot(gs[r,0])
        ax.hist(all_times[r], bins=bins)
        ax.set_title(region_names[regions[r]], fontsize=12)
        if r==0:
            ax.set_ylabel('# simulations', fontsize=10)
        if r==1:
            ax.set_xlabel('years since emissions stopped', fontsize=10)
    plt.suptitle('Tipping points reached after climate stabilisation', fontsize=14)
    finished_plot(fig, fig_name=fig_name)


# After downloading some more variables from MASS for one suite, merge these files (in a subdirectory) into the main files.
def merge_sfc_files (suite='cx209', subdir='sfc'):

    for f in os.listdir(suite+'/'+subdir):
        if not f.endswith('.nc'):
            continue
        print('Processing '+f)
        ds1 = xr.open_dataset(suite+'/'+f)
        ds2 = xr.open_dataset(suite+'/'+subdir+'/'+f)
        ds = xr.merge([ds1, ds2])
        ds.to_netcdf(suite+'/'+f+'_tmp')
        ds.close()
        ds1.close()
        ds2.close()
        os.rename(suite+'/'+f+'_tmp', suite+'/'+f)    


# Calculate some extra timeseries for one suite
def sfc_FW_timeseries (suite='cx209', base_dir='./'):

    update_simulation_timeseries(suite, ['all_iceberg_melt', 'all_pminuse', 'all_runoff', 'all_seaice_meltfreeze'], timeseries_file='timeseries_sfc.nc', sim_dir=base_dir+'/'+suite+'/', freq='m', halo=True, gtype='T')


# Timeseries of various freshwater fluxes, relative to preindustrial baseline, for one trajectory.
def plot_FW_timeseries (base_dir='./'):

    suite_string = 'cx209-cz377-dc130'
    suite_list = suite_string.split('-')
    pi_suite = 'cs568'  # Want evolving ice PI control so the ice mass is partitioned the same way into calving, basal melting, runoff.
    pi_years = 50
    var_names = ['all_massloss', 'all_seaice_meltfreeze', 'all_pminuse', 'all_runoff', 'all_iceberg_melt']
    var_titles = ['Ice shelves', 'Sea ice', 'Precip - evap', 'Surface melt',  'Icebergs']
    timeseries_files = ['timeseries.nc'] + 4*['timeseries_sfc.nc']
    factors = [rho_fw/rho_ice*1e6/sec_per_year] + 4*[1e-3/sec_per_year]
    units = 'mSv'
    colours = [(0.6,0.6,0.6), (0.8,0.47,0.65), (0,0.62,0.45), (0,0.45,0.7), (0.9,0.62,0)]   
    #fw_colours = [(0.6,0.6,0.6), '#e41a1c', '#4daf4a', '#984ea3', '#dede00']
    num_vars = len(var_names)
    #ismr_regions = ['ross', 'filchner_ronne', 'west_antarctica', 'east_antarctica', 'all']
    #ismr_titles = [region_names[region] for region in ismr_regions[:-1]] + ['Total']
    #ismr_colours = [(0,0.45,0.7), (0.8,0.47,0.65), (0,0.62,0.45), (0.9,0.62,0), (0.6,0.6,0.6)]
    tip_regions = ['ross', 'filchner_ronne'] 
    tip_titles = ['Ross', 'FRIS']
    smooth = 20*months_per_year
    stage_colours = ['Crimson', 'white', 'DodgerBlue']
    label_y = 120
    #sample_file = base_dir+'/time_averaged/piControl_grid-T.nc'
    #ds_grid = xr.open_dataset(sample_file).squeeze()
    
    # Inner function to read the variable from the main trajectory and subtract the PI baseline
    def read_var_anomaly (var, fname):
        data = build_timeseries_trajectory(suite_list, var, base_dir=base_dir, timeseries_file=fname)
        ds = xr.open_dataset(base_dir+'/'+pi_suite+'/'+fname)
        baseline = ds[var].isel(time_centered=slice(0,pi_years*months_per_year)).mean(dim='time_centered')
        ds.close()
        return data-baseline

    # Loop over variables and read all the data
    data_plot = []
    for v in range(num_vars):
        data = read_var_anomaly(var_names[v], timeseries_files[v])
        if v == 0:
            # Save first year of simulation, before smoothing
            year0 = data.time_centered[0].dt.year.item()
        # Convert units and smooth in time
        data = moving_average(data, smooth)*factors[v]
        data_plot.append(data)
    #ismr_plot = []
    #for region in ismr_regions:
    #    data = read_var_anomaly(region+'_massloss', timeseries_files[0])
    #    data = moving_average(data, smooth)*factors[0]
    #    ismr_plot.append(data)

    # Find the first and last year of each stage in the simulation
    stage_start = []
    stage_end = []
    for suite in suite_list:
        file_path = base_dir+'/'+suite+'/'+timeseries_files[0]
        ds = xr.open_dataset(file_path)
        stage_start.append(ds.time_centered[0].dt.year.item()-year0)
        stage_end.append(ds.time_centered[-1].dt.year.item()-year0)
        ds.close()
    # Now deal with overlaps
    stage_end[:-1] = stage_start[1:]

    # Find the time of tipping and recovery for each cavity
    tip_times = [check_tip(suite=suite_string, region=region, return_date=True, base_dir=base_dir)[1] for region in tip_regions]
    recover_times = [check_recover(suite=suite_string, region=region, return_date=True, base_dir=base_dir)[1] for region in tip_regions]

    # Plot
    fig = plt.figure(figsize=(8,3.5))
    gs = plt.GridSpec(1,1)
    gs.update(left=0.1, right=0.95, bottom=0.15, top=0.9, hspace=0.05)
    ax = plt.subplot(gs[0,0])
    for v in range(len(data_plot)):
        # Get time axis in years since beginning
        years = np.array([(date.dt.year.item() - year0) + (date.dt.month.item() - 1)/months_per_year + 0.5 for date in data_plot[v].time_centered])
        ax.plot(years, data_plot[v], '-', color=colours[v], label=var_titles[v], linewidth=1)
    # Shade each stage
    for t in range(len(stage_start)):
        ax.axvspan(stage_start[t], stage_end[t], alpha=0.1, color=stage_colours[t])
    # Dashed lines to show tipping and recovery
    for r in range(len(tip_regions)):
        if tip_times[r] is not None:
            tip_year = tip_times[r].dt.year.item() - year0
            ax.axvline(tip_year, color='black', linestyle='dashed', linewidth=1)
            plt.text(tip_year, label_y, tip_titles[r]+' tips', ha='left', va='top', rotation=-90)
            if recover_times[r] is not None:
                recover_year = recover_times[r].dt.year.item() - year0
                ax.axvline(recover_year, color='black', linestyle='dashed', linewidth=1)
                plt.text(recover_year, label_y, tip_titles[r]+' recovers', ha='left', va='top', rotation=-90)
    ax.grid(linestyle='dotted')
    ax.axhline(0, color='black')
    ax.set_ylabel(units)
    ax.set_xlim([stage_start[0], stage_end[-1]])
    ax.set_title('Antarctic freshwater fluxes (anomalies from preindustrial)', fontsize=14)    
    plt.text(616, -12.5, 'years', ha='left', va='top')
    plt.text(0.5, 0.01, trajectory_title(suite_string), ha='center', va='bottom', transform=fig.transFigure, fontsize=12)
    ax.legend(loc='upper left')
    '''n == 1:
        # Little map showing regions
        masks = [region_mask(region, ds_grid, option='cavity')[0] for region in ismr_regions]
        ax2 = inset_axes(ax, '15%', '35%', loc='lower left', borderpad=2)
        ax2.axis('equal')
        for r in range(len(ismr_regions)-1):
            mask = region_mask(ismr_regions[r], ds_grid, option='cavity')[0]
            circumpolar_plot(mask, ds_grid, ax=ax2, make_cbar=False, ctype=colours[r], lat_max=-66, shade_land=(r==0), title='')
        ax2.axis('on')
        ax2.set_xticks([])
        ax2.set_yticks([])'''
    finished_plot(fig, fig_name='figures/FW_timeseries.png', dpi=300)


# Plot shelf bwsalt and its time-derivative for the Ross and FRIS regions in untipped trajectories, with the given level of smoothing (in years).
def plot_untipped_salinity (smooth=30, base_dir='./'):

    regions = ['ross', 'filchner_ronne']
    stype = [1, 0, -1]
    colours = ['Crimson', 'Grey', 'DodgerBlue']
    labels = ['ramp-up', 'stabilise', 'ramp-down']
    suite_sequences = all_suite_trajectories()

    # Make one figure per untipped trajectory per region
    for region in regions:
        # Find untipped trajectories
        for suite_list in suite_sequences:
            suite_string = '-'.join(suite_list)
            if not check_tip(suite=suite_string, region=region):
                bwsalt = moving_average(build_timeseries_trajectory(suite_list, region+'_shelf_bwsalt', base_dir=base_dir), smooth*months_per_year)
                # Time-derivative of smoothed array; convert from days to centuries (30-day months in UKESM)
                ds_dt = bwsalt.differentiate('time_centered', datetime_unit='D')*360*1e2
                # Smooth it again
                ds_dt = moving_average(ds_dt, smooth*months_per_year)
                # Plot different phases in different colours
                fig = plt.figure(figsize=(5,7))
                gs = plt.GridSpec(2,1)
                gs.update(left=0.15, right=0.95, bottom=0.1, top=0.9, hspace=0.2)
                ax1 = plt.subplot(gs[0,0])
                ax2 = plt.subplot(gs[1,0])
                for m in range(len(stype)):
                    index = bwsalt.scenario_type == stype[m]
                    ax1.plot_date(bwsalt.time_centered.where(bwsalt.scenario_type==stype[m], drop=True), bwsalt.where(bwsalt.scenario_type==stype[m], drop=True), '-', color=colours[m], linewidth=1)
                    ax2.plot_date(ds_dt.time_centered.where(ds_dt.scenario_type==stype[m], drop=True), ds_dt.where(ds_dt.scenario_type==stype[m], drop=True), '-', color=colours[m], linewidth=1)
                ax1.set_title('Shelf bottom salinity', fontsize=12)
                ax1.grid(linestyle='dotted')
                ax2.set_title('Time derivative', fontsize=12)
                ax2.axhline(0, color='black', linewidth=0.5)
                ax2.grid(linestyle='dotted')
                plt.suptitle(suite_string+', '+region_names[region], fontsize=14)
                ax1.set_ylabel('psu')
                ax2.set_ylabel('psu/century')
                # Manual legend
                handles = []
                for m in range(len(stype)):
                    handles.append(Line2D([0], [0], color=colours[m], label=labels[m], linestyle='-'))
                ax2.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5,-0.25), ncol=3)
                finished_plot(fig)

# Calculate the timescales for different stages in the tipping process; plot 1x3 timeseries (shelf salinity, cavity temperature, mass loss) for each trajectory and region (optional), and histograms of 4 different timescales.
def stage_timescales (base_dir='./', fig_dir=None, plot_traj=False):

    regions = ['ross', 'filchner_ronne']
    smooth = 10*months_per_year
    titles = ['Bottom salinity on continental shelf', 'Temperature in cavity', 'Basal mass loss']
    units = ['psu', deg_string+'C', 'Gt/y']
    stypes = [1, 0, -1]
    colours = ['Crimson', 'Grey', 'DodgerBlue']
    labels = ['ramp-up', 'stabilise', 'ramp-down']
    num_bins = 10

    # Inner function to calculate years between two dates; timedelta is returning negative weirdly (overflow with dates?)
    def years_between (date1, date2):
        years = date2.dt.year.item() - date1.dt.year.item()
        months = date2.dt.month.item() - date1.dt.month.item()
        return years + months/months_per_year

    # Inner function to find the date at which the given scenario type starts
    def stype_date (data, stype, return_index=False):
        if np.count_nonzero(data.scenario_type==stype)==0:
            t0 = None
            date = None
        else:
            t0 = np.argwhere(data.scenario_type.data==stype)[0][0]
            date = data.time_centered[t0]
        if return_index:
            return date, t0
        else:
            return date

    stab_to_tip_all = []
    tip_to_melt_max_all = []
    ramp_down_to_recovery_all = []
    ramp_down_to_min_s_all = []
    # Loop over regions
    for region in regions:
        all_shelf_bwsalt, suite_strings = all_timeseries_trajectories(region+'_shelf_bwsalt', base_dir=base_dir)
        all_cavity_temp = all_timeseries_trajectories(region+'_cavity_temp', base_dir=base_dir)[0]
        all_massloss = all_timeseries_trajectories(region+'_massloss', base_dir=base_dir)[0]
        stab_to_tip = []
        tip_to_melt_max = []
        ramp_down_to_recovery = []
        ramp_down_to_min_s = []
        # Loop over trajectories
        for suite_string, shelf_bwsalt, cavity_temp, massloss in zip(suite_strings, all_shelf_bwsalt, all_cavity_temp, all_massloss):
            print(suite_string+', '+region)
            tip_time = None
            stab_time = None
            melt_max_time = None
            ramp_down_time = None
            recover_time = None
            min_s_time = None
            shelf_bwsalt_smooth = moving_average(shelf_bwsalt, smooth)
            cavity_temp_smooth = moving_average(cavity_temp, smooth)
            massloss_smooth = moving_average(massloss, smooth)
            # Check for trajectories which tip
            tips, tip_time = check_tip(cavity_temp=cavity_temp, smoothed=False, return_date=True, base_dir=base_dir)
            if tips:
                # Check for trajectories which tip after stabilisation
                stab_time = stype_date(cavity_temp, 0)
                if stab_time is not None and tip_time > stab_time:
                    # Save years between stabilisation and tipping
                    stab_to_tip.append(years_between(stab_time, tip_time))
                # Find time of max smoothed mass loss
                melt_max_time = massloss_smooth.time_centered[massloss_smooth.argmax()]
                # Save years between tipping and max mass loss
                tip_to_melt_max.append(years_between(tip_time, melt_max_time))
                # Check for trajectories which recover
                recovers, recover_time = check_recover(cavity_temp=cavity_temp, smoothed=False, return_date=True, base_dir=base_dir)
                # Check for trajectories which have a ramp-down
                ramp_down_time = stype_date(cavity_temp, -1)
                if ramp_down_time is not None and recovers:
                    # Save years between ramp-down and recovery
                    ramp_down_to_recovery.append(years_between(ramp_down_time, recover_time))                
            else:
                # Untipped
                recovers = False
                # Check for trajectories which have a ramp-down
                ramp_down_time, t0 = stype_date(cavity_temp, -1, return_index=True)
                if ramp_down_time is not None:
                    # Find time of min smoothed salinity after ramp down starts, unless it's in the last 10 years (not stabilised yet)
                    ts = shelf_bwsalt_smooth.isel(time_centered=slice(t0,None)).argmin() + t0
                    if ts < shelf_bwsalt_smooth.sizes['time_centered']-10*months_per_year:
                        min_s_time = shelf_bwsalt_smooth.time_centered[ts]
                        # Save years between ramp-down and min salinity
                        ramp_down_to_min_s.append(years_between(ramp_down_time, min_s_time))
            if plot_traj:
                # Plot
                fig = plt.figure(figsize=(5,8))
                gs = plt.GridSpec(3,1)
                gs.update(left=0.15, right=0.95, bottom=0.05, top=0.92, hspace=0.3)
                data_plot = [shelf_bwsalt_smooth, cavity_temp_smooth, massloss_smooth]
                time = data_plot[0].time_centered
                for n in range(3):
                    ax = plt.subplot(gs[n,0])
                    # Plot different phases in different colours
                    for m in range(len(stypes)):
                        index = shelf_bwsalt_smooth.scenario_type == stypes[m]
                        ax.plot_date(time.where(index, drop=True), data_plot[n].where(index, drop=True), '-', color=colours[m], linewidth=1)
                    ax.set_title(titles[n], fontsize=12)
                    ax.set_ylabel(units[n], fontsize=10)
                    ax.grid(linestyle='dotted')
                    ax.set_xlim([time[0], time[-1]])
                    ymax = ax.get_ylim()[-1]
                    if tips:
                        ax.axvline(tip_time.item(), color='black', linestyle='dashed', linewidth=1)
                        if n==0:
                            plt.text(tip_time.item(), ymax, ' tips', ha='left', va='top')
                    if recovers:
                        ax.axvline(recover_time.item(), color='black', linestyle='dashed', linewidth=1)
                        if n==0:
                            plt.text(recover_time.item(), ymax, ' recovers', ha='left', va='top')
                plt.suptitle(suite_string+', '+region_names[region], fontsize=14)
                if fig_dir is not None:
                    fig_name = fig_dir+'/'+suite_string+'_'+region+'.png'
                else:
                    fig_name = None
                finished_plot(fig, fig_name=fig_name)
        stab_to_tip_all.append(np.unique(stab_to_tip))
        tip_to_melt_max_all.append(np.unique(tip_to_melt_max))
        ramp_down_to_recovery_all.append(np.unique(ramp_down_to_recovery))
        ramp_down_to_min_s_all.append(np.unique(ramp_down_to_min_s))

    # Plot histograms
    def plot_histogram (all_times, title, abbrev):
        tmax = 0
        for times in all_times:
            if len(times) > 0:
                tmax = max(tmax, np.amax(times))
        bins = np.linspace(0, tmax, num=num_bins)
        fig = plt.figure(figsize=(5,5))
        gs = plt.GridSpec(2,1)
        gs.update(left=0.1, right=0.99, bottom=0.1, top=0.85, hspace=0.4)
        for r in range(len(regions)):
            ax = plt.subplot(gs[r,0])
            ax.hist(all_times[r], bins=bins)
            ax.set_title(region_names[regions[r]], fontsize=12)
            if r==0:
                ax.set_ylabel('# simulations', fontsize=10)
            if r==1:
                ax.set_xlabel('years', fontsize=10)
            if len(all_times[r]) > 0:
                plt.text(0.02, 0.98, 'Mean '+str(int(np.round(np.mean(all_times[r]))))+' years', ha='left', va='top', transform=ax.transAxes)
                plt.text(0.02, 0.88, 'Range '+str(int(round(np.amin(all_times[r]))))+'-'+str(int(round(np.amax(all_times[r]))))+' years', ha='left', va='top', transform=ax.transAxes)
                plt.text(0.02, 0.78, 'n='+str(np.size(all_times[r])), ha='left', va='top', transform=ax.transAxes)
        plt.suptitle(title, fontsize=14)
        if fig_dir is not None:
            fig_name = fig_dir+'/histogram_'+abbrev+'.png'
        else:
            fig_name = None
        finished_plot(fig, fig_name=fig_name)

    for all_times, title, abbrev in zip([stab_to_tip_all, tip_to_melt_max_all, ramp_down_to_recovery_all, ramp_down_to_min_s_all], ['climate stabilisation and tipping', 'tipping and maximum basal mass loss', 'ramp-down and recovery', 'ramp-down and minimum salinity'], ['stab_to_tip', 'tip_to_melt_max', 'ramp_down_to_recovery', 'ramp_down_to_min_s']):
        plot_histogram(all_times, 'Time between '+title, abbrev)

        
def fix_all_missing_months (base_dir='./'):

    file_names = ['timeseries.nc', 'timeseries_um.nc']
    for scenario in suites_by_scenario:
        for suite in suites_by_scenario[scenario]:
            print('Checking '+suite)
            for timeseries_file in file_names:                
                fix_missing_months(base_dir+'/'+suite+'/'+timeseries_file)


def check_all_nans (base_dir='./'):

    file_names = ['timeseries.nc', 'timeseries_um.nc']
    var_lists = [['all_massloss', 'all_bwtemp'], ['global_mean_sat']]

    for scenario in suites_by_scenario:
        for suite in suites_by_scenario[scenario]:
            print('Checking '+suite)
            for timeseries_file, var_names in zip(file_names, var_lists):
                check_nans(base_dir+'/'+suite+'/'+timeseries_file, var_names=var_names)


# Plot a series of maps showing snapshots of the given variable in each cavity for selected scenarios: initial state, tipping point, 100 years later, recovery point. Works for bwtemp, bwsalt, ismr. Later add icevel (BISICLES output).
def map_snapshots (var_name='bwtemp', base_dir='./'):

    regions = ['ross', 'filchner_ronne']
    subfig = ['a) ', 'b) ']
    num_regions = len(regions)
    num_snapshots = 4
    suite_strings = ['cx209-cz376-dc123', 'cx209-cz377-dc130']
    year_titles = [['Initial', 'Tipping', '100 years later', 'Recovery'] for n in range(num_regions)]
    mask_pad = 5e4

    # Set variable title, NEMO name, units
    if var_name == 'bwtemp':
        var_title = 'Bottom ocean temperature'
        units = deg_string+'C'
        nemo_var = 'tob'
        vmin = -2
        vmax = 3.5
        ctype = 'RdBu_r'
        colour_GL = 'yellow'
    elif var_name == 'bwsalt':
        var_title = 'Bottom ocean salinity'
        units = 'psu'
        nemo_var = 'sob'
        vmin = 33
        vmax = 34.9
        ctype = 'viridis'
        colour_GL = 'white'
    elif var_name == 'ismr':
        var_title = 'Ice shelf melt rate'
        units = 'm/y'
        nemo_var = 'sowflisf'
        vmin = -2
        vmax = 20
        ctype = 'ismr'
        colour_GL = 'blue'
    elif var_name == 'icevel':
        raise Exception('Not yet coded icevel case')
    else:
        raise Exception('Invalid variable '+var_name)

    # Construct suite titles describing each trajectory
    suite_titles = [trajectory_title(suites) for suites in suite_strings]

    # Inner function to figure out which suite in the trajectory the given year is, given a list of suites and corresponding starting years
    def find_suite (year, suite_list, start_years):
        # Loop backwards through suites
        for suite, year0 in zip(suite_list[::-1], start_years[::-1]):
            if year >= year0:
                return suite
        raise Exception('Year '+str(year)+' is too early')

    # Inner function to find the most retreated mask in the given direction from the list of masks, and set the plotting bound to be a bit larger than that
    def set_bound (masks, coord, option):
        if option == 'min':
            bound = coord.max()
        elif option == 'max':
            bound = coord.min()
        else:
            raise Exception('Invalid option '+option)
        for mask in masks:
            if option == 'min':
                bound = min(bound, coord.where(mask).min())
            elif option == 'max':
                bound = max(bound, coord.where(mask).max())
        if option == 'min':
            bound -= mask_pad
        elif option == 'max':
            bound += mask_pad
        return bound
    
    # Loop over regions and read all the things we need
    data_plot = []
    omask_plot = []
    omask_GL = []
    imask_front = []
    x_bounds = []
    y_bounds = []
    for n in range(num_regions):
        # Find initial, tipping, and recovery years and suites
        tips, date_tip = check_tip(suite=suite_strings[n], region=regions[n], return_date=True, base_dir=base_dir)
        if not tips:
            raise Exception(suite_string+' does not tip')
        recovers, date_recover = check_recover(suite=suite_strings[n], region=regions[n], return_date=True, base_dir=base_dir)
        if not recovers:
            raise Exception(suite_strings[n]+' does not recover')
        # Find starting year of each suite in trajectory
        start_years = []
        suite_list = suite_strings[n].split('-')
        for suite in suite_list:
            file_names = []
            for f in os.listdir(base_dir+'/'+suite):
                if f.startswith('nemo_'+suite+'o_1m_') and f.endswith('-T.nc'):
                    file_names.append(f)
            file_names.sort()
            date_code = re.findall(r'\d{4}\d{2}\d{2}', file_names[0])
            start_years.append(int(date_code[0][:4]))
        # Now locate the years and suites we want to plot
        plot_years = [start_years[0], date_tip.dt.year.item(), date_tip.dt.year.item()+100, date_recover.dt.year.item()]
        plot_suites = [find_suite(year, suite_list, start_years) for year in plot_years]
        # Add the tipping and recovery years (since initial) to the titles
        for m in [1, 3]:
            year_titles[n][m] += ' (year '+str(plot_years[m]-plot_years[0])+')'
        # Now read the data for the four years, annually averaging
        data_region = []
        omask_region = []
        imask_region = []
        for year, suite in zip(plot_years, plot_suites):
            if var_name == 'icevel':
                raise Exception('Not yet coded icevel case')
            else:
                # Build a list of the NEMO file patterns to read for this year
                files_to_read = []
                for f in os.listdir(base_dir+'/'+suite):
                    if f.startswith('nemo_'+suite+'o_1m_'+str(year)) and f.endswith('-T.nc'):
                        date_codes = re.findall(r'\d{4}\d{2}\d{2}', f)
                        file_pattern = base_dir+'/'+suite+'/nemo_'+suite+'o_1m_'+date_codes[0]+'-'+date_codes[1]+'*-T.nc'
                        if file_pattern not in files_to_read:
                            files_to_read.append(file_pattern)
                files_to_read.sort()
                # Check there are 12 files
                if len(files_to_read) != months_per_year:
                    raise Exception(str(len(files_to_read))+' files found for '+suite+', year '+str(year))
                # Annually average, carefully with masks
                data_accum = None
                ocean_mask = None
                ice_mask = None
                for file_pattern in files_to_read:
                    ds = xr.open_mfdataset(file_pattern)
                    ds.load()
                    ds = ds.swap_dims({'time_counter':'time_centered'}).drop_vars(['time_counter'])
                    data_tmp = ds[nemo_var].where(ds[nemo_var]!=0)
                    ocean_mask_tmp = region_mask(regions[n], ds)[0]
                    ice_mask_tmp = region_mask(regions[n], ds, option='cavity')[0]
                    if data_accum is None:
                        data_accum = data_tmp
                        ocean_mask = ocean_mask_tmp
                        ice_mask = ice_mask_tmp
                        if n==0:
                            # Set up plotting coordinates
                            x, y = polar_stereo(ds['nav_lon'], ds['nav_lat'])
                            lon_edges = cfxr.bounds_to_vertices(ds['bounds_lon'], 'nvertex')
                            lat_edges = cfxr.bounds_to_vertices(ds['bounds_lat'], 'nvertex')
                            x_edges, y_edges = polar_stereo(lon_edges, lat_edges)
                            x_bg, y_bg = np.meshgrid(np.linspace(x_edges.min(), x_edges.max()), np.linspace(y_edges.min(), y_edges.max()))
                            mask_bg = np.ones(x_bg.shape)
                    else:
                        data_accum = xr.concat([data_accum, data_tmp], dim='time_centered')
                        # Save most retreated GL
                        ocean_mask = xr.where(ocean_mask_tmp+ocean_mask, True, False)
                        ice_mask = xr.where(ice_mask_tmp+ice_mask, True, False)
                    ds.close()
                data_mean = data_accum.mean(dim='time_centered')
                if var_name == 'ismr':
                    data_mean = convert_ismr(data_mean)
                data_region.append(data_mean)
                omask_region.append(ocean_mask)
                imask_region.append(ice_mask)
        data_plot.append(data_region)
        omask_plot.append(omask_region)
        # Set plotting bounds on x (based on cavity) and y (based on cavity and shelf)
        xmin = set_bound(imask_region, x, 'min')
        xmax = set_bound(imask_region, x, 'max')
        ymin = set_bound(omask_region, y, 'min')
        ymax = set_bound(omask_region, y, 'max')
        x_bounds.append([xmin, xmax])
        y_bounds.append([ymin, ymax])            
        # Prepare initial GL and ice front for contouring
        omask_ini = omask_region[0].astype('float')
        imask_ini = imask_region[0].astype('float')
        mask_sum = omask_ini + imask_ini  # 0 in grounded ice, 1 in open ocean, 2 in cavity
        # Mask open ocean for GL
        omask_GL.append(omask_ini.where(mask_sum!=1))
        # Mask grounded ice for ice front
        imask_front.append(imask_ini.where(mask_sum!=0))
        # Print actual min and max in this region to help choosing manual bounds
        plot_region = (x>=xmin)*(x<=xmax)*(y>=ymin)*(y<=ymax)
        vmin_real = np.amin([data.where(plot_region).min() for data in data_region])
        vmax_real = np.amax([data.where(plot_region).max() for data in data_region])
        print(regions[n]+' bounds from '+str(vmin_real)+' to '+str(vmax_real))

    # Plot
    cmap = set_colours(data_plot[0][0], ctype=ctype, vmin=vmin, vmax=vmax)[0]
    fig = plt.figure(figsize=(7,5))
    gs = plt.GridSpec(num_regions, num_snapshots)
    gs.update(left=0.02, right=0.98, bottom=0.1, top=0.86, wspace=0.1, hspace=0.55)
    for n in range(num_regions):
        for m in range(num_snapshots):
            ax = plt.subplot(gs[n,m])
            if var_name == 'icevel':
                raise Exception('Not yet coded icevel case')
            else:            
                # Shade land in grey
                omask = omask_plot[n][m].where(omask_plot[n][m])
                ax.pcolormesh(x_bg, y_bg, mask_bg, cmap=cl.ListedColormap(['DarkGrey']))
                # Clear open ocean back to white
                # TODO: this is only continental shelf
                ax.pcolormesh(x_edges, y_edges, omask, cmap=cl.ListedColormap(['white']))
                # Plot the data
                img = ax.pcolormesh(x_edges, y_edges, data_plot[n][m], cmap=cmap, vmin=vmin, vmax=vmax)
                # Contour initial GL in given colour
                ax.contour(x, y, omask_GL[n], levels=[0.5], colors=(colour_GL), linewidths=0.5)
                # Contour ice front in black
                ax.contour(x, y, imask_front[n], levels=[0.5], colors=('black'), linewidths=0.5)
                ax.set_xlim(x_bounds[n])
                ax.set_ylim(y_bounds[n])
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title(year_titles[n][m], fontsize=12)
        plt.text(0.5, 0.99-0.46*n, subfig[n]+region_names[regions[n]]+' Ice Shelf', ha='center', va='top', fontsize=14, transform=fig.transFigure)
        plt.text(0.5, 0.95-0.46*n, suite_titles[n], ha='center', va='top', fontsize=10, transform=fig.transFigure)
    cax = fig.add_axes([0.51, 0.05, 0.4, 0.02])
    cbar = plt.colorbar(img, cax=cax, orientation='horizontal', extend='both')
    cbar.ax.tick_params(labelsize=8)
    plt.text(0.49, 0.03, var_title+' ('+units+')', ha='right', va='bottom', fontsize=12, transform=fig.transFigure)
    finished_plot(fig, fig_name='figures/map_snapshots_'+var_name+'.png', dpi=300)


def plot_SLR_timeseries (base_dir='./'):

    vaf_dir = '/gws/nopw/j04/terrafirma/tm17544/TerraFIRMA_overshoots/processed_data/netcdf_files/'
    file_head = 'vaf_'
    file_tail = '_timeseries.nc'
    pi_suite = 'cs568'  # Evolving ice
    baseline_suite = 'cx209'  # First member ramp-up
    regions = ['ross', 'filchner_ronne']
    num_regions = len(regions)
    colours = ['DarkGrey', 'Crimson', 'DodgerBlue']
    labels = ['untipped', 'tipped', 'recovered']
    trend_years = 200  # Calculate drift in control over the last 200 years
    total_years = 1000  # Extend control to be 1000 years long (just needs to be longer than any other trajectory)

    # Inner function to add the given DataArray to the axes with the given colour
    def add_line (data, ax, colour, year0):
        ax.plot(data.coords['time']-year0, data, '-', color=colour, linewidth=0.8)

    # Set up plot
    fig = plt.figure(figsize=(4.5,6))
    gs = plt.GridSpec(num_regions, 1)
    gs.update(left=0.13, right=0.87, bottom=0.15, top=0.9, hspace=0.3)
    for n in range(num_regions):
        ax = plt.subplot(gs[n,0])
        # Get baseline initial VAF from first member ramp-up (should be consistent between members as evolving ice has just been switched on)
        ds = xr.open_dataset(vaf_dir+'/'+file_head+baseline_suite+file_tail)
        vaf0 = ds[regions[n]+'_vaf'].isel(time=0)
        year0 = ds['time'].isel(time=0)
        ds.close()
        num_tip = 0
        num_recover = 0
        pi_slr = None
        # Loop over scenarios and check if file exists
        for scenario in suites_by_scenario:
            if scenario in ['piControl', 'ramp_up_static_ice']:
                continue
            for suite in suites_by_scenario[scenario]:
                file_path = vaf_dir + '/' + file_head + suite + file_tail
                if not os.path.isfile(file_path):
                    print('Warning: '+suite+' missing')
                    continue
                # Read data
                ds = xr.open_dataset(file_path)
                # Offset of 1 year as per Tom's email (BISICLES output is snapshot at beginning of next year)
                time = ds['time'] - 1
                vaf = ds[regions[n]+'_vaf']
                # Convert from VAF to sea level rise in cm
                if suite == pi_suite:
                    # Different initial state to the rest
                    baseline = vaf.isel(time=0)
                else:
                    baseline = vaf0
                slr = (vaf-baseline)*vaf_to_gmslr*1e2
                if suite == pi_suite:
                    # Calculate linear trend over last 200 years of control
                    pi_trend = linregress(time.data[-trend_years:], slr.data[-trend_years:])[0]
                    print(regions[n]+' drift at end of control simulation is '+str(pi_trend)+' cm/y')
                    # Extend the control suite with this linear trend                    
                    time_extend = np.arange(time[-1]+1, time[0]+total_years)
                    slr_extend = slr.data[-1] + pi_trend*(time_extend - time[-1].data)
                    slr_extend = xr.DataArray(data=slr_extend, dims=['time'], coords=dict(time=time_extend+1))
                    pi_slr = xr.concat([slr, slr_extend], dim='time')
                else:
                    # Subtract drift
                    slr_trim, pi_slr_trim = align_timeseries(slr, pi_slr, time_coord='time')
                    if not (slr_trim==slr).all():
                        # Shouldn't have needed to trim the base timeseries
                        raise Exception('Problem with aligning timeseries')
                    slr = slr_trim - pi_slr_trim
                    tips, date_tip = check_tip(suite=suite, region=regions[n], return_date=True, base_dir=base_dir)
                    if tips:
                        year_tip = date_tip.dt.year
                        if year_tip <= time[-1]:
                            num_tip += 1
                        else:
                            print('Warning: '+suite+' does not extend to tipping date')
                        # Select untipped section
                        untipped = slr.where(time < year_tip, drop=True)  # 1-year offset as before
                        recovers, date_recover = check_recover(suite=suite, region=regions[n], return_date=True, base_dir=base_dir)
                        if recovers:
                            year_recover = date_recover.dt.year
                            if year_recover <= time[-1]:
                                num_recover += 1
                            else:
                                print('Warning: '+suite+' does not extend to recovery date')
                            tipped = slr.where((time >= year_tip)*(time < year_recover), drop=True)
                            recovered = slr.where(time >= year_recover, drop=True)
                            add_line(recovered, ax, colours[labels.index('recovered')], year0)
                        else:
                            tipped = slr.where(time >= year_tip, drop=True)
                        add_line(tipped, ax, colours[labels.index('tipped')], year0)
                    else:
                        untipped = slr
                        recovers = False
                    add_line(untipped, ax, colours[labels.index('untipped')], year0)
        print(regions[n]+': '+str(num_tip)+' tipped, '+str(num_recover)+' recovered')
        ax.grid(linestyle='dotted')
        ax.axhline(0, color='black', linewidth=0.5)
        if n == 1:
            ax.set_xlabel('Years')
        ax.set_ylabel('cm')
        ax.set_xlim([0, None])
        ax.set_title(region_names[regions[n]]+' catchment', fontsize=12)
        plt.suptitle('Sea level contribution', fontsize=14)
    # Manual legend
    handles = []
    for m in range(len(colours)):
        handles.append(Line2D([0], [0], color=colours[m], label=labels[m], linestyle='-'))
    ax.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, -0.45), ncol=3)
    finished_plot(fig, fig_name='figures/SLR_timeseries.png', dpi=300)
        
                

    
        
    
    
    
        
    

    
    


    
    
    

    

    
            
            

    



        
                
                  
            
    
       

    
        



                    

    
