# Analyse / check output from runs with adapted eANT025 (NEMO 4.2)

import xarray as xr
import matplotlib.pyplot as plt
import glob

from ..plots import finished_plot, plot_hovmoeller, animate_2D_circumpolar
from ..timeseries import calc_hovmoeller_region
from ..projects.evaluation import bottom_TS_vs_obs


# Calculate and plot domain-wide sea surface height time series:
def plot_SSH_trend(run_folder, fig_name, dpi=None):

    # Load SSH into dataset and calculate domain-wide average, minimum, maximum:
    gridT_files = glob.glob(f'{run_folder}/*grid_T*')
    SSH_ds  = xr.open_mfdataset(gridT_files) # load all the gridT files in the run folder
    SSH_ave = ((SSH_ds.zos*SSH_ds.area_grid_T).sum(dim=('x_grid_T', 'y_grid_T')) / SSH_ds.area_grid_T.sum(dim=('x_grid_T', 'y_grid_T')))
    SSH_max = SSH_ds.zos.max(dim=('x_grid_T','y_grid_T'))
    SSH_min = SSH_ds.zos.min(dim=('x_grid_T','y_grid_T'))

    # Visualize:
    fig, ax = plt.subplots(3,1, figsize=(12,10))
    SSH_ave.plot(ax=ax[0], xlim=(SSH_ave.time_counter[0], SSH_ave.time_counter[-1]))
    SSH_max.plot(ax=ax[1], xlim=(SSH_max.time_counter[0], SSH_max.time_counter[-1]))
    SSH_min.plot(ax=ax[2], xlim=(SSH_min.time_counter[0], SSH_min.time_counter[-1]))
    # horizontal reference lines:
    ax[0].hlines(y=SSH_ave[0], xmin=SSH_ave.time_counter[0], xmax=SSH_ave.time_counter[-1], color='k', linestyle='--')
    ax[1].hlines(y=SSH_max[0], xmin=SSH_max.time_counter[0], xmax=SSH_max.time_counter[-1], color='k', linestyle='--')
    ax[2].hlines(y=SSH_min[0], xmin=SSH_min.time_counter[0], xmax=SSH_min.time_counter[-1], color='k', linestyle='--')
    # labels:
    ax[0].set_ylabel('Domain average SSH')
    ax[1].set_ylabel('Domain maximum SSH')
    ax[2].set_ylabel('Domain minimum SSH')
    for axis in ax.ravel():
        axis.set_xlabel('')

    finished_plot(fig, fig_name=fig_name, dpi=dpi)
    
    return

# Evaluate bottom temperature and salinity with WOA output (two plots: (1) average over time series (2) end state of timeseries)
def plot_WOA_eval(run_folder, figname1, figname2):

    # Load gridT files into dataset:
    gridT_files = glob.glob(f'{run_folder}/*grid_T*')
    nemo_ds     = xr.open_mfdataset(gridT_files) # load all the gridT files in the run folder

    nemo_ds = nemo_ds.rename({'e3t':'thkcello', 'x_grid_T':'x', 'y_grid_T':'y', 'area_grid_T':'area', 'e3t':'thkcello',
                              'nav_lon_grid_T':'nav_lon', 'nav_lat_grid_T':'nav_lat', 
                              'bounds_nav_lon_grid_T':'bounds_lon', 'bounds_nav_lat_grid_T':'bounds_lat',
                              'nvertex_grid_T':'nvertex'})
    
    # Average full time series:
    bottom_TS_vs_obs(nemo_ds.mean(dim='time_counter'), time_ave=False, fig_name=figname1)
    # End state (last year of run):
    bottom_TS_vs_obs(nemo_ds.isel(time_counter=slice(-12,None)).mean(dim='time_counter'), time_ave=False, fig_name=figname2)
    return

# Visualize the timeseries of variable averaged over a region to see convection
def plot_hovmoeller_convect(run_folder, region, figname1, figname2, title='', tlim=(-1.5, 0.5), slim=(34.8, 34.86)):

    T_region = calc_hovmoeller_region('thetao', region, run_folder=run_folder)    
    S_region = calc_hovmoeller_region('so', region, run_folder=run_folder)

    plot_hovmoeller(T_region, title=title, ylim=(5500,0), vlim=tlim, fig_name=figname1, varname='Temperature (C)')
    plot_hovmoeller(S_region, title=title, ylim=(5500,0), vlim=slim, fig_name=figname2, varname='Salinity')

    return

# Create animations of some standard variables that are useful to look at
def animate_vars(run_folder):
    import cmocean 

    var   = ['mldr10_1', 'siconc', 'zos', 'sbt', 'sbs', 'sosst', 'sosss']
    vlims = [(0,1000), (0,1), (-3,3), (-2,2), (34.4,35.0), (-5,5), (30,35)]
    stub  = ['grid_T', 'icemod', 'grid_T', 'grid_T', 'grid_T', 'grid_T', 'grid_T']
    cmaps = ['viridis', 'viridis', cmocean.cm.balance, cmocean.cm.balance, cmocean.cm.haline, cmocean.cm.balance, cmocean.cm.haline]

    for v in range(len(var)):
        animate_2D_circumpolar(run_folder, var[v], stub[v], vlim=vlims[v], cmap=cmaps[v])

    return