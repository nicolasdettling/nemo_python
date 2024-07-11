# Analyse / check output from runs with adapted eANT025 (NEMO 4.2)

import xarray as xr
import matplotlib.pyplot as plt
import glob
import numpy as np
import cmocean

from ..plots import finished_plot, plot_hovmoeller, animate_2D_circumpolar, plot_transect
from ..utils import moving_average, distance_along_transect
from ..timeseries import calc_hovmoeller_region
from ..projects.evaluation import bottom_TS_vs_obs
from ..constants import region_names, transect_amundsen
from ..file_io import read_dutrieux
from ..grid import transect_coords_from_latlon_waypoints

# Calculate and plot domain-wide sea surface height time series:
# Function plots two timeseries as one extended timeseries figure (ds1, then ds2)
# Inputs:
# path_ts1 : string of path to timeseries1 (for example, dataset of spin up timeseries)
# path_ts2 : string of path to timeseries2 (for example, dataset of results timeseries)
# var      : string name of variable to plot
# regions  : list of strings of regions to visualize
# smooth   : length of window for moving average
def plot_extended_timeseries(path_ts1, path_ts2, var, fig_name, title='',
                             regions=['amundsen_sea','bellingshausen_sea','larsen','filchner_ronne','ross', 'amery', 'all'],
                             colours=['IndianRed', 'SandyBrown', 'LightGreen', 'MediumTurquoise', 'Plum', 'Pink', 'gray'], dpi=None, smooth=0):
    # load timeseries datasets
    ds1 = xr.open_dataset(f'{path_ts1}')
    ds2 = xr.open_dataset(f'{path_ts2}')

    fig, (ax, ax2) = plt.subplots(1,2, figsize=(12,7), sharey=True, facecolor='w')
    plt.subplots_adjust(wspace=0)
    
    # plot the same data on both axes
    for region, colour in zip(regions, colours):    
        ds1_plot = moving_average(ds1[f'{region}_{var}'], smooth)
        ds2_plot = moving_average(ds2[f'{region}_{var}'], smooth)
        ax.plot(ds1['time_centered'][0:-smooth], ds1_plot, c=colour)
        ax2.plot(ds2['time_centered'][0:-smooth], ds2_plot, c=colour, label=region_names[region])
    
    # hide the spines between ax and ax2
    ax.spines['right'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    ax.yaxis.tick_left()
    ax2.yaxis.tick_right()
    ax2.legend()
    # print(ds1[f'{region}_{var}'].long_name.split(f'for {region_names[region]}')[0])
    ylabel = f"{ds1[f'{region}_{var}'].long_name.split('for')[0]} ({ds1[f'{region}_{var}'].units})"
    ax.set_ylabel(ylabel)
    fig.suptitle(title)
    
    finished_plot(fig, fig_name=fig_name, dpi=dpi)
    
    return

def plot_SSH_trend(run_folder, fig_name, style='lineplot', dpi=None, nemo_mesh='/gws/nopw/j04/anthrofail/birgal/NEMO_AIS/bathymetry/mesh_mask-20240305.nc'):
    # Load meshmask
    mesh_ds    = xr.open_dataset(nemo_mesh)

    # Load SSH into dataset and calculate domain-wide average, minimum, maximum:
    gridT_files = glob.glob(f'{run_folder}files/*grid_T*')
    SSH_ds      = xr.open_mfdataset(gridT_files) # load all the gridT files in the run folder
    ocean_area  = xr.where(mesh_ds.tmask.isel(time_counter=0, nav_lev=0).values==0, 0, SSH_ds.area_grid_T)

    SSH_ave = ((SSH_ds.zos*ocean_area).sum(dim=('x_grid_T', 'y_grid_T')) / ocean_area.sum(dim=('x_grid_T', 'y_grid_T')))
    SSH_max = SSH_ds.zos.max(dim=('x_grid_T','y_grid_T'))
    SSH_min = SSH_ds.zos.min(dim=('x_grid_T','y_grid_T'))

    if style=='lineplot':
        # Visualize:
        fig, ax = plt.subplots(3,1, figsize=(12,10), dpi=300)
        SSH_ave.plot(ax=ax[0], xlim=(SSH_ave.time_counter[0], SSH_ave.time_counter[-1] + np.timedelta64(30,'D')))
        SSH_max.plot(ax=ax[1], xlim=(SSH_max.time_counter[0], SSH_max.time_counter[-1] + np.timedelta64(30,'D')))
        SSH_min.plot(ax=ax[2], xlim=(SSH_min.time_counter[0], SSH_min.time_counter[-1] + np.timedelta64(30,'D')))
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
    elif style=='histogram':
        import seaborn as sns
        import cmocean
        time  = np.repeat(SSH_ds.time_counter.values, SSH_ds.dims['x_grid_T']*SSH_ds.dims['y_grid_T'])
        SSH   = SSH_ds.zos.stack(cells=['x_grid_T','y_grid_T']).values.flatten()
        timem = np.delete(time,(SSH==0) | (np.isnan(SSH)))
        SSHm  = np.delete(SSH, (SSH==0) | (np.isnan(SSH)))
        aream = np.delete(ocean_area.values, (SSH==0) | (np.isnan(SSH)))

        fig, ax = plt.subplots(1,1, figsize=(12,6))
        sns.histplot(x=timem, y=SSHm, ax=ax, bins=(SSH_ds.dims['time_counter'], 50), weights=aream, \
                     cmap=cmocean.cm.matter,cbar=True, cbar_kws=dict(shrink=.75))
        SSH_ave.plot(ax=ax, c='k')
        ax.set_ylim(-2, 0.5)

    else:
        raise Exception('Style can be either lineplot or histogram')

    finished_plot(fig, fig_name=fig_name, dpi=dpi)
    
    return

# Evaluate bottom temperature and salinity with WOA output (two plots: (1) average over time series (2) end state of timeseries)
def plot_WOA_eval(run_folder, figname1, figname2, figname3, nemo_mesh='/gws/nopw/j04/anthrofail/birgal/NEMO_AIS/bathymetry/mesh_mask-20240305.nc', dpi=None):

    # Load gridT files into dataset:
    gridT_files = glob.glob(f'{run_folder}files/*grid_T*')
    nemo_ds     = xr.open_mfdataset(gridT_files) # load all the gridT files in the run folder

    nemo_ds = nemo_ds.rename({'e3t':'thkcello', 'x_grid_T':'x', 'y_grid_T':'y', 'area_grid_T':'area', 'e3t':'thkcello',
                              'nav_lon_grid_T':'nav_lon', 'nav_lat_grid_T':'nav_lat', 
                              'bounds_nav_lon_grid_T':'bounds_lon', 'bounds_nav_lat_grid_T':'bounds_lat',
                              'nvertex_grid_T':'nvertex'})
    
    # Average full time series:
    bottom_TS_vs_obs(nemo_ds.mean(dim='time_counter'), time_ave=False, fig_name=figname1, dpi=dpi)
    bottom_TS_vs_obs(nemo_ds.mean(dim='time_counter'), time_ave=False, nemo_mesh=nemo_mesh, amundsen=True, fig_name=figname2, dpi=dpi)
    # End state (last year of run):
    bottom_TS_vs_obs(nemo_ds.isel(time_counter=slice(-12,None)).mean(dim='time_counter'), time_ave=False, fig_name=figname3, dpi=dpi)
    return

# Visualize the timeseries of variable averaged over a region to see convection
def plot_hovmoeller_convect(run_folder, region, figname1, figname2, title='', tlim=(-1.5, 0.5), slim=(34.8, 34.86)):

    T_region = calc_hovmoeller_region('thetao', region, run_folder=run_folder)    
    S_region = calc_hovmoeller_region('so', region, run_folder=run_folder)

    plot_hovmoeller(T_region, title=title, ylim=(5500,0), vlim=tlim, fig_name=figname1, varname='Temperature (C)')
    plot_hovmoeller(S_region, title=title, ylim=(5500,0), vlim=slim, fig_name=figname2, varname='Salinity')

    return

# Create animations of some standard variables that are useful to look at
def animate_vars(run_folder, nemo_mesh='/gws/nopw/j04/anthrofail/birgal/NEMO_AIS/bathymetry/mesh_mask-20240305.nc'):
    import cmocean 

    var   = ['mldr10_1', 'siconc', 'zos', 'sbt', 'sbs', 'sosst', 'sosss']
    vlims = [(0,1000), (0,1), (-3,3), (-2,2), (34.4,35.0), (-5,5), (30,35)]
    stub  = ['grid_T', 'icemod', 'grid_T', 'grid_T', 'grid_T', 'grid_T', 'grid_T']
    cmaps = ['viridis', 'viridis', cmocean.cm.balance, cmocean.cm.balance, cmocean.cm.haline, cmocean.cm.balance, cmocean.cm.haline]

    for v in range(len(var)):
        animate_2D_circumpolar(run_folder, var[v], stub[v], vlim=vlims[v], cmap=cmaps[v], nemo_mesh=nemo_mesh)

    return

# transections of U and V velocities, but still need to be rotated from the grid to universal (!!)
def transects_UV_Amundsen(run_folder, savefig=False, nemo_mesh='/gws/nopw/j04/anthrofail/birgal/NEMO_AIS/bathymetry/mesh_mask-20240305.nc'):
    # load nemo simulations
    gridU_files  = glob.glob(f'{run_folder}*grid_U*')
    gridV_files  = glob.glob(f'{run_folder}*grid_V*')
    nemoU_ds     = xr.open_mfdataset(gridU_files) # load all the grid files in the run folder
    nemoV_ds     = xr.open_mfdataset(gridV_files)  
    nemoU_ds     = nemoU_ds.rename({'depthu':'depth'})
    nemoV_ds     = nemoV_ds.rename({'depthv':'depth'})
    nemoU_results = nemoU_ds.isel(time_counter=slice(180,None)).mean(dim='time_counter')  # Average time series
    nemoV_results = nemoV_ds.isel(time_counter=slice(180,None)).mean(dim='time_counter')
    nemo_mesh_ds  = xr.open_dataset(nemo_mesh).isel(time_counter=0)

    # calculate transects and plot:
    for transect in ['shelf_west', 'shelf_mid', 'shelf_east', 'shelf_edge']:
        # get coordinates for the transect:
        xU_sim, yU_sim = transect_coords_from_latlon_waypoints(nemoU_results, transect_amundsen[transect], opt_float=False)
        xV_sim, yV_sim = transect_coords_from_latlon_waypoints(nemoV_results, transect_amundsen[transect], opt_float=False)

        # subset the datasets and nemo_mesh to the coordinates of the transect:
        simU_transect = nemoU_results.isel(x=xr.DataArray(xU_sim, dims='n'), y=xr.DataArray(yU_sim, dims='n'))
        simV_transect = nemoV_results.isel(x=xr.DataArray(xV_sim, dims='n'), y=xr.DataArray(yV_sim, dims='n'))
        nemo_mesh_transectU = nemo_mesh_ds.isel(x=xr.DataArray(xU_sim, dims='n'), y=xr.DataArray(yU_sim, dims='n')).rename({'nav_lev':'depth'})
        nemo_mesh_transectV = nemo_mesh_ds.isel(x=xr.DataArray(xV_sim, dims='n'), y=xr.DataArray(yV_sim, dims='n')).rename({'nav_lev':'depth'})

        # add tmask, iceshelfmask and depths to the simulation dataset
        simU_transect = simU_transect.assign({'gdept_0':nemo_mesh_transectU.gdept_0, 'tmask':nemo_mesh_transectU.umask, 'isfdraft':nemo_mesh_transectU.isfdraft})
        simV_transect = simV_transect.assign({'gdept_0':nemo_mesh_transectV.gdept_0, 'tmask':nemo_mesh_transectV.vmask, 'isfdraft':nemo_mesh_transectV.isfdraft})

        # calculate the distance of each point along the transect relative to the start of the transect:
        simU_distance = distance_along_transect(simU_transect)
        simV_distance = distance_along_transect(simV_transect)

        # visualize the transect:
        fig, ax = plt.subplots(2,1, figsize=(8,6), dpi=300)
        kwagsU    ={'vmin':-0.15,'vmax':0.15,'cmap':cmocean.cm.balance,'label':'U velocity'}
        kwagsV    ={'vmin':-0.15,'vmax':0.15,'cmap':cmocean.cm.balance,'label':'V velocity'}
        kwags_mask={'mask_land':True, 'mask_iceshelf':True}
        plot_transect(ax[0], simU_distance, simU_transect, 'uo', **kwagsU, **kwags_mask)
        plot_transect(ax[1], simV_distance, simV_transect, 'vo', **kwagsV, **kwags_mask)
        ax[1].set_xlabel('Distance (km)')

        if savefig:
            finished_plot(fig, fig_name=f'{run_folder}figures/transect_UV_{transect}.jpg')

    return

def plot_Amundsen_2d_slices(nemo_ds, var_name='thetao', vlim=(-1.5, 0.5), savefig=False, figname='', region=''):

    import cartopy.crs as ccrs

    # add a line to mark location of shelf west transect
    obs               = read_dutrieux(eos='teos10')
    dutrieux_obs      = obs.assign({'nav_lon':obs.lon, 'nav_lat':obs.lat}).rename_dims({'lat':'y', 'lon':'x'})
    x_obs, y_obs      = transect_coords_from_latlon_waypoints(dutrieux_obs, transect_amundsen['shelf_west'], opt_float=False)    
    obs_transect_west = dutrieux_obs.isel(x=xr.DataArray(x_obs, dims='n'), y=xr.DataArray(y_obs, dims='n'))

    if region=='crosson':
        fig, ax     = plt.subplots(2,2, figsize=(12,7), subplot_kw={'projection': ccrs.Mercator(latitude_true_scale=-70)}, dpi=150)
        depthind    = [32, 39, 50, 62]
        axis_extent = [-110, -127, -75.1, -72.5]
        cax         = fig.add_axes([0.00, 0.52, 0.02, 0.35])
    else:
        fig, ax     = plt.subplots(3,3, figsize=(12,10), subplot_kw={'projection': ccrs.Mercator(latitude_true_scale=-70)}, dpi=150)
        depthind    = [30,35,40,45,50,55,60,65,68]
        axis_extent = [-95, -125, -75.5, -70]
        cax         = fig.add_axes([0.00, 0.67, 0.02, 0.2])

    # plot each depth slice
    for n, axis in enumerate(ax.ravel()):
        axis.set_extent(axis_extent, ccrs.PlateCarree())
        gl = axis.gridlines(draw_labels=True);
        gl.xlines=None; gl.ylines=None; gl.top_labels=None; gl.right_labels=None;

        var_plot = np.ma.masked_where(nemo_ds[var_name].isel(depth=depthind[n]).values == 0, 
                                      nemo_ds[var_name].isel(depth=depthind[n]).values)
        img1 = axis.pcolormesh(nemo_ds.nav_lon.values, nemo_ds.nav_lat.values, 
                               var_plot, transform=ccrs.PlateCarree(),rasterized=True,
                               cmap=cmocean.cm.dense, vmin=vlim[0], vmax=vlim[1], zorder=1)
    
        axis.plot(obs_transect_west.nav_lon.values, obs_transect_west.nav_lat.values, '--r', linewidth=0.8,
                  transform=ccrs.PlateCarree(), zorder=3)
        axis.set_title(f'Depth: {nemo_ds.depth.isel(depth=depthind[n]).values:.0f} m')
    
    fig.colorbar(img1, cax=cax, extend='both', label='Conservative Temperature')
    fig.suptitle(figname.split('T_')[1].split('.jpg')[0])

    plt.close()
    if savefig:
        finished_plot(fig, fig_name=figname)
        
    return


def frames_Amundsen_shelf_T_slices(run_folder, region='', savefig=True, vlim=(-1.5, 0.5)):
    
    gridT_files  = glob.glob(f'{run_folder}files/*grid_T*')
    nemo_ds      = xr.open_mfdataset(gridT_files, engine='netcdf4')
    nemo_ds      = nemo_ds.rename({'x_grid_T':'x', 'y_grid_T':'y', 'nav_lon_grid_T':'nav_lon', 'nav_lat_grid_T':'nav_lat', 'deptht':'depth'})
    if region=='crosson':
        nemo_ds  = nemo_ds.isel(x=slice(520,750), y=slice(160,280))
    else:
        nemo_ds  = nemo_ds.isel(x=slice(500,880), y=slice(150,300)) # subset region to speed up the plotting
    
    for time in nemo_ds.time_counter:
        year  = time.dt.year.values
        month = time.dt.month.values
        day   = time.dt.day.values
        plot_Amundsen_2d_slices(nemo_ds.sel(time_counter=time), var_name='thetao', vlim=vlim, savefig=savefig, region=region,
                                figname=f'{run_folder}animations/frames/amundsen_{region}_T_y{year}m{month:02}d{day:02}.jpg')
    
    return

