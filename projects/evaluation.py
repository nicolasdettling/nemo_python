import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import glob
import cmocean
from ..utils import select_bottom, distance_along_transect
from ..constants import deg_string, gkg_string, transect_amundsen
from ..plots import circumpolar_plot, finished_plot, plot_ts_distribution, plot_transect
from ..interpolation import interp_latlon_cf
from ..file_io import read_schmidtko, read_woa, read_dutrieux
from ..grid import extract_var_region, transect_coords_from_latlon_waypoints

# Compare the bottom temperature and salinity in NEMO (time-averaged over the given xarray Dataset) to observations: Schmidtko on the continental shelf, World Ocean Atlas 2018 in the deep ocean.
def bottom_TS_vs_obs (nemo, time_ave=True,
                      schmidtko_file='/gws/nopw/j04/terrafirma/kaight/input_data/schmidtko_TS.txt', 
                      woa_files='/gws/nopw/j04/terrafirma/kaight/input_data/WOA18/woa18_decav_*00_04.nc', 
                      nemo_mesh='/gws/nopw/j04/anthrofail/birgal/NEMO_AIS/bathymetry/mesh_mask-20240305.nc',
                      fig_name=None, amundsen=False, dpi=None):

    obs = read_schmidtko(schmidtko_file=schmidtko_file, eos='teos10')
    woa = read_woa(woa_files=woa_files, eos='teos10')

    # Regrid to the NEMO grid
    obs_interp = interp_latlon_cf(obs, nemo, method='bilinear')
    woa_interp = interp_latlon_cf(woa, nemo, method='bilinear')
    # Now combine them, giving precedence to the Schmidtko obs where both datasets exist
    obs_plot = xr.where(obs_interp.isnull(), woa_interp, obs_interp)

    # Select the NEMO variables we need and time-average
    if time_ave:
        nemo_plot = xr.Dataset({'temp':nemo['sbt'], 'salt':nemo['sbs']}).mean(dim='time_counter')
    else:
        nemo_plot = xr.Dataset({'temp':nemo['sbt'], 'salt':nemo['sbs']})
    nemo_plot = nemo_plot.rename({'x_grid_T_inner':'x', 'y_grid_T_inner':'y'})
    # Apply NEMO land mask to both
    nemo_plot = nemo_plot.where(nemo_plot['temp']!=0)
    obs_plot = obs_plot.where(nemo_plot['temp'].notnull()*obs_plot.notnull())
    obs_plot = obs_plot.where(nemo_plot['temp']!=0)
    nemo_plot = nemo_plot.where(nemo_plot['temp']!=0)
    # Get difference from obs
    bias = nemo_plot - obs_plot

    if amundsen:
       import cartopy.crs as ccrs
    
       nemo_mesh_ds = xr.open_dataset(nemo_mesh)
       # These indices are based on eANT025; eventually should generalize based on lat, lon
       mesh_sub  = nemo_mesh_ds.isel(x=slice(450, 900), y=slice(130,350), time_counter=0)
       nemo_plt  = nemo_plot.isel(x=slice(450, 900), y=slice(130,350))
       obs_plt   = obs_plot.isel(x=slice(450, 900), y=slice(130,350))
       bias_plt  = bias.isel(x=slice(450, 900), y=slice(130,350))
       # Little helper function to help cartopy with landmasking
       def mask_land(nemo_mesh, file_var):
          lon_plot = np.ma.masked_where(mesh_sub.tmask.isel(nav_lev=0) == 0, mesh_sub.nav_lon.values)
          lat_plot = np.ma.masked_where(mesh_sub.tmask.isel(nav_lev=0) == 0, mesh_sub.nav_lat.values)
          plot_var = np.ma.masked_where(mesh_sub.tmask.isel(nav_lev=0) == 0, file_var.values)
          return lon_plot, lat_plot, plot_var 
      
       data_plot  = [nemo_plt, obs_plt, bias_plt]
       var_titles = ['Bottom temperature ('+deg_string+'C)', 'Bottom salinity ('+gkg_string+')']
       vmin = [-2, -2, -1, 34.2, 34.2, -0.4]
       vmax = [2, 2, 1, 35, 35, 0.4]

       # fig, ax = plt.subplots(2,3, figsize=(20,8), subplot_kw={'projection': ccrs.Mercator(latitude_true_scale=-70)})
       fig, ax = plt.subplots(2,3, figsize=(18,10), subplot_kw={'projection': ccrs.Mercator(latitude_true_scale=-70)})

       for axis in ax.ravel():
          axis.set_extent([-95, -135, -76, -68], ccrs.PlateCarree())
          # axis.set_extent([-95, -160, -78, -67], ccrs.PlateCarree())
          gl = axis.gridlines(draw_labels=True);
          gl.xlines=None; gl.ylines=None; gl.top_labels=None; gl.right_labels=None;

       i=0
       for v, var in enumerate(['temp', 'salt']):
          for n, name in enumerate(['Model', 'Observations', 'Model bias']):
             lon_plt, lat_plt, var_plt = mask_land(mesh_sub, data_plot[n][var])
             img = ax[v,n].pcolormesh(lon_plt, lat_plt, var_plt, transform=ccrs.PlateCarree(), rasterized=True, cmap='RdBu_r', vmin=vmin[i], vmax=vmax[i])
             ax[v,n].set_title(name)
             i+=1
             if n != 1:
                cax = fig.add_axes([0.04+0.44*n, 0.56-0.41*v, 0.02, 0.3])
                plt.colorbar(img, cax=cax, extend='both' if n==0 else 'neither', label=var_titles[v])
       finished_plot(fig, fig_name=fig_name, dpi=dpi)
    else:
       # Make the plot
       fig = plt.figure(figsize=(10,7))
       gs = plt.GridSpec(2,3)
       gs.update(left=0.1, right=0.9, bottom=0.05, top=0.95, hspace=0.2, wspace=0.1)
       data_plot = [nemo_plot, obs_plot, bias]
       var_plot = ['temp', 'salt']
       var_titles = ['Bottom temperature ('+deg_string+'C)', 'Bottom salinity ('+gkg_string+')']
       alt_titles = [None, 'Observations', 'Model bias']
       vmin = [-2, -2, -0.5, 34.5, 34.5, -0.2]
       vmax = [2, 2, 0.5, 35, 35, 0.2]
       ctype = ['RdBu_r', 'RdBu_r', 'plusminus']
       i=0
       for v in range(2):
           for n in range(3):
               ax = plt.subplot(gs[v,n])
               ax.axis('equal')
               img = circumpolar_plot(data_plot[n][var_plot[v]], nemo, ax=ax, masked=True, make_cbar=False, 
                                      title=(var_titles[v] if n==0 else alt_titles[n]), 
                                      vmin=vmin[i], vmax=vmax[i], ctype=ctype[n], shade_land=False)
               i+=1
               if n != 1:
                   cax = fig.add_axes([0.01+0.46*n, 0.58-0.48*v, 0.02, 0.3])
                   plt.colorbar(img, cax=cax, extend='both' if n==0 else 'neither')
       finished_plot(fig, fig_name=fig_name, dpi=dpi)
                
            
# Function creates a figure with T-S diagram for simulations and for Pierre's observations in the Amundsen Sea
# Inputs:
# run_folder : string path to simulation folder
# time_slice : (optional) slice to subset time_counter for averaging simulation
# depth_slice: (optional) slice to subset deptht from simulation
# fig_name   : (optional) string for path to save figure if you want to save it
# return_fig : (optional) boolean for returning fig and ax
def TS_diagrams_Amundsen (run_folder, time_slice=None, depth_slice=None, fig_name=None, return_fig=False, smin=30, smax=35.25, tmin=-3, tmax=2.25):
    # --- get data ----
    
    # load observations
    obs = read_dutrieux(eos='teos10')
    
    # load nemo simulations
    gridT_files  = glob.glob(f'{run_folder}/*grid_T*') # load all the gridT files in the run folder
    nemo_ds      = xr.open_mfdataset(gridT_files).rename({'x_grid_T':'x','y_grid_T':'y'}) 
    if time_slice:
        nemo_average = nemo_ds.isel(time_counter=time_slice).mean(dim='time_counter') 
    else:
        nemo_average = nemo_ds.mean(dim='time_counter')
    # extract specific region
    amundsen_so = extract_var_region(nemo_average, 'so'    , 'amundsen_sea')
    amundsen_to = extract_var_region(nemo_average, 'thetao', 'amundsen_sea')
    if depth_slice:
        amundsen_so = amundsen_so.isel(deptht=depth_slice)
        amundsen_to = amundsen_to.isel(deptht=depth_slice)

    # --- plot distributions -----
    fig, ax = plt.subplots(1,2,figsize=(18,7), dpi=300)
    ax[0].set_title('Amundsen Sea simulations')
    plot_ts_distribution(ax[0], amundsen_so.values.flatten(), amundsen_to.values.flatten(), plot_density=True, plot_freeze=True, bins=150, smin=smin, smax=smax, tmin=tmin, tmax=tmax)
    ax[1].set_title('Amundsen Sea observations Pierre')
    plot_ts_distribution(ax[1], obs.AbsSal.values.flatten(), obs.ConsTemp.values.flatten(), plot_density=True, plot_freeze=True, smin=smin, smax=smax, tmin=tmin, tmax=tmax)

    if fig_name:
        finished_plot(fig, fig_name=fig_name)
    if return_fig:
        return fig, ax
    else:
        return
    
    
# Function produces figures of transects of observations on the Amundsen Sea shelf and simulation results    
# Inputs:
# run_folder : string path to folder containing NEMO simulations (gridT files)
# savefig    : (optional) boolean whether to save figure within figures sub-directory in run_folder
def transects_Amundsen(run_folder, savefig=False, nemo_mesh='/gws/nopw/j04/anthrofail/birgal/NEMO_AIS/bathymetry/mesh_mask-20240305.nc'):
    # load nemo simulations
    gridT_files  = glob.glob(f'{run_folder}*grid_T*')
    nemo_ds      = xr.open_mfdataset(gridT_files) # load all the gridT files in the run folder
    nemo_ds      = nemo_ds.rename({'x_grid_T':'x', 'y_grid_T':'y', 'nav_lon_grid_T':'nav_lon', 'nav_lat_grid_T':'nav_lat', 'deptht':'depth'}) 
    nemo_results = nemo_ds.isel(time_counter=slice(180,None)).mean(dim='time_counter')  # Average time series
    nemo_mesh_ds  = xr.open_dataset(nemo_mesh).isel(time_counter=0)
    
    # load observations:
    obs          = read_dutrieux(eos='teos10')
    dutrieux_obs = obs.assign({'nav_lon':obs.lon, 'nav_lat':obs.lat}).rename_dims({'lat':'y', 'lon':'x'})
    
    # calculate transects and plot:
    for transect in ['shelf_west', 'shelf_mid', 'shelf_east']:
        # get coordinates for the transect:
        x_obs, y_obs = transect_coords_from_latlon_waypoints(dutrieux_obs, transect_amundsen[transect], opt_float=False)
        x_sim, y_sim = transect_coords_from_latlon_waypoints(nemo_results, transect_amundsen[transect], opt_float=False)

        # subset the datasets and nemo_mesh to the coordinates of the transect:
        obs_transect = dutrieux_obs.isel(x=xr.DataArray(x_obs, dims='n'), y=xr.DataArray(y_obs, dims='n'))
        sim_transect = nemo_results.isel(x=xr.DataArray(x_sim, dims='n'), y=xr.DataArray(y_sim, dims='n'))
        nemo_mesh_transect  = nemo_mesh_ds.isel(x=xr.DataArray(x_sim, dims='n'), y=xr.DataArray(y_sim, dims='n')).rename({'nav_lev':'depth'})

        # add tmask, iceshelfmask and depths to the simulation dataset
        sim_transect = sim_transect.assign({'gdept_0':nemo_mesh_transect.gdept_0, 'tmask':nemo_mesh_transect.tmask, 'isfdraft':nemo_mesh_transect.isfdraft})

        # calculate the distance of each point along the transect relative to the start of the transect:
        obs_distance = distance_along_transect(obs_transect)
        sim_distance = distance_along_transect(sim_transect)

        # visualize the transect:
        fig, ax = plt.subplots(2,2, figsize=(15,6), dpi=300)
        kwagsT    ={'vmin':-2,'vmax':1,'cmap':cmocean.cm.dense,'label':'Conservative Temp.'}
        kwagsS    ={'vmin':33,'vmax':35,'cmap':cmocean.cm.haline,'label':'Absolute Salinity'}
        kwags_mask={'mask_land':True, 'mask_iceshelf':True}
        ax[0,0].set_title('Observations Dutrieux')
        ax[0,1].set_title('Observations Dutrieux')
        ax[1,0].set_title('Model simulations')
        ax[1,1].set_title('Model simulations')
        plot_transect(ax[0,0], obs_distance, obs_transect, 'ConsTemp', **kwagsT)
        plot_transect(ax[1,0], sim_distance, sim_transect, 'thetao', **kwagsT, **kwags_mask) 
        plot_transect(ax[0,1], obs_distance, obs_transect, 'AbsSal', **kwagsS)
        plot_transect(ax[1,1], sim_distance, sim_transect, 'so', **kwagsS, **kwags_mask) 
        ax[1,0].set_xlabel('Distance (km)')
        ax[1,1].set_xlabel('Distance (km)')

        if savefig:
            finished_plot(fig, fig_name=f'{run_folder}figures/evaluation_transect_{transect}.jpg')

    return
    
    
