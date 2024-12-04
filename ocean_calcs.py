# common ocean calculations, think stratification etc.
import numpy as np
import xarray as xr
from .constants import gravity, rho_fw

# Function to calculate the convective resistance --- the buoyance loss required to overturn the water column 
# to a given depth, H (Campbell et al. 2019; Gulk et al., 2024)
# Inputs:
# dsT    : xarray dataset containing variables "so", "thetao", and "deptht"
# ds_mesh: xarray dataset containing variable "e3t_0", the grid cell vertical thicknesses
# H      : (optional) water column convection depth
def convective_resistance(dsT, ds_mesh, H=830):
    import gsw

    # calculate density referenced to 0 dbar from conservative temperature and absolute salinity
    rho = gsw.density.sigma0(dsT.so, dsT.thetao)
    # find depth level index closest to H:
    Hind = np.argmin(abs(dsT.deptht.values - H))

    # calculate potential density anomaly w.r.t reference pressure of 0 dbar
    rho_0H = rho.where(dsT.so!=0).isel(deptht=Hind)
    rho_0z = rho.where(dsT.so!=0).isel(deptht=slice(0,Hind))
    integrand = (rho_0H - rho_0z)*ds_mesh.e3t_0.isel(deptht=slice(0,Hind)) 
    # calculate convective resistance:
    CR = (gravity/rho_fw) * integrand.sum(dim='deptht') 

    return CR



