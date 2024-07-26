# px07380
from os.path import join as pj

import numpy as np
import rioxarray
import xarray as xr

from bias_correction.postprocessing.arome_interpolation import (
    recompute_coordinates_on_unet_output,
)


qgis_dir = "/home/merzisenh/Projets/qgis_inbox"
arome_dir = "/home/merzisenh/NO_SAVE/wind_alp"
data_dir = "/home/merzisenh/Projets/bias_correction_data"


dem_gr = xr.open_dataset(pj(data_dir, "domaine_grande_rousse_large.nc"))
dem_gr_small = xr.open_dataset(pj(data_dir, "domaine_grande_rousse_small.nc"))
dem_eaudolle = xr.open_dataset(pj(data_dir, "eaudolle_large.nc"))
dem_eaudolle_small = xr.open_dataset(pj(data_dir, "eaudolle_30m.nc"))
unet_old = xr.open_dataset(pj(data_dir, "unet_output_on_small_dem.nc"))

unet_new = xr.open_dataset(pj(data_dir, "eaudolle_large_corrections.nc"))
# le nouveau unet est brut il faut lui attacher des coordonnées géographiques
unet_new = recompute_coordinates_on_unet_output(unet_new, dem_eaudolle.Band1, 2154)

unet_new

# on visualise ça dans qgis et on constate qu'il manque encore des bouts :-|
# combien?


unet_new.isel(angle=45).to_netcdf(pj(qgis_dir, "unet_new_45.nc"))

unet_new.isel(angle=45).rio.to_raster(pj(qgis_dir, "unet_new_45.tiff"))

# OK cf obsidian

# pour trouver les coordonnées souhaitées pour le domaine final

dem_eaudolle_small.x.min().item()
dem_eaudolle_small.x.max().item()

dem_eaudolle_small.y.min().item()
dem_eaudolle_small.y.max().item()

dem_eaudolle_small
dem_eaudolle.x.min().item()
