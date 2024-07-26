# px07380
from os.path import join as pj

import numpy as np
import rioxarray
import xarray as xr

from bias_correction.postprocessing.arome_interpolation import (
    recompute_coordinates_on_unet_output,
)


qgis_dir = "/home/merzisenh/Projets/qgis_inbox"
data_dir = "/home/merzisenh/Projets/bias_correction_data"


dem_eaudolle_v2 = xr.open_dataset(pj(data_dir, "dem_grandmaison_large.nc"))
dem_eaudolle_v2

unet_old = xr.open_dataset(pj(data_dir, "unet_output_on_small_dem.nc"))

unet_new = xr.open_dataset(pj(data_dir, "dem_grandmaison_large_corrections.nc"))
# le nouveau unet est brut il faut lui attacher des coordonnées géographiques
unet_new = recompute_coordinates_on_unet_output(
    unet_new, dem_eaudolle_v2.band_data, 2154
)

unet_new

# on visualise ça dans qgis et on constate qu'il manque encore des bouts :-|
# combien?


unet_new.isel(angle=45).acceleration.rio.to_raster(pj(qgis_dir, "unet_new_45_acc.tiff"))
unet_old.isel(angle=45).acceleration.rio.to_raster(pj(qgis_dir, "unet_old_45_acc.tiff"))


x_min = 929865.0
x_max = 961305.0
y_min = 6445485.0
y_max = 6477405.0

unet_new.isel(angle=45).sel(
    x=slice(x_min, x_max), y=slice(y_max, y_min)
).acceleration.rio.to_raster(pj(qgis_dir, "unet_new_45_acc_cropped.tiff"))

unet_new.isel(angle=45).alpha.rio.to_raster(pj(qgis_dir, "unet_new_45_alpha.tiff"))


unet_new.isel(angle=55).sel(
    x=slice(x_min, x_max), y=slice(y_max, y_min)
).acceleration.rio.to_raster(pj(qgis_dir, "unet_new_55_acc_cropped.tiff"))


diff = unet_new - unet_old

diff45 = diff.isel(angle=45).sel(x=slice(x_min, x_max), y=slice(y_max, y_min))

diff45.alpha.rio.to_raster(pj(qgis_dir, "new_minus_old_45_alpha.tiff"))
diff.isel(angle=50).sel(
    x=slice(x_min, x_max), y=slice(y_max, y_min)
).alpha.rio.to_raster(pj(qgis_dir, "new_minus_old_50_alpha.tiff"))
diff.isel(angle=190).sel(
    x=slice(x_min, x_max), y=slice(y_max, y_min)
).alpha.rio.to_raster(pj(qgis_dir, "new_minus_old_190_alpha.tiff"))


np.abs(diff45.acceleration).mean()
np.abs(diff45.alpha).mean()

# OK ça va le faire!


# sauvons unet_new avec les coordonnées

unet_new = unet_new.sel(x=slice(x_min, x_max), y=slice(y_max, y_min))
unet_new.to_netcdf(pj(data_dir, "dem_grandmaison_corrections.nc"))


dem_eaudolle_v2
dem_eaudolle_v2.sel(x=slice(x_min, x_max), y=slice(y_max, y_min)).to_netcdf(
    pj(data_dir, "dem_grandmaison.nc")
)
