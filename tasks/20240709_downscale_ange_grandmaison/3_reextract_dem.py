# px07380
from os.path import join as pj

import numpy as np
import rioxarray
import xarray as xr


dem_path = (
    "/rd/cenfic3/cenmod/dem_cen/definitive/france_30m/DEM_FRANCE_L93_30m_bilinear.tif"
)
data_dir = "/home/merzisenh/Projets/bias_correction_data"
qgis_dir = "/home/merzisenh/Projets/qgis_inbox"


whole_dem = xr.open_dataset(dem_path)

x_min = 929865.0 - 30 * 1200 / 2
x_max = 961305.0 + 30 * 1200 / 2
y_min = 6445485.0 - 30 * 1200 / 2
y_max = 6477405.0 + 30 * 1200 / 2

whole_dem

whole_dem.sel(x=slice(x_min, x_max), y=slice(y_max, y_min)).isel(band=0)
subzone = whole_dem.sel(x=slice(x_min, x_max), y=slice(y_max, y_min)).isel(band=0)
subzone.spatial_ref

subzone.to_netcdf(pj(data_dir, "dem_grandmaison_large.nc"))


subzone.band_data.rio.to_raster(pj(qgis_dir, "dem_grandmaison.tiff"))

# it will be cropped in the future steps to allow devine to work ...
