import rioxarray
import os
import xarray as xr
import numpy as np
import gc

from bias_correction.postprocessing.arome_interpolation import get_arome_downscaled_loop, get_arome_interpolated, compute_epsg_2154


# mail de Matthieu Vernay: il manque les 6 premières heures du mois d'octobre 2021

# on va aller vérifier

arome_dir="/cnrm/mrns/users/NO_SAVE/merzisenh/extraction_matthieu_v_2021/HendrixExtraction_2023_11_29_11_ID_1a9b1/alp/"
data_dir = "/home/merzisenh/NO_SAVE/arome_2021_2022_downscaled_devine"

arome_reference_for_lonlat = xr.open_dataset( os.path.join(arome_dir, "AROME_alp_2022_12.nc"))
x_var, y_var = compute_epsg_2154(arome_reference_for_lonlat)
unet_output = xr.open_dataset(os.path.join(
    data_dir, 'unet_output_on_small_dem.nc'))
small_dem = xr.open_dataarray(os.path.join(
    data_dir, "domaine_grande_rousse_small.nc"))

period = "2021_10"

interpolated_arome_fullpath = os.path.join(
    data_dir, "interpolated", "arome_interpolated_beginning_of_october.nc")
downscaled_arome_filepath = os.path.join(
    data_dir, "downscaled", f"arome_downscaled_beginning_of_october.nc")



arome_filepath = os.path.join(arome_dir, f"AROME_alp_{period}.nc")
arome = xr.open_dataset(arome_filepath)

arome['x'] = x_var
arome['y'] = y_var

arome_interpolated = get_arome_interpolated(arome, small_dem)
arome_interpolated.to_netcdf(interpolated_arome_fullpath)

print(f"Downscaling Arome winds for period {period}")
windspeed, direction = get_arome_downscaled_loop(
    arome_interpolated,
    unet_output
)
downscaled_wind = xr.Dataset(
    {'speed': windspeed, 'direction': direction})
downscaled_wind.to_netcdf(downscaled_arome_filepath)
