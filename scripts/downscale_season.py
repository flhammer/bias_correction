
import rioxarray
import os
import xarray as xr
import numpy as np

from bias_correction.postprocessing.arome_interpolation import get_arome_downscaled_loop, get_arome_interpolated, compute_epsg_2154


arome_dir = "/cnrm/cen/users/NO_SAVE/letoumelinl/AROME/AROME_for_Ange_17_10_2023/HendrixExtraction_2023_11_17_13_ID_9a741/alp/"
data_dir = "/home/merzisenh/GIT/bias_correction/data"


unet_output = xr.open_dataset(os.path.join(
    data_dir, 'unet_output_on_small_dem.nc'))
small_dem = xr.open_dataarray(os.path.join(
    data_dir, "domaine_grande_rousse_small.nc"))
# itérateur sur les fichiers arome renvoie -> arome_path, period
# ouh c'est laid! tous les fichiers arome n'ont pas de lon/lat on prend celui là parce qu'on a constaté qu'il en a...
arome_reference_for_lonlat = xr.open_dataset(
    os.path.join(arome_dir, "AROME_alp_2022_12.nc"))
x_var, y_var = compute_epsg_2154(arome_reference_for_lonlat)

period = "2021_10"
filepath = os.path.join(arome_dir, f"AROME_alp_{period}.nc")


interpolated_arome_fullpath = os.path.join(
    data_dir, "interpolated", f"interpolated_arome{period}.nc")

if not os.path.exists(interpolated_arome_fullpath):
    arome = xr.open_dataset(filepath).isel(time=slice(0, 50))
    arome['x'] = x_var
    arome['y'] = y_var
    arome_interpolated = get_arome_interpolated(arome, small_dem)
    arome_interpolated.to_netcdf(interpolated_arome_fullpath)

arome_interpolated = xr.open_dataset(interpolated_arome_fullpath)

windspeed, direction = get_arome_downscaled_loop(
    arome_interpolated.isel(time=slice(0, 20)),
    unet_output
)

downscaled_wind = xr.Dataset({'speed': windspeed, 'direction': direction})
downscaled_wind.to_netcdf(os.path.join(
    data_dir, "downscaled", f"arome_downscaled_{period}.nc"))

downscaled_wind
