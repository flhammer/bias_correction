# sxcen

import gc
import os

import numpy as np
import rioxarray
import xarray as xr

from bias_correction.postprocessing.arome_interpolation import (
    compute_epsg_2154, get_arome_downscaled_loop, get_arome_interpolated)

arome_dir="/cnrm/mrns/users/NO_SAVE/merzisenh/extraction_matthieu_v_2021/HendrixExtraction_2023_11_29_11_ID_1a9b1/alp/"
data_dir = "/home/merzisenh/NO_SAVE/arome_2021_2022_downscaled_devine"


unet_output = xr.open_dataset(os.path.join(
    data_dir, 'unet_output_on_small_dem.nc'))
small_dem = xr.open_dataarray(os.path.join(
    data_dir, "domaine_grande_rousse_small.nc"))
# itérateur sur les fichiers arome renvoie -> arome_path, period
# ouh c'est laid! tous les fichiers arome n'ont pas de lon/lat on prend celui là parce qu'on a constaté qu'il en a...
arome_reference_for_lonlat = xr.open_dataset(
    os.path.join(arome_dir, "AROME_alp_2022_12.nc"))
x_var, y_var = compute_epsg_2154(arome_reference_for_lonlat)


list_periods = ["2021_07", "2021_08", "2021_09", "2021_10", "2021_11", "2021_12", "2022_01",
                "2022_02", "2022_03", "2022_04", "2022_05", "2022_06", "2022_07"]

for period in list_periods:
    filepath = os.path.join(arome_dir, f"AROME_alp_{period}.nc")
    interpolated_arome_fullpath = os.path.join(
        data_dir, "interpolated", f"arome_interpolated_{period}.nc")
    downscaled_arome_filepath = os.path.join(
        data_dir, "downscaled", f"arome_downscaled_{period}.nc")
    if not os.path.exists(downscaled_arome_filepath):
        if not os.path.exists(interpolated_arome_fullpath):
            print(f"Interpolating Arome winds for period {period}")
            arome = xr.open_dataset(filepath)
            arome['x'] = x_var
            arome['y'] = y_var
            arome_interpolated = get_arome_interpolated(arome, small_dem)
            arome_interpolated.to_netcdf(interpolated_arome_fullpath)
            del arome
        arome_interpolated = xr.open_dataset(interpolated_arome_fullpath)
        print(f"Downscaling Arome winds for period {period}")
        windspeed, direction = get_arome_downscaled_loop(
            arome_interpolated,
            unet_output
        )
        downscaled_wind = xr.Dataset(
            {'speed': windspeed, 'direction': direction})
        downscaled_wind.to_netcdf(downscaled_arome_filepath)
        del downscaled_wind, windspeed, direction
    del arome_interpolated
    gc.collect()
