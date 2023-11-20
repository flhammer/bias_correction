import rioxarray
import os
import xarray as xr
import numpy as np

from bias_correction.postprocessing.arome_interpolation import get_arome_downscaled_loop, get_arome_interpolated


data_dir = "/home/merzisenh/Projets/LIVE/bias_correction/data"


def fullpath(path):
    return os.path.join(data_dir, path)


unet_output = xr.open_dataset(fullpath('results/unet_output_on_small_dem.nc'))
small_dem = xr.open_dataarray(os.path.join(data_dir, "domaine_grande_rousse_small.nc"))
# itérateur sur les fichiers arome renvoie -> arome_path, period
arome_path = "arome0/arome0_2017_10_23_15___2017_12_4_6.nc"
period = "201711"
interpolated_arome_filepath = f"arome0/interpolated_arome{period}.nc"

if not os.path.exists(fullpath(interpolated_arome_filepath)):
    arome_data = xr.open_dataset(fullpath(arome_path))
    arome_interpolated = get_arome_interpolated(arome_data, small_dem)
    arome_interpolated.to_netcdf(fullpath(interpolated_arome_filepath))

arome_interpolated = xr.open_dataset(fullpath(interpolated_arome_filepath))

windspeed, direction = get_arome_downscaled_loop(
    arome_interpolated.isel(time=slice(0, 20)),
    unet_output
)

downscaled_wind = xr.Dataset({'speed': windspeed, 'direction': direction})

downscaled_wind.to_netcdf(fullpath(f"downscaled/arome_downscaled_{period}.nc"))

# ouvrir small_dem, unet_output (déjà reprojetés sur le domaine on a sauvé le résultat dans un fichier)
