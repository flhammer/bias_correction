import os
import xarray as xr
import numpy as np

from bias_correction.postprocessing.arome_interpolation import get_arome_downscaled_direction_vectorized, get_arome_downscaled_direction_loop


data_dir = "/home/merzisenh/Projets/LIVE/bias_correction/data"
interpolated_arome_filepath = os.path.join(
    data_dir, 'arome0/interpolated_arome0_2017_10_23_15___2017_12_4_6.nc')
arome_interpolated = xr.open_dataset(interpolated_arome_filepath)
unet_output = xr.open_dataset(
    os.path.join(data_dir, 'results/unet_output_on_small_dem.nc'))


direction, speed = get_arome_downscaled_direction_loop(
    arome_interpolated,
    unet_output.isel(angle=slice(0, 10))
)
