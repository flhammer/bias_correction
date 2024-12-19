# dezm
# source: https://git.meteo.fr/cnrm-cen/louisletoumelin/devine_tutorial/-/blob/master/Tutorial_Pre_Computed_Method_1.ipynb

import pandas as pd
import tensorflow as tf
import xarray as xr
import numpy as np

from bias_correction.config.config_custom_devine import config
from bias_correction.train.dataloader import CustomDataHandler
from bias_correction.train.experience_manager import ExperienceManager
from bias_correction.train.model import CustomModel
from bias_correction.train.wind_utils import comp2dir, comp2speed, wind2comp
from bias_correction.utils_bc.context_manager import timer_context
from bias_correction.dem import dem_txt_to_numpy

config["use_scaling"] = False
config["type_of_output"] = (
    "map_speed_alpha"  # The outputs of DEVINE will be a map for speed and a map for angular deviations ("alpha")
)
# @Louis, The only change I made to the config was to adjust the following path.
config["unet_path"] = (
    "coeffs_unet/date_21_12_2021_name_simu_classic_all_low_epochs_0_model_UNet/"
)
batch_size = (
    1  # Batch size control how many maps are sent to the GPU at once for prediction
)


def process(demdataarray):
    # @Louis, the process function expects an xarray dataset, but I am providing a numpy array.
    # Hence, I changed the original demdataarray.values to demdataarray.
    # No other changes were made.
    dem_array = np.expand_dims(demdataarray, axis=-1)

    # Create topo_dict
    dict_topo_custom = {"custom": {"data": dem_array}}
    # Create time_series
    custom_time_series = pd.DataFrame()
    # Speed = [3, 3, 3, ...] and direction = [0, 1, 2, ...]
    custom_time_series["Wind"] = [3] * 360
    custom_time_series["Wind_DIR"] = list(range(360))
    # Order of 'Wind' and 'Wind_DIR' in dataset is important
    assert custom_time_series.columns[0] == "Wind"
    assert custom_time_series.columns[1] == "Wind_DIR"
    data_loader = CustomDataHandler(config)
    cm = CustomModel(None, config)
    min_shape = np.intp(np.min(np.squeeze(dem_array).shape) / np.sqrt(2)) + 1
    output_shape = list(
        tuple([2]) + tuple([len(custom_time_series)]) + tuple([min_shape, min_shape])
    )
    config["custom_input_shape"] = list(dem_array.shape)
    data_loader.prepare_custom_devine_data(custom_time_series, dict_topo_custom)
    batch_size = 1
    inputs = data_loader.get_tf_zipped_inputs(
        mode="custom", output_shapes=config["custom_input_shape"]
    ).batch(batch_size)
    with timer_context("Predict taggest set"):
        results = cm.predict_multiple_batches(
            inputs, batch_size=batch_size, output_shape=output_shape, force_build=True
        )
        print(results.shape)
    return results


# @Louis, my understanding is that the following DEM was used during the DEVINE model training.
# My goal is to make a prediction here and then directly compare it to the ARPS flow field.
dem_path = "gaussiandem_N5451_dx30_xi800_sigma71_r000.txt"

# @Louis, calls a helper function from the bias_correction.dem module, which simply returns a numpy array
dem = dem_txt_to_numpy(dem_path)


######
# @ Louis, the following lines are commented out as we load the data from a text file and not NetCDF.
# dem_path = "/app/data/dem_grandmaison_large.nc"
#
# # ------------- dem has been cropped to avoid exception on patch dimensions ------------
# dem = xr.open_dataset(dem_path)
# dem = dem.band_data.isel(x=slice(3, -2))
# dem.to_netcdf(dem_path + "_cropped")
# # manuellement renommé en dem_path par la suite
#
#
# dem = xr.open_dataset(dem_path)
######

# @Louis, the process function above expected an xarray dataset, but I am passing a numpy array.
# See the respective comment in the process function.
result = process(dem)

# @Louis, no changes were made past this point apart from adjusting the 'save_path' variable.
# normalize
with timer_context("results[0, :, :, :] = results[0, :, :, :]/3"):
    result[0, :, :, :] = result[0, :, :, :] / 3

# downcast
with timer_context("Change dtype"):
    result[0, :, :, :] = np.float16(result[0, :, :, :])
    result[1, :, :, :] = np.float16(result[1, :, :, :])

save_path = "results.nc"

with timer_context("Save results"):
    ds = xr.Dataset(
        data_vars={
            "acceleration": (("angle", "y", "x"), result[0, :, :, :]),
            "alpha": (("angle", "y", "x"), result[1, :, :, :]),
        }
    )
    comp = dict(zlib=True, complevel=3)
    encoding = {
        "acceleration": {"zlib": True, "complevel": 3},
        "alpha": {"zlib": True, "complevel": 3},
    }
    ds.to_netcdf(save_path, encoding=encoding)  # type:ignore
