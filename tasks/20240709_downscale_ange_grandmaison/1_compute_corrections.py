# source: https://git.meteo.fr/cnrm-cen/louisletoumelin/devine_tutorial/-/blob/master/Tutorial_Pre_Computed_Method_1.ipynb

import hvplot.xarray
import numpy as np
import pandas as pd
import tensorflow as tf
import xarray as xr

from bias_correction.config.config_custom_devine import config
from bias_correction.train.dataloader import CustomDataHandler
from bias_correction.train.experience_manager import ExperienceManager
from bias_correction.train.model import CustomModel
from bias_correction.train.wind_utils import comp2dir, comp2speed, wind2comp
from bias_correction.utils_bc.context_manager import timer_context

config["use_scaling"] = False
config["type_of_output"] = "map_speed_alpha" # The outputs of DEVINE will be a map for speed and a map for angular deviations ("alpha")
config["unet_path"] =  '../data/date_21_12_2021_name_simu_classic_all_low_epochs_0_model_UNet/'
batch_size = 1 # Batch size control how many maps are sent to the GPU at once for prediction


def process(demdataarray):
    dem_array = np.expand_dims(demdataarray.values, axis=-1)
    # Create topo_dict
    dict_topo_custom = {"custom": {"data": dem_array}}
    # Create time_series
    custom_time_series = pd.DataFrame()
    # Speed = [3, 3, 3, ...] and direction = [0, 1, 2, ...]
    custom_time_series["Wind"] = [3]*360
    custom_time_series["Wind_DIR"] = list(range(360))
    # Order of 'Wind' and 'Wind_DIR' in dataset is important
    assert custom_time_series.columns[0] == "Wind"
    assert custom_time_series.columns[1] == "Wind_DIR"
    data_loader = CustomDataHandler(config)
    cm = CustomModel(None, config)
    min_shape = np.intp(np.min(np.squeeze(dem_array).shape) / np.sqrt(2)) + 1
    output_shape = list(tuple([2]) + tuple([len(custom_time_series)]) + tuple([min_shape, min_shape]))
    config["custom_input_shape"] = list(dem_array.shape)
    data_loader.prepare_custom_devine_data(custom_time_series, dict_topo_custom)
    batch_size = 1
    inputs = data_loader.get_tf_zipped_inputs(mode="custom", output_shapes=config["custom_input_shape"]).batch(batch_size)
    with timer_context("Predict taggest set"):
        results = cm.predict_multiple_batches(inputs, batch_size=batch_size, output_shape=output_shape, force_build=True)
        print(results.shape)
    return results



eaudolle_path = "/app/data/eaudolle_large.nc"
# eaudolle_path = '/home/merzisenh/Projets/bias_correction_data/eaudolle_large.nc'
grandesrousses_path = "/app/data/domaine_grande_rousse_large.nc"


# on va comparer avec la structure des DEM précédents
grandesrousses = xr.open_dataarray(grandesrousses_path)
eaudolle = xr.open_dataset(eaudolle_path)
eaudolle.Band1
# process(grandesrousses.isel(band=0))

# ------------- dem has been cropped to avoid exception on patch dimensions ------------
# eaudolle = eaudolle.isel(y=slice(2,-1))
# eaudolle.to_netcdf(eaudolle_path+"_cropped")

# result = process(eaudolle.Band1.isel(y=slice(2, -1)))
result = process(eaudolle.Band1)

# normalize
with timer_context("results[0, :, :, :] = results[0, :, :, :]/3"):
    result[0, :, :, :] = result[0, :, :, :] / 3

# downcast
with timer_context("Change dtype"):
    result[0, :, :, :] = np.float16(result[0, :, :, :])
    result[1, :, :, :] = np.float16(result[1, :, :, :])

save_path = "/app/data/eaudolle_large_corrections.nc"

with timer_context("Save results"):
    ds = xr.Dataset(
        data_vars={"acceleration": (("angle", "y", "x"), result[0, :, :, :]),
                   "alpha": (("angle", "y", "x"), result[1, :, :, :])})
    comp = dict(zlib=True, complevel=3)
    encoding = {"acceleration": {"zlib": True, "complevel": 3},
                "alpha": {"zlib": True, "complevel": 3}}
    ds.to_netcdf(save_path, encoding=encoding) #type:ignore

# TODO: il faut réattacher les coordonnées, mais Ange ne nous a pas donné le CRS |-(
