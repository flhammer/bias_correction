from bias_correction.postprocessing.arome_interpolation import get_arome_interpolated, compute_epsg_2154
import os
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib
import pyproj
import numpy as np

matplotlib.use("TkAgg")
plt.ion()

data_dir = "/cnrm/cen/users/NO_SAVE/letoumelinl/AROME/AROME_for_Ange_17_10_2023/HendrixExtraction_2023_11_17_13_ID_9a741/alp/"
os.listdir(data_dir)
period = "2022_12"
filepath = os.path.join(data_dir, f"AROME_alp_{period}.nc")
arome = xr.open_dataset(filepath)
# arome.isel(time=0).Tair.plot()
compute_epsg_2154(arome)



