# px07380
from os.path import join as pj

import numpy as np
import rioxarray
import xarray as xr

from bias_correction.postprocessing.arome_interpolation import (
    recompute_coordinates_on_unet_output,
    convert_4324_to_2154,
)
import matplotlib.pyplot as plt
import matplotlib

from bias_correction.plots.utils import (
    FigureManager,
    add_boundaries_to_plot,
    get_subdomain,
    compute_devine_wind,
)

matplotlib.use("TkAgg")
plt.ion()
fm = FigureManager()


epsg = "EPSG:2154"

qgis_dir = "/home/merzisenh/Projets/qgis_inbox"
data_dir = "/home/merzisenh/NO_SAVE/projects/bias_correction/"

dem = xr.open_dataset(pj(data_dir, "eaudolle_large.nc")).rio.write_crs(epsg)
dem = dem.Band1
dem

downscaled = xr.open_dataset(pj(data_dir, "wind_downscaling_2019_12.nc"))
downscaled.rio.set_crs("EPSG:2154", inplace=True)
downscaled = downscaled.rio.write_crs("EPSG:2154")

interpolated = xr.open_dataset(pj(data_dir, "wind_interpolation_2019_12.nc"))
interpolated.rio.set_crs("EPSG:2154", inplace=True)
interpolated = interpolated.rio.write_crs("EPSG:2154")


arome = xr.open_dataset(pj(data_dir, "uv_2019_2020.nc"))
# arome.rio.set_crs("EPSG:2154", inplace=True)


def attach_epsg2154(array):
    lon, lat = np.meshgrid(array.longitude.data, array.latitude.data)
    x, y = convert_4324_to_2154(xr.DataArray(lon), xr.DataArray(lat))
    x = x.rename(dict(dim_0="latitude", dim_1="longitude"))
    y = y.rename(dict(dim_0="latitude", dim_1="longitude"))
    array.coords["x"] = x
    array.coords["y"] = y
    return array


arome = attach_epsg2154(arome)

arome_raw = arome.isel(time=0)
arome_interp = interpolated.isel(time=0)
arome_down = downscaled.isel(time=0)


arome_down["u"] = np.sin(np.deg2rad(arome_down.direction)) * arome_down.force
arome_down["v"] = np.cos(np.deg2rad(arome_down.direction)) * arome_down.force


# rangex = [937755, 973515]
# rangey = [6439005, 6464265]
# domain = dict(minx=rangex[0], maxx=rangex[1], miny=rangey[0], maxy=rangey[1])

fig_all, ax_all = fm.create_new()
dem.plot(ax=ax_all, levels=20)
# x0 = 0.95e6
# y0 = 6.44e6
# add_boundaries_to_plot(ax, **get_subdomain(x0, y0))

ax_all.quiver(
    arome_raw.x.values,
    arome_raw.y.values,
    arome_raw.u.values,
    arome_raw.v.values,
    scale=80,
)

# fm.close_all()


subdomain = get_subdomain(0.94e6, 6.455e6)
# visualise subdomain on exiting plot
add_boundaries_to_plot(ax_all, **subdomain)

############## new plot: interpolated and downscaled data on subdomain
fig, ax = fm.create_new()
interp_sub = arome_interp.rio.clip_box(**subdomain)  # .isel(time=0)
down_sub = arome_down.rio.clip_box(**subdomain)  # .isel(time=0)
down_sub
x, y = np.meshgrid(interp_sub.x.values, interp_sub.y.values)
dem.rio.clip_box(**subdomain).plot(ax=ax)
ax.quiver(x, y, interp_sub.u.values, interp_sub.v.values)
ax.quiver(x, y, down_sub.u.values, down_sub.v.values, color="green")
