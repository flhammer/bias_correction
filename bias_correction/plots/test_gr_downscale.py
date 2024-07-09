from bias_correction.plots.utils import FigureManager, add_boundaries_to_plot, get_subdomain, compute_devine_wind
import numpy as np
import matplotlib.pyplot as plt
import rioxarray
import xarray as xr
from os.path import join as pj

import matplotlib
matplotlib.use("TkAgg")


plt.ion()
fm = FigureManager()

# root = "/home/merzisenh/Projets/LIVE/bias_correction"
arome_dir = "/cnrm/mrns/users/NO_SAVE/merzisenh/extraction_matthieu_v_2021/HendrixExtraction_2023_11_29_11_ID_1a9b1/alp/"
data_dir = "/home/merzisenh/NO_SAVE/arome_2021_2022_downscaled_devine"

epsg = "EPSG:2153"

unet = xr.open_dataset(
    pj(data_dir, 'unet_output_on_small_dem.nc')).rio.write_crs(epsg)
dem = xr.open_dataarray(
    pj(data_dir, "domaine_grande_rousse_small.nc")).rio.write_crs(epsg)
period = "2021_07"

arome = xr.open_dataset(pj(arome_dir, f"AROME_alp_{period}.nc"))
arome_interp = xr.open_dataset(
    pj(data_dir, "interpolated", f"arome_interpolated_{period}.nc")).rio.write_crs(epsg)
arome_down = xr.open_dataset(
    pj(data_dir, "downscaled", f"arome_downscaled_{period}.nc")).rio.write_crs(epsg)
arome_epsg = xr.open_dataset(
    pj(data_dir, "ref_epsg2154_coords_arome_alp.nc")).rio.write_crs(epsg)


arome = arome.isel(time=0)
arome_interp = arome_interp.isel(time=0)
arome_down = arome_down.isel(time=0)


arome['x'] = arome_epsg.x
arome['y'] = arome_epsg.y
arome['u'] = np.sin(np.deg2rad(arome.Wind_DIR)) * arome.Wind
arome['v'] = np.cos(np.deg2rad(arome.Wind_DIR)) * arome.Wind
arome_down['u'] = np.sin(np.deg2rad(arome_down.direction)) * arome_down.speed
arome_down['v'] = np.cos(np.deg2rad(arome_down.direction)) * arome_down.speed


rangex = [937755, 973515]
rangey = [6439005, 6464265]
domain = dict(minx=rangex[0], maxx=rangex[1], miny=rangey[0], maxy=rangey[1])

# unet.isel(angle=0).acceleration.plot(ax=fm.create_new()[1])


# # restreindre à une sous-zone
# subdomain = dict(minx=0.94e6, maxx=0.95e6, miny=6.44e6, maxy=6.45e6)

fig, ax = fm.create_new()
dem.plot(ax=ax, levels=20)
x0 = 0.95e6
y0 = 6.44e6
add_boundaries_to_plot(ax, **get_subdomain(x0, y0))

subunet = unet.rio.clip_box(**get_subdomain(x0, y0))
fig, ax = fm.create_new()
dem.rio.clip_box(**get_subdomain(x0, y0)).plot(ax=ax, levels=20)
X, Y, U, V = compute_devine_wind(subunet, 90)
ax.quiver(X, Y, U, V)

fm.close_all()
arome
arome_down.direction.max()

subdomain = dict(minx=0.9e6, maxx=0.95e6, miny=6.4e6, maxy=6.45e6)


########################## Visualisation AROME sur le grand domaine ########################
x_flat = arome.x.values.ravel()
y_flat = arome.y.values.ravel()


def subdomain_to_condition(x, y, minx, maxx, miny, maxy):
    return (x >= minx) & (x <= maxx) & (y >= miny) & (y <= maxy)


condition = subdomain_to_condition(x_flat, y_flat, **domain)
xyuv = {
    var: arome[var].values.ravel(
    )[subdomain_to_condition(x_flat, y_flat, **domain)]
    for var in ['x', 'y', 'u', 'v']
}

fig_arome_native, ax_arome_native = fm.create_new()
dem.plot(ax=ax_arome_native)
ax_arome_native.quiver(*xyuv.values())

fm.close_all()


# on choisit un sous-domaine sur lequel visualiser le downscaling
subdomain = get_subdomain(0.96e6, 6.451e6)
add_boundaries_to_plot(ax_arome_native, **subdomain)

subdomain = get_subdomain(0.946e6, 6.454e6)
add_boundaries_to_plot(ax_arome_native, **subdomain)


# Nouveau plot sur le sous-domaine précédent
interp_sub = arome_interp.rio.clip_box(**subdomain)  # .isel(time=0)
down_sub = arome_down.rio.clip_box(**subdomain)  # .isel(time=0)

x, y = np.meshgrid(interp_sub.x.values, interp_sub.y.values)

fig, ax = fm.create_new()
dem.rio.clip_box(**subdomain).plot(ax=ax)

ax.quiver(x, y, interp_sub.u.values, interp_sub.v.values)
ax.quiver(x, y, down_sub.u.values, down_sub.v.values, color="green")

fm.close_all()
