import numpy as np
import xarray as xr
from scipy.interpolate import griddata
from tqdm import tqdm
from typing import Tuple
import pyproj


def recompute_coordinates_on_unet_output(
        unet_output,
        unet_dem_input,
        epsg
):
    unet_output_with_coordinates = unet_output.copy()
    length_y, length_x = unet_dem_input.shape
    min_length = np.min([length_y, length_x])
    y_diff = np.intp((min_length/np.sqrt(2))) // 2
    x_diff = y_diff
    y_min = length_y//2 - y_diff
    y_max = length_y//2 + y_diff + 1
    x_min = length_x//2 - x_diff
    x_max = length_x//2 + x_diff + 1
    unet_output_with_coordinates.coords['x'] = (
        ('x',),
        unet_dem_input.coords['x'].data[x_min:x_max]
    )
    unet_output_with_coordinates.coords['y'] = (
        ('y',),
        unet_dem_input.coords['y'].data[y_min:y_max]
    )
    unet_output_with_coordinates = unet_output_with_coordinates.rio.write_crs(
        epsg)
    return unet_output_with_coordinates


def compute_epsg_2154(arome: xr.Dataset):
    gps_to_l93 = pyproj.Transformer.from_crs(4324, 2154, always_xy=True)
    lon = arome.LON.isel(time=0).values.ravel()
    lat = arome.LAT.isel(time=0).values.ravel()
    shape_ini = arome.LAT.isel(time=0).values.shape
    x, y = np.array(list(gps_to_l93.itransform(
        np.array([lon, lat]).swapaxes(0, 1)))).swapaxes(0, 1)
    x_var = (("yy", "xx"), x.reshape(*shape_ini))
    y_var = (("yy", "xx"), y.reshape(*shape_ini))
    return x_var, y_var


def get_arome_interpolated(arome_data: xr.Dataset, target_dem: xr.DataArray) -> xr.Dataset:
    arome_data['u'] = np.sin(np.deg2rad(arome_data.Wind_DIR)) * arome_data.Wind
    arome_data['v'] = np.cos(np.deg2rad(arome_data.Wind_DIR)) * arome_data.Wind

    size_t = arome_data.time.shape[0]
    size_y = target_dem.y.shape[0]
    size_x = target_dem.x.shape[0]

    arome_interpolated = xr.Dataset(
        data_vars=dict(
            u=(["time", "y", "x"], np.zeros(
                (size_t, size_y, size_x), dtype=np.float32)),
            v=(["time", "y", "x"], np.zeros(
                (size_t, size_y, size_x), dtype=np.float32)),
        ),
        coords=dict(
            x=(["x"], target_dem.x.data),
            y=(["y"], target_dem.y.data),
            time=(['time'], arome_data.time.data)
        )
    )
    epsg = "EPSG:2154"
    arome_interpolated = arome_interpolated.rio.write_crs(epsg)

    # Calcul de l'interpolation d'Arome sur la grille de large_dem (u et v)
    grid_x = np.tile(target_dem.x, (size_y, 1))
    grid_y = np.tile(target_dem.y, (size_x, 1)).T
    x_flat = arome_data.isel(time=0)['x'].values.ravel()
    y_flat = arome_data.isel(time=0)['y'].values.ravel()
    points = np.array([x_flat, y_flat]).T

    for time in tqdm(arome_data.time):
        u_arome = arome_data.sel(time=time)['u'].values.ravel()
        v_arome = arome_data.sel(time=time)['v'].values.ravel()
        arome_interpolated.u.loc[dict(time=time)] = griddata(
            points, u_arome, (grid_x, grid_y), method='linear').astype("float32")
        arome_interpolated.v.loc[dict(time=time)] = griddata(
            points, v_arome, (grid_x, grid_y), method='linear').astype("float32")

    # Calcul de force et direction du vent interpolé
    u = arome_interpolated.u
    v = arome_interpolated.v
    norm = np.sqrt(u**2 + v**2)
    arome_interpolated['windspeed'] = norm
    direction = np.arcsin(u/norm)
    direction = xr.where(
        v < 0,
        np.pi-direction,
        direction
    )
    direction = xr.where(
        direction < 0,
        2*np.pi+direction,
        direction
    )
    arome_interpolated['dir'] = np.rad2deg(direction)
    return arome_interpolated


def get_arome_downscaled_direction_vectorized(arome_interpolated, unet_output):
    size_t = arome_interpolated.time.shape[-1]
    size_y = unet_output.y.shape[0]
    size_x = unet_output.x.shape[0]
    size_angles = unet_output.sizes['angle']

    rounded_dir = np.round(arome_interpolated.dir, 0).expand_dims(
        dim={"angle": size_angles})  # type:ignore
    angles = xr.DataArray(np.arange(0, size_angles, 1), dims=("angle",))
    angles = angles.expand_dims(
        dim={"x": size_x, "y": size_y, "time": size_t})
    direction = arome_interpolated['dir'].expand_dims(
        dim={"angle": size_angles})
    windspeed = arome_interpolated['windspeed'].expand_dims(
        dim={"angle": size_angles})
    match_matrix = rounded_dir == angles
    # remarquablement rapide...
    dir_corr = (
        xr.where(match_matrix, direction, 0)
        + xr.where(match_matrix, unet_output.alpha, 0)
    ).sum(dim="angle")
    return dir_corr


def get_arome_downscaled_direction_loop(arome_interpolated, unet_output):
    """
    direction only - used to benchmark in comparison with the vectorized version
    """
    rounded_dir = np.mod(np.round(arome_interpolated.dir, 0), 360)
    size_t = arome_interpolated.time.shape[0]
    size_y = unet_output.y.shape[0]
    size_x = unet_output.x.shape[0]
    size_angles = unet_output.sizes['angle']
    dir_corr = xr.DataArray(data=np.zeros((size_x, size_y, size_t),
                            dtype="float32"), dims=("x", "y", "time"))
    for angle in tqdm(range(size_angles)):
        dir_corr = xr.where(
            rounded_dir == angle,
            np.mod(arome_interpolated.dir -
                   unet_output.alpha.isel(angle=angle), 360),
            dir_corr
        )
    return dir_corr


def get_arome_downscaled_loop(arome_interpolated: xr.Dataset, unet_output: xr.Dataset) -> Tuple[xr.DataArray, xr.DataArray]:
    """
    une option: vu que c'est très très long
    traiter un maximum de temps en même temps
    à intervalles réguliers dans la boucle sur les angles, dumper les résultats
    dans un fichier de résultat temporaire
    (et avoir la possibilité de reprendre en cours de route)
    """
    rounded_dir = np.mod(np.round(arome_interpolated.dir, 0), 360)
    size_t = arome_interpolated.time.shape[0]
    size_y = unet_output.y.shape[0]
    size_x = unet_output.x.shape[0]
    size_angles = unet_output.sizes['angle']
    dir_corr = xr.DataArray(data=np.zeros((size_x, size_y, size_t),
                            dtype="float32"), dims=("x", "y", "time"))
    windspeed_corr = xr.DataArray(data=np.zeros((size_x, size_y, size_t),
                                                dtype="float32"), dims=("x", "y", "time"))
    for angle in tqdm(range(size_angles)):
        dir_corr = xr.where(
            rounded_dir == angle,
            np.mod(arome_interpolated.dir -
                   np.rad2deg(unet_output.alpha.isel(angle=angle)),
                   360),
            dir_corr
        )
        windspeed_corr = xr.where(
            rounded_dir == angle,
            unet_output.acceleration.isel(
                angle=angle)*arome_interpolated.windspeed,
            windspeed_corr
        )
    return (windspeed_corr, dir_corr)
