import numpy as np
import xarray as xr
from typing import Tuple
import pyproj

from tqdm import tqdm
from scipy.interpolate import LinearNDInterpolator, griddata
from scipy.spatial import Delaunay


def recompute_coordinates_on_unet_output(unet_output, unet_dem_input, epsg):
    unet_output_with_coordinates = unet_output.copy()
    length_y, length_x = unet_dem_input.shape
    min_length = np.min([length_y, length_x])
    y_diff = np.intp((min_length / np.sqrt(2))) // 2
    x_diff = y_diff
    y_min = length_y // 2 - y_diff
    y_max = length_y // 2 + y_diff + 1
    x_min = length_x // 2 - x_diff
    x_max = length_x // 2 + x_diff + 1
    unet_output_with_coordinates.coords["x"] = (
        ("x",),
        unet_dem_input.coords["x"].data[x_min:x_max],
    )
    unet_output_with_coordinates.coords["y"] = (
        ("y",),
        unet_dem_input.coords["y"].data[y_min:y_max],
    )
    unet_output_with_coordinates = unet_output_with_coordinates.rio.write_crs(epsg)
    return unet_output_with_coordinates


def convert_4324_to_2154(longitude: xr.DataArray, latitude: xr.DataArray):
    gps_to_l93 = pyproj.Transformer.from_crs(4324, 2154, always_xy=True)
    this_is_a_generator = gps_to_l93.itransform(
        np.array([longitude.values.ravel(), latitude.values.ravel()]).swapaxes(0, 1)
    )
    this_is_an_array = np.array(list(this_is_a_generator)).swapaxes(0, 1)
    this_is_an_array.shape
    x2154 = xr.DataArray(
        this_is_an_array[0].reshape(longitude.shape), dims=longitude.dims
    )
    y2154 = xr.DataArray(
        this_is_an_array[1].reshape(longitude.shape), dims=longitude.dims
    )
    return x2154, y2154


def reshape_as_grid(x_var, y_var):
    """
    x_var, y_var coordonnées 1D sur une grille régulière
    on retourne les valeurs de x et y sur une grille y,x en
    répétant les valeurs
    """
    size_x = x_var.shape[0]
    size_y = y_var.shape[0]
    x_values = np.tile(x_var.values, (size_y, 1))
    y_values = np.tile(y_var.values, (size_x, 1)).T
    x_dim_name = x_var.dims[0]
    y_dim_name = y_var.dims[0]
    x_reshaped = xr.DataArray(x_values, dims=[y_dim_name, x_dim_name])
    y_reshaped = xr.DataArray(y_values, dims=[y_dim_name, x_dim_name])
    return (x_reshaped, y_reshaped)


class DelaunayFixGridsInterpolator:
    def __init__(
        self,
        x_source,
        y_source,
        x_target,
        y_target,
    ):
        """
        On suppose qu'on n'est pas dans un cas tordu ou les coordonnées sont inversées,
        i.e. x(yy,xx) ...
        Pour les grib extraits de la BDAP on sera sur latitude(y) longitude(x) par exemple comme noms de dimensions
        """
        self.x_source = x_source
        self.y_source = y_source
        self.x_target = x_target
        self.y_target = y_target
        self.points_source = self._shape_coord_(x_source, y_source)
        self.points_target = self._shape_coord_(x_target, y_target)

    def _shape_coord_(self, x_array, y_array):
        return np.array([x_array.values.ravel(), y_array.values.ravel()]).T

    def _values_to_numpy_(self, values_array):
        """
        L'array de valeurs est supposé bidimensionnel (dépend de la dim associée
        à x et de celle associée à y
        """
        if values_array.dims == self.xy_source_dim_names:
            return values_array.values.ravel()
        if values_array.dims == self.xy_source_dim_names[::-1]:
            return values_array.values.T.ravel()
        raise Exception("non implémenté")

    def compute_interpolation_coeffs(self):
        triangulation = Delaunay(self.points_source)
        indexes = triangulation.find_simplex(self.points_target)
        # pour chaque index
        T = triangulation.transform[
            indexes, :2
        ]  # pour chaque index, la transformation affine du triangle
        r = triangulation.transform[indexes, 2]  # le barycentre pour chaque index
        ab = np.array(
            [np.dot(T[i], self.points_target[i] - r[i]) for i in range(T.shape[0])]
        )
        c = 1 - ab.sum(axis=1)
        self.index_interpolating_triangle_vertex = triangulation.simplices[indexes]
        self.regression_coeff_on_interpolating_triangle = np.c_[ab, c]

    def compute_interpolated_values_on_target(self, values_array: xr.DataArray):
        values = values_array.values.ravel()
        try:
            values_shaped = values[self.index_interpolating_triangle_vertex]
            values_on_target = (
                self.regression_coeff_on_interpolating_triangle * values_shaped
            ).sum(axis=1)
            return values_on_target.reshape(self.x_target.shape)
        except AttributeError:
            print("Please call compute_interpolation_coeffs method first")


def get_arome_interpolated(
    arome_data: xr.Dataset, target_dem: xr.DataArray
) -> xr.Dataset:
    arome_data["u"] = np.sin(np.deg2rad(arome_data.Wind_DIR)) * arome_data.Wind
    arome_data["v"] = np.cos(np.deg2rad(arome_data.Wind_DIR)) * arome_data.Wind

    size_t = arome_data.time.shape[0]
    size_y = target_dem.y.shape[0]
    size_x = target_dem.x.shape[0]

    arome_interpolated = xr.Dataset(
        data_vars=dict(
            u=(
                ["time", "y", "x"],
                np.zeros((size_t, size_y, size_x), dtype=np.float32),
            ),
            v=(
                ["time", "y", "x"],
                np.zeros((size_t, size_y, size_x), dtype=np.float32),
            ),
        ),
        coords=dict(
            x=(["x"], target_dem.x.data),
            y=(["y"], target_dem.y.data),
            time=(["time"], arome_data.time.data),
        ),
    )
    epsg = "EPSG:2154"
    arome_interpolated = arome_interpolated.rio.write_crs(epsg)

    # Calcul de l'interpolation d'Arome sur la grille de large_dem (u et v)
    grid_x = np.tile(target_dem.x, (size_y, 1))
    grid_y = np.tile(target_dem.y, (size_x, 1)).T
    x_flat = arome_data.isel(time=0)["x"].values.ravel()
    y_flat = arome_data.isel(time=0)["y"].values.ravel()
    points = np.array([x_flat, y_flat]).T

    for time in tqdm(arome_data.time):
        u_arome = arome_data.sel(time=time)["u"].values.ravel()
        v_arome = arome_data.sel(time=time)["v"].values.ravel()
        arome_interpolated.u.loc[dict(time=time)] = griddata(
            points, u_arome, (grid_x, grid_y), method="linear"
        ).astype("float32")
        arome_interpolated.v.loc[dict(time=time)] = griddata(
            points, v_arome, (grid_x, grid_y), method="linear"
        ).astype("float32")

    # Calcul de force et direction du vent interpolé
    u = arome_interpolated.u
    v = arome_interpolated.v
    norm = np.sqrt(u ** 2 + v ** 2)
    arome_interpolated["windspeed"] = norm
    direction = np.arcsin(u / norm)
    direction = xr.where(v < 0, np.pi - direction, direction)
    direction = xr.where(direction < 0, 2 * np.pi + direction, direction)
    arome_interpolated["dir"] = np.rad2deg(direction)
    return arome_interpolated


def get_arome_downscaled_direction_vectorized(arome_interpolated, unet_output):
    size_t = arome_interpolated.time.shape[-1]
    size_y = unet_output.y.shape[0]
    size_x = unet_output.x.shape[0]
    size_angles = unet_output.sizes["angle"]

    rounded_dir = np.round(arome_interpolated.dir, 0).expand_dims(
        dim={"angle": size_angles}
    )  # type:ignore
    angles = xr.DataArray(np.arange(0, size_angles, 1), dims=("angle",))
    angles = angles.expand_dims(dim={"x": size_x, "y": size_y, "time": size_t})
    direction = arome_interpolated["dir"].expand_dims(dim={"angle": size_angles})
    windspeed = arome_interpolated["windspeed"].expand_dims(dim={"angle": size_angles})
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
    size_angles = unet_output.sizes["angle"]
    dir_corr = xr.DataArray(
        data=np.zeros((size_x, size_y, size_t), dtype="float32"),
        dims=("x", "y", "time"),
    )
    for angle in tqdm(range(size_angles)):
        dir_corr = xr.where(
            rounded_dir == angle,
            np.mod(arome_interpolated.dir - unet_output.alpha.isel(angle=angle), 360),
            dir_corr,
        )
    return dir_corr


def get_arome_downscaled_loop(
    arome_interpolated: xr.Dataset, unet_output: xr.Dataset
) -> Tuple[xr.DataArray, xr.DataArray]:
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
    size_angles = unet_output.sizes["angle"]
    dir_corr = xr.DataArray(
        data=np.zeros((size_x, size_y, size_t), dtype="float32"),
        dims=("x", "y", "time"),
    )
    windspeed_corr = xr.DataArray(
        data=np.zeros((size_x, size_y, size_t), dtype="float32"),
        dims=("x", "y", "time"),
    )
    for angle in tqdm(range(size_angles)):
        dir_corr = xr.where(
            rounded_dir == angle,
            np.mod(
                arome_interpolated.dir
                - np.rad2deg(unet_output.alpha.isel(angle=angle)),
                360,
            ),
            dir_corr,
        )
        windspeed_corr = xr.where(
            rounded_dir == angle,
            unet_output.acceleration.isel(angle=angle) * arome_interpolated.windspeed,
            windspeed_corr,
        )
    return (windspeed_corr, dir_corr)


def force_dir_from_u_v(u, v):
    force = np.sqrt(u ** 2 + v ** 2)
    direction = np.arcsin(u / force)
    direction = xr.where(v < 0, np.pi - direction, direction)
    direction = xr.where(direction < 0, 2 * np.pi + direction, direction)
    direction = np.rad2deg(direction)
    return force, direction


def get_downscaled_wind(
    force: xr.DataArray, direction: xr.DataArray, unet_output: xr.Dataset
) -> Tuple[xr.DataArray, xr.DataArray]:
    rounded_dir = np.mod(np.round(direction, 0), 360)
    size_t = force.time.shape[0]
    size_s = force.step.shape[0]
    size_y = unet_output.y.shape[0]
    size_x = unet_output.x.shape[0]
    size_angles = unet_output.sizes["angle"]
    dir_corr = xr.DataArray(
        data=np.zeros((size_x, size_y, size_t, size_s), dtype="float32"),
        dims=("x", "y", "time", "step"),
    )
    force_corr = xr.DataArray(
        data=np.zeros((size_x, size_y, size_t, size_s), dtype="float32"),
        dims=("x", "y", "time", "step"),
    )
    for angle in tqdm(range(size_angles)):
        dir_corr = xr.where(
            rounded_dir == angle,
            np.mod(
                direction - np.rad2deg(unet_output.alpha.isel(angle=angle)),
                360,
            ),
            dir_corr,
        )
        force_corr = xr.where(
            rounded_dir == angle,
            unet_output.acceleration.isel(angle=angle) * force,
            force_corr,
        )
    return (force_corr, dir_corr)
