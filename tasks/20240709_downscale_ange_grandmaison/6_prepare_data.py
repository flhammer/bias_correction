from os.path import join as pj
import xarray as xr
import gc


from obsdatacen.bdap.database import session
from obsdatacen.bdap.models_orm_style import (
    Extraction,
    get_request_by_id,
)


def reshape_to_timeserie(bdap_array):
    shape = bdap_array.data.shape
    time = bdap_array.valid_time.data.reshape(shape[0] * shape[1])
    reshaped_array = xr.DataArray(
        bdap_array.data.reshape(shape[0] * shape[1], shape[2], shape[3]),
        dims=["time", "latitude", "longitude"],
        coords=dict(
            latitude=bdap_array.latitude, longitude=bdap_array.longitude, time=time
        ),
    )
    return reshaped_array


data_dir = "/home/merzisenh/NO_SAVE/bias_correction_data"

Extraction.print_all_requests(session)

req_u = get_request_by_id(session, 51)  # U, 2019-08-01 à 2019-01-31
req_v = get_request_by_id(session, 53)  # V, 2019-08-01 à 2019-01-31
u_arome_2019 = req_u.get_data_array()  # U, 2019-08-01 à 2019-01-31
v_arome_2019 = req_v.get_data_array()  # V, 2019-08-01 à 2019-01-31


req = get_request_by_id(session, 52)  # U, 2019-08-01 à 2019-01-31
req.param
req.run_min
req.run_max
u_arome_2020 = req.get_data_array()  # U, 2019-08-01 à 2019-01-31

req = get_request_by_id(session, 54)  # U, 2019-08-01 à 2019-01-31
req.param
req.run_min
req.run_max

v_arome_2020 = req.get_data_array()  # U, 2019-08-01 à 2019-01-31


u_arome_2019 = reshape_to_timeserie(u_arome_2019)
u_arome_2020 = reshape_to_timeserie(u_arome_2020)
v_arome_2019 = reshape_to_timeserie(v_arome_2019)
v_arome_2020 = reshape_to_timeserie(v_arome_2020)

u = xr.concat([u_arome_2019, u_arome_2020], dim="time")
v = xr.concat([v_arome_2019, v_arome_2020], dim="time")

arome = xr.Dataset(dict(u=u, v=v))

arome.to_netcdf(pj(data_dir, "arome_wind", "uv_2019_2020.nc"))
