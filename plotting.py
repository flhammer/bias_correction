import matplotlib.pyplot as plt
import xarray as xr
from bias_correction.dem import dem_txt_to_numpy

prediction = True


if prediction:
    ds = xr.open_dataset("results.nc")

    accels = ds.sel(angle=270).acceleration.values
    plt.imshow(accels)
    plt.colorbar()
    plt.savefig(f"tf_prediction.png", dpi=300)
    plt.close()


dem_path = "gaussiandem_N5451_dx30_xi800_sigma71_r000.txt"
dem = dem_txt_to_numpy(dem_path)
plt.imshow(dem)
plt.colorbar()
plt.savefig(f"tf_input.png", dpi=300)
plt.close()

flow_path = "gaussianu_N5451_dx30_xi800_sigma71_r000.txt"
flow = dem_txt_to_numpy(flow_path)
plt.imshow(flow)
plt.colorbar()
plt.savefig(f"tf_flow_field.png", dpi=300)
plt.close()
