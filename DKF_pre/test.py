import torch
import xarray as xr
import numpy as np

ds = xr.open_dataset("../data/data_0511/gebco0_1_land_only.nc")
print(ds)
land_data = ds["elevation"].values
land_data = torch.from_numpy(land_data)
print(land_data.shape)
