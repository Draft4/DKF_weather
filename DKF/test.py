import xarray as xr
import numpy as np
import torch
import glob

path = "../data/data_0511/nc_NZL/*.nc"
files = sorted(glob.glob(path))
# print(files)

data_tensor = []
for i in files:
    ds = xr.open_dataset(i)
    num_array = ds["GPM_3IMERGHH_07_precipitation"].values
    num_tensor = torch.from_numpy(num_array)
    # num_tensor = torch.nn.functional.pad(
    #     num_tensor, pad=[0, 3, 0, 2], mode="constant", value=0
    # )
    data_tensor.append(num_tensor)
print(data_tensor[0].shape)
data_tensor = torch.cat(data_tensor, dim=0)
data_tensor = torch.nn.functional.pad(
    data_tensor, pad=[0, 3, 0, 2], mode="constant", value=0
)
print(data_tensor.shape)


# ds2 = xr.open_dataset(
#     "../data/data_0511/nc_NZL/scrubbed.GPM_3IMERGHH_07_precipitation.20200703000000.nc"
# )
# # print(ds2)
# data_array = ds2["GPM_3IMERGHH_07_precipitation"]
# num_array = data_array.values
# num_tensor = torch.from_numpy(num_array)
# print(num_tensor)
# num_tensor = torch.nn.functional.pad(
#     num_tensor, pad=[0, 3, 0, 2], mode="constant", value=0
# )
# print(num_tensor.shape)
# ds2 = xr.open_dataset("../data/data_0511/gebco0_1_land_only.nc")
# num_array = ds2["elevation"].values
# print(num_array)
# # for i in range(158):
# #     for j in range(165):
# #         if num_array[i, j] != 0:
# #             print(num_array[i, j])
