import xarray as xr
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os
import glob


class WeatherDataset(Dataset):
    def __init__(self, precip_data, land_data, n_past=10, n_future=10):
        self.n_past = n_past
        self.n_future = n_future
        self.n_Total = n_past + n_future

        self.num_samples = len(precip_data) - self.n_Total

        # 地形数据归一化（只需做一次）
        land_min, land_max = land_data.min(), land_data.max()
        self.land_data = (land_data - land_min) / (land_max - land_min)

        # 预处理降水数据：对数变换（只需做一次）
        self.precip_data = torch.log1p(precip_data)

        # 预处理地形数据：扩展时间维度并增加通道维度（只需做一次）
        # land_data: (H, W) -> (1, 1, H, W)
        self.land_data = self.land_data.unsqueeze(0).unsqueeze(0)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        start_idx = idx
        end_idx = start_idx + self.n_Total

        # 切片获取降水序列: (T_total, 1, H, W)
        precip_seq = self.precip_data[start_idx:end_idx].unsqueeze(1)

        # 地形数据直接复用预处理好的: (1, 1, H, W) -> (T_total, 1, H, W)
        land_seq = self.land_data.expand(self.n_Total, -1, -1, -1)

        # 拼接: (T_total, 2, H, W)
        combined_seq = torch.cat([precip_seq, land_seq], dim=1)

        return combined_seq


def get_dataloaders(
    batch_size=32,
    n_past=10,
    n_future=10,
    precip_data_path="../data/data_0511/nc_NZL/*.nc",
    land_data_path="../data/data_0511/gebco0_1_land_only.nc",
    num_workers=4,
):
    precip_files = sorted(glob.glob(precip_data_path))
    precip_data = []
    for precip_file in precip_files:
        precip_ds = xr.open_dataset(precip_file)
        precip_numpy = precip_ds["GPM_3IMERGHH_07_precipitation"].values
        precip_tensor = torch.from_numpy(precip_numpy)
        precip_data.append(precip_tensor)
    precip_data = torch.cat(precip_data, dim=0)

    land_ds = xr.open_dataset(land_data_path)
    land_numpy = land_ds["elevation"].values
    land_data = torch.from_numpy(land_numpy)

    train_predip_data = precip_data[:750]
    val_predip_data = precip_data[750:900]
    test_predip_data = precip_data[900:]

    train_dataset = WeatherDataset(train_predip_data, land_data, n_past, n_future)
    val_dataset = WeatherDataset(val_predip_data, land_data, n_past, n_future)
    test_dataset = WeatherDataset(test_predip_data, land_data, n_past, n_future)

    train_loader = DataLoader(
        train_dataset, batch_size, True, num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size, False, num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size, False, num_workers=num_workers, pin_memory=True
    )

    return train_loader, val_loader, test_loader
