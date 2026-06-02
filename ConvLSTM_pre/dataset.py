import glob

import torch
import xarray as xr
from torch.utils.data import DataLoader, Dataset


class WeatherConvLSTMDataset(Dataset):
    def __init__(self, precip_data, land_data, n_past=10, n_future=10):
        self.n_past = n_past
        self.n_future = n_future
        self.n_total = n_past + n_future
        self.num_samples = len(precip_data) - self.n_total

        land_min, land_max = land_data.min(), land_data.max()
        self.land_data = (land_data - land_min) / (land_max - land_min)
        self.land_data = self.land_data.float().unsqueeze(0).unsqueeze(0)

        self.precip_data = torch.log1p(precip_data.float())

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        start_idx = idx
        end_idx = start_idx + self.n_total

        precip_seq = self.precip_data[start_idx:end_idx].unsqueeze(1)
        land_seq = self.land_data.expand(self.n_total, -1, -1, -1)
        combined_seq = torch.cat([precip_seq, land_seq], dim=1)

        x_past = combined_seq[: self.n_past]
        y_future = precip_seq[self.n_past :]
        return x_past, y_future


def load_precip_data(
    precip_data_path="../data/data_0511/nc_NZL/*.nc",
    variable_name="GPM_3IMERGHH_07_precipitation",
):
    precip_files = sorted(glob.glob(precip_data_path))
    if not precip_files:
        raise FileNotFoundError(f"No precipitation files matched: {precip_data_path}")

    precip_data = []
    for precip_file in precip_files:
        with xr.open_dataset(precip_file) as precip_ds:
            precip_numpy = precip_ds[variable_name].values
        precip_data.append(torch.from_numpy(precip_numpy))
    return torch.cat(precip_data, dim=0)


def load_land_data(
    land_data_path="../data/data_0511/gebco0_1_land_only.nc",
    variable_name="elevation",
):
    with xr.open_dataset(land_data_path) as land_ds:
        land_numpy = land_ds[variable_name].values
    return torch.from_numpy(land_numpy)


def get_dataloaders(
    batch_size=32,
    n_past=10,
    n_future=10,
    precip_data_path="../data/data_0511/nc_NZL/*.nc",
    land_data_path="../data/data_0511/gebco0_1_land_only.nc",
    num_workers=4,
):
    precip_data = load_precip_data(precip_data_path)
    land_data = load_land_data(land_data_path)

    train_precip_data = precip_data[:750]
    val_precip_data = precip_data[750:900]
    test_precip_data = precip_data[900:]

    train_dataset = WeatherConvLSTMDataset(
        train_precip_data, land_data, n_past, n_future
    )
    val_dataset = WeatherConvLSTMDataset(val_precip_data, land_data, n_past, n_future)
    test_dataset = WeatherConvLSTMDataset(test_precip_data, land_data, n_past, n_future)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader, test_loader
