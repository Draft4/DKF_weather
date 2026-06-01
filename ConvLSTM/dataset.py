import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os


def patchify(x, patch_size=4):
    """
    将图像分割为 patch
    输入: (seq_len, C, H, W)
    输出: (seq_len, C * patch_size^2, H//patch_size, W//patch_size)
    """
    seq_len, C, H, W = x.shape
    assert H % patch_size == 0 and W % patch_size == 0
    h_patches = H // patch_size
    w_patches = W // patch_size

    # (seq_len, C, h_patches, patch_size, w_patches, patch_size)
    x = x.reshape(seq_len, C, h_patches, patch_size, w_patches, patch_size)
    # (seq_len, h_patches, w_patches, C, patch_size, patch_size)
    x = x.permute(0, 2, 4, 1, 3, 5)
    # (seq_len, h_patches, w_patches, C * patch_size * patch_size)
    x = x.reshape(seq_len, h_patches, w_patches, C * patch_size * patch_size)
    # (seq_len, C * patch_size^2, h_patches, w_patches)
    x = x.permute(0, 3, 1, 2)
    return x


def unpatchify(x, patch_size=4, H=64, W=64):
    """
    将 patch 恢复为图像
    输入: (seq_len, C * patch_size^2, H//patch_size, W//patch_size)  numpy 或 tensor
    输出: (seq_len, C, H, W)
    """
    # 统一转换为 torch tensor
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)

    seq_len, _, h_patches, w_patches = x.shape
    C = x.shape[1] // (patch_size * patch_size)

    # (seq_len, C, patch_size, patch_size, h_patches, w_patches)
    x = x.reshape(seq_len, C, patch_size, patch_size, h_patches, w_patches)
    # (seq_len, C, h_patches, patch_size, w_patches, patch_size)
    x = x.permute(0, 1, 4, 2, 5, 3)
    # (seq_len, C, H, W)
    x = x.reshape(seq_len, C, H, W)
    return x


class MovingMNISTDataset(Dataset):
    def __init__(
        self,
        data_path="../data/MovingMNIST/mnist_test_seq.npy",
        T_past=10,
        T_future=10,
        patch_size=4,
    ):
        super(MovingMNISTDataset, self).__init__()
        self.T_past = T_past
        self.T_future = T_future
        self.T_total = T_past + T_future
        self.patch_size = patch_size

        if os.path.exists(data_path):
            data = np.load(data_path)
            # 数据形状: (20, 10000, 64, 64)
            # 转置为: (10000, 20, 64, 64)
            data = np.transpose(data, (1, 0, 2, 3))
            self.data = torch.from_numpy(data).float() / 255.0
        else:
            raise FileNotFoundError(f"数据文件未找到: {data_path}")

        if self.data.dim() == 4:
            self.data = self.data.unsqueeze(2)

        self.num_sequences = self.data.size(0)
        self.seq_len = self.data.size(1)

        print(
            f"数据集加载完成: {self.num_sequences} 个序列, "
            f"每序列 {self.seq_len} 帧, 形状: {self.data.shape}"
        )

        if self.seq_len < self.T_total:
            raise ValueError(f"序列长度 {self.seq_len} 小于所需长度 {self.T_total}")

    def __len__(self):
        return self.num_sequences

    def __getitem__(self, idx):
        sample = self.data[idx, : self.T_total]
        # 将图像转换为 patch
        sample = patchify(sample, self.patch_size)
        x_past = sample[: self.T_past]
        x_future = sample[self.T_past :]

        return x_past, x_future


def get_dataloaders(
    data_path="../data/MovingMNIST/mnist_test_seq.npy",
    batch_size=32,
    T_past=10,
    T_future=10,
    num_workers=4,
    train_split=0.8,
):
    dataset = MovingMNISTDataset(data_path, T_past, T_future)

    train_size = int(len(dataset) * train_split)
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, test_size], torch.Generator().manual_seed(42)
    )
    print(f"训练集大小: {len(train_dataset)}, 测试集大小: {len(test_dataset)}")
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, test_loader
