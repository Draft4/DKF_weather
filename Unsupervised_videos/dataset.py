import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os


class MovingMNISTDataset(Dataset):
    """
    Moving MNIST 数据集加载器。
    使用预生成的 mnist_test_seq.npy 文件。
    数据格式: (T, N, H, W) = (20, 10000, 64, 64)
    需要转置为 (N, T, H, W) = (10000, 20, 64, 64)
    """

    def __init__(
        self, data_path="../data/MovingMNIST/mnist_test_seq.npy", T_past=10, T_future=10
    ):
        super(MovingMNISTDataset, self).__init__()
        self.T_past = T_past
        self.T_future = T_future
        self.T_total = T_past + T_future

        # 加载数据
        if os.path.exists(data_path):
            data = np.load(data_path)
            # 数据形状: (20, 10000, 64, 64)
            # 转置为: (10000, 20, 64, 64)
            data = np.transpose(data, (1, 0, 2, 3))
            self.data = torch.from_numpy(data).float() / 255.0
        else:
            raise FileNotFoundError(f"数据文件未找到: {data_path}")

        # 确保数据形状为 (N, T, 1, 64, 64)
        if self.data.dim() == 4:
            # (N, T, 64, 64) -> (N, T, 1, 64, 64)
            self.data = self.data.unsqueeze(2)

        self.num_sequences = self.data.size(0)
        self.seq_len = self.data.size(1)

        print(
            f"数据集加载完成: {self.num_sequences} 个序列, "
            f"每序列 {self.seq_len} 帧, 形状: {self.data.shape}"
        )

        # 检查序列长度是否足够
        if self.seq_len < self.T_total:
            raise ValueError(f"序列长度 {self.seq_len} 小于所需长度 {self.T_total}")

    def __len__(self):
        # 每个完整序列作为一个样本
        return self.num_sequences

    def __getitem__(self, idx):
        # 提取连续 T_total 帧
        sample = self.data[idx, : self.T_total]  # (T_total, 1, 64, 64)

        # 划分为 past 和 future
        x_past = sample[: self.T_past]  # (T_past, 1, 64, 64)
        x_future = sample[self.T_past :]  # (T_future, 1, 64, 64)

        return x_past, x_future


def get_dataloaders(
    data_path="../data/MovingMNIST/mnist_test_seq.npy",
    batch_size=32,
    T_past=10,
    T_future=10,
    num_workers=4,
    train_split=0.8,
):
    """
    创建训练和测试 DataLoader。
    将数据集按 train_split 比例划分为训练集和测试集。
    """
    dataset = MovingMNISTDataset(data_path=data_path, T_past=T_past, T_future=T_future)

    # 划分训练集和测试集
    train_size = int(len(dataset) * train_split)
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, test_size], generator=torch.Generator().manual_seed(42)
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
