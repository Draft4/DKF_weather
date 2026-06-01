import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader

# 导入你自己的模块
from modules_dkf import DeepKalmanFilter
from dataset import get_dataloaders


def load_model(model_path, device):
    """
    初始化模型并加载训练好的权重
    """
    model = DeepKalmanFilter().to(device)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"找不到模型权重文件: {model_path}")

    # 加载状态字典，严格匹配参数
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()  # 必须设置为评估模式！
    print(f"成功加载模型权重: {model_path}")
    return model


def visualize_trajectory(
    model, dataloader, device, n_past=10, n_future=10, save_path="prediction_result.png"
):
    """
    在测试集上抽取一个 Batch，进行自回归预测，并将结果拼接保存为高清图片
    """
    print("开始执行推断与可视化...")

    with torch.no_grad():  # 禁用梯度，节省显存
        # 1. 从 dataloader 中只取出一个 Batch 用于可视化
        x_batch = next(iter(dataloader))
        x_batch = x_batch.to(device)

        # 2. 运行模型的 predict 模式，进行开环推演
        # predictions shape: [Batch, n_future, Channels, Height, Width]
        predictions = model(x_batch, n_past, n_future, mode="predict")

        # 3. 取出 Batch 中的第一个样本 (Sample 0) 转换到 CPU 并转为 NumPy 数组
        # 真实序列 shape: [T_total, 1, 64, 64]
        x_true_np = x_batch[0].cpu().numpy()
        # 预测序列 shape: [n_future, 1, 64, 64]
        pred_np = predictions[0].cpu().numpy()

    # ================= 开始绘图 =================
    # 创建一个 3 行、列数为 max(n_past, n_future) 的画布
    cols = max(n_past, n_future)
    fig, axes = plt.subplots(nrows=3, ncols=cols, figsize=(2 * cols, 6))
    fig.suptitle("Deep Kalman Filter: Moving MNIST Forecasting", fontsize=16)

    # 绘制第一行：真实的过去帧 (Burn-in Context)
    for t in range(n_past):
        ax = axes[0, t]
        # vmin=0, vmax=1 是极其关键的参数！强制规定黑白界限，防止 matplotlib 乱缩放
        ax.imshow(x_true_np[t, 0], cmap="gray", vmin=0.0, vmax=1.0)
        ax.set_title(f"Past t={t+1}")
        ax.axis("off")
    # 如果未来帧比过去帧多，隐藏第一行多出来的子图
    for t in range(n_past, cols):
        axes[0, t].axis("off")

    # 绘制第二行：真实的未来帧 (Ground Truth)
    for t in range(n_future):
        ax = axes[1, t]
        ax.imshow(x_true_np[n_past + t, 0], cmap="gray", vmin=0.0, vmax=1.0)
        ax.set_title(f"True t={n_past + t + 1}")
        ax.axis("off")
    for t in range(n_future, cols):
        axes[1, t].axis("off")

    # 绘制第三行：模型预测的未来帧 (Predictions)
    for t in range(n_future):
        ax = axes[2, t]
        ax.imshow(pred_np[t, 0], cmap="gray", vmin=0.0, vmax=1.0)
        ax.set_title(f"Pred t={n_past + t + 1}")
        ax.axis("off")
    for t in range(n_future, cols):
        axes[2, t].axis("off")

    # 调整布局并保存
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)  # 给主标题留点空间
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f">>> 可视化结果已成功保存为高清图片: {save_path}")


def visualize_train_reconstruction(
    model,
    dataloader,
    device,
    n_past=10,
    n_future=10,
    save_path="train_reconstruction_result.png",
):
    """
    在训练集上抽取一个 Batch，使用 train 模式进行重构，并将结果拼接保存为高清图片
    train 模式会对整个序列 (past + future) 进行编码-推断-解码重构
    """
    print("开始执行 train 模式重构与可视化...")

    with torch.no_grad():
        x_batch = next(iter(dataloader))
        x_batch = x_batch.to(device)

        # 运行模型的 train 模式，返回整个序列的重构
        # recon_preds shape: [Batch, T_total, 1, 64, 64]
        # kld_loss: KL 散度损失
        recon_preds, kld_loss = model(x_batch, n_past, n_future, mode="train")

        # 取出第一个样本
        x_true_np = x_batch[0].cpu().numpy()  # [T_total, 1, 64, 64]
        recon_np = recon_preds[0].cpu().numpy()  # [T_total, 1, 64, 64]

    T_total = n_past + n_future
    cols = T_total

    # 创建 2 行、T_total 列的画布
    fig, axes = plt.subplots(nrows=2, ncols=cols, figsize=(2 * cols, 4))
    fig.suptitle(
        f"Deep Kalman Filter: Train Mode Reconstruction (KLD: {kld_loss.item():.4f})",
        fontsize=16,
    )

    # 第一行：原始序列
    for t in range(T_total):
        ax = axes[0, t]
        ax.imshow(x_true_np[t, 0], cmap="gray", vmin=0.0, vmax=1.0)
        if t < n_past:
            ax.set_title(f"Past {t+1}", fontsize=10)
        else:
            ax.set_title(f"Future {t+1}", fontsize=10, color="green")
        ax.axis("off")

    # 第二行：重构序列
    for t in range(T_total):
        ax = axes[1, t]
        ax.imshow(recon_np[t, 0], cmap="gray", vmin=0.0, vmax=1.0)
        if t < n_past:
            ax.set_title(f"Recon {t+1}", fontsize=10)
        else:
            ax.set_title(f"Recon {t+1}", fontsize=10, color="blue")
        ax.axis("off")

    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f">>> Train 模式重构可视化结果已保存: {save_path}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 你的模型权重保存路径 (确保与 train.py 中保存的一致)
    MODEL_PATH = "./checkpoints/best_model.pth"

    # ================= 数据加载准备 =================
    train_loader, test_loader = get_dataloaders(
        "../data/MovingMNIST/mnist_test_seq.npy"
    )

    try:
        model = DeepKalmanFilter().to(device)
        checkpoint = torch.load(MODEL_PATH, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])

        # 1. 测试集 predict 模式可视化
        visualize_trajectory(
            model,
            test_loader,
            device,
            n_past=10,
            n_future=10,
            save_path="prediction_result.png",
        )

        # 2. 训练集 train 模式重构可视化
        visualize_train_reconstruction(
            model,
            train_loader,
            device,
            n_past=10,
            n_future=10,
            save_path="train_reconstruction_result.png",
        )

    except Exception as e:
        print(f"执行可视化时出错: {e}")


if __name__ == "__main__":
    main()
