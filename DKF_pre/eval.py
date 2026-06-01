import os
import argparse
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import cm

from modules import DeepKalmanFilter
from dataset import get_dataloaders


def inverse_log_transform(x):
    """将对数变换后的数据还原为原始降水值 (mm/h)"""
    return torch.expm1(x)


def create_precip_cmap():
    """创建降水专用颜色映射：白->蓝->绿->黄->橙->红->紫"""
    colors = [
        (1.0, 1.0, 1.0),  # 0: 白色 (无降水)
        (0.8, 0.9, 1.0),  # 微量
        (0.4, 0.7, 1.0),  # 小雨
        (0.0, 0.5, 1.0),  # 中雨
        (0.0, 0.8, 0.4),  # 大雨
        (1.0, 1.0, 0.0),  # 暴雨
        (1.0, 0.6, 0.0),  # 大暴雨
        (1.0, 0.0, 0.0),  # 特大暴雨
        (0.6, 0.0, 0.6),  # 极端
    ]
    return LinearSegmentedColormap.from_list("precipitation", colors)


def visualize_precip_comparison(
    true_past, true_future, pred_future, n_past, n_future, save_path, sample_idx=0
):
    """
    可视化降水对比图
    true_past: (n_past, H, W) 真实历史降水
    true_future: (n_future, H, W) 真实未来降水
    pred_future: (n_future, H, W) 预测未来降水
    """
    cols = max(n_past, n_future)
    fig, axes = plt.subplots(nrows=3, ncols=cols, figsize=(2.5 * cols, 7))
    fig.suptitle(f"Precipitation Forecasting - Sample {sample_idx}", fontsize=14)

    cmap = create_precip_cmap()
    # 统一颜色范围，基于真实数据的最大值
    vmax = max(true_past.max(), true_future.max(), pred_future.max())
    vmin = 0

    # 第一行：历史降水
    for t in range(n_past):
        ax = axes[0, t]
        im = ax.imshow(true_past[t], cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(f"Past t={t+1}", fontsize=9)
        ax.axis("off")
    for t in range(n_past, cols):
        axes[0, t].axis("off")

    # 第二行：真实未来降水
    for t in range(n_future):
        ax = axes[1, t]
        im = ax.imshow(true_future[t], cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(f"True t={n_past+t+1}", fontsize=9)
        ax.axis("off")
    for t in range(n_future, cols):
        axes[1, t].axis("off")

    # 第三行：预测未来降水
    for t in range(n_future):
        ax = axes[2, t]
        im = ax.imshow(pred_future[t], cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(f"Pred t={n_past+t+1}", fontsize=9)
        ax.axis("off")
    for t in range(n_future, cols):
        axes[2, t].axis("off")

    # 添加 colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax, label="Precipitation (mm/h)")

    plt.tight_layout(rect=[0, 0, 0.9, 0.95])
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f">>> 降水对比图已保存: {save_path}")


def visualize_error_map(true_future, pred_future, n_future, save_path, sample_idx=0):
    """
    可视化误差空间分布图
    """
    cols = n_future
    fig, axes = plt.subplots(nrows=2, ncols=cols, figsize=(2.5 * cols, 5))
    fig.suptitle(f"Prediction Error Map - Sample {sample_idx}", fontsize=14)

    # 计算误差
    error = np.abs(pred_future - true_future)
    vmax_err = error.max()

    for t in range(n_future):
        # 真实值
        ax = axes[0, t]
        im1 = ax.imshow(true_future[t], cmap="Blues", vmin=0)
        ax.set_title(f"True t={t+1}", fontsize=9)
        ax.axis("off")

        # 误差
        ax = axes[1, t]
        im2 = ax.imshow(error[t], cmap="Reds", vmin=0, vmax=vmax_err)
        ax.set_title(f"Error t={t+1}", fontsize=9)
        ax.axis("off")

    cbar_ax1 = fig.add_axes([0.92, 0.55, 0.02, 0.35])
    fig.colorbar(im1, cax=cbar_ax1, label="True (mm/h)")

    cbar_ax2 = fig.add_axes([0.92, 0.15, 0.02, 0.35])
    fig.colorbar(im2, cax=cbar_ax2, label="Abs Error (mm/h)")

    plt.tight_layout(rect=[0, 0, 0.9, 0.95])
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f">>> 误差分布图已保存: {save_path}")


def visualize_scatter(true_future, pred_future, save_path):
    """
    散点图：真实值 vs 预测值
    """
    fig, ax = plt.subplots(figsize=(6, 6))

    true_flat = true_future.flatten()
    pred_flat = pred_future.flatten()

    # 过滤掉无降水区域，聚焦有降水的点
    mask = true_flat > 0.1
    if mask.sum() > 0:
        ax.scatter(true_flat[mask], pred_flat[mask], alpha=0.3, s=1, c="blue")

    # 对角线
    max_val = max(true_flat.max(), pred_flat.max())
    ax.plot([0, max_val], [0, max_val], "r--", lw=2, label="Perfect Prediction")

    ax.set_xlabel("True Precipitation (mm/h)", fontsize=12)
    ax.set_ylabel("Predicted Precipitation (mm/h)", fontsize=12)
    ax.set_title("True vs Predicted Scatter Plot", fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f">>> 散点图已保存: {save_path}")


def plot_mse_curve(mse_per_step, save_path):
    """
    绘制每步预测的 MSE 曲线
    """
    plt.figure(figsize=(10, 6))
    steps = range(1, len(mse_per_step) + 1)
    plt.plot(steps, mse_per_step, marker="o", linewidth=2, markersize=6)
    plt.xlabel("Prediction Step", fontsize=12)
    plt.ylabel("MSE (mm/h)^2", fontsize=12)
    plt.title("MSE over Prediction Steps", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.xticks(steps)

    # 标注数值
    for i, mse in enumerate(mse_per_step):
        plt.annotate(
            f"{mse:.4f}",
            (steps[i], mse),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontsize=8,
        )

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f">>> MSE 曲线已保存: {save_path}")


def evaluate_model(
    model, test_loader, device, save_dir, n_past=10, n_future=10, num_samples=3
):
    """
    评估模型并生成可视化结果
    """
    model.eval()
    os.makedirs(save_dir, exist_ok=True)

    total_mse = 0
    total_mae = 0
    sample_count = 0
    mse_per_step = [[] for _ in range(n_future)]

    with torch.no_grad():
        for batch_idx, x in enumerate(tqdm(test_loader, desc="Evaluating")):
            x = x.to(device)

            # 预测
            predictions = model(x, n_past, n_future, mode="predict")

            # 提取真实未来降水 (对数域)
            true_future_log = x[:, n_past:, 0, :, :]

            # 转换回原始降水值 (mm/h)
            true_future = inverse_log_transform(true_future_log)
            pred_future = inverse_log_transform(predictions.squeeze(2))

            # 计算指标 (在原始值域)
            mse = torch.mean((pred_future - true_future) ** 2).item()
            mae = torch.mean(torch.abs(pred_future - true_future)).item()
            total_mse += mse
            total_mae += mae

            for t in range(n_future):
                step_mse = torch.mean(
                    (pred_future[:, t] - true_future[:, t]) ** 2
                ).item()
                mse_per_step[t].append(step_mse)

            # 可视化样本
            if batch_idx == 0 and sample_count < num_samples:
                B = x.size(0)
                for i in range(min(B, num_samples - sample_count)):
                    # 提取单个样本
                    true_past = (
                        inverse_log_transform(x[i, :n_past, 0, :, :]).cpu().numpy()
                    )
                    true_future_i = true_future[i].cpu().numpy()
                    pred_future_i = pred_future[i].cpu().numpy()

                    # 1. 降水对比图
                    visualize_precip_comparison(
                        true_past,
                        true_future_i,
                        pred_future_i,
                        n_past,
                        n_future,
                        os.path.join(
                            save_dir, f"sample_{sample_count}_precip_comparison.png"
                        ),
                        sample_idx=sample_count,
                    )

                    # 2. 误差分布图
                    visualize_error_map(
                        true_future_i,
                        pred_future_i,
                        n_future,
                        os.path.join(save_dir, f"sample_{sample_count}_error_map.png"),
                        sample_idx=sample_count,
                    )

                    # 3. 散点图
                    visualize_scatter(
                        true_future_i,
                        pred_future_i,
                        os.path.join(save_dir, f"sample_{sample_count}_scatter.png"),
                    )

                    # 打印每步 MSE
                    print(f"\nSample {sample_count}:")
                    for t in range(n_future):
                        step_mse = float(
                            np.mean((true_future_i[t] - pred_future_i[t]) ** 2)
                        )
                        print(f"  Step {t+1}: MSE = {step_mse:.6f} (mm/h)^2")

                    sample_count += 1

    n_batches = len(test_loader)
    avg_mse = total_mse / n_batches
    avg_mae = total_mae / n_batches
    avg_mse_per_step = [np.mean(mse_list) for mse_list in mse_per_step]

    print(f"\n{'='*60}")
    print(f"Evaluation Results:")
    print(f"  Average MSE: {avg_mse:.6f} (mm/h)^2")
    print(f"  Average MAE: {avg_mae:.6f} mm/h")
    print(f"\nMSE per step:")
    for t, mse in enumerate(avg_mse_per_step):
        print(f"  Step {t+1}: {mse:.6f} (mm/h)^2")
    print(f"{'='*60}")

    # 绘制 MSE 曲线
    plot_mse_curve(avg_mse_per_step, os.path.join(save_dir, "mse_curve.png"))

    return avg_mse, avg_mae, avg_mse_per_step


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate DKF on Weather Precipitation Data"
    )
    parser.add_argument("--checkpoint", type=str, required=True, help="模型检查点路径")
    parser.add_argument(
        "--precip_data_path", type=str, default="../data/data_0511/nc_NZL/*.nc"
    )
    parser.add_argument(
        "--save_dir", type=str, default="./results", help="结果保存目录"
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--n_past", type=int, default=10)
    parser.add_argument("--n_future", type=int, default=10)
    parser.add_argument("--num_samples", type=int, default=3, help="可视化样本数")
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    _, _, test_loader = get_dataloaders(
        batch_size=args.batch_size,
        n_past=args.n_past,
        n_future=args.n_future,
        precip_data_path=args.precip_data_path,
        num_workers=args.num_workers,
    )

    model = DeepKalmanFilter().to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"加载模型: {args.checkpoint}, Epoch: {checkpoint.get('epoch', 'N/A')}")

    evaluate_model(
        model,
        test_loader,
        device,
        args.save_dir,
        args.n_past,
        args.n_future,
        args.num_samples,
    )

    print(f"\n所有结果已保存到: {args.save_dir}")


if __name__ == "__main__":
    main()
