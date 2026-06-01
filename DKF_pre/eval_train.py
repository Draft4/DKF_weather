import os
import argparse
import torch
import numpy as np
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from modules import DeepKalmanFilter
from dataset import get_dataloaders


def inverse_log_transform(x):
    """将对数变换后的数据还原为原始降水值 (mm/h)"""
    return torch.expm1(x)


def create_precip_cmap():
    """创建降水专用颜色映射：白->蓝->绿->黄->橙->红->紫"""
    colors = [
        (1.0, 1.0, 1.0),
        (0.8, 0.9, 1.0),
        (0.4, 0.7, 1.0),
        (0.0, 0.5, 1.0),
        (0.0, 0.8, 0.4),
        (1.0, 1.0, 0.0),
        (1.0, 0.6, 0.0),
        (1.0, 0.0, 0.0),
        (0.6, 0.0, 0.6),
    ]
    return LinearSegmentedColormap.from_list("precipitation", colors)


def visualize_train_reconstruction(
    original_seq, recon_seq, n_past, n_future, save_path, sample_idx=0, kld=0.0
):
    """
    可视化 train 模式下的重构效果
    original_seq: (T_total, H, W) 原始序列（降水通道）
    recon_seq: (T_total, H, W) 重构序列
    """
    T_total = n_past + n_future
    cols = T_total
    fig, axes = plt.subplots(nrows=2, ncols=cols, figsize=(2.2 * cols, 5))
    fig.suptitle(
        f"Train Mode Reconstruction - Sample {sample_idx} (KLD: {kld:.4f})",
        fontsize=14
    )

    cmap = create_precip_cmap()
    vmax = max(original_seq.max(), recon_seq.max())
    vmin = 0

    # 第一行：原始序列
    for t in range(T_total):
        ax = axes[0, t]
        im = ax.imshow(original_seq[t], cmap=cmap, vmin=vmin, vmax=vmax)
        if t < n_past:
            ax.set_title(f"Past {t+1}", fontsize=8)
        else:
            ax.set_title(f"Future {t+1}", fontsize=8, color="green")
        ax.axis("off")

    # 第二行：重构序列
    for t in range(T_total):
        ax = axes[1, t]
        im = ax.imshow(recon_seq[t], cmap=cmap, vmin=vmin, vmax=vmax)
        if t < n_past:
            ax.set_title(f"Recon {t+1}", fontsize=8)
        else:
            ax.set_title(f"Recon {t+1}", fontsize=8, color="blue")
        ax.axis("off")

    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax, label="Precipitation (mm/h)")

    plt.tight_layout(rect=[0, 0, 0.9, 0.92])
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f">>> 重构对比图已保存: {save_path}")


def visualize_recon_error(
    original_seq, recon_seq, n_past, n_future, save_path, sample_idx=0
):
    """
    可视化重构误差分布
    """
    T_total = n_past + n_future
    error = np.abs(recon_seq - original_seq)
    cols = T_total

    fig, axes = plt.subplots(nrows=2, ncols=cols, figsize=(2.2 * cols, 5))
    fig.suptitle(f"Reconstruction Error Map - Sample {sample_idx}", fontsize=14)

    vmax_err = error.max()

    for t in range(T_total):
        # 原始值
        ax = axes[0, t]
        im1 = ax.imshow(original_seq[t], cmap="Blues", vmin=0)
        ax.set_title(f"Orig {t+1}", fontsize=8)
        ax.axis("off")

        # 误差
        ax = axes[1, t]
        im2 = ax.imshow(error[t], cmap="Reds", vmin=0, vmax=vmax_err)
        ax.set_title(f"Error {t+1}", fontsize=8)
        ax.axis("off")

    cbar_ax1 = fig.add_axes([0.92, 0.55, 0.02, 0.35])
    fig.colorbar(im1, cax=cbar_ax1, label="Original (mm/h)")

    cbar_ax2 = fig.add_axes([0.92, 0.15, 0.02, 0.35])
    fig.colorbar(im2, cax=cbar_ax2, label="Abs Error (mm/h)")

    plt.tight_layout(rect=[0, 0, 0.9, 0.92])
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f">>> 重构误差图已保存: {save_path}")


def visualize_scatter(original_seq, recon_seq, save_path, sample_idx=0):
    """
    散点图：真实值 vs 重构值
    """
    fig, ax = plt.subplots(figsize=(6, 6))

    true_flat = original_seq.flatten()
    pred_flat = recon_seq.flatten()

    # 过滤掉无降水区域，聚焦有降水的点
    mask = true_flat > 0.1
    if mask.sum() > 0:
        ax.scatter(true_flat[mask], pred_flat[mask], alpha=0.3, s=1, c="blue")

    # 对角线
    max_val = max(true_flat.max(), pred_flat.max())
    ax.plot([0, max_val], [0, max_val], "r--", lw=2, label="Perfect Reconstruction")

    ax.set_xlabel("True Precipitation (mm/h)", fontsize=12)
    ax.set_ylabel("Reconstructed Precipitation (mm/h)", fontsize=12)
    ax.set_title(f"True vs Reconstructed Scatter Plot - Sample {sample_idx}", fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f">>> 散点图已保存: {save_path}")


def plot_train_mse_curve(mse_per_step, save_path, n_past, n_future):
    """
    绘制 train 模式下每步重构的 MSE 曲线
    """
    T_total = n_past + n_future
    plt.figure(figsize=(12, 6))
    steps = range(1, T_total + 1)

    # 区分 past 和 future
    past_steps = steps[:n_past]
    future_steps = steps[n_past:]
    past_mse = mse_per_step[:n_past]
    future_mse = mse_per_step[n_past:]

    plt.plot(past_steps, past_mse, marker="o", linewidth=2, markersize=6,
             label="Past Reconstruction", color="blue")
    plt.plot(future_steps, future_mse, marker="s", linewidth=2, markersize=6,
             label="Future Reconstruction", color="green")

    plt.axvline(x=n_past + 0.5, color="gray", linestyle="--", alpha=0.5, label="Past/Future Boundary")

    plt.xlabel("Time Step", fontsize=12)
    plt.ylabel("MSE (mm/h)^2", fontsize=12)
    plt.title("Reconstruction MSE over Time Steps (Train Mode)", fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(steps)

    for i, mse in enumerate(mse_per_step):
        plt.annotate(f"{mse:.3f}", (steps[i], mse), textcoords="offset points",
                     xytext=(0, 10), ha="center", fontsize=7)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f">>> MSE 曲线已保存: {save_path}")


def evaluate_train_mode(model, loader, device, save_dir, n_past=10, n_future=10, num_samples=3, split_name="train"):
    """
    使用 train 模式评估重构效果
    """
    model.eval()
    split_save_dir = os.path.join(save_dir, f"{split_name}_train_mode")
    os.makedirs(split_save_dir, exist_ok=True)

    total_mse = 0
    total_kld = 0
    sample_count = 0
    mse_per_step = [[] for _ in range(n_past + n_future)]

    with torch.no_grad():
        for batch_idx, x in enumerate(tqdm(loader, desc=f"Evaluating {split_name} train mode")):
            x = x.to(device)

            # Train 模式：重构整个序列
            recon_preds, kld_loss = model(x, n_past, n_future, mode="train")

            # 提取原始降水序列（对数域 -> 原始值）
            original_log = x[:, :, 0, :, :]  # (B, T_total, H, W)
            original = inverse_log_transform(original_log)
            recon = inverse_log_transform(recon_preds.squeeze(2))  # (B, T_total, H, W)

            B, T_total, H, W = original.shape

            # 计算指标
            mse = torch.mean((recon - original) ** 2).item()
            total_mse += mse
            total_kld += kld_loss.item()

            for t in range(T_total):
                step_mse = torch.mean((recon[:, t] - original[:, t]) ** 2).item()
                mse_per_step[t].append(step_mse)

            # 可视化样本
            if batch_idx == 0 and sample_count < num_samples:
                for i in range(min(B, num_samples - sample_count)):
                    original_i = original[i].cpu().numpy()
                    recon_i = recon[i].cpu().numpy()

                    # 1. 重构对比图
                    visualize_train_reconstruction(
                        original_i, recon_i, n_past, n_future,
                        os.path.join(split_save_dir, f"sample_{sample_count}_recon.png"),
                        sample_idx=sample_count, kld=kld_loss
                    )

                    # 2. 误差分布图
                    visualize_recon_error(
                        original_i, recon_i, n_past, n_future,
                        os.path.join(split_save_dir, f"sample_{sample_count}_error.png"),
                        sample_idx=sample_count
                    )

                    # 3. 散点图
                    visualize_scatter(
                        original_i, recon_i,
                        os.path.join(split_save_dir, f"sample_{sample_count}_scatter.png"),
                        sample_idx=sample_count
                    )

                    # 打印每步 MSE
                    print(f"\n{split_name.upper()} Sample {sample_count}:")
                    for t in range(T_total):
                        prefix = "Past" if t < n_past else "Future"
                        step_mse = float(np.mean((original_i[t] - recon_i[t]) ** 2))
                        print(f"  {prefix} Step {t+1}: MSE = {step_mse:.6f} (mm/h)^2")

                    sample_count += 1

    n_batches = len(loader)
    avg_mse = total_mse / n_batches
    avg_kld = total_kld / n_batches
    avg_mse_per_step = [np.mean(mse_list) for mse_list in mse_per_step]

    print(f"\n{'='*60}")
    print(f"{split_name.upper()} Train Mode Results:")
    print(f"  Average Recon MSE: {avg_mse:.6f} (mm/h)^2")
    print(f"  Average KLD: {avg_kld:.6f}")
    print(f"\nMSE per step:")
    for t, mse in enumerate(avg_mse_per_step):
        prefix = "Past" if t < n_past else "Future"
        print(f"  {prefix} Step {t+1}: {mse:.6f} (mm/h)^2")
    print(f"{'='*60}")

    # 绘制 MSE 曲线
    plot_train_mse_curve(
        avg_mse_per_step,
        os.path.join(split_save_dir, "mse_curve.png"),
        n_past, n_future
    )

    return avg_mse, avg_kld, avg_mse_per_step


def main():
    parser = argparse.ArgumentParser(description="Evaluate DKF Train Mode on Weather Data")
    parser.add_argument("--checkpoint", type=str, required=True, help="模型检查点路径")
    parser.add_argument("--precip_data_path", type=str, default="../data/data_0511/nc_NZL/*.nc")
    parser.add_argument("--save_dir", type=str, default="./results", help="结果保存目录")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--n_past", type=int, default=10)
    parser.add_argument("--n_future", type=int, default=10)
    parser.add_argument("--num_samples", type=int, default=3, help="可视化样本数")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--eval_split", type=str, default="train", choices=["train", "val", "test"],
                        help="在哪个数据集上评估: train/val/test")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=args.batch_size,
        n_past=args.n_past,
        n_future=args.n_future,
        precip_data_path=args.precip_data_path,
        num_workers=args.num_workers,
    )

    # 选择评估的数据集
    if args.eval_split == "train":
        loader = train_loader
    elif args.eval_split == "val":
        loader = val_loader
    else:
        loader = test_loader

    print(f"评估数据集: {args.eval_split}, 样本数: {len(loader.dataset)}")

    model = DeepKalmanFilter().to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"加载模型: {args.checkpoint}, Epoch: {checkpoint.get('epoch', 'N/A')}")

    evaluate_train_mode(
        model, loader, device, args.save_dir,
        args.n_past, args.n_future, args.num_samples, args.eval_split
    )

    print(f"\n所有结果已保存到: {os.path.join(args.save_dir, f'{args.eval_split}_train_mode')}")


if __name__ == "__main__":
    main()
