import os
import argparse
import torch
import torch.nn as nn
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from modules_dkf import DeepKalmanFilter
from dataset import get_dataloaders


def add_label_to_image(img_array, label, font_size=14):
    """在图像顶部添加标签"""
    if isinstance(img_array, torch.Tensor):
        img_array = img_array.cpu().numpy()
    img = Image.fromarray((img_array * 255).astype(np.uint8))
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", font_size)
    except:
        try:
            font = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size
            )
        except:
            font = ImageFont.load_default()

    bbox = draw.textbbox((0, 0), label, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    draw.rectangle([0, 0, text_width + 4, text_height + 4], fill=128)
    draw.text((2, 0), label, fill=255, font=font)

    return np.array(img) / 255.0


def create_comparison_grid(past_frames, true_future, pred_future, T_past, T_future):
    """
    创建对比网格：
    第一行：历史输入帧（灰色标签）
    第二行：真实未来帧（绿色标签）
    第三行：预测未来帧（蓝色标签）
    """
    labeled_frames = []

    for t in range(T_past):
        label = f"Past {t+1}"
        labeled = add_label_to_image(past_frames[t], label, font_size=12)
        labeled_frames.append(labeled)

    for t in range(T_future):
        label = f"True {t+1}"
        labeled = add_label_to_image(true_future[t], label, font_size=12)
        labeled_frames.append(labeled)

    for t in range(T_future):
        label = f"Pred {t+1}"
        labeled = add_label_to_image(pred_future[t], label, font_size=12)
        labeled_frames.append(labeled)

    n_cols = max(T_past, T_future)
    frame_h, frame_w = labeled_frames[0].shape[:2]
    grid = np.ones((3 * frame_h, n_cols * frame_w, 3))

    for i, frame in enumerate(labeled_frames):
        row = i // n_cols
        col = i % n_cols
        if frame.ndim == 2:
            frame_rgb = np.stack([frame] * 3, axis=-1)
        else:
            frame_rgb = frame
        grid[
            row * frame_h : (row + 1) * frame_h, col * frame_w : (col + 1) * frame_w
        ] = frame_rgb

    return grid


def create_true_vs_pred(true_future, pred_future, T_future):
    """创建真实 vs 预测对比图"""
    labeled_frames = []

    for t in range(T_future):
        label = f"True {t+1}"
        labeled = add_label_to_image(true_future[t], label, font_size=14)
        labeled_frames.append(labeled)

        label = f"Pred {t+1}"
        labeled = add_label_to_image(pred_future[t], label, font_size=14)
        labeled_frames.append(labeled)

    frame_h, frame_w = labeled_frames[0].shape[:2]
    grid = np.ones((T_future * frame_h, 2 * frame_w, 3))

    for i, frame in enumerate(labeled_frames):
        row = i // 2
        col = i % 2
        if frame.ndim == 2:
            frame_rgb = np.stack([frame] * 3, axis=-1)
        else:
            frame_rgb = frame
        grid[
            row * frame_h : (row + 1) * frame_h, col * frame_w : (col + 1) * frame_w
        ] = frame_rgb

    return grid


def tensor_to_image(x):
    """
    将 tensor 转换为可显示的图像 numpy 数组
    输入: (T, 1, H, W) tensor
    输出: (T, H, W) numpy 数组，值域 [0, 1]
    """
    if isinstance(x, torch.Tensor):
        x = x.cpu().numpy()
    return x.squeeze(1)


def visualize_train_mode(
    model,
    loader,
    device,
    save_dir,
    num_samples=5,
    n_past=10,
    n_future=10,
    split_name="train",
):
    """
    使用 train 模式可视化重构效果
    train 模式会对整个序列 (past + future) 进行重构
    """
    model.eval()
    train_save_dir = os.path.join(save_dir, f"{split_name}_mode")
    os.makedirs(train_save_dir, exist_ok=True)

    sample_count = 0

    with torch.no_grad():
        for batch_idx, x in enumerate(
            tqdm(loader, desc=f"Visualizing {split_name} mode")
        ):
            x = x.to(device)

            # DKF train 模式：返回整个序列的重构
            recon_preds, kld_loss = model(x, n_past, n_future, mode="train")

            B, T_total, _, H, W = recon_preds.shape

            # 保存可视化样本
            if batch_idx == 0 and sample_count < num_samples:
                for i in range(min(B, num_samples - sample_count)):
                    # 原始序列
                    original = tensor_to_image(x[i])
                    # 重构序列
                    recon = tensor_to_image(recon_preds[i])

                    # 创建对比图：原始 vs 重构
                    labeled_frames = []

                    for t in range(T_total):
                        label = f"Orig {t+1}"
                        labeled = add_label_to_image(original[t], label, font_size=10)
                        labeled_frames.append(labeled)

                    for t in range(T_total):
                        label = f"Recon {t+1}"
                        labeled = add_label_to_image(recon[t], label, font_size=10)
                        labeled_frames.append(labeled)

                    n_cols = T_total
                    frame_h, frame_w = labeled_frames[0].shape[:2]
                    grid = np.ones((2 * frame_h, n_cols * frame_w, 3))

                    for idx, frame in enumerate(labeled_frames):
                        row = idx // n_cols
                        col = idx % n_cols
                        if frame.ndim == 2:
                            frame_rgb = np.stack([frame] * 3, axis=-1)
                        else:
                            frame_rgb = frame
                        grid[
                            row * frame_h : (row + 1) * frame_h,
                            col * frame_w : (col + 1) * frame_w,
                        ] = frame_rgb

                    plt.figure(figsize=(24, 6))
                    plt.imshow(grid)
                    plt.axis("off")
                    plt.title(
                        f"Sample {sample_count} - {split_name.upper()} Mode Reconstruction\n"
                        f"Row 1: Original | Row 2: Reconstructed | "
                        f"Past: 1-{n_past} | Future: {n_past+1}-{T_total} | "
                        f"KLD: {kld_loss.item():.4f}",
                        fontsize=12,
                    )
                    plt.tight_layout()
                    plt.savefig(
                        os.path.join(
                            train_save_dir,
                            f"sample_{sample_count}_reconstruction.png",
                        ),
                        dpi=150,
                        bbox_inches="tight",
                    )
                    plt.close()

                    # 计算并打印每帧 MSE
                    print(f"\n{split_name.upper()} Mode Sample {sample_count}:")
                    for t in range(T_total):
                        step_mse = float(np.mean((original[t] - recon[t]) ** 2))
                        prefix = "Past" if t < n_past else "Future"
                        print(f"  {prefix} Step {t+1}: MSE = {step_mse:.6f}")

                    sample_count += 1

            if sample_count >= num_samples:
                break

    print(f"\n{split_name.upper()} mode visualizations saved to: {train_save_dir}")


def evaluate_model(
    model,
    test_loader,
    criterion,
    device,
    save_dir,
    num_samples=5,
    n_past=10,
    n_future=10,
):
    model.eval()
    os.makedirs(save_dir, exist_ok=True)

    total_loss = 0
    total_mse = 0
    sample_count = 0

    mse_per_step = [[] for _ in range(n_future)]

    with torch.no_grad():
        for batch_idx, x in enumerate(tqdm(test_loader, desc="Evaluating")):
            x = x.to(device)

            # DKF 预测模式：输入完整序列，返回未来预测
            pred_future = model(x, n_past, n_future, mode="predict")

            # 提取真实未来帧
            true_future = x[:, n_past:, :, :, :]

            # 计算损失 (DKF 输出已是 [0,1] 概率，直接使用 MSE)
            loss = criterion(pred_future, true_future)
            total_loss += loss.item()

            # 计算 MSE
            mse = torch.mean((pred_future - true_future) ** 2).item()
            total_mse += mse

            B, T_future_actual, _, H, W = true_future.shape

            for t in range(T_future_actual):
                step_mse = torch.mean(
                    (pred_future[:, t] - true_future[:, t]) ** 2
                ).item()
                mse_per_step[t].append(step_mse)

            # 保存可视化样本
            if batch_idx == 0 and sample_count < num_samples:
                for i in range(min(B, num_samples - sample_count)):
                    # 提取 past, true, pred 并转换为图像
                    past = tensor_to_image(x[i, :n_past])
                    true = tensor_to_image(true_future[i])
                    pred = tensor_to_image(pred_future[i])

                    T_past = past.shape[0]
                    T_future = true.shape[0]

                    # 创建对比图
                    grid = create_comparison_grid(past, true, pred, T_past, T_future)

                    plt.figure(figsize=(20, 8))
                    plt.imshow(grid)
                    plt.axis("off")
                    plt.title(
                        f"Sample {sample_count} - Comparison Grid\n"
                        f"Row 1: Input Past | Row 2: True Future | Row 3: Predicted Future",
                        fontsize=12,
                    )
                    plt.tight_layout()
                    plt.savefig(
                        os.path.join(save_dir, f"sample_{sample_count}_comparison.png"),
                        dpi=150,
                        bbox_inches="tight",
                    )
                    plt.close()

                    # 创建 True vs Pred 对比图
                    grid2 = create_true_vs_pred(true, pred, T_future)

                    plt.figure(figsize=(8, 20))
                    plt.imshow(grid2)
                    plt.axis("off")
                    plt.title(
                        f"Sample {sample_count} - True vs Predicted Future", fontsize=12
                    )
                    plt.tight_layout()
                    plt.savefig(
                        os.path.join(
                            save_dir, f"sample_{sample_count}_true_vs_pred.png"
                        ),
                        dpi=150,
                        bbox_inches="tight",
                    )
                    plt.close()

                    # 打印每步 MSE
                    print(f"\nSample {sample_count}:")
                    for t in range(T_future):
                        step_mse = float(np.mean((true[t] - pred[t]) ** 2))
                        print(f"  Step {t+1}: MSE = {step_mse:.6f}")

                    sample_count += 1

    n_batches = len(test_loader)
    avg_loss = total_loss / n_batches
    avg_mse = total_mse / n_batches

    avg_mse_per_step = [
        np.mean(mse_list) if mse_list else 0 for mse_list in mse_per_step
    ]

    print(f"\n{'='*50}")
    print(f"Evaluation Results:")
    print(f"  Average Loss: {avg_loss:.6f}")
    print(f"  Average MSE:  {avg_mse:.6f}")
    print(f"\nMSE per step:")
    for t, mse in enumerate(avg_mse_per_step):
        print(f"  Step {t+1}: {mse:.6f}")
    print(f"{'='*50}")

    # 绘制 MSE 曲线
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(avg_mse_per_step) + 1), avg_mse_per_step, marker="o")
    plt.xlabel("Prediction Step")
    plt.ylabel("MSE")
    plt.title("MSE over Prediction Steps")
    plt.grid(True)
    plt.savefig(
        os.path.join(save_dir, "mse_over_steps.png"), dpi=150, bbox_inches="tight"
    )
    plt.close()

    return avg_loss, avg_mse, avg_mse_per_step


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Deep Kalman Filter on Moving MNIST"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="../data/MovingMNIST/mnist_test_seq.npy",
        help="Moving MNIST 数据路径",
    )
    parser.add_argument("--checkpoint", type=str, required=True, help="模型检查点路径")
    parser.add_argument("--batch_size", type=int, default=32, help="批次大小")
    parser.add_argument("--n_past", type=int, default=10, help="输入帧数")
    parser.add_argument("--n_future", type=int, default=10, help="预测帧数")
    parser.add_argument(
        "--save_dir", type=str, default="./results", help="结果保存目录"
    )
    parser.add_argument("--num_samples", type=int, default=5, help="可视化样本数")
    parser.add_argument("--num_workers", type=int, default=4, help="数据加载线程数")
    parser.add_argument(
        "--visualize_train",
        action="store_true",
        help="是否同时可视化 train 模式下的重构效果",
    )
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 数据加载器 - DKF 使用 get_dataloaders 返回完整序列
    train_loader, test_loader = get_dataloaders(
        data_path=args.data_path,
        batch_size=args.batch_size,
        n_past=args.n_past,
        n_future=args.n_future,
        num_workers=args.num_workers,
        train_split=0.8,
    )

    # 模型
    model = DeepKalmanFilter().to(device)

    # 加载检查点
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"加载模型: {args.checkpoint}, Epoch: {checkpoint.get('epoch', 'N/A')}")

    # 损失函数 - DKF 使用 MSELoss
    criterion = nn.MSELoss()

    # 评估 predict 模式
    evaluate_model(
        model,
        test_loader,
        criterion,
        device,
        args.save_dir,
        args.num_samples,
        args.n_past,
        args.n_future,
    )

    # 如果指定了 --visualize_train，同时可视化 train 模式
    if args.visualize_train:
        print("\n" + "=" * 50)
        print("Visualizing TRAIN mode (reconstruction)")
        print("=" * 50)

        # 在训练集上可视化 train 模式
        visualize_train_mode(
            model,
            train_loader,
            device,
            args.save_dir,
            args.num_samples,
            args.n_past,
            args.n_future,
            split_name="train",
        )

        # 在测试集上也可视化 train 模式，方便对比
        visualize_train_mode(
            model,
            test_loader,
            device,
            args.save_dir,
            args.num_samples,
            args.n_past,
            args.n_future,
            split_name="test",
        )

    print(f"\n结果已保存到: {args.save_dir}")


if __name__ == "__main__":
    main()
