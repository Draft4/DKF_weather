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

from modules import EncodingForecastingConvLSTM
from dataset import get_dataloaders, unpatchify


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


def patch_to_image(x, patch_size=4, H=64, W=64):
    """
    将 patch 数据转换为可显示的图像
    输入: (seq_len, patch_dim, h_patches, w_patches) 或 (seq_len, 1, H, W)
    输出: (seq_len, H, W) numpy 数组，值域 [0, 1]
    """
    if isinstance(x, torch.Tensor):
        x = x.cpu().numpy()

    # 如果已经是图像格式 (seq_len, 1, H, W)
    if x.shape[1] == 1 and x.shape[2] == H and x.shape[3] == W:
        return x.squeeze(1)

    # 如果是 patch 格式，先 unpatchify
    x = unpatchify(x, patch_size=patch_size, H=H, W=W)
    if isinstance(x, torch.Tensor):
        x = x.cpu().numpy()
    return x.squeeze(1)


def evaluate_model(
    model, test_loader, criterion, device, save_dir, num_samples=5, patch_size=4
):
    model.eval()
    os.makedirs(save_dir, exist_ok=True)

    total_loss = 0
    total_mse = 0
    sample_count = 0

    mse_per_step = [[] for _ in range(100)]

    with torch.no_grad():
        for batch_idx, (x_past, x_future) in enumerate(
            tqdm(test_loader, desc="Evaluating")
        ):
            x_past = x_past.to(device)
            x_future = x_future.to(device)

            # 前向传播 - 预测未来帧
            pred_future = model(x_past, x_future.shape[1])

            # 应用 sigmoid 将 logits 转换为概率 [0, 1]
            pred_future_prob = torch.sigmoid(pred_future)

            # 计算损失 (在 patch 空间，使用概率值)
            loss = criterion(pred_future_prob, x_future)
            total_loss += loss.item()

            # 计算 MSE (在 patch 空间，使用概率值)
            mse = torch.mean((pred_future_prob - x_future) ** 2).item()
            total_mse += mse

            B, T_future_actual, _, H, W = x_future.shape

            for t in range(T_future_actual):
                step_mse = torch.mean(
                    (pred_future_prob[:, t] - x_future[:, t]) ** 2
                ).item()
                mse_per_step[t].append(step_mse)

            # 保存可视化样本
            if batch_idx == 0 and sample_count < num_samples:
                for i in range(min(B, num_samples - sample_count)):
                    # 转换为图像空间进行可视化 (使用 sigmoid 后的概率值)
                    past = patch_to_image(x_past[i], patch_size=patch_size)
                    true = patch_to_image(x_future[i], patch_size=patch_size)
                    pred = patch_to_image(pred_future_prob[i], patch_size=patch_size)

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

                    # 打印每步 MSE (在图像空间)
                    print(f"\nSample {sample_count}:")
                    for t in range(T_future):
                        step_mse = float(np.mean((true[t] - pred[t]) ** 2))
                        print(f"  Step {t+1}: MSE = {step_mse:.6f}")

                    sample_count += 1

    n_batches = len(test_loader)
    avg_loss = total_loss / n_batches
    avg_mse = total_mse / n_batches

    avg_mse_per_step = [
        np.mean(mse_list) if mse_list else 0
        for mse_list in mse_per_step[:T_future_actual]
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
    parser = argparse.ArgumentParser(description="Evaluate ConvLSTM on Moving MNIST")
    parser.add_argument(
        "--data_path",
        type=str,
        default="../data/MovingMNIST/mnist_test_seq.npy",
        help="Moving MNIST 数据路径",
    )
    parser.add_argument("--checkpoint", type=str, required=True, help="模型检查点路径")
    parser.add_argument("--batch_size", type=int, default=32, help="批次大小")
    parser.add_argument(
        "--hidden_dim_list",
        type=int,
        nargs="+",
        default=[128, 64, 64],
        help="ConvLSTM 隐藏维度列表",
    )
    parser.add_argument("--T_past", type=int, default=10, help="输入帧数")
    parser.add_argument("--T_future", type=int, default=10, help="预测帧数")
    parser.add_argument(
        "--save_dir", type=str, default="./results", help="结果保存目录"
    )
    parser.add_argument("--num_samples", type=int, default=5, help="可视化样本数")
    parser.add_argument("--num_workers", type=int, default=4, help="数据加载线程数")
    parser.add_argument("--patch_size", type=int, default=4, help="Patch 大小")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 数据加载器
    _, test_loader = get_dataloaders(
        data_path=args.data_path,
        batch_size=args.batch_size,
        T_past=args.T_past,
        T_future=args.T_future,
        num_workers=args.num_workers,
        train_split=0.8,
    )

    # 模型
    patch_dim = 1 * args.patch_size * args.patch_size
    model = EncodingForecastingConvLSTM(patch_dim, args.hidden_dim_list, 5, 2).to(
        device
    )

    # 加载检查点
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"加载模型: {args.checkpoint}, Epoch: {checkpoint.get('epoch', 'N/A')}")

    # 损失函数 - 使用 MSELoss 而不是 BCEWithLogitsLoss，因为我们在 patch 空间计算
    criterion = nn.MSELoss()

    # 评估
    evaluate_model(
        model,
        test_loader,
        criterion,
        device,
        args.save_dir,
        args.num_samples,
        args.patch_size,
    )

    print(f"\n结果已保存到: {args.save_dir}")


if __name__ == "__main__":
    main()
