import os
import argparse
import torch
import torch.nn as nn
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

from model import CompositeModel
from dataset import get_dataloaders


def add_label_to_image(img_array, label, font_size=14):
    """在图像顶部添加标签"""
    img = Image.fromarray((img_array * 255).astype(np.uint8))
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", font_size)
    except:
        font = ImageFont.load_default()

    # 获取标签尺寸
    bbox = draw.textbbox((0, 0), label, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    # 绘制背景
    draw.rectangle([0, 0, text_width + 4, text_height + 4], fill=128)
    draw.text((2, 0), label, fill=255, font=font)

    return np.array(img) / 255.0


def create_comparison_grid(past_frames, true_future, recon_past, pred_future, T_past, T_future):
    """
    创建对比网格：
    第一行：历史输入帧（灰色标签）
    第二行：逆序重构帧（橙色标签）
    第三行：真实未来帧（绿色标签）
    第四行：预测未来帧（蓝色标签）
    """
    labeled_frames = []

    # 历史帧（灰色标签）
    for t in range(T_past):
        label = f"Past {t+1}"
        labeled = add_label_to_image(past_frames[t], label, font_size=12)
        labeled_frames.append(labeled)

    # 重构帧（橙色标签）- 注意是逆序重构
    for t in range(T_past):
        label = f"Recon {T_past-t}"
        labeled = add_label_to_image(recon_past[t], label, font_size=12)
        labeled_frames.append(labeled)

    # 真实未来帧（绿色标签）
    for t in range(T_future):
        label = f"True {t+1}"
        labeled = add_label_to_image(true_future[t], label, font_size=12)
        labeled_frames.append(labeled)

    # 预测未来帧（蓝色标签）
    for t in range(T_future):
        label = f"Pred {t+1}"
        labeled = add_label_to_image(pred_future[t], label, font_size=12)
        labeled_frames.append(labeled)

    # 创建网格 (4 行, max(T_past, T_future) 列)
    n_cols = max(T_past, T_future)
    frame_h, frame_w = labeled_frames[0].shape[:2]

    grid = np.ones((4 * frame_h, n_cols * frame_w, 3))

    for i, frame in enumerate(labeled_frames):
        row = i // n_cols
        col = i % n_cols
        if row < 4 and col < n_cols:
            # 转换为 RGB
            if frame.ndim == 2:
                frame_rgb = np.stack([frame]*3, axis=-1)
            else:
                frame_rgb = frame
            grid[row*frame_h:(row+1)*frame_h, col*frame_w:(col+1)*frame_w] = frame_rgb

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
            frame_rgb = np.stack([frame]*3, axis=-1)
        else:
            frame_rgb = frame
        grid[row*frame_h:(row+1)*frame_h, col*frame_w:(col+1)*frame_w] = frame_rgb

    return grid


def evaluate_model(model, test_loader, criterion, device, save_dir, num_samples=5):
    model.eval()

    os.makedirs(save_dir, exist_ok=True)

    total_loss = 0
    total_recon_loss = 0
    total_pred_loss = 0
    total_mse = 0
    sample_count = 0

    with torch.no_grad():
        for batch_idx, (x_past, x_future) in enumerate(tqdm(test_loader, desc="Evaluating")):
            x_past = x_past.to(device)
            x_future = x_future.to(device)

            # 前向传播 - 使用 predict 模式进行真实预测
            recon_input, pred_future = model(x_past, x_future, mode='predict')

            # 计算损失
            x_past_reversed = torch.flip(x_past, dims=[1])
            recon_loss = criterion(recon_input, x_past_reversed)
            pred_loss = criterion(pred_future, x_future)
            loss = recon_loss + pred_loss

            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_pred_loss += pred_loss.item()

            # 计算 MSE
            mse = torch.mean((pred_future - x_future) ** 2).item()
            total_mse += mse

            # 保存可视化样本
            if batch_idx == 0 and sample_count < num_samples:
                B = x_past.size(0)
                for i in range(min(B, num_samples - sample_count)):
                    past = x_past[i].cpu().numpy().squeeze()  # (T_past, 64, 64)
                    true = x_future[i].cpu().numpy().squeeze()  # (T_future, 64, 64)
                    recon = recon_input[i].cpu().numpy().squeeze()  # (T_past, 64, 64)
                    pred = pred_future[i].cpu().numpy().squeeze()  # (T_future, 64, 64)

                    # 创建对比图
                    grid = create_comparison_grid(
                        past, true, recon, pred,
                        T_past=x_past.size(1),
                        T_future=x_future.size(1)
                    )

                    import matplotlib
                    matplotlib.use('Agg')
                    import matplotlib.pyplot as plt

                    plt.figure(figsize=(20, 12))
                    plt.imshow(grid)
                    plt.axis('off')
                    plt.title(f'Sample {sample_count} - Comparison Grid\n'
                              f'Row 1: Input | Row 2: Reversed Recon | '
                              f'Row 3: True Future | Row 4: Predicted Future',
                              fontsize=12)
                    plt.tight_layout()
                    plt.savefig(os.path.join(save_dir, f'sample_{sample_count}_comparison.png'),
                                dpi=150, bbox_inches='tight')
                    plt.close()

                    # 创建 True vs Pred 对比图
                    grid2 = create_true_vs_pred(true, pred, T_future=x_future.size(1))

                    plt.figure(figsize=(8, 20))
                    plt.imshow(grid2)
                    plt.axis('off')
                    plt.title(f'Sample {sample_count} - True vs Predicted Future', fontsize=12)
                    plt.tight_layout()
                    plt.savefig(os.path.join(save_dir, f'sample_{sample_count}_true_vs_pred.png'),
                                dpi=150, bbox_inches='tight')
                    plt.close()

                    # 打印每步 MSE
                    print(f"\nSample {sample_count}:")
                    for t in range(x_future.size(1)):
                        step_mse = np.mean((true[t] - pred[t]) ** 2)
                        print(f"  Step {t+1}: MSE = {step_mse:.6f}")

                    sample_count += 1

    n_batches = len(test_loader)
    avg_loss = total_loss / n_batches
    avg_recon = total_recon_loss / n_batches
    avg_pred = total_pred_loss / n_batches
    avg_mse = total_mse / n_batches

    print(f"\n{'='*50}")
    print(f"Evaluation Results:")
    print(f"  Total Loss: {avg_loss:.6f}")
    print(f"  Recon Loss: {avg_recon:.6f}")
    print(f"  Pred Loss:  {avg_pred:.6f}")
    print(f"  Pred MSE:   {avg_mse:.6f}")
    print(f"{'='*50}")

    return avg_loss, avg_recon, avg_pred, avg_mse


def main():
    parser = argparse.ArgumentParser(description='Evaluate Composite LSTM Model')
    parser.add_argument('--data_path', type=str, default='../data/MovingMNIST/mnist_test_seq.npy',
                        help='Moving MNIST 数据路径')
    parser.add_argument('--checkpoint', type=str, required=True, help='模型检查点路径')
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('--hidden_size', type=int, default=2048, help='LSTM 隐藏层大小')
    parser.add_argument('--num_layers', type=int, default=1, help='LSTM 层数')
    parser.add_argument('--T_past', type=int, default=10, help='输入帧数')
    parser.add_argument('--T_future', type=int, default=10, help='预测帧数')
    parser.add_argument('--save_dir', type=str, default='./results', help='结果保存目录')
    parser.add_argument('--num_samples', type=int, default=5, help='可视化样本数')
    parser.add_argument('--num_workers', type=int, default=4, help='数据加载线程数')
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 数据加载器
    _, test_loader = get_dataloaders(
        data_path=args.data_path,
        batch_size=args.batch_size,
        T_past=args.T_past,
        T_future=args.T_future,
        num_workers=args.num_workers,
        train_split=0.8
    )

    # 模型
    model = CompositeModel(
        input_size=64*64,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers
    ).to(device)

    # 加载检查点
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"加载模型: {args.checkpoint}, Epoch: {checkpoint.get('epoch', 'N/A')}")

    # 损失函数
    criterion = nn.MSELoss()

    # 评估
    evaluate_model(model, test_loader, criterion, device, args.save_dir, args.num_samples)

    print(f"\n结果已保存到: {args.save_dir}")


if __name__ == '__main__':
    main()
