import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from model import CompositeModel
from dataset import get_dataloaders


def train_epoch(model, train_loader, criterion, optimizer, device, epoch, writer):
    model.train()
    total_loss = 0
    total_recon_loss = 0
    total_pred_loss = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    for batch_idx, (x_past, x_future) in enumerate(pbar):
        x_past = x_past.to(device)
        x_future = x_future.to(device)

        optimizer.zero_grad()

        # 前向传播
        recon_input, pred_future = model(x_past, x_future, mode="train")

        # 计算损失
        # 重构损失: 逆序重构 vs 逆序输入
        x_past_reversed = torch.flip(x_past, dims=[1])
        recon_loss = criterion(recon_input, x_past_reversed)

        # 预测损失
        pred_loss = criterion(pred_future, x_future)

        # 总损失 (1:1 权重)
        loss = recon_loss + pred_loss

        # 反向传播
        loss.backward()
        # 梯度裁剪，与源码一致 (gradient_clip: 0.0001)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.0001)
        optimizer.step()

        total_loss += loss.item()
        total_recon_loss += recon_loss.item()
        total_pred_loss += pred_loss.item()

        # 更新进度条
        pbar.set_postfix(
            {"loss": loss.item(), "recon": recon_loss.item(), "pred": pred_loss.item()}
        )

    # 计算平均损失
    n_batches = len(train_loader)
    avg_loss = total_loss / n_batches
    avg_recon = total_recon_loss / n_batches
    avg_pred = total_pred_loss / n_batches

    # 记录到 tensorboard
    writer.add_scalar("Loss/train", avg_loss, epoch)
    writer.add_scalar("Loss/recon", avg_recon, epoch)
    writer.add_scalar("Loss/pred", avg_pred, epoch)

    return avg_loss, avg_recon, avg_pred


def evaluate(model, test_loader, criterion, device, epoch, writer):
    model.eval()
    total_loss = 0
    total_recon_loss = 0
    total_pred_loss = 0

    with torch.no_grad():
        for x_past, x_future in test_loader:
            x_past = x_past.to(device)
            x_future = x_future.to(device)

            # 前向传播
            recon_input, pred_future = model(x_past, x_future, mode="train")

            # 计算损失
            x_past_reversed = torch.flip(x_past, dims=[1])
            recon_loss = criterion(recon_input, x_past_reversed)
            pred_loss = criterion(pred_future, x_future)
            loss = recon_loss + pred_loss

            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_pred_loss += pred_loss.item()

    n_batches = len(test_loader)
    avg_loss = total_loss / n_batches
    avg_recon = total_recon_loss / n_batches
    avg_pred = total_pred_loss / n_batches

    # 记录到 tensorboard
    writer.add_scalar("Loss/test", avg_loss, epoch)
    writer.add_scalar("Loss/test_recon", avg_recon, epoch)
    writer.add_scalar("Loss/test_pred", avg_pred, epoch)

    return avg_loss, avg_recon, avg_pred


def main():
    parser = argparse.ArgumentParser(
        description="Train Composite LSTM Model on Moving MNIST"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="../data/MovingMNIST/mnist_test_seq.npy",
        help="Moving MNIST 数据路径",
    )
    parser.add_argument("--batch_size", type=int, default=32, help="批次大小")
    parser.add_argument("--epochs", type=int, default=100, help="训练轮数")
    parser.add_argument("--lr", type=float, default=1e-4, help="学习率")
    parser.add_argument("--hidden_size", type=int, default=2048, help="LSTM 隐藏层大小")
    parser.add_argument("--num_layers", type=int, default=1, help="LSTM 层数")
    parser.add_argument("--T_past", type=int, default=10, help="输入帧数")
    parser.add_argument("--T_future", type=int, default=10, help="预测帧数")
    parser.add_argument(
        "--save_dir", type=str, default="./checkpoints", help="模型保存目录"
    )
    parser.add_argument(
        "--log_dir", type=str, default="./logs", help="TensorBoard 日志目录"
    )
    parser.add_argument("--num_workers", type=int, default=4, help="数据加载线程数")
    args = parser.parse_args()

    # 创建目录
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    # 设备
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"使用设备: {device}")

    # 数据加载器
    train_loader, test_loader = get_dataloaders(
        data_path=args.data_path,
        batch_size=args.batch_size,
        T_past=args.T_past,
        T_future=args.T_future,
        num_workers=args.num_workers,
    )
    print(
        f"训练样本数: {len(train_loader.dataset)}, 测试样本数: {len(test_loader.dataset)}"
    )

    # 模型
    model = CompositeModel(
        input_size=64 * 64, hidden_size=args.hidden_size, num_layers=args.num_layers
    ).to(device)

    print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")

    # 损失函数 - 使用 BCELoss 与源码一致 (binary_data: true)
    criterion = nn.BCELoss()

    # 优化器 - 使用 SGD + momentum 与源码一致
    # 源码参数: epsilon=0.0001, momentum=0.9, l2_decay=0.0001
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=0.9,
        weight_decay=0.0001,
    )

    # 学习率衰减 - 与源码一致 (eps_decay_factor: 0.9, eps_decay_after: 10000)
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=10, gamma=0.9
    )

    # TensorBoard
    writer = SummaryWriter(args.log_dir)

    # 训练循环
    best_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        train_loss, train_recon, train_pred = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch, writer
        )

        test_loss, test_recon, test_pred = evaluate(
            model, test_loader, criterion, device, epoch, writer
        )

        # 学习率衰减
        scheduler.step()

        print(
            f"Epoch {epoch}: Train Loss={train_loss:.6f} (Recon={train_recon:.6f}, Pred={train_pred:.6f})"
        )
        print(
            f"           Test Loss={test_loss:.6f} (Recon={test_recon:.6f}, Pred={test_pred:.6f})"
        )
        print(f"           LR={optimizer.param_groups[0]['lr']:.6f}")

        # 保存最佳模型
        if test_loss < best_loss:
            best_loss = test_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": test_loss,
                },
                os.path.join(args.save_dir, "best_model.pth"),
            )
            print(f"保存最佳模型，损失: {best_loss:.6f}")

        # 每 10 个 epoch 保存检查点
        if epoch % 10 == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": test_loss,
                },
                os.path.join(args.save_dir, f"checkpoint_epoch_{epoch}.pth"),
            )

    writer.close()
    print("训练完成!")


if __name__ == "__main__":
    main()
