import argparse
import os

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset import get_dataloaders
from modules import PrecipConvLSTM


def asymmetric_heavy_rain_loss(pred_log, target_log, alpha=2.0, penalty_factor=3.0):
    squared_error_log = (pred_log - target_log) ** 2
    target = torch.expm1(target_log)
    pred = torch.expm1(pred_log)

    base_weight_mask = 1.0 + alpha * target
    underestimate_mask = target > pred
    heavy_rain_underestimated = underestimate_mask & (target > 10.0)
    asymmetry_mask = torch.ones_like(target)
    asymmetry_mask[heavy_rain_underestimated] = penalty_factor

    return torch.mean(squared_error_log * base_weight_mask * asymmetry_mask)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_epoch(
    model,
    train_loader,
    criterion,
    optimizer,
    device,
    epoch,
    writer,
    n_future=10,
    grad_clip=5.0,
):
    model.train()
    total_loss = 0.0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    for x_past, y_future in pbar:
        x_past = x_past.to(device)
        y_future = y_future.to(device)

        optimizer.zero_grad()
        predictions = model(x_past, n_future)
        loss = criterion(predictions, y_future)
        loss.backward()

        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)

        optimizer.step()
        total_loss += loss.item()
        pbar.set_postfix({"loss": loss.item()})

    avg_loss = total_loss / len(train_loader)
    writer.add_scalar("Loss/train", avg_loss, epoch)
    return avg_loss


def evaluate(model, val_loader, criterion, device, epoch, writer, n_future=10):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for x_past, y_future in val_loader:
            x_past = x_past.to(device)
            y_future = y_future.to(device)

            predictions = model(x_past, n_future)
            loss = criterion(predictions, y_future)
            total_loss += loss.item()

    avg_loss = total_loss / len(val_loader)
    writer.add_scalar("Loss/val", avg_loss, epoch)
    return avg_loss


def load_resume_checkpoint(model, optimizer, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    if "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    start_epoch = checkpoint.get("epoch", 0) + 1
    best_loss = checkpoint.get("loss", float("inf"))
    return start_epoch, best_loss


def parse_args():
    parser = argparse.ArgumentParser(description="Train ConvLSTM on precipitation data")
    parser.add_argument(
        "--precip_data_path",
        type=str,
        default="../data/data_0511/nc_NZL/*.nc",
        help="降水数据路径",
    )
    parser.add_argument(
        "--land_data_path",
        type=str,
        default="../data/data_0511/gebco0_1_land_only.nc",
        help="地形数据路径",
    )
    parser.add_argument("--batch_size", type=int, default=32, help="批次大小")
    parser.add_argument("--epochs", type=int, default=100, help="训练轮数")
    parser.add_argument("--lr", type=float, default=1e-4, help="学习率")
    parser.add_argument("--n_past", type=int, default=10, help="输入帧数")
    parser.add_argument("--n_future", type=int, default=10, help="预测帧数")
    parser.add_argument(
        "--hidden_dim_list",
        type=int,
        nargs="+",
        default=[128, 64, 64],
        help="ConvLSTM 隐藏维度列表",
    )
    parser.add_argument("--patch_size", type=int, default=4, help="Patch 大小")
    parser.add_argument("--grad_clip", type=float, default=5.0, help="梯度裁剪阈值")
    parser.add_argument(
        "--save_dir", type=str, default="./checkpoints", help="模型保存目录"
    )
    parser.add_argument(
        "--log_dir", type=str, default="./logs", help="TensorBoard 日志目录"
    )
    parser.add_argument("--num_workers", type=int, default=4, help="数据加载线程数")
    parser.add_argument(
        "--resume_checkpoint",
        type=str,
        default=None,
        help="可选：从已有 checkpoint 继续训练",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    device = get_device()
    print(f"使用设备: {device}")

    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=args.batch_size,
        n_past=args.n_past,
        n_future=args.n_future,
        precip_data_path=args.precip_data_path,
        land_data_path=args.land_data_path,
        num_workers=args.num_workers,
    )
    print(
        f"训练样本数: {len(train_loader.dataset)}, "
        f"验证样本数: {len(val_loader.dataset)}, "
        f"测试样本数: {len(test_loader.dataset)}"
    )

    model = PrecipConvLSTM(
        hidden_dim_list=args.hidden_dim_list,
        patch_size=args.patch_size,
    ).to(device)
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")

    criterion = asymmetric_heavy_rain_loss
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    start_epoch = 1
    best_loss = float("inf")
    if args.resume_checkpoint:
        start_epoch, best_loss = load_resume_checkpoint(
            model, optimizer, args.resume_checkpoint, device
        )
        print(
            f"从 checkpoint 继续训练: {args.resume_checkpoint}, "
            f"start_epoch={start_epoch}, best_loss={best_loss:.6f}"
        )

    writer = SummaryWriter(log_dir=os.path.join(args.save_dir, "logs"))

    for epoch in range(start_epoch, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        train_loss = train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            epoch,
            writer,
            args.n_future,
            args.grad_clip,
        )
        val_loss = evaluate(
            model,
            val_loader,
            criterion,
            device,
            epoch,
            writer,
            args.n_future,
        )

        print(f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": val_loss,
                    "hidden_dim_list": args.hidden_dim_list,
                    "patch_size": args.patch_size,
                    "n_past": args.n_past,
                    "n_future": args.n_future,
                },
                os.path.join(args.save_dir, "best_model.pth"),
            )
            print(f"保存最佳模型，损失: {best_loss:.6f}")

        if epoch == args.epochs:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": train_loss,
                    "hidden_dim_list": args.hidden_dim_list,
                    "patch_size": args.patch_size,
                    "n_past": args.n_past,
                    "n_future": args.n_future,
                },
                os.path.join(args.save_dir, "last_model.pth"),
            )
            print(f"保存最终模型，损失: {train_loss:.6f}")

    writer.close()
    print("\nTraining completed!")


if __name__ == "__main__":
    main()
