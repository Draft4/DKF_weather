import os
import argparse
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from modules import DeepKalmanFilter
from dataset import get_dataloaders


def weighted_mse_loss(
    pred, target, thresholds=[2.0, 5.0, 10.0, 30.0], weights=[1, 2, 5, 10, 30]
):
    squared_errpr = (pred - target) ** 2
    weight_mask = torch.ones_like(target)

    for i in range(len(thresholds)):
        if i == 0:
            mask = target < thresholds[0]
            weight_mask[mask] = weights[0]
        else:
            mask = (target >= thresholds[i - 1]) & (target < thresholds[i])
            weight_mask[mask] = weights[i]
    mask = target >= thresholds[-1]
    weight_mask[mask] = weights[-1]

    weighted_loss = torch.mean(squared_errpr * weight_mask)
    return weighted_loss


def meteorology_weighted_mse(
    pred_log,
    target_log,
    thresholds=[2.0, 5.0, 10.0, 30.0, 40.0, 50.0],
    weights=[1, 2, 5, 10, 30, 40, 50],
):
    squared_error_log = (pred_log - target_log) ** 2

    target = torch.expm1(target_log)

    weight_mask = torch.ones_like(target)

    for i in range(len(thresholds)):
        if i == 0:
            mask = target < thresholds[0]
            weight_mask[mask] = weights[0]
        else:
            mask = (target >= thresholds[i - 1]) & (target < thresholds[i])
            weight_mask[mask] = weights[i]
    mask = target >= thresholds[-1]
    weight_mask[mask] = weights[-1]

    weighted_loss = torch.mean(squared_error_log * weight_mask)
    return weighted_loss


def asymmetric_heavy_rain_loss(pred_log, target_log, alpha=2.0, penalty_factor=3.0):
    squared_error_log = (pred_log - target_log) ** 2
    target = torch.expm1(target_log)
    pred = torch.expm1(pred_log)

    base_weigt_mask = 1.0 + alpha * target
    underestimate_mask = target > pred
    heavy_rain_underestimated = underestimate_mask & (target > 10.0)
    asymmetry_mask = torch.ones_like(target)
    asymmetry_mask[heavy_rain_underestimated] = penalty_factor

    final_loss = torch.mean(squared_error_log * base_weigt_mask * asymmetry_mask)
    return final_loss


def get_beta_kld(current_epoch, warmup_epochs=30, max_beta=0.06):
    # if current_epoch < warmup_epochs:
    #     return max_beta * (current_epoch / warmup_epochs)
    return max_beta


def train_epoch(
    model,
    train_loader,
    criterion,
    optimizer,
    device,
    epoch,
    writer,
    n_past=10,
    n_future=10,
):
    model.train()
    total_loss = 0
    total_recon_pred = 0.0
    total_kld = 0.0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    for batch_idx, x in enumerate(pbar):
        x = x.to(device)
        precip_x = x[:, :, 0, :, :].unsqueeze(2)

        optimizer.zero_grad()

        recon_preds, kld_loss = model(x, n_past, n_future, "train")
        recon_pred_loss = criterion(recon_preds, precip_x)
        loss = recon_pred_loss + get_beta_kld(epoch) * kld_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss += loss.item()
        total_recon_pred += recon_pred_loss.item()
        total_kld += kld_loss.item()

        pbar.set_postfix(
            {
                "loss": loss.item(),
                "recon_pred": recon_pred_loss.item(),
                "KLD": kld_loss.item(),
            }
        )

    avg_loss = total_loss / len(train_loader)
    avg_recon_pred = total_recon_pred / len(train_loader)
    avg_kld = total_kld / len(train_loader)

    writer.add_scalar("Loss/train", avg_loss, epoch)
    writer.add_scalar("Loss/recon", avg_recon_pred, epoch)
    writer.add_scalar("Loss/pred", avg_kld, epoch)
    return avg_loss, avg_recon_pred, avg_kld


def evaluate_predict(
    model, val_loader, criterion, device, epoch, writer, n_past=10, n_future=10
):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for x in val_loader:
            x = x.to(device)

            predictions = model(x, n_past, n_future, mode="predict")

            true_future = x[:, n_past:, 0, :, :].unsqueeze(2)

            val_loss = criterion(predictions, true_future)
            total_loss += val_loss.item()
    avg_loss = total_loss / len(val_loader)
    writer.add_scalar("Loss/valuate", avg_loss, epoch)
    return avg_loss


def evaluate_train(
    model, val_loader, criterion, device, epoch, writer, n_past=10, n_future=10
):
    model.eval()
    total_loss = 0.0
    total_recon_pred = 0.0
    total_kld = 0.0

    with torch.no_grad():
        for x in val_loader:
            x = x.to(device)
            precip_x = x[:, :, 0, :, :].unsqueeze(2)

            recon_preds, kld_loss = model(x, n_past, n_future, "train")
            recon_pred_loss = criterion(recon_preds, precip_x)
            loss = recon_pred_loss + get_beta_kld(500) * kld_loss

            total_loss += loss.item()
            total_recon_pred += recon_pred_loss.item()
            total_kld += kld_loss.item()

        length = len(val_loader)
        avg_loss = total_loss / length
        avg_recon_pred = total_recon_pred / length
        avg_kld = total_kld / length

        writer.add_scalar("Loss/valuate", avg_loss, epoch)
        return avg_loss, avg_recon_pred, avg_kld


def main():
    parser = argparse.ArgumentParser(
        description="Train DKF ConvLSTM Model on Weather Dataset"
    )
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

    train_loader, val_loader, test_loader = get_dataloaders(
        args.batch_size, args.n_past, args.n_future, args.precip_data_path
    )
    print(
        f"训练样本数: {len(train_loader.dataset)}, 验证样本数: {len(val_loader.dataset)}, 测试样本数: {len(test_loader.dataset)}"
    )

    model = DeepKalmanFilter()
    model = model.to(device)

    ### 加载已训练的模型继续训练
    checkpoint = torch.load("./checkpoints/best_model.pth", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")

    criterion = asymmetric_heavy_rain_loss
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    writer = SummaryWriter(log_dir=os.path.join(args.save_dir, "logs"))

    best_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        train_loss, train_recon_pred, train_kld = train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            epoch,
            writer,
            args.n_past,
            args.n_future,
        )
        print(
            f"Train Loss: {train_loss:.6f}, Train Recon_Pred: {train_recon_pred:.6f}, Train KLD: {train_kld:.6f}"
        )
        if epoch == 71:
            best_loss = float("inf")
        if epoch <= 70:
            val_loss, val_recon_pred, val_kld = evaluate_train(
                model,
                val_loader,
                criterion,
                device,
                epoch,
                writer,
                args.n_past,
                args.n_future,
            )
            print(
                f"Valuate Loss: {val_loss:.6f}, Valuate Recon_Pred: {val_recon_pred:.6f}, Valuate KLD: {val_kld:.6f}"
            )
        else:
            val_loss = evaluate_predict(
                model,
                val_loader,
                criterion,
                device,
                epoch,
                writer,
                args.n_past,
                args.n_future,
            )
            print(f"Valuate Loss: {val_loss:.6f}")

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": val_loss,
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
                },
                os.path.join(args.save_dir, "last_model.pth"),
            )
            print(f"保存最终模型，损失: {train_loss:.6f}")

    writer.close()
    print("\nTraining completed!")


if __name__ == "__main__":
    main()
