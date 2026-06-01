import os
import argparse
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from modules_dkf import DeepKalmanFilter
from dataset import get_dataloaders


def get_beta_kld(current_epoch, warmup_epochs=50, max_beta=0.06):
    if current_epoch < warmup_epochs:
        return max_beta * (current_epoch / warmup_epochs)
    # elif current_epoch >= 200:
    #     return 2 * max_beta
    return max_beta
    # return max_beta


def get_scheduled_sampling_ratio(
    current_epoch, total_epochs=500, warmup_epochs=100, max_ratio=0.4
):
    if current_epoch < warmup_epochs:
        return 0.0

    ratio = max_ratio * (
        (current_epoch - warmup_epochs) / (total_epochs - warmup_epochs)
    )
    return min(ratio, max_ratio)


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

        optimizer.zero_grad()

        ss_ratio = get_scheduled_sampling_ratio(epoch)

        recon_preds, kld_loss = model(x, n_past, n_future, "train", ss_ratio)
        recon_pred_loss = criterion(recon_preds, x)
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
    model, test_loader, criterion, device, epoch, writer, n_past=10, n_future=10
):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for x in test_loader:
            x = x.to(device)

            predictions = model(x, n_past, n_future, mode="predict")

            true_future = x[:, n_past:, :, :, :]

            val_loss = criterion(predictions, true_future)
            total_loss += val_loss.item()

        avg_loss = total_loss / len(test_loader)
        writer.add_scalar("Loss/test", avg_loss, epoch)
        return avg_loss


def evaluate_train(
    model, test_loader, criterion, device, epoch, writer, n_past=10, n_future=10
):
    model.eval()
    total_loss = 0.0
    total_recon_pred = 0.0
    total_kld = 0.0

    with torch.no_grad():
        for x in test_loader:
            x = x.to(device)

            recon_preds, kld_loss = model(x, n_past, n_future, "train")
            recon_pred_loss = criterion(recon_preds, x)
            loss = recon_pred_loss + 0.06 * kld_loss

            total_loss += loss.item()
            total_recon_pred += recon_pred_loss.item()
            total_kld += kld_loss.item()

        length = len(test_loader)
        avg_loss = total_loss / length
        avg_recon_pred = total_recon_pred / length
        avg_kld = total_kld / length

        writer.add_scalar("Loss/test", avg_loss, epoch)
        return avg_loss, avg_recon_pred, avg_kld


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
    parser.add_argument("--batch_size", type=int, default=64, help="批次大小")
    parser.add_argument("--epochs", type=int, default=350, help="训练轮数")
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

    # 数据加载器
    train_loader, test_loader = get_dataloaders(
        data_path=args.data_path,
        batch_size=args.batch_size,
        n_past=args.n_past,
        n_future=args.n_future,
        num_workers=args.num_workers,
    )
    print(
        f"训练样本数: {len(train_loader.dataset)}, 测试样本数: {len(test_loader.dataset)}"
    )

    model = DeepKalmanFilter()
    model = model.to(device)

    # ### 加载已训练的模型继续训练
    # checkpoint = torch.load("./checkpoints/best_model.pth", map_location=device)
    # model.load_state_dict(checkpoint["model_state_dict"])

    print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")

    criterion = torch.nn.functional.binary_cross_entropy
    # criterion = torch.nn.BCEWithLogitsLoss()
    # criterion = torch.nn.MSELoss()
    # optimizer = optim.RMSprop(model.parameters(), lr=1e-3, alpha=0.9)
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
            f"Train Loss: {train_loss:.6f}, Train Recon_Pred: {train_recon_pred:.6f}, Train KLD: {train_kld:.6f}, KLD Beta: {get_beta_kld(epoch)}"
        )

        if epoch == 201:
            best_loss = float("inf")

        if epoch < 200:
            test_loss, test_recon_pred, test_kld = evaluate_train(
                model,
                test_loader,
                criterion,
                device,
                epoch,
                writer,
                args.n_past,
                args.n_future,
            )
            print(
                f"Test Loss: {test_loss:.6f}, Test Recon_Pred: {test_recon_pred:.6f}, Test KLD: {test_kld:.6f}"
            )
        else:
            test_loss = evaluate_predict(
                model,
                test_loader,
                criterion,
                device,
                epoch,
                writer,
                args.n_past,
                args.n_future,
            )
            print(f"Test Loss: {test_loss:.6f}")

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
