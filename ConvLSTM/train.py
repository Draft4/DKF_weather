import os
import argparse
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from modules import EncodingForecastingConvLSTM
from dataset import get_dataloaders


def train_epoch(model, train_loader, criterion, optimizer, device, epoch, writer):
    model.train()
    total_loss = 0
    loss = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    for batch_idx, (x_past, x_future) in enumerate(pbar):
        x_past = x_past.to(device)
        x_future = x_future.to(device)

        optimizer.zero_grad()

        predictions = model(x_past, x_future.shape[1])
        loss = criterion(predictions, x_future)
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.0001)
        optimizer.step()

        total_loss += loss.item()

        pbar.set_postfix({"loss": loss.item()})

    n_batches = len(train_loader)
    avg_loss = total_loss / n_batches

    writer.add_scalar("Loss/train", avg_loss, epoch)
    return avg_loss


def evaluate(model, test_loader, criterion, device, epoch, writer):
    model.eval()
    total_loss = 0
    loss = 0

    with torch.no_grad():
        for x_past, x_future in test_loader:
            x_past = x_past.to(device)
            x_future = x_future.to(device)

            predictions = model(x_past, x_future.shape[1])

            loss = criterion(predictions, x_future)
            total_loss += loss.item()

        n_batches = len(test_loader)
        avg_loss = total_loss / n_batches

        writer.add_scalar("Loss/test", avg_loss, epoch)
        return avg_loss


def main():
    parser = argparse.ArgumentParser(description="Train ConvLSTM on Moving MNIST")
    parser.add_argument(
        "--data_path",
        type=str,
        default="../data/MovingMNIST/mnist_test_seq.npy",
        help="Moving MNIST 数据路径",
    )
    parser.add_argument("--batch_size", type=int, default=32, help="批次大小")
    parser.add_argument("--epochs", type=int, default=100, help="训练轮数")
    parser.add_argument("--lr", type=float, default=1e-4, help="学习率")
    parser.add_argument("--hidden_dim_list", type=int, nargs="+", default=[128, 64, 64])
    parser.add_argument("--T_past", type=int, default=10)
    parser.add_argument("--T_future", type=int, default=10)
    parser.add_argument(
        "--save_dir", type=str, default="./checkpoints", help="模型保存目录"
    )
    parser.add_argument(
        "--log_dir", type=str, default="./logs", help="TensorBoard 日志目录"
    )
    parser.add_argument("--num_workers", type=int, default=4, help="数据加载线程数")
    parser.add_argument("--train_split", type=float, default=0.8)
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

    print("Loading data...")
    train_loader, test_loader = get_dataloaders(
        args.data_path,
        args.batch_size,
        args.T_past,
        args.T_future,
        args.num_workers,
        args.train_split,
    )
    print(
        f"训练样本数: {len(train_loader.dataset)}, 测试样本数: {len(test_loader.dataset)}"
    )

    print("Initializing model...")
    # Moving MNIST 4x4 patch: 1 channel * 4 * 4 = 16
    patch_dim = 1 * 4 * 4
    model = EncodingForecastingConvLSTM(patch_dim, args.hidden_dim_list, 5, 2)
    model = model.to(device)
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")

    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = optim.RMSprop(model.parameters(), lr=1e-3, alpha=0.9)

    writer = SummaryWriter(log_dir=os.path.join(args.save_dir, "logs"))

    best_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")

        train_loss = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch, writer
        )
        test_loss = evaluate(model, test_loader, criterion, device, epoch, writer)

        print(f"Train Loss: {train_loss:.6f}, Test Loss: {test_loss:.6f}")

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
            print(f"Saved best model with test loss: {test_loss:.6f}")
    writer.close()
    print("\nTraining completed!")


if __name__ == "__main__":
    main()
