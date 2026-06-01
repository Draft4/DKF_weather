import argparse
import importlib.util
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


ROOT = Path(__file__).resolve().parent


def load_class(module_path, class_name):
    spec = importlib.util.spec_from_file_location(module_path.stem, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, class_name)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_moving_mnist(data_path, n_past, n_future):
    data = np.load(data_path)
    data = np.transpose(data, (1, 0, 2, 3))
    data = torch.from_numpy(data).float().div(255.0).unsqueeze(2)
    return data[:, : n_past + n_future]


def make_test_loader(
    data, batch_size, train_split=0.8, seed=42, num_workers=0, max_test_samples=None
):
    num_samples = data.size(0)
    train_size = int(num_samples * train_split)
    perm = torch.randperm(num_samples, generator=torch.Generator().manual_seed(seed))
    test_indices = perm[train_size:]
    if max_test_samples is not None:
        test_indices = test_indices[:max_test_samples]
    test_data = data[test_indices]
    dataset = TensorDataset(test_data)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def patchify_batch(x, patch_size=4):
    # x: (B, T, C, H, W) -> (B, T, C * patch_size^2, H/patch, W/patch)
    batch, seq_len, channels, height, width = x.shape
    h_patches = height // patch_size
    w_patches = width // patch_size
    x = x.reshape(
        batch,
        seq_len,
        channels,
        h_patches,
        patch_size,
        w_patches,
        patch_size,
    )
    x = x.permute(0, 1, 3, 5, 2, 4, 6)
    x = x.reshape(
        batch,
        seq_len,
        h_patches,
        w_patches,
        channels * patch_size * patch_size,
    )
    return x.permute(0, 1, 4, 2, 3).contiguous()


def unpatchify_batch(x, patch_size=4, height=64, width=64):
    # x: (B, T, C * patch_size^2, H/patch, W/patch) -> (B, T, C, H, W)
    batch, seq_len, patch_dim, h_patches, w_patches = x.shape
    channels = patch_dim // (patch_size * patch_size)
    x = x.reshape(
        batch,
        seq_len,
        channels,
        patch_size,
        patch_size,
        h_patches,
        w_patches,
    )
    x = x.permute(0, 1, 2, 5, 3, 6, 4)
    return x.reshape(batch, seq_len, channels, height, width).contiguous()


def load_models(args, device):
    dkf_cls = load_class(ROOT / "DKF" / "modules_dkf.py", "DeepKalmanFilter")
    convlstm_cls = load_class(
        ROOT / "ConvLSTM" / "modules.py", "EncodingForecastingConvLSTM"
    )

    dkf = dkf_cls().to(device)
    dkf_checkpoint = torch.load(args.dkf_checkpoint, map_location=device)
    dkf.load_state_dict(dkf_checkpoint["model_state_dict"])
    dkf.eval()

    patch_dim = args.patch_size * args.patch_size
    convlstm = convlstm_cls(patch_dim, args.hidden_dim_list, 5, 2).to(device)
    conv_checkpoint = torch.load(args.convlstm_checkpoint, map_location=device)
    convlstm.load_state_dict(conv_checkpoint["model_state_dict"])
    convlstm.eval()

    return dkf, convlstm, dkf_checkpoint, conv_checkpoint


def gaussian_window(window_size=11, sigma=1.5, device="cpu"):
    coords = torch.arange(window_size, dtype=torch.float32, device=device)
    coords -= window_size // 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g /= g.sum()
    window = torch.outer(g, g)
    return window.view(1, 1, window_size, window_size)


def ssim_score(pred, target, window, data_range=1.0):
    # pred/target: (B, T, 1, H, W), values in [0, 1].
    pred = pred.reshape(-1, 1, pred.size(-2), pred.size(-1)).clamp(0.0, 1.0)
    target = target.reshape(-1, 1, target.size(-2), target.size(-1)).clamp(0.0, 1.0)
    pad = window.size(-1) // 2
    pred_pad = F.pad(pred, (pad, pad, pad, pad), mode="reflect")
    target_pad = F.pad(target, (pad, pad, pad, pad), mode="reflect")

    mu_x = F.conv2d(pred_pad, window)
    mu_y = F.conv2d(target_pad, window)
    mu_x_sq = mu_x.pow(2)
    mu_y_sq = mu_y.pow(2)
    mu_xy = mu_x * mu_y

    sigma_x_sq = F.conv2d(pred_pad * pred_pad, window) - mu_x_sq
    sigma_y_sq = F.conv2d(target_pad * target_pad, window) - mu_y_sq
    sigma_xy = F.conv2d(pred_pad * target_pad, window) - mu_xy

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    ssim_map = ((2 * mu_xy + c1) * (2 * sigma_xy + c2)) / (
        (mu_x_sq + mu_y_sq + c1) * (sigma_x_sq + sigma_y_sq + c2)
    )
    return ssim_map.mean()


def update_metrics(acc, pred, target, ssim_window):
    batch = pred.size(0)
    pred = pred.clamp(0.0, 1.0)
    target = target.clamp(0.0, 1.0)

    acc["samples"] += batch
    acc["mse"] += F.mse_loss(pred, target, reduction="mean").item() * batch
    acc["mae"] += F.l1_loss(pred, target, reduction="mean").item() * batch
    acc["bce"] += F.binary_cross_entropy(
        pred.clamp(1e-6, 1.0 - 1e-6), target, reduction="mean"
    ).item() * batch
    acc["ssim"] += ssim_score(pred, target, ssim_window).item() * batch

    step_mse = ((pred - target) ** 2).mean(dim=(0, 2, 3, 4)).detach().cpu().numpy()
    step_mae = (pred - target).abs().mean(dim=(0, 2, 3, 4)).detach().cpu().numpy()
    step_bce = F.binary_cross_entropy(
        pred.clamp(1e-6, 1.0 - 1e-6), target, reduction="none"
    )
    step_bce = step_bce.mean(dim=(0, 2, 3, 4)).detach().cpu().numpy()
    acc["step_mse_sum"] += step_mse * batch
    acc["step_mae_sum"] += step_mae * batch
    acc["step_bce_sum"] += step_bce * batch


def finalize_metrics(acc):
    samples = acc["samples"]
    return {
        "MSE": acc["mse"] / samples,
        "MAE": acc["mae"] / samples,
        "BCE": acc["bce"] / samples,
        "SSIM": acc["ssim"] / samples,
        "step_mse": acc["step_mse_sum"] / samples,
        "step_mae": acc["step_mae_sum"] / samples,
        "step_bce": acc["step_bce_sum"] / samples,
    }


def init_accumulator(n_future):
    return {
        "samples": 0,
        "mse": 0.0,
        "mae": 0.0,
        "bce": 0.0,
        "ssim": 0.0,
        "step_mse_sum": np.zeros(n_future, dtype=np.float64),
        "step_mae_sum": np.zeros(n_future, dtype=np.float64),
        "step_bce_sum": np.zeros(n_future, dtype=np.float64),
    }


def collect_sample(sample_store, sample_count, past, target, conv_pred, dkf_pred):
    if len(sample_store) >= sample_count:
        return
    remaining = sample_count - len(sample_store)
    take = min(remaining, past.size(0))
    for i in range(take):
        sample_store.append(
            {
                "past": past[i].detach().cpu(),
                "target": target[i].detach().cpu(),
                "ConvLSTM": conv_pred[i].detach().cpu(),
                "DKF": dkf_pred[i].detach().cpu(),
            }
        )


def evaluate(args, dkf, convlstm, test_loader, device):
    dkf_acc = init_accumulator(args.n_future)
    conv_acc = init_accumulator(args.n_future)
    samples = []
    ssim_window = gaussian_window(device=device)

    with torch.no_grad():
        for (x,) in tqdm(test_loader, desc="Evaluating MovingMNIST"):
            x = x.to(device)
            past = x[:, : args.n_past]
            target = x[:, args.n_past :]

            dkf_pred = dkf(x, args.n_past, args.n_future, mode="predict")

            conv_input = patchify_batch(past, args.patch_size)
            conv_patch_logits = convlstm(conv_input, args.n_future)
            conv_pred = torch.sigmoid(
                unpatchify_batch(
                    conv_patch_logits,
                    patch_size=args.patch_size,
                    height=args.image_size,
                    width=args.image_size,
                )
            )

            update_metrics(dkf_acc, dkf_pred, target, ssim_window)
            update_metrics(conv_acc, conv_pred, target, ssim_window)
            collect_sample(samples, args.num_samples, past, target, conv_pred, dkf_pred)

    return finalize_metrics(conv_acc), finalize_metrics(dkf_acc), samples


def image_np(x):
    return x.squeeze(0).numpy().clip(0.0, 1.0)


def add_image(ax, img, title="", cmap="gray", vmin=0.0, vmax=1.0):
    ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=8)
    ax.axis("off")


def save_selected_comparison(sample, out_path, sample_idx):
    future_indices = [0, 2, 4, 6, 9]
    col_titles = ["Input t=10"] + [f"t+{idx + 1}" for idx in future_indices]
    row_labels = ["Input", "Ground Truth", "ConvLSTM", "DKF"]

    fig, axes = plt.subplots(4, 6, figsize=(8.4, 5.6))
    fig.suptitle(f"MovingMNIST Forecast Comparison - Sample {sample_idx}", fontsize=12)

    target = sample["target"]
    conv = sample["ConvLSTM"]
    dkf = sample["DKF"]

    for row in range(4):
        for col in range(6):
            axes[row, col].axis("off")

    add_image(axes[0, 0], image_np(sample["past"][-1]), col_titles[0])
    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=8)

    for j, idx in enumerate(future_indices, start=1):
        add_image(axes[1, j], image_np(target[idx]))
        add_image(axes[2, j], image_np(conv[idx]))
        add_image(axes[3, j], image_np(dkf[idx]))

    for row, label in enumerate(row_labels):
        axes[row, 0].set_ylabel(label, fontsize=9, rotation=0, labelpad=34, va="center")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_full_sequence_comparison(sample, out_path, sample_idx):
    cols = sample["past"].size(0) + sample["target"].size(0)
    fig, axes = plt.subplots(4, cols, figsize=(18, 4.2))
    fig.suptitle(
        f"MovingMNIST Full Sequence Forecast Comparison - Sample {sample_idx}",
        fontsize=12,
    )

    for row in range(4):
        for col in range(cols):
            axes[row, col].axis("off")

    for t in range(sample["past"].size(0)):
        add_image(axes[0, t], image_np(sample["past"][t]), f"Past {t + 1}")

    offset = sample["past"].size(0)
    for t in range(sample["target"].size(0)):
        add_image(axes[1, offset + t], image_np(sample["target"][t]), f"True {t + 1}")
        add_image(axes[2, offset + t], image_np(sample["ConvLSTM"][t]), f"Pred {t + 1}")
        add_image(axes[3, offset + t], image_np(sample["DKF"][t]), f"Pred {t + 1}")

    for row, label in enumerate(["Input", "Ground Truth", "ConvLSTM", "DKF"]):
        axes[row, 0].set_ylabel(label, fontsize=9, rotation=0, labelpad=34, va="center")

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_step_plot(conv_metrics, dkf_metrics, out_path):
    steps = np.arange(1, len(conv_metrics["step_mse"]) + 1)
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    ax.plot(steps, conv_metrics["step_mse"], marker="o", label="ConvLSTM")
    ax.plot(steps, dkf_metrics["step_mse"], marker="s", label="DKF")
    ax.set_xlabel("Prediction step")
    ax.set_ylabel("MSE")
    ax.set_title("MSE over Prediction Steps")
    ax.set_xticks(steps)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_metric_step_plot(conv_metrics, dkf_metrics, metric_key, ylabel, title, out_path):
    steps = np.arange(1, len(conv_metrics[metric_key]) + 1)
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    ax.plot(steps, conv_metrics[metric_key], marker="o", label="ConvLSTM")
    ax.plot(steps, dkf_metrics[metric_key], marker="s", label="DKF")
    ax.set_xlabel("Prediction step")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(steps)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_metrics_bar(conv_metrics, dkf_metrics, out_path):
    names = ["MSE", "MAE", "BCE"]
    x = np.arange(len(names))
    width = 0.36
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.bar(x - width / 2, [conv_metrics[n] for n in names], width, label="ConvLSTM")
    ax.bar(x + width / 2, [dkf_metrics[n] for n in names], width, label="DKF")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("Lower is better")
    ax.set_title("Forecast Error Metrics")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def format_float(value):
    return f"{value:.6f}"


def write_markdown(conv_metrics, dkf_metrics, args, out_path):
    better = {
        "MSE": "DKF" if dkf_metrics["MSE"] < conv_metrics["MSE"] else "ConvLSTM",
        "MAE": "DKF" if dkf_metrics["MAE"] < conv_metrics["MAE"] else "ConvLSTM",
        "BCE": "DKF" if dkf_metrics["BCE"] < conv_metrics["BCE"] else "ConvLSTM",
        "SSIM": "DKF" if dkf_metrics["SSIM"] > conv_metrics["SSIM"] else "ConvLSTM",
    }
    lines = [
        "# MovingMNIST Model Comparison",
        "",
        f"- Dataset: `{args.data_path}`",
        f"- Split: train/test = `{args.train_split:.2f}/{1 - args.train_split:.2f}`, seed = `{args.seed}`",
        f"- Input frames: `{args.n_past}`",
        f"- Prediction frames: `{args.n_future}`",
        f"- DKF checkpoint: `{args.dkf_checkpoint}`",
        f"- ConvLSTM checkpoint: `{args.convlstm_checkpoint}`",
        "",
        "## Summary Metrics",
        "",
        "| Metric | ConvLSTM | DKF | Better |",
        "|---|---:|---:|---|",
    ]
    for metric in ["MSE", "MAE", "BCE", "SSIM"]:
        lines.append(
            f"| {metric} | {format_float(conv_metrics[metric])} | "
            f"{format_float(dkf_metrics[metric])} | {better[metric]} |"
        )

    lines.extend(
        [
            "",
            "## MSE Per Prediction Step",
            "",
            "| Step | ConvLSTM | DKF |",
            "|---:|---:|---:|",
        ]
    )
    for idx, (conv_mse, dkf_mse) in enumerate(
        zip(conv_metrics["step_mse"], dkf_metrics["step_mse"]), start=1
    ):
        lines.append(f"| {idx} | {format_float(conv_mse)} | {format_float(dkf_mse)} |")

    lines.extend(
        [
            "",
            "## MAE Per Prediction Step",
            "",
            "| Step | ConvLSTM | DKF |",
            "|---:|---:|---:|",
        ]
    )
    for idx, (conv_mae, dkf_mae) in enumerate(
        zip(conv_metrics["step_mae"], dkf_metrics["step_mae"]), start=1
    ):
        lines.append(f"| {idx} | {format_float(conv_mae)} | {format_float(dkf_mae)} |")

    lines.extend(
        [
            "",
            "## BCE Per Prediction Step",
            "",
            "BCE matches the reconstruction criterion used by the MovingMNIST DKF training script, evaluated here on future-frame predictions.",
            "",
            "| Step | ConvLSTM | DKF |",
            "|---:|---:|---:|",
        ]
    )
    for idx, (conv_bce, dkf_bce) in enumerate(
        zip(conv_metrics["step_bce"], dkf_metrics["step_bce"]), start=1
    ):
        lines.append(f"| {idx} | {format_float(conv_bce)} | {format_float(dkf_bce)} |")

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- MSE, MAE, and BCE are lower-is-better metrics computed on predicted future frames.",
            "- SSIM is higher-is-better and is computed with an 11x11 Gaussian window on grayscale frames.",
            "- ConvLSTM logits are converted with sigmoid before metric computation.",
        ]
    )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def print_metrics(conv_metrics, dkf_metrics):
    print("\n=== MovingMNIST Comparison ===")
    print(f"{'Metric':<10} {'ConvLSTM':>12} {'DKF':>12} {'Better':>12}")
    for metric in ["MSE", "MAE", "BCE", "SSIM"]:
        if metric == "SSIM":
            better = "DKF" if dkf_metrics[metric] > conv_metrics[metric] else "ConvLSTM"
        else:
            better = "DKF" if dkf_metrics[metric] < conv_metrics[metric] else "ConvLSTM"
        print(
            f"{metric:<10} {conv_metrics[metric]:>12.6f} "
            f"{dkf_metrics[metric]:>12.6f} {better:>12}"
        )
    print("\nMSE per prediction step:")
    print(f"{'Step':<6} {'ConvLSTM':>12} {'DKF':>12}")
    for idx, (conv_mse, dkf_mse) in enumerate(
        zip(conv_metrics["step_mse"], dkf_metrics["step_mse"]), start=1
    ):
        print(f"{idx:<6} {conv_mse:>12.6f} {dkf_mse:>12.6f}")

    print("\nMAE per prediction step:")
    print(f"{'Step':<6} {'ConvLSTM':>12} {'DKF':>12}")
    for idx, (conv_mae, dkf_mae) in enumerate(
        zip(conv_metrics["step_mae"], dkf_metrics["step_mae"]), start=1
    ):
        print(f"{idx:<6} {conv_mae:>12.6f} {dkf_mae:>12.6f}")

    print("\nBCE per prediction step:")
    print(f"{'Step':<6} {'ConvLSTM':>12} {'DKF':>12}")
    for idx, (conv_bce, dkf_bce) in enumerate(
        zip(conv_metrics["step_bce"], dkf_metrics["step_bce"]), start=1
    ):
        print(f"{idx:<6} {conv_bce:>12.6f} {dkf_bce:>12.6f}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare DKF and ConvLSTM on MovingMNIST with unified metrics."
    )
    parser.add_argument(
        "--data_path",
        type=Path,
        default=ROOT / "data" / "MovingMNIST" / "mnist_test_seq.npy",
    )
    parser.add_argument(
        "--dkf_checkpoint",
        type=Path,
        default=ROOT / "DKF" / "checkpoints" / "best_model.pth",
    )
    parser.add_argument(
        "--convlstm_checkpoint",
        type=Path,
        default=ROOT / "ConvLSTM" / "checkpoints" / "best_model.pth",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=ROOT / "comparison_results" / "moving_mnist",
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--num_samples", type=int, default=3)
    parser.add_argument(
        "--max_test_samples",
        type=int,
        default=None,
        help="Optional cap for quick smoke tests. By default, evaluate all test samples.",
    )
    parser.add_argument("--n_past", type=int, default=10)
    parser.add_argument("--n_future", type=int, default=10)
    parser.add_argument("--train_split", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patch_size", type=int, default=4)
    parser.add_argument("--image_size", type=int, default=64)
    parser.add_argument(
        "--hidden_dim_list", type=int, nargs="+", default=[128, 64, 64]
    )
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = get_device()
    print(f"Using device: {device}")

    data = load_moving_mnist(args.data_path, args.n_past, args.n_future)
    test_loader = make_test_loader(
        data,
        args.batch_size,
        train_split=args.train_split,
        seed=args.seed,
        num_workers=args.num_workers,
        max_test_samples=args.max_test_samples,
    )
    print(f"Loaded MovingMNIST: {tuple(data.shape)}")
    print(f"Test samples: {len(test_loader.dataset)}")

    dkf, convlstm, dkf_ckpt, conv_ckpt = load_models(args, device)
    print(
        f"Loaded DKF checkpoint: epoch={dkf_ckpt.get('epoch', 'N/A')}, "
        f"loss={dkf_ckpt.get('loss', 'N/A')}"
    )
    print(
        f"Loaded ConvLSTM checkpoint: epoch={conv_ckpt.get('epoch', 'N/A')}, "
        f"loss={conv_ckpt.get('loss', 'N/A')}"
    )

    conv_metrics, dkf_metrics, samples = evaluate(args, dkf, convlstm, test_loader, device)
    print_metrics(conv_metrics, dkf_metrics)

    metrics_path = args.output_dir / "metrics_summary.md"
    write_markdown(conv_metrics, dkf_metrics, args, metrics_path)
    save_step_plot(conv_metrics, dkf_metrics, args.output_dir / "mse_per_step.png")
    save_metric_step_plot(
        conv_metrics,
        dkf_metrics,
        "step_mae",
        "MAE",
        "MAE over Prediction Steps",
        args.output_dir / "mae_per_step.png",
    )
    save_metric_step_plot(
        conv_metrics,
        dkf_metrics,
        "step_bce",
        "BCE",
        "BCE over Prediction Steps",
        args.output_dir / "bce_per_step.png",
    )
    save_metrics_bar(conv_metrics, dkf_metrics, args.output_dir / "metrics_bar_chart.png")

    for idx, sample in enumerate(samples):
        save_selected_comparison(
            sample,
            args.output_dir / f"sample_{idx}_selected_comparison.png",
            idx,
        )
        save_full_sequence_comparison(
            sample,
            args.output_dir / f"sample_{idx}_full_sequence_comparison.png",
            idx,
        )

    print(f"\nSaved markdown summary: {metrics_path}")
    print(f"Saved figures to: {args.output_dir}")


if __name__ == "__main__":
    main()
