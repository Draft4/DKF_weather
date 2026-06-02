import argparse
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LinearSegmentedColormap
from tqdm import tqdm

from dataset import get_dataloaders
from modules import PrecipConvLSTM


def inverse_log_transform(x):
    return torch.expm1(x).clamp_min(0.0)


def create_precip_cmap():
    colors = [
        (1.0, 1.0, 1.0),
        (0.80, 0.90, 1.0),
        (0.35, 0.65, 1.0),
        (0.00, 0.45, 0.95),
        (0.00, 0.75, 0.35),
        (1.00, 0.90, 0.00),
        (1.00, 0.55, 0.00),
        (0.90, 0.00, 0.00),
        (0.55, 0.00, 0.55),
    ]
    return LinearSegmentedColormap.from_list("precipitation", colors)


def safe_div(num, den):
    if den == 0:
        return float("nan")
    return num / den


def init_metrics(n_future, thresholds):
    return {
        "pixels": 0,
        "sum_error": 0.0,
        "sum_abs_error": 0.0,
        "sum_sq_error": 0.0,
        "sum_true": 0.0,
        "sum_pred": 0.0,
        "sum_true_sq": 0.0,
        "sum_pred_sq": 0.0,
        "sum_true_pred": 0.0,
        "step_pixels": np.zeros(n_future, dtype=np.float64),
        "step_sum_error": np.zeros(n_future, dtype=np.float64),
        "step_sum_abs_error": np.zeros(n_future, dtype=np.float64),
        "step_sum_sq_error": np.zeros(n_future, dtype=np.float64),
        "thresholds": {
            threshold: {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
            for threshold in thresholds
        },
    }


def update_metrics(metrics, pred, target, thresholds):
    error = pred - target
    metrics["pixels"] += pred.numel()
    metrics["sum_error"] += float(error.sum().item())
    metrics["sum_abs_error"] += float(error.abs().sum().item())
    metrics["sum_sq_error"] += float((error * error).sum().item())
    metrics["sum_true"] += float(target.sum().item())
    metrics["sum_pred"] += float(pred.sum().item())
    metrics["sum_true_sq"] += float((target * target).sum().item())
    metrics["sum_pred_sq"] += float((pred * pred).sum().item())
    metrics["sum_true_pred"] += float((target * pred).sum().item())

    step_pixels = pred[:, 0].numel()
    metrics["step_pixels"] += step_pixels
    metrics["step_sum_error"] += error.sum(dim=(0, 2, 3)).detach().cpu().numpy()
    metrics["step_sum_abs_error"] += (
        error.abs().sum(dim=(0, 2, 3)).detach().cpu().numpy()
    )
    metrics["step_sum_sq_error"] += (
        (error * error).sum(dim=(0, 2, 3)).detach().cpu().numpy()
    )

    for threshold in thresholds:
        pred_event = pred >= threshold
        true_event = target >= threshold
        counts = metrics["thresholds"][threshold]
        counts["tp"] += int((pred_event & true_event).sum().item())
        counts["fp"] += int((pred_event & ~true_event).sum().item())
        counts["fn"] += int((~pred_event & true_event).sum().item())
        counts["tn"] += int((~pred_event & ~true_event).sum().item())


def finalize_threshold_scores(counts):
    tp = counts["tp"]
    fp = counts["fp"]
    fn = counts["fn"]
    tn = counts["tn"]
    pod = safe_div(tp, tp + fn)
    far = safe_div(fp, tp + fp)
    csi = safe_div(tp, tp + fp + fn)
    precision = safe_div(tp, tp + fp)
    recall = pod
    f1 = safe_div(2 * precision * recall, precision + recall)
    frequency_bias = safe_div(tp + fp, tp + fn)
    accuracy = safe_div(tp + tn, tp + fp + fn + tn)
    return {
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
        "POD": pod,
        "FAR": far,
        "CSI": csi,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "Frequency Bias": frequency_bias,
        "Accuracy": accuracy,
    }


def finalize_metrics(metrics):
    n = metrics["pixels"]
    mse = metrics["sum_sq_error"] / n
    mae = metrics["sum_abs_error"] / n
    bias = metrics["sum_error"] / n
    mean_true = metrics["sum_true"] / n
    mean_pred = metrics["sum_pred"] / n

    numerator = n * metrics["sum_true_pred"] - metrics["sum_true"] * metrics["sum_pred"]
    true_var = n * metrics["sum_true_sq"] - metrics["sum_true"] ** 2
    pred_var = n * metrics["sum_pred_sq"] - metrics["sum_pred"] ** 2
    pearson_r = safe_div(numerator, math.sqrt(max(true_var * pred_var, 0.0)))

    step_mse = metrics["step_sum_sq_error"] / metrics["step_pixels"]
    step_mae = metrics["step_sum_abs_error"] / metrics["step_pixels"]
    step_bias = metrics["step_sum_error"] / metrics["step_pixels"]

    threshold_scores = {
        threshold: finalize_threshold_scores(counts)
        for threshold, counts in metrics["thresholds"].items()
    }
    return {
        "MSE": mse,
        "RMSE": math.sqrt(mse),
        "MAE": mae,
        "Bias": bias,
        "Mean Obs": mean_true,
        "Mean Pred": mean_pred,
        "Pearson r": pearson_r,
        "step_mse": step_mse,
        "step_rmse": np.sqrt(step_mse),
        "step_mae": step_mae,
        "step_bias": step_bias,
        "threshold_scores": threshold_scores,
    }


def update_top_samples(top_samples, x_past, target, pred, args, sample_offset):
    past = inverse_log_transform(x_past[:, :, 0])
    scores = target.sum(dim=(1, 2, 3)).detach().cpu().numpy()

    for i, score in enumerate(scores):
        top_samples.append(
            {
                "score": float(score),
                "index": sample_offset + i,
                "past": past[i].detach().cpu(),
                "target": target[i].detach().cpu(),
                "pred": pred[i].detach().cpu(),
            }
        )
    top_samples.sort(key=lambda item: item["score"], reverse=True)
    del top_samples[args.num_samples :]


def update_scatter_store(store, target, pred, args, rng):
    if len(store["true"]) >= args.scatter_points:
        return

    true_np = target.detach().cpu().numpy().reshape(-1)
    pred_np = pred.detach().cpu().numpy().reshape(-1)
    mask = (true_np >= args.scatter_threshold) | (pred_np >= args.scatter_threshold)
    if mask.any():
        true_np = true_np[mask]
        pred_np = pred_np[mask]

    needed = args.scatter_points - len(store["true"])
    take = min(needed, len(true_np))
    if take <= 0:
        return
    indices = rng.choice(len(true_np), size=take, replace=False)
    store["true"].extend(true_np[indices].tolist())
    store["pred"].extend(pred_np[indices].tolist())


def figure_vmax(*arrays):
    values = np.concatenate([arr.reshape(-1) for arr in arrays])
    positive = values[values > 0]
    if len(positive) == 0:
        return 1.0
    return max(5.0, float(np.percentile(positive, 99.5)))


def add_precip_image(ax, image, title, cmap, vmax):
    im = ax.imshow(image, cmap=cmap, vmin=0.0, vmax=vmax)
    ax.set_title(title, fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])
    return im


def save_sample_figure(sample, out_path):
    past_indices = np.linspace(0, sample["past"].shape[0] - 1, 5, dtype=int)
    future_indices = np.array([0, 2, 4, 6, sample["target"].shape[0] - 1])
    cmap = create_precip_cmap()

    past = sample["past"].numpy()
    target = sample["target"].numpy()
    pred = sample["pred"].numpy()
    vmax = figure_vmax(past[past_indices], target[future_indices], pred[future_indices])

    fig, axes = plt.subplots(3, 5, figsize=(8.5, 4.5), constrained_layout=True)
    fig.suptitle(
        f"ConvLSTM Precipitation Forecast - Test Sample {sample['index']}",
        fontsize=11,
    )

    for col, idx in enumerate(past_indices):
        lead = sample["past"].shape[0] - idx - 1
        title = "Input t" if lead == 0 else f"Input t-{lead}"
        im = add_precip_image(axes[0, col], past[idx], title, cmap, vmax)
    for col, idx in enumerate(future_indices):
        im = add_precip_image(axes[1, col], target[idx], f"Obs t+{idx + 1}", cmap, vmax)
        im = add_precip_image(axes[2, col], pred[idx], f"Pred t+{idx + 1}", cmap, vmax)

    for row, label in enumerate(["History", "Observed", "Predicted"]):
        axes[row, 0].set_ylabel(label, fontsize=9, rotation=0, labelpad=34, va="center")

    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.82, pad=0.02)
    cbar.set_label("Precipitation (mm/h)", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_step_metrics_plot(metrics, out_path):
    steps = np.arange(1, len(metrics["step_mae"]) + 1)
    fig, axes = plt.subplots(3, 1, figsize=(6.7, 6.8), sharex=True)

    axes[0].plot(steps, metrics["step_mae"], marker="o", linewidth=1.8)
    axes[0].set_ylabel("MAE (mm/h)")
    axes[0].set_title("ConvLSTM Prediction Error over Lead Time")

    axes[1].plot(steps, metrics["step_rmse"], marker="o", linewidth=1.8, color="#d95f02")
    axes[1].set_ylabel("RMSE (mm/h)")

    axes[2].plot(steps, metrics["step_bias"], marker="o", linewidth=1.8, color="#1b9e77")
    axes[2].axhline(0.0, color="black", linewidth=0.8, alpha=0.6)
    axes[2].set_ylabel("Bias (mm/h)")
    axes[2].set_xlabel("Prediction step")

    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.set_xticks(steps)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_threshold_scores_plot(metrics, out_path):
    thresholds = list(metrics["threshold_scores"].keys())
    labels = [f">={threshold:g}" for threshold in thresholds]
    x = np.arange(len(thresholds))

    fig, ax = plt.subplots(figsize=(6.7, 4.0))
    for key, marker in [("POD", "o"), ("FAR", "s"), ("CSI", "^"), ("F1", "D")]:
        values = [metrics["threshold_scores"][threshold][key] for threshold in thresholds]
        ax.plot(x, values, marker=marker, linewidth=1.8, label=key)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel("Rainfall threshold (mm/h)")
    ax.set_ylabel("Score")
    ax.set_title("ConvLSTM Categorical Forecast Scores")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=4, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_scatter_plot(scatter_store, metrics, out_path):
    true = np.asarray(scatter_store["true"], dtype=np.float32)
    pred = np.asarray(scatter_store["pred"], dtype=np.float32)

    fig, ax = plt.subplots(figsize=(6.0, 5.0))
    if len(true) > 0:
        max_val = max(float(true.max()), float(pred.max()), 1.0)
        hb = ax.hexbin(true, pred, gridsize=65, mincnt=1, bins="log", cmap="viridis")
        cbar = fig.colorbar(hb, ax=ax, fraction=0.045, pad=0.03)
        cbar.set_label("log10(count)", fontsize=8)
        cbar.ax.tick_params(labelsize=7)
        ax.plot([0, max_val], [0, max_val], "r--", linewidth=1.2, label="1:1 line")
        ax.set_xlim(0, max_val)
        ax.set_ylim(0, max_val)
    ax.set_xlabel("Observed precipitation (mm/h)")
    ax.set_ylabel("Predicted precipitation (mm/h)")
    ax.set_title(f"ConvLSTM Observed vs Predicted Rainfall (r={metrics['Pearson r']:.3f})")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def format_value(value):
    if isinstance(value, (float, np.floating)) and math.isnan(value):
        return "N/A"
    return f"{float(value):.6f}"


def write_markdown(metrics, args, out_path):
    lines = [
        "# ConvLSTM_pre Precipitation Forecast Evaluation",
        "",
        f"- Checkpoint: `{args.checkpoint}`",
        f"- Precipitation files: `{args.precip_data_path}`",
        f"- Land data: `{args.land_data_path}`",
        f"- Input frames: `{args.n_past}`",
        f"- Prediction frames: `{args.n_future}`",
        "",
        "## Summary Metrics",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    for metric in ["MSE", "RMSE", "MAE", "Bias", "Mean Obs", "Mean Pred", "Pearson r"]:
        lines.append(f"| {metric} | {format_value(metrics[metric])} |")

    lines.extend(
        [
            "",
            "## Metrics Per Prediction Step",
            "",
            "| Step | MAE (mm/h) | RMSE (mm/h) | Bias (mm/h) |",
            "|---:|---:|---:|---:|",
        ]
    )
    for idx, (mae, rmse, bias) in enumerate(
        zip(metrics["step_mae"], metrics["step_rmse"], metrics["step_bias"]), start=1
    ):
        lines.append(
            f"| {idx} | {format_value(mae)} | {format_value(rmse)} | {format_value(bias)} |"
        )

    lines.extend(
        [
            "",
            "## Threshold-Based Scores",
            "",
            "| Threshold (mm/h) | TP | FP | FN | TN | POD | FAR | CSI | Precision | Recall | F1 | Frequency Bias | Accuracy |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for threshold, scores in metrics["threshold_scores"].items():
        lines.append(
            f"| >= {threshold:g} | {scores['TP']} | {scores['FP']} | {scores['FN']} | {scores['TN']} | "
            f"{format_value(scores['POD'])} | {format_value(scores['FAR'])} | "
            f"{format_value(scores['CSI'])} | {format_value(scores['Precision'])} | "
            f"{format_value(scores['Recall'])} | {format_value(scores['F1'])} | "
            f"{format_value(scores['Frequency Bias'])} | {format_value(scores['Accuracy'])} |"
        )

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Continuous metrics are computed in the original precipitation scale after `expm1` inverse transformation.",
            "- Bias is `prediction - observation`; negative values indicate underestimation.",
            "- POD is probability of detection, FAR is false alarm ratio, and CSI is critical success index.",
            "- Threshold scores are computed pixel-wise over all predicted future frames.",
        ]
    )

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def evaluate_model(model, test_loader, device, args):
    model.eval()
    metrics_acc = init_metrics(args.n_future, args.thresholds)
    top_samples = []
    scatter_store = {"true": [], "pred": []}
    rng = np.random.default_rng(42)
    sample_offset = 0

    with torch.no_grad():
        for x_past, y_future in tqdm(test_loader, desc="Evaluating ConvLSTM_pre"):
            x_past = x_past.to(device)
            y_future = y_future.to(device)

            predictions_log = model(x_past, args.n_future)
            pred = inverse_log_transform(predictions_log.squeeze(2))
            target = inverse_log_transform(y_future.squeeze(2))

            update_metrics(metrics_acc, pred, target, args.thresholds)
            update_scatter_store(scatter_store, target, pred, args, rng)
            update_top_samples(top_samples, x_past, target, pred, args, sample_offset)
            sample_offset += x_past.size(0)

    return finalize_metrics(metrics_acc), top_samples, scatter_store


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate ConvLSTM on precipitation data")
    parser.add_argument("--checkpoint", type=str, required=True, help="模型检查点路径")
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
    parser.add_argument(
        "--save_dir", type=str, default="./results", help="结果保存目录"
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--n_past", type=int, default=10)
    parser.add_argument("--n_future", type=int, default=10)
    parser.add_argument(
        "--hidden_dim_list",
        type=int,
        nargs="+",
        default=[128, 64, 64],
    )
    parser.add_argument("--patch_size", type=int, default=4)
    parser.add_argument("--num_samples", type=int, default=3)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=[0.1, 2.0, 5.0, 10.0, 30.0],
    )
    parser.add_argument(
        "--scatter_points",
        type=int,
        default=50000,
        help="Maximum number of points used in the observed-vs-predicted scatter plot.",
    )
    parser.add_argument(
        "--scatter_threshold",
        type=float,
        default=0.1,
        help="Keep scatter points where observed or predicted rain exceeds this value.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    device = get_device()
    print(f"使用设备: {device}")

    _, _, test_loader = get_dataloaders(
        batch_size=args.batch_size,
        n_past=args.n_past,
        n_future=args.n_future,
        precip_data_path=args.precip_data_path,
        land_data_path=args.land_data_path,
        num_workers=args.num_workers,
    )

    model = PrecipConvLSTM(
        hidden_dim_list=args.hidden_dim_list,
        patch_size=args.patch_size,
    ).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"加载模型: {args.checkpoint}, Epoch: {checkpoint.get('epoch', 'N/A')}")

    metrics, top_samples, scatter_store = evaluate_model(model, test_loader, device, args)

    print("\n=== ConvLSTM_pre Evaluation ===")
    for metric in ["MSE", "RMSE", "MAE", "Bias", "Mean Obs", "Mean Pred", "Pearson r"]:
        print(f"{metric}: {format_value(metrics[metric])}")

    metrics_path = os.path.join(args.save_dir, "metrics_summary.md")
    write_markdown(metrics, args, metrics_path)
    save_step_metrics_plot(metrics, os.path.join(args.save_dir, "step_metrics.png"))
    save_threshold_scores_plot(
        metrics, os.path.join(args.save_dir, "threshold_scores.png")
    )
    save_scatter_plot(
        scatter_store, metrics, os.path.join(args.save_dir, "true_vs_pred_scatter.png")
    )

    for idx, sample in enumerate(top_samples):
        save_sample_figure(
            sample,
            os.path.join(args.save_dir, f"sample_{idx}_precip_forecast.png"),
        )

    print(f"\n结果已保存到: {args.save_dir}")


if __name__ == "__main__":
    main()
