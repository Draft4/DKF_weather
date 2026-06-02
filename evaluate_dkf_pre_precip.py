import argparse
import glob
import importlib.util
import math
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import xarray as xr
from matplotlib.colors import LinearSegmentedColormap
from torch.utils.data import DataLoader
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


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_precip_tensor(path_pattern, variable_name):
    precip_files = sorted(glob.glob(str(path_pattern)))
    if not precip_files:
        raise FileNotFoundError(f"No precipitation files matched: {path_pattern}")

    tensors = []
    for precip_file in tqdm(precip_files, desc="Loading precipitation files"):
        with xr.open_dataset(precip_file) as ds:
            values = ds[variable_name].values
        tensors.append(torch.from_numpy(values).float())
    return torch.cat(tensors, dim=0), precip_files


def load_land_tensor(path, variable_name):
    with xr.open_dataset(path) as ds:
        values = ds[variable_name].values
    return torch.from_numpy(values).float()


def make_test_loader(args):
    weather_dataset_cls = load_class(ROOT / "DKF_pre" / "dataset.py", "WeatherDataset")
    precip_data, precip_files = load_precip_tensor(
        args.precip_data_path, args.precip_variable
    )
    land_data = load_land_tensor(args.land_data_path, args.land_variable)

    if args.train_end >= args.val_end:
        raise ValueError("Expected train_end to be smaller than val_end.")
    if args.val_end + args.n_past + args.n_future >= len(precip_data):
        raise ValueError(
            "The configured validation end leaves too few frames for the test split."
        )

    test_precip_data = precip_data[args.val_end :]
    if args.max_test_frames is not None:
        test_precip_data = test_precip_data[: args.max_test_frames]
    if len(test_precip_data) <= args.n_past + args.n_future:
        raise ValueError("The test split is too short for the configured sequence length.")

    test_dataset = weather_dataset_cls(
        test_precip_data,
        land_data,
        n_past=args.n_past,
        n_future=args.n_future,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return test_loader, precip_data, precip_files


def load_model(args, device):
    model_cls = load_class(ROOT / "DKF_pre" / "modules.py", "DeepKalmanFilter")
    model = model_cls().to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def inverse_log_transform(x):
    return torch.expm1(x).clamp_min(0.0)


def predict_ensemble(model, x, args, batch_idx):
    preds = []
    for ensemble_idx in range(args.num_ensembles):
        set_seed(args.seed + batch_idx * args.num_ensembles + ensemble_idx)
        pred_log = model(x, args.n_past, args.n_future, mode="predict").squeeze(2)
        preds.append(inverse_log_transform(pred_log))
    return torch.stack(preds, dim=0).mean(dim=0)


def safe_div(num, den):
    if den == 0:
        return float("nan")
    return num / den


def init_metrics(n_future, thresholds):
    return {
        "pixels": 0,
        "sum_true": 0.0,
        "sum_pred": 0.0,
        "sum_true_sq": 0.0,
        "sum_pred_sq": 0.0,
        "sum_true_pred": 0.0,
        "sum_error": 0.0,
        "sum_abs_error": 0.0,
        "sum_sq_error": 0.0,
        "step_pixels": np.zeros(n_future, dtype=np.float64),
        "step_sum_error": np.zeros(n_future, dtype=np.float64),
        "step_sum_abs_error": np.zeros(n_future, dtype=np.float64),
        "step_sum_sq_error": np.zeros(n_future, dtype=np.float64),
        "thresholds": {
            threshold: {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
            for threshold in thresholds
        },
    }


def update_threshold_counts(metrics, pred, target, thresholds):
    for threshold in thresholds:
        pred_event = pred >= threshold
        true_event = target >= threshold
        counts = metrics["thresholds"][threshold]
        counts["tp"] += int((pred_event & true_event).sum().item())
        counts["fp"] += int((pred_event & ~true_event).sum().item())
        counts["fn"] += int((~pred_event & true_event).sum().item())
        counts["tn"] += int((~pred_event & ~true_event).sum().item())


def update_metrics(metrics, pred, target, thresholds):
    error = pred - target
    pixels = pred.numel()

    metrics["pixels"] += pixels
    metrics["sum_true"] += float(target.sum().item())
    metrics["sum_pred"] += float(pred.sum().item())
    metrics["sum_true_sq"] += float((target * target).sum().item())
    metrics["sum_pred_sq"] += float((pred * pred).sum().item())
    metrics["sum_true_pred"] += float((target * pred).sum().item())
    metrics["sum_error"] += float(error.sum().item())
    metrics["sum_abs_error"] += float(error.abs().sum().item())
    metrics["sum_sq_error"] += float((error * error).sum().item())

    step_pixels = pred[:, 0].numel()
    metrics["step_pixels"] += step_pixels
    metrics["step_sum_error"] += error.sum(dim=(0, 2, 3)).detach().cpu().numpy()
    metrics["step_sum_abs_error"] += (
        error.abs().sum(dim=(0, 2, 3)).detach().cpu().numpy()
    )
    metrics["step_sum_sq_error"] += (
        (error * error).sum(dim=(0, 2, 3)).detach().cpu().numpy()
    )

    update_threshold_counts(metrics, pred, target, thresholds)


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


def update_top_samples(top_samples, x, target, pred, args, sample_offset):
    past = inverse_log_transform(x[:, : args.n_past, 0])
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


def evaluate(model, test_loader, args, device):
    metrics = init_metrics(args.n_future, args.thresholds)
    top_samples = []
    scatter_store = {"true": [], "pred": []}
    rng = np.random.default_rng(args.seed)
    sample_offset = 0

    with torch.no_grad():
        for batch_idx, x in enumerate(tqdm(test_loader, desc="Evaluating DKF_pre")):
            x = x.to(device)
            target = inverse_log_transform(x[:, args.n_past :, 0])
            pred = predict_ensemble(model, x, args, batch_idx)

            update_metrics(metrics, pred, target, args.thresholds)
            update_scatter_store(scatter_store, target, pred, args, rng)
            update_top_samples(top_samples, x, target, pred, args, sample_offset)
            sample_offset += x.size(0)

    return finalize_metrics(metrics), top_samples, scatter_store


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
    return LinearSegmentedColormap.from_list("paper_precip", colors)


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
        f"DKF Precipitation Forecast - Test Sample {sample['index']}",
        fontsize=11,
    )

    for col, idx in enumerate(past_indices):
        lead = sample["past"].shape[0] - idx - 1
        title = "Input t" if lead == 0 else f"Input t-{lead}"
        im = add_precip_image(
            axes[0, col], past[idx], title, cmap, vmax
        )
    for col, idx in enumerate(future_indices):
        im = add_precip_image(axes[1, col], target[idx], f"Obs t+{idx + 1}", cmap, vmax)
        im = add_precip_image(axes[2, col], pred[idx], f"Pred t+{idx + 1}", cmap, vmax)

    row_labels = ["History", "Observed", "Predicted"]
    for row, label in enumerate(row_labels):
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
    axes[0].set_title("Prediction Error over Lead Time")

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
    ax.set_title("Categorical Forecast Scores by Rainfall Intensity")
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
        cbar = fig.colorbar(hb, ax=ax, fraction=0.045, pad=0.03, label="log10(count)")
        ax.plot([0, max_val], [0, max_val], "r--", linewidth=1.2, label="1:1 line")
        ax.set_xlim(0, max_val)
        ax.set_ylim(0, max_val)
    ax.set_xlabel("Observed precipitation (mm/h)")
    ax.set_ylabel("Predicted precipitation (mm/h)")
    ax.set_title(f"Observed vs Predicted Rainfall (r={metrics['Pearson r']:.3f})")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def format_value(value):
    if isinstance(value, (float, np.floating)) and math.isnan(value):
        return "N/A"
    if isinstance(value, (int, np.integer)):
        return str(value)
    return f"{float(value):.6f}"


def write_markdown(metrics, args, out_path):
    lines = [
        "# DKF_pre Precipitation Forecast Evaluation",
        "",
        f"- Checkpoint: `{args.checkpoint}`",
        f"- Precipitation files: `{args.precip_data_path}`",
        f"- Land data: `{args.land_data_path}`",
        f"- Split indices: train `< {args.train_end}`, validation `{args.train_end}:{args.val_end}`, test `>= {args.val_end}`",
        f"- Input frames: `{args.n_past}`",
        f"- Prediction frames: `{args.n_future}`",
        f"- Ensemble forward passes: `{args.num_ensembles}`",
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
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def print_metrics(metrics):
    print("\n=== DKF_pre Precipitation Evaluation ===")
    print(f"{'Metric':<14} {'Value':>14}")
    for metric in ["MSE", "RMSE", "MAE", "Bias", "Mean Obs", "Mean Pred", "Pearson r"]:
        print(f"{metric:<14} {format_value(metrics[metric]):>14}")

    print("\nMetrics per prediction step:")
    print(f"{'Step':<6} {'MAE':>12} {'RMSE':>12} {'Bias':>12}")
    for idx, (mae, rmse, bias) in enumerate(
        zip(metrics["step_mae"], metrics["step_rmse"], metrics["step_bias"]), start=1
    ):
        print(
            f"{idx:<6} {format_value(mae):>12} {format_value(rmse):>12} {format_value(bias):>12}"
        )

    print("\nThreshold-based scores:")
    print(
        f"{'Threshold':<12} {'POD':>10} {'FAR':>10} {'CSI':>10} {'F1':>10} {'FreqBias':>10}"
    )
    for threshold, scores in metrics["threshold_scores"].items():
        print(
            f">={threshold:g} mm/h  {format_value(scores['POD']):>10} "
            f"{format_value(scores['FAR']):>10} {format_value(scores['CSI']):>10} "
            f"{format_value(scores['F1']):>10} {format_value(scores['Frequency Bias']):>10}"
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate DKF_pre precipitation forecasts with paper-ready metrics and figures."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "DKF_pre" / "checkpoints" / "best_model.pth",
    )
    parser.add_argument(
        "--precip_data_path",
        type=Path,
        default=ROOT / "data" / "data_0511" / "nc_NZL" / "*.nc",
    )
    parser.add_argument(
        "--land_data_path",
        type=Path,
        default=ROOT / "data" / "data_0511" / "gebco0_1_land_only.nc",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=ROOT / "comparison_results" / "dkf_pre_precip",
    )
    parser.add_argument(
        "--precip_variable",
        type=str,
        default="GPM_3IMERGHH_07_precipitation",
    )
    parser.add_argument("--land_variable", type=str, default="elevation")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--n_past", type=int, default=10)
    parser.add_argument("--n_future", type=int, default=10)
    parser.add_argument("--train_end", type=int, default=750)
    parser.add_argument("--val_end", type=int, default=900)
    parser.add_argument("--num_samples", type=int, default=3)
    parser.add_argument("--num_ensembles", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=[0.1, 2.0, 5.0, 10.0, 30.0],
        help="Rainfall thresholds in mm/h for categorical forecast scores.",
    )
    parser.add_argument(
        "--scatter_points",
        type=int,
        default=50000,
        help="Maximum number of points used in the observed-vs-predicted plot.",
    )
    parser.add_argument(
        "--scatter_threshold",
        type=float,
        default=0.1,
        help="Keep scatter points where observed or predicted rain exceeds this value.",
    )
    parser.add_argument(
        "--max_test_frames",
        type=int,
        default=None,
        help="Optional cap for quick smoke tests. By default, use the full test split.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)

    device = get_device()
    print(f"Using device: {device}")

    test_loader, precip_data, precip_files = make_test_loader(args)
    print(f"Loaded precipitation frames: {len(precip_data)} from {len(precip_files)} files")
    print(f"Test samples: {len(test_loader.dataset)}")

    model, checkpoint = load_model(args, device)
    print(
        f"Loaded checkpoint: {args.checkpoint}, "
        f"epoch={checkpoint.get('epoch', 'N/A')}, loss={checkpoint.get('loss', 'N/A')}"
    )

    metrics, top_samples, scatter_store = evaluate(model, test_loader, args, device)
    print_metrics(metrics)

    metrics_path = args.output_dir / "metrics_summary.md"
    write_markdown(metrics, args, metrics_path)
    save_step_metrics_plot(metrics, args.output_dir / "step_metrics.png")
    save_threshold_scores_plot(metrics, args.output_dir / "threshold_scores.png")
    save_scatter_plot(scatter_store, metrics, args.output_dir / "true_vs_pred_scatter.png")

    for idx, sample in enumerate(top_samples):
        save_sample_figure(
            sample,
            args.output_dir / f"sample_{idx}_precip_forecast.png",
        )

    print(f"\nSaved markdown summary: {metrics_path}")
    print(f"Saved figures to: {args.output_dir}")


if __name__ == "__main__":
    main()
