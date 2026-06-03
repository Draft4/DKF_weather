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


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

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


def inverse_log_transform(x):
    return torch.expm1(x).clamp_min(0.0)


def safe_div(num, den):
    if den == 0:
        return float("nan")
    return num / den


def format_value(value):
    if isinstance(value, (float, np.floating)) and math.isnan(value):
        return "N/A"
    if isinstance(value, (int, np.integer)):
        return str(value)
    return f"{float(value):.6f}"


# ---------------------------------------------------------------------------
# Precipitation colormap
# ---------------------------------------------------------------------------

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
    ax.set_title(title, fontsize=7)
    ax.set_xticks([])
    ax.set_yticks([])
    return im


# ---------------------------------------------------------------------------
# Data loading  (unified – uses DKF_pre WeatherDataset for 2-channel sequences)
# ---------------------------------------------------------------------------

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
    WeatherDataset = load_class(ROOT / "DKF_pre" / "dataset.py", "WeatherDataset")
    precip_data, precip_files = load_precip_tensor(
        args.precip_data_path, args.precip_variable
    )
    land_data = load_land_tensor(args.land_data_path, args.land_variable)

    if args.val_end + args.n_past + args.n_future >= len(precip_data):
        raise ValueError("val_end leaves too few frames for the test split.")

    test_precip_data = precip_data[args.val_end:]
    if args.max_test_frames is not None:
        test_precip_data = test_precip_data[: args.max_test_frames]
    if len(test_precip_data) <= args.n_past + args.n_future:
        raise ValueError("Test split is too short for the configured sequence length.")

    test_dataset = WeatherDataset(
        test_precip_data, land_data,
        n_past=args.n_past, n_future=args.n_future,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return test_loader, precip_data, precip_files


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_dkf_model(args, device):
    DKF = load_class(ROOT / "DKF_pre" / "modules.py", "DeepKalmanFilter")
    model = DKF().to(device)
    checkpoint = torch.load(args.dkf_checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def load_convlstm_model(args, device):
    PrecipConvLSTM = load_class(ROOT / "ConvLSTM_pre" / "modules.py", "PrecipConvLSTM")
    model = PrecipConvLSTM(
        hidden_dim_list=args.hidden_dim_list,
        patch_size=args.patch_size,
    ).to(device)
    checkpoint = torch.load(args.convlstm_checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


# ---------------------------------------------------------------------------
# Prediction wrappers
# ---------------------------------------------------------------------------

def dkf_predict(model, x, args, batch_idx):
    """Ensemble prediction for DKF."""
    preds = []
    for ensemble_idx in range(args.num_ensembles):
        set_seed(args.seed + batch_idx * args.num_ensembles + ensemble_idx)
        pred_log = model(x, args.n_past, args.n_future, mode="predict").squeeze(2)
        preds.append(inverse_log_transform(pred_log))
    return torch.stack(preds, dim=0).mean(dim=0)


def convlstm_predict(model, x_past, args):
    """Single-pass prediction for ConvLSTM."""
    pred_log = model(x_past, args.n_future)
    return inverse_log_transform(pred_log.squeeze(2))


# ---------------------------------------------------------------------------
# Metrics accumulators
# ---------------------------------------------------------------------------

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
    metrics["step_sum_abs_error"] += error.abs().sum(dim=(0, 2, 3)).detach().cpu().numpy()
    metrics["step_sum_sq_error"] += (error * error).sum(dim=(0, 2, 3)).detach().cpu().numpy()

    for threshold in thresholds:
        pred_event = pred >= threshold
        true_event = target >= threshold
        counts = metrics["thresholds"][threshold]
        counts["tp"] += int((pred_event & true_event).sum().item())
        counts["fp"] += int((pred_event & ~true_event).sum().item())
        counts["fn"] += int((~pred_event & true_event).sum().item())
        counts["tn"] += int((~pred_event & ~true_event).sum().item())


def finalize_threshold_scores(counts):
    tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
    pod = safe_div(tp, tp + fn)
    far = safe_div(fp, tp + fp)
    csi = safe_div(tp, tp + fp + fn)
    precision = safe_div(tp, tp + fp)
    recall = pod
    f1 = safe_div(2 * precision * recall, precision + recall)
    fbias = safe_div(tp + fp, tp + fn)
    acc = safe_div(tp + counts["tn"], tp + fp + fn + counts["tn"])
    return {
        "TP": tp, "FP": fp, "FN": fn, "TN": counts["tn"],
        "POD": pod, "FAR": far, "CSI": csi,
        "Precision": precision, "Recall": recall, "F1": f1,
        "Frequency Bias": fbias, "Accuracy": acc,
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
        t: finalize_threshold_scores(counts)
        for t, counts in metrics["thresholds"].items()
    }
    return {
        "MSE": mse, "RMSE": math.sqrt(mse),
        "MAE": mae, "Bias": bias,
        "Mean Obs": mean_true, "Mean Pred": mean_pred,
        "Pearson r": pearson_r,
        "step_mse": step_mse, "step_rmse": np.sqrt(step_mse),
        "step_mae": step_mae, "step_bias": step_bias,
        "threshold_scores": threshold_scores,
    }


# ---------------------------------------------------------------------------
# Scatter store
# ---------------------------------------------------------------------------

def update_scatter_store(store, target, pred, args, rng):
    if len(store["true"]) >= args.scatter_points:
        return
    true_np = target.detach().cpu().numpy().reshape(-1)
    pred_np = pred.detach().cpu().numpy().reshape(-1)
    mask = (true_np >= args.scatter_threshold) | (pred_np >= args.scatter_threshold)
    if mask.any():
        true_np, pred_np = true_np[mask], pred_np[mask]
    needed = args.scatter_points - len(store["true"])
    take = min(needed, len(true_np))
    if take <= 0:
        return
    indices = rng.choice(len(true_np), size=take, replace=False)
    store["true"].extend(true_np[indices].tolist())
    store["pred"].extend(pred_np[indices].tolist())


# ---------------------------------------------------------------------------
# Sample collection (shared across models – the *same* precipitation events)
# ---------------------------------------------------------------------------

def update_top_samples(samples, x_past, target, conv_pred, dkf_pred, args, offset):
    past = inverse_log_transform(x_past[:, :, 0])
    scores = target.sum(dim=(1, 2, 3)).detach().cpu().numpy()
    for i, score in enumerate(scores):
        samples.append({
            "score": float(score),
            "index": offset + i,
            "past": past[i].detach().cpu(),
            "target": target[i].detach().cpu(),
            "ConvLSTM": conv_pred[i].detach().cpu(),
            "DKF": dkf_pred[i].detach().cpu(),
        })
    samples.sort(key=lambda item: item["score"], reverse=True)
    del samples[args.num_samples:]


# ---------------------------------------------------------------------------
# Unified evaluation loop
# ---------------------------------------------------------------------------

def evaluate(dkf_model, conv_model, test_loader, args, device):
    dkf_metrics = init_metrics(args.n_future, args.thresholds)
    conv_metrics = init_metrics(args.n_future, args.thresholds)
    top_samples = []
    dkf_scatter = {"true": [], "pred": []}
    conv_scatter = {"true": [], "pred": []}
    rng = np.random.default_rng(args.seed)
    sample_offset = 0

    with torch.no_grad():
        for batch_idx, x in enumerate(
            tqdm(test_loader, desc="Comparing DKF vs ConvLSTM")
        ):
            x = x.to(device)
            x_past = x[:, : args.n_past]                  # (B, n_past, 2, H, W)
            target = inverse_log_transform(x[:, args.n_past :, 0])  # (B, n_future, H, W)

            dkf_pred = dkf_predict(dkf_model, x, args, batch_idx)
            conv_pred = convlstm_predict(conv_model, x_past, args)

            update_metrics(dkf_metrics, dkf_pred, target, args.thresholds)
            update_metrics(conv_metrics, conv_pred, target, args.thresholds)
            update_scatter_store(dkf_scatter, target, dkf_pred, args, rng)
            update_scatter_store(conv_scatter, target, conv_pred, args, rng)
            update_top_samples(
                top_samples, x_past, target, conv_pred, dkf_pred, args, sample_offset
            )
            sample_offset += x.size(0)

    return (
        finalize_metrics(conv_metrics),
        finalize_metrics(dkf_metrics),
        top_samples,
        conv_scatter,
        dkf_scatter,
    )


# ---------------------------------------------------------------------------
# Figures  (sized for dual-column paper)
# ---------------------------------------------------------------------------

# Helper: single-column ≈ 3.5″, full-width ≈ 7.2″

def image_np(x):
    return x.squeeze(0).numpy()


def save_precip_forecast_comparison(sample, out_path, sample_idx):
    """4-row comparison: Input / GT / ConvLSTM / DKF  —  full-width figure."""
    past_indices = np.linspace(0, sample["past"].shape[0] - 1, 5, dtype=int)
    future_indices = np.array([0, 2, 4, 6, sample["target"].shape[0] - 1])
    cmap = create_precip_cmap()

    past = sample["past"].numpy()
    target = sample["target"].numpy()
    conv = sample["ConvLSTM"].numpy()
    dkf = sample["DKF"].numpy()

    all_arrays = [
        past[past_indices],
        target[future_indices],
        conv[future_indices],
        dkf[future_indices],
    ]
    vmax = figure_vmax(*all_arrays)

    fig, axes = plt.subplots(4, 5, figsize=(7.2, 5.5), constrained_layout=True)
    fig.suptitle(
        f"Precipitation Forecast Comparison – Sample {sample_idx}",
        fontsize=10, fontweight="bold",
    )

    for col, idx in enumerate(past_indices):
        lead = sample["past"].shape[0] - idx - 1
        title = "Input t" if lead == 0 else f"Input t−{lead}"
        add_precip_image(axes[0, col], past[idx], title, cmap, vmax)

    for col, idx in enumerate(future_indices):
        add_precip_image(axes[1, col], target[idx], f"Obs t+{idx + 1}", cmap, vmax)
        add_precip_image(axes[2, col], conv[idx], f"Pred t+{idx + 1}", cmap, vmax)
        add_precip_image(axes[3, col], dkf[idx], f"Pred t+{idx + 1}", cmap, vmax)

    for row, label in enumerate(["Input", "Observed", "ConvLSTM", "DKF"]):
        axes[row, 0].set_ylabel(label, fontsize=8, rotation=0, labelpad=32, va="center")

    cbar = fig.colorbar(
        axes[3, -1].images[0], ax=axes.ravel().tolist(),
        shrink=0.82, pad=0.02,
    )
    cbar.set_label("Precipitation (mm/h)", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_step_comparison_plot(conv_m, dkf_m, out_path):
    """Per-step MAE / RMSE / Bias  —  full-width 3-panel figure."""
    steps = np.arange(1, len(conv_m["step_mae"]) + 1)
    fig, axes = plt.subplots(3, 1, figsize=(7.2, 6.0), sharex=True)

    colors = {"ConvLSTM": "#0072B2", "DKF": "#D55E00"}
    for model_name, metrics, ls in [
        ("ConvLSTM", conv_m, "-"), ("DKF", dkf_m, "--"),
    ]:
        axes[0].plot(steps, metrics["step_mae"], marker="o", linewidth=1.5,
                     linestyle=ls, color=colors[model_name], label=model_name)
        axes[1].plot(steps, metrics["step_rmse"], marker="s", linewidth=1.5,
                     linestyle=ls, color=colors[model_name], label=model_name)
        axes[2].plot(steps, metrics["step_bias"], marker="^", linewidth=1.5,
                     linestyle=ls, color=colors[model_name], label=model_name)

    axes[0].set_ylabel("MAE (mm/h)", fontsize=9)
    axes[1].set_ylabel("RMSE (mm/h)", fontsize=9)
    axes[2].set_ylabel("Bias (mm/h)", fontsize=9)
    axes[2].set_xlabel("Prediction step", fontsize=9)
    axes[2].axhline(0.0, color="black", linewidth=0.8, alpha=0.4)

    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.set_xticks(steps)
        ax.legend(fontsize=8, loc="upper left")

    fig.suptitle("Prediction Error over Lead Time", fontsize=10, fontweight="bold")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_threshold_comparison_plot(conv_m, dkf_m, out_path):
    """Categorical scores at thresholds  —  single-column figure."""
    thresholds = list(conv_m["threshold_scores"].keys())
    labels = [f"≥{t:g}" for t in thresholds]
    x = np.arange(len(thresholds))

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.0), sharex=True)
    score_keys = [
        ("POD", "Probability of Detection"),
        ("FAR", "False Alarm Ratio"),
        ("CSI", "Critical Success Index"),
        ("F1", "F1 Score"),
    ]

    colors = {"ConvLSTM": "#0072B2", "DKF": "#D55E00"}
    for (key, title), ax in zip(score_keys, axes.flat):
        for model_name, metrics, marker in [
            ("ConvLSTM", conv_m, "o"), ("DKF", dkf_m, "s"),
        ]:
            values = [metrics["threshold_scores"][t][key] for t in thresholds]
            ax.plot(x, values, marker=marker, linewidth=1.5,
                    color=colors[model_name], label=model_name)
        ax.set_title(title, fontsize=9)
        ax.set_ylabel("Score", fontsize=8)
        ax.set_ylim(-0.03, 1.03)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=7)

    axes[1, 0].set_xlabel("Rainfall threshold (mm/h)", fontsize=9)
    axes[1, 1].set_xlabel("Rainfall threshold (mm/h)", fontsize=9)
    fig.suptitle("Categorical Forecast Scores", fontsize=10, fontweight="bold")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_scatter_comparison(conv_store, dkf_store, conv_m, dkf_m, out_path):
    """Side-by-side hexbin scatter  —  full-width figure."""
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.5))

    for ax, store, metrics, name in [
        (axes[0], conv_store, conv_m, "ConvLSTM"),
        (axes[1], dkf_store, dkf_m, "DKF"),
    ]:
        true = np.asarray(store["true"], dtype=np.float32)
        pred = np.asarray(store["pred"], dtype=np.float32)
        if len(true) > 0:
            max_val = max(float(true.max()), float(pred.max()), 1.0)
            hb = ax.hexbin(true, pred, gridsize=65, mincnt=1, bins="log", cmap="viridis")
            cbar = fig.colorbar(hb, ax=ax, fraction=0.045, pad=0.03)
            cbar.set_label("log10(count)", fontsize=7)
            cbar.ax.tick_params(labelsize=6)
            ax.plot([0, max_val], [0, max_val], "r--", linewidth=1.0, label="1:1")
            ax.set_xlim(0, max_val)
            ax.set_ylim(0, max_val)
        ax.set_xlabel("Observed (mm/h)", fontsize=8)
        ax.set_ylabel("Predicted (mm/h)", fontsize=8)
        ax.set_title(f"{name}  (r = {metrics['Pearson r']:.3f})", fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc="upper left")

    fig.suptitle("Observed vs Predicted Precipitation", fontsize=10, fontweight="bold")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_summary_bar(conv_m, dkf_m, out_path):
    """Grouped bar chart of summary metrics  —  single-column figure."""
    names = ["MSE", "RMSE", "MAE", "Bias", "Pearson r"]
    x = np.arange(len(names))
    width = 0.36

    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    conv_vals = [conv_m[n] for n in names]
    dkf_vals = [dkf_m[n] for n in names]
    bars1 = ax.bar(x - width / 2, conv_vals, width, label="ConvLSTM",
                   color="#0072B2", edgecolor="white", linewidth=0.5)
    bars2 = ax.bar(x + width / 2, dkf_vals, width, label="DKF",
                   color="#D55E00", edgecolor="white", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=8)
    ax.set_ylabel("Value", fontsize=9)
    ax.set_title("Summary Metrics Comparison", fontsize=10, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8)

    # Annotate bar values
    for bar in bars1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h, f"{h:.3f}",
                ha="center", va="bottom", fontsize=6, rotation=90)
    for bar in bars2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h, f"{h:.3f}",
                ha="center", va="bottom", fontsize=6, rotation=90)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------

def write_markdown(conv_m, dkf_m, args, out_path):
    better = {}
    for m in ["MSE", "RMSE", "MAE", "Bias"]:
        better[m] = "DKF" if abs(dkf_m[m]) < abs(conv_m[m]) else "ConvLSTM"
    better["Pearson r"] = "DKF" if dkf_m["Pearson r"] > conv_m["Pearson r"] else "ConvLSTM"

    lines = [
        "# DKF vs ConvLSTM — Precipitation Forecast Comparison",
        "",
        f"- DKF checkpoint: `{args.dkf_checkpoint}`",
        f"- ConvLSTM checkpoint: `{args.convlstm_checkpoint}`",
        f"- Precipitation files: `{args.precip_data_path}`",
        f"- Land data: `{args.land_data_path}`",
        f"- Test split: indices ≥ `{args.val_end}`",
        f"- Input frames: `{args.n_past}`",
        f"- Prediction frames: `{args.n_future}`",
        f"- DKF ensemble passes: `{args.num_ensembles}`",
        "",
        "## Summary Metrics",
        "",
        "| Metric | ConvLSTM | DKF | Better |",
        "|---|---:|---:|---|",
    ]
    for metric in ["MSE", "RMSE", "MAE", "Bias", "Pearson r"]:
        lines.append(
            f"| {metric} | {format_value(conv_m[metric])} | "
            f"{format_value(dkf_m[metric])} | {better[metric]} |"
        )

    # Per-step tables
    for metric_label, key in [
        ("MAE (mm/h)", "step_mae"),
        ("RMSE (mm/h)", "step_rmse"),
        ("Bias (mm/h)", "step_bias"),
    ]:
        lines.extend([
            "",
            f"## {metric_label} Per Prediction Step",
            "",
            "| Step | ConvLSTM | DKF |",
            "|---:|---:|---:|",
        ])
        for idx, (cv, dk) in enumerate(
            zip(conv_m[key], dkf_m[key]), start=1
        ):
            lines.append(f"| {idx} | {format_value(cv)} | {format_value(dk)} |")

    # Threshold scores
    lines.extend([
        "",
        "## Threshold-Based Scores",
        "",
        "### ConvLSTM",
        "",
        "| Thresh (mm/h) | TP | FP | FN | TN | POD | FAR | CSI | F1 | Bias |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for t, s in conv_m["threshold_scores"].items():
        lines.append(
            f"| ≥{t:g} | {s['TP']} | {s['FP']} | {s['FN']} | {s['TN']} | "
            f"{format_value(s['POD'])} | {format_value(s['FAR'])} | "
            f"{format_value(s['CSI'])} | {format_value(s['F1'])} | "
            f"{format_value(s['Frequency Bias'])} |"
        )

    lines.extend([
        "",
        "### DKF",
        "",
        "| Thresh (mm/h) | TP | FP | FN | TN | POD | FAR | CSI | F1 | Bias |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for t, s in dkf_m["threshold_scores"].items():
        lines.append(
            f"| ≥{t:g} | {s['TP']} | {s['FP']} | {s['FN']} | {s['TN']} | "
            f"{format_value(s['POD'])} | {format_value(s['FAR'])} | "
            f"{format_value(s['CSI'])} | {format_value(s['F1'])} | "
            f"{format_value(s['Frequency Bias'])} |"
        )

    lines.extend([
        "",
        "## Notes",
        "",
        "- Continuous metrics are computed in the original precipitation scale after `expm1`.",
        "- Bias = prediction − observation; negative values → underestimation.",
        "- POD = probability of detection; FAR = false alarm ratio; CSI = critical success index.",
        "- Threshold scores are computed pixel-wise over all predicted future frames.",
        "- DKF uses 5-ensemble prediction; ConvLSTM uses a single deterministic pass.",
    ])

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Terminal output
# ---------------------------------------------------------------------------

def print_comparison(conv_m, dkf_m):
    print("\n" + "=" * 62)
    print("  DKF vs ConvLSTM — Precipitation Forecast Comparison")
    print("=" * 62)
    print(f"{'Metric':<14} {'ConvLSTM':>14} {'DKF':>14} {'Better':>12}")
    print("-" * 56)
    for metric in ["MSE", "RMSE", "MAE", "Bias", "Pearson r"]:
        b = "DKF" if (metric == "Pearson r" and dkf_m[metric] > conv_m[metric]) \
            or (metric != "Pearson r" and abs(dkf_m[metric]) < abs(conv_m[metric])) \
            else "ConvLSTM"
        print(f"{metric:<14} {format_value(conv_m[metric]):>14} "
              f"{format_value(dkf_m[metric]):>14} {b:>12}")

    print("\nPer-step MAE:")
    print(f"{'Step':<6} {'ConvLSTM':>14} {'DKF':>14}")
    for idx, (c, d) in enumerate(zip(conv_m["step_mae"], dkf_m["step_mae"]), 1):
        print(f"{idx:<6} {format_value(c):>14} {format_value(d):>14}")

    print("\nThreshold scores (POD / FAR / CSI / F1):")
    print(f"{'Thresh':<10} {'Conv POD':>10} {'DKF POD':>10} "
          f"{'Conv CSI':>10} {'DKF CSI':>10}")
    for t in conv_m["threshold_scores"]:
        cs = conv_m["threshold_scores"][t]
        ds = dkf_m["threshold_scores"][t]
        print(f"≥{t:<9g} {format_value(cs['POD']):>10} {format_value(ds['POD']):>10} "
              f"{format_value(cs['CSI']):>10} {format_value(ds['CSI']):>10}")


# ---------------------------------------------------------------------------
# CLI and main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Comprehensive DKF vs ConvLSTM comparison on precipitation data."
    )
    parser.add_argument(
        "--dkf_checkpoint", type=Path,
        default=ROOT / "DKF_pre" / "checkpoints" / "best_model.pth",
    )
    parser.add_argument(
        "--convlstm_checkpoint", type=Path,
        default=ROOT / "ConvLSTM_pre" / "checkpoints" / "best_model.pth",
    )
    parser.add_argument(
        "--precip_data_path", type=Path,
        default=ROOT / "data" / "data_0511" / "nc_NZL" / "*.nc",
    )
    parser.add_argument(
        "--land_data_path", type=Path,
        default=ROOT / "data" / "data_0511" / "gebco0_1_land_only.nc",
    )
    parser.add_argument(
        "--output_dir", type=Path,
        default=ROOT / "comparison_results" / "precip_comparison",
    )
    parser.add_argument("--precip_variable", type=str,
                        default="GPM_3IMERGHH_07_precipitation")
    parser.add_argument("--land_variable", type=str, default="elevation")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--n_past", type=int, default=10)
    parser.add_argument("--n_future", type=int, default=10)
    parser.add_argument("--val_end", type=int, default=900)
    parser.add_argument("--num_samples", type=int, default=3)
    parser.add_argument("--num_ensembles", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patch_size", type=int, default=4)
    parser.add_argument("--hidden_dim_list", type=int, nargs="+",
                        default=[128, 64, 64])
    parser.add_argument("--thresholds", type=float, nargs="+",
                        default=[0.1, 2.0, 5.0, 10.0, 30.0])
    parser.add_argument("--scatter_points", type=int, default=50000)
    parser.add_argument("--scatter_threshold", type=float, default=0.1)
    parser.add_argument("--max_test_frames", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)

    device = get_device()
    print(f"Using device: {device}")

    # Data
    test_loader, precip_data, precip_files = make_test_loader(args)
    print(f"Loaded {len(precip_data)} precip frames from {len(precip_files)} files")
    print(f"Test samples: {len(test_loader.dataset)}")

    # Models
    dkf, dkf_ckpt = load_dkf_model(args, device)
    print(f"DKF checkpoint: epoch={dkf_ckpt.get('epoch', 'N/A')}, "
          f"loss={dkf_ckpt.get('loss', 'N/A'):.6f}"
          if isinstance(dkf_ckpt.get('loss'), float) else
          f"DKF checkpoint: epoch={dkf_ckpt.get('epoch', 'N/A')}")

    conv, conv_ckpt = load_convlstm_model(args, device)
    print(f"ConvLSTM checkpoint: epoch={conv_ckpt.get('epoch', 'N/A')}, "
          f"loss={conv_ckpt.get('loss', 'N/A'):.6f}"
          if isinstance(conv_ckpt.get('loss'), float) else
          f"ConvLSTM checkpoint: epoch={conv_ckpt.get('epoch', 'N/A')}")

    # Evaluate
    conv_m, dkf_m, samples, conv_scatter, dkf_scatter = evaluate(
        dkf, conv, test_loader, args, device
    )
    print_comparison(conv_m, dkf_m)

    # Save markdown
    md_path = args.output_dir / "metrics_summary.md"
    write_markdown(conv_m, dkf_m, args, md_path)

    # Save figures
    save_step_comparison_plot(conv_m, dkf_m,
                              args.output_dir / "step_metrics_comparison.png")
    save_threshold_comparison_plot(conv_m, dkf_m,
                                   args.output_dir / "threshold_scores_comparison.png")
    save_scatter_comparison(conv_scatter, dkf_scatter, conv_m, dkf_m,
                            args.output_dir / "true_vs_pred_scatter_comparison.png")
    save_summary_bar(conv_m, dkf_m,
                     args.output_dir / "summary_metrics_bar.png")

    for idx, sample in enumerate(samples):
        save_precip_forecast_comparison(
            sample,
            args.output_dir / f"sample_{idx}_precip_forecast_comparison.png",
            sample["index"],
        )

    print(f"\nResults saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
