#!/usr/bin/env python3
"""
Analyze SEMIT and FUNIT runs from epoch_metrics.csv files.

Expected files (adjust paths if needed):
    semit_epoch_metrics.csv
    funit_epoch_metrics.csv

Outputs:
    ./plots/*.png
"""

import os
import pandas as pd
import matplotlib.pyplot as plt


# ---------- config ----------

SEMIT_CSV = "results/semit_epoch_metrics.csv"
FUNIT_CSV = "results/funit_epoch_metrics.csv"
PLOT_DIR = "plots"

# Fraction of epochs we treat as "tail" for zooming loss plots
TAIL_FRAC_FOR_ZOOM = 0.8   # use last 80% of epochs for y-limit calculation


# ---------- helpers ----------

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_metrics(path: str, label: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"{label} metrics CSV not found at: {path}")
    df = pd.read_csv(path)
    if "epoch" not in df.columns:
        raise ValueError(f"{label} CSV must contain an 'epoch' column")
    return df


def plot_lines(xs, series_dict, title, xlabel, ylabel, out_path):
    """Generic multi-series plot with default y-limits."""
    plt.figure()
    for name, ys in series_dict.items():
        plt.plot(xs, ys, label=name)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_loss_lines(xs, series_dict, title, xlabel, ylabel, out_path):
    """
    Plot loss curves but automatically zoom the y-axis based on the
    later part of training, so if they converge near zero we see detail.
    """
    plt.figure()
    for name, ys in series_dict.items():
        plt.plot(xs, ys, label=name)

    # Compute zoomed y-limits from the tail of training
    import numpy as np

    xs_arr = np.array(xs)
    tail_start_idx = int(len(xs_arr) * (1.0 - TAIL_FRAC_FOR_ZOOM))
    tail_start_idx = max(0, min(tail_start_idx, len(xs_arr) - 1))

    all_tail_values = []
    for ys in series_dict.values():
        ys_arr = np.array(ys)
        all_tail_values.append(ys_arr[tail_start_idx:])

    if all_tail_values:
        all_tail_values = np.concatenate(all_tail_values)
        # Filter out NaNs just in case
        all_tail_values = all_tail_values[~np.isnan(all_tail_values)]
        if all_tail_values.size > 0:
            ymin = float(all_tail_values.min())
            ymax = float(all_tail_values.max())
            if ymax > ymin:
                # Add a small margin and clamp at 0
                margin = 0.1 * (ymax - ymin)
                ymin = max(0.0, ymin - margin)
                ymax = ymax + margin
                plt.ylim(ymin, ymax)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


# ---------- main plotting logic ----------

def plot_per_model(df: pd.DataFrame, model_name: str, out_dir: str) -> None:
    epochs = df["epoch"]

    # 1) Total losses (zoomed)
    plot_loss_lines(
        epochs,
        {
            "loss_dis_total": df["loss_dis_total"],
            "loss_gen_total": df["loss_gen_total"],
        },
        title=f"{model_name}: Generator & Discriminator Total Loss (zoomed)",
        xlabel="Epoch",
        ylabel="Loss",
        out_path=os.path.join(out_dir, f"{model_name.lower()}_loss_total_zoomed.png"),
    )

    # 2) Adversarial accuracies (full range)
    plot_lines(
        epochs,
        {
            "accuracy_dis_adv": df["accuracy_dis_adv"],
            "accuracy_gen_adv": df["accuracy_gen_adv"],
        },
        title=f"{model_name}: Adversarial Accuracies",
        xlabel="Epoch",
        ylabel="Accuracy",
        out_path=os.path.join(out_dir, f"{model_name.lower()}_adv_accuracy.png"),
    )

    # 3) Reconstruction losses: c, s, x (zoomed)
    plot_loss_lines(
        epochs,
        {
            "loss_gen_recon_c": df["loss_gen_recon_c"],
            "loss_gen_recon_s": df["loss_gen_recon_s"],
            "loss_gen_recon_x": df["loss_gen_recon_x"],
        },
        title=f"{model_name}: Reconstruction Losses (c, s, x, zoomed)",
        xlabel="Epoch",
        ylabel="Reconstruction loss",
        out_path=os.path.join(out_dir, f"{model_name.lower()}_recon_losses_zoomed.png"),
    )

    # 4) Individual recon losses (also zoomed)
    for key in ["loss_gen_recon_c", "loss_gen_recon_s", "loss_gen_recon_x"]:
        plot_loss_lines(
            epochs,
            {key: df[key]},
            title=f"{model_name}: {key} (zoomed)",
            xlabel="Epoch",
            ylabel="Loss",
            out_path=os.path.join(out_dir, f"{model_name.lower()}_{key}_zoomed.png"),
        )


def plot_comparisons(semit_df: pd.DataFrame, funit_df: pd.DataFrame, out_dir: str) -> None:
    # Align on overlapping epochs just in case
    common_epochs = sorted(set(semit_df["epoch"]).intersection(set(funit_df["epoch"])))
    if not common_epochs:
        print("No common epochs between SEMIT and FUNIT; skipping comparison plots.")
        return

    s = semit_df.set_index("epoch").loc[common_epochs]
    f = funit_df.set_index("epoch").loc[common_epochs]
    epochs = common_epochs

    # 1) Total losses (zoomed)
    plot_loss_lines(
        epochs,
        {
            "SEMIT loss_gen_total": s["loss_gen_total"],
            "FUNIT loss_gen_total": f["loss_gen_total"],
        },
        title="SEMIT vs FUNIT: Generator Total Loss (zoomed)",
        xlabel="Epoch",
        ylabel="Loss",
        out_path=os.path.join(out_dir, "compare_loss_gen_total_zoomed.png"),
    )

    plot_loss_lines(
        epochs,
        {
            "SEMIT loss_dis_total": s["loss_dis_total"],
            "FUNIT loss_dis_total": f["loss_dis_total"],
        },
        title="SEMIT vs FUNIT: Discriminator Total Loss (zoomed)",
        xlabel="Epoch",
        ylabel="Loss",
        out_path=os.path.join(out_dir, "compare_loss_dis_total_zoomed.png"),
    )

    # 2) Adversarial accuracies (full range)
    plot_lines(
        epochs,
        {
            "SEMIT accuracy_dis_adv": s["accuracy_dis_adv"],
            "FUNIT accuracy_dis_adv": f["accuracy_dis_adv"],
        },
        title="SEMIT vs FUNIT: Discriminator Adversarial Accuracy",
        xlabel="Epoch",
        ylabel="Accuracy",
        out_path=os.path.join(out_dir, "compare_accuracy_dis_adv.png"),
    )

    plot_lines(
        epochs,
        {
            "SEMIT accuracy_gen_adv": s["accuracy_gen_adv"],
            "FUNIT accuracy_gen_adv": f["accuracy_gen_adv"],
        },
        title="SEMIT vs FUNIT: Generator Adversarial Accuracy",
        xlabel="Epoch",
        ylabel="Accuracy",
        out_path=os.path.join(out_dir, "compare_accuracy_gen_adv.png"),
    )

    # 3) Reconstruction losses (zoomed)
    for key in ["loss_gen_recon_c", "loss_gen_recon_s", "loss_gen_recon_x"]:
        plot_loss_lines(
            epochs,
            {
                f"SEMIT {key}": s[key],
                f"FUNIT {key}": f[key],
            },
            title=f"SEMIT vs FUNIT: {key} (zoomed)",
            xlabel="Epoch",
            ylabel="Reconstruction loss",
            out_path=os.path.join(out_dir, f"compare_{key}_zoomed.png"),
        )


def main():
    ensure_dir(PLOT_DIR)

    # Load CSVs
    semit_df = load_metrics(SEMIT_CSV, "SEMIT")
    funit_df = load_metrics(FUNIT_CSV, "FUNIT")

    # Per-model plots
    plot_per_model(semit_df, "SEMIT", PLOT_DIR)
    plot_per_model(funit_df, "FUNIT", PLOT_DIR)

    # Cross-model comparison plots
    plot_comparisons(semit_df, funit_df, PLOT_DIR)

    print(f"Done. Plots written to: {os.path.abspath(PLOT_DIR)}")


if __name__ == "__main__":
    main()
