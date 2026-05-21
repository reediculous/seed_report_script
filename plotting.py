"""Plotting functions for seed experiment report."""

import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Optional

from stats_analysis import METRIC_TITLES

_FONTS_DIR = Path(__file__).resolve().parent / "fonts"


def _register_bundled_fonts() -> None:
    for name in ("DejaVuSans.ttf", "DejaVuSans-Bold.ttf"):
        path = _FONTS_DIR / name
        if path.is_file():
            fm.fontManager.addfont(str(path))


def _setup_style():
    """Grayscale theme for print-friendly (black and white) figures."""
    _register_bundled_fonts()
    sns.set_theme(style="whitegrid", font_scale=1.0)
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "black",
        "axes.labelcolor": "black",
        "text.color": "black",
        "xtick.color": "black",
        "ytick.color": "black",
        "grid.color": "#bfbfbf",
        "axes.prop_cycle": plt.cycler(
            color=["black", "#333333", "#555555", "#777777"]
        ),
    })
    sns.set_palette(["#4d4d4d", "#737373", "#999999", "#b3b3b3", "#cccccc"])


def _shared_bin_edges(
    values: np.ndarray, min_bins: int = 20, max_bins: int = 50,
) -> np.ndarray:
    """Freedman-Diaconis bin edges over `values`, lower bound at 0, count clamped to [min_bins, max_bins]."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size < 2 or np.ptp(v) == 0:
        hi = float(v.max()) if v.size else 1.0
        return np.linspace(0.0, max(1.0, hi), min_bins + 1)
    q75, q25 = np.percentile(v, [75, 25])
    iqr = q75 - q25
    h = 2 * iqr / (v.size ** (1 / 3)) if iqr > 0 else 0.0
    lo = max(0.0, float(v.min()))
    hi = float(v.max())
    if h <= 0 or hi <= lo:
        n_bins = min_bins
    else:
        n_bins = int(np.clip(np.ceil((hi - lo) / h), min_bins, max_bins))
    return np.linspace(lo, hi, n_bins + 1)


def plot_histogram_per_variant(
    analysis_df: pd.DataFrame,
    metric: str,
    variant_order: List[str],
    output_path: str,
) -> str:
    """Histogram per variant (pooled replicates) with shared bin edges and per-replicate rug.

    Returns path to saved figure.
    """
    _setup_style()
    n_variants = len(variant_order)
    n_cols = min(n_variants, 3)
    n_rows = (n_variants + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    if n_variants == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    title = METRIC_TITLES.get(metric, metric)
    fig.suptitle(f"Распределение: {title}", fontsize=14, y=1.02)

    all_vals = (
        analysis_df.loc[analysis_df["variant"].isin(variant_order), metric]
        .dropna()
        .to_numpy()
    )
    bin_edges = _shared_bin_edges(all_vals)

    rep_linestyles = ["-", "--", "-.", ":"]

    for i, variant in enumerate(variant_order):
        ax = axes[i]
        sub = analysis_df.loc[analysis_df["variant"] == variant]
        data = sub[metric].dropna()
        if len(data) == 0:
            ax.set_title(variant)
            continue

        ax.hist(
            data.to_numpy(), bins=bin_edges, density=True,
            color="#b8b8b8", edgecolor="black", linewidth=0.6, alpha=0.55,
            label=f"Все (N={len(data)})",
        )

        replicates = sorted(sub["replicate"].dropna().unique().tolist())
        for r_idx, rep in enumerate(replicates):
            rep_data = sub.loc[sub["replicate"] == rep, metric].dropna()
            if len(rep_data) == 0:
                continue
            ax.hist(
                rep_data.to_numpy(), bins=bin_edges, density=True,
                histtype="step", color="black", linewidth=1.3,
                linestyle=rep_linestyles[r_idx % len(rep_linestyles)],
                label=f"п{rep} (N={len(rep_data)})",
            )

        ax.set_title(f"{variant}\nN = {len(data)}")
        ax.set_xlabel(metric)
        ax.set_ylabel("Плотность")
        ax.set_xlim(left=max(0.0, float(bin_edges[0])))
        ax.grid(False)
        ax.legend(fontsize=7, loc="best", frameon=True)

    for i in range(n_variants, len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_kde_per_variant(
    analysis_df: pd.DataFrame,
    metric: str,
    variant_order: List[str],
    output_path: str,
) -> str:
    """KDE per variant (pooled replicates) with per-replicate overlays.

    Density is clipped to [0, +inf) since the metrics are physical lengths.
    Returns path to saved figure.
    """
    _setup_style()
    n_variants = len(variant_order)
    n_cols = min(n_variants, 3)
    n_rows = (n_variants + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    if n_variants == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    title = METRIC_TITLES.get(metric, metric)
    fig.suptitle(f"Распределение (KDE): {title}", fontsize=14, y=1.02)

    rep_linestyles = ["-", "--", "-.", ":"]

    for i, variant in enumerate(variant_order):
        ax = axes[i]
        sub = analysis_df.loc[analysis_df["variant"] == variant]
        data = sub[metric].dropna()
        if len(data) == 0:
            ax.set_title(variant)
            continue

        replicates = sorted(sub["replicate"].dropna().unique().tolist())
        for r_idx, rep in enumerate(replicates):
            rep_data = sub.loc[sub["replicate"] == rep, metric].dropna()
            if len(rep_data) < 2 or rep_data.nunique() < 2:
                continue
            try:
                ls = rep_linestyles[r_idx % len(rep_linestyles)]
                sns.kdeplot(
                    data=rep_data, ax=ax, fill=False,
                    clip=(0, None),
                    linewidth=1.1, color="black", linestyle=ls,
                    label=f"п{rep} (N={len(rep_data)})",
                )
            except Exception:
                pass

        if len(data) >= 2 and data.nunique() >= 2:
            sns.kdeplot(
                data=data, ax=ax, fill=True, alpha=0.22,
                clip=(0, None),
                linewidth=2.2, color="black",
                label=f"Все (N={len(data)})",
            )

        ax.set_title(f"{variant}\nN = {len(data)}")
        ax.set_xlabel(metric)
        ax.set_ylabel("Плотность")
        ax.set_xlim(left=0.0)
        ax.legend(fontsize=7, loc="best", frameon=True)

    for i in range(n_variants, len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_boxplots_per_variant(
    analysis_df: pd.DataFrame,
    metric: str,
    variant_order: List[str],
    output_path: str,
) -> str:
    """Box plots of a metric, one box per variant.

    Returns path to saved figure.
    """
    _setup_style()
    fig, ax = plt.subplots(figsize=(max(6, 2 * len(variant_order)), 5))

    title = METRIC_TITLES.get(metric, metric)
    ax.set_title(f"Боксплоты: {title}")

    plot_data = analysis_df[analysis_df["variant"].isin(variant_order)].copy()
    plot_data["variant"] = pd.Categorical(plot_data["variant"], categories=variant_order, ordered=True)

    n_var = len(variant_order)
    box_gray = sns.color_palette("Greys", n_colors=max(n_var + 1, 3))[1 : n_var + 1]
    sns.boxplot(
        data=plot_data, x="variant", y=metric, hue="variant", ax=ax,
        order=variant_order, palette=box_gray, width=0.5, legend=False,
    )
    ax.set_xlabel("Вариант")
    ax.set_ylabel(metric)

    plt.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_boxplots_per_replicate(
    analysis_df: pd.DataFrame,
    metric: str,
    variant_order: List[str],
    output_path: str,
) -> str:
    """Box plots showing individual replicates within each variant.

    Returns path to saved figure.
    """
    _setup_style()
    plot_data = analysis_df[analysis_df["variant"].isin(variant_order)].copy()
    plot_data["variant"] = pd.Categorical(plot_data["variant"], categories=variant_order, ordered=True)

    fig, ax = plt.subplots(figsize=(max(8, 2.5 * len(variant_order)), 5))

    title = METRIC_TITLES.get(metric, metric)
    ax.set_title(f"Боксплоты по повторностям: {title}")

    n_hue = int(plot_data["replicate"].nunique()) if len(plot_data) else 1
    rep_grays = sns.color_palette("Greys", n_colors=max(n_hue + 2, 4))[2:]
    sns.boxplot(
        data=plot_data, x="variant", y=metric, hue="replicate",
        ax=ax, order=variant_order, palette=rep_grays, width=0.7,
    )
    ax.set_xlabel("Вариант")
    ax.set_ylabel(metric)
    ax.legend(
        title="Повторность",
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
        frameon=True,
    )

    plt.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_summary_with_cld(
    analysis_df: pd.DataFrame,
    metric: str,
    variant_order: List[str],
    cld_letters: Dict[str, str],
    output_path: str,
) -> str:
    """Horizontal bar chart of variant means with Tukey HSD CLD letters."""
    _setup_style()
    plot_data = analysis_df[analysis_df["variant"].isin(variant_order)].copy().dropna(subset=[metric])

    means = plot_data.groupby("variant")[metric].mean()
    sems = plot_data.groupby("variant")[metric].sem()

    ordered = [v for v in variant_order if v in means.index]

    y_pos = np.arange(len(ordered))
    vals = [means[v] for v in ordered]
    errs = [sems[v] if not np.isnan(sems[v]) else 0.0 for v in ordered]

    fig_h = max(3.5, 0.5 * len(ordered) + 1.5)
    fig, ax = plt.subplots(figsize=(8, fig_h))
    bar_gray = "#c4c4c4"
    ax.barh(
        y_pos, vals, xerr=errs,
        color=bar_gray, edgecolor="black", linewidth=0.7, capsize=3,
    )
    ax.set_yticks(y_pos)
    ax.set_yticklabels(ordered)

    title = METRIC_TITLES.get(metric, metric)
    ax.set_title(f"Средние по вариантам с группами Tukey HSD — {title}")
    ax.set_xlabel(metric)
    ax.set_ylabel("Вариант")

    x_max = max(vals[i] + errs[i] for i in range(len(vals))) if vals else 1.0
    offset = 0.02 * (x_max if x_max > 0 else 1.0)
    for i, v in enumerate(ordered):
        letter = cld_letters.get(v, "")
        if letter:
            ax.text(vals[i] + errs[i] + offset, y_pos[i], letter,
                    va="center", ha="left", fontsize=10, fontweight="bold")

    plt.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def generate_all_plots(
    analysis_df: pd.DataFrame,
    variant_order: List[str],
    output_dir: str,
    metrics: Optional[List[str]] = None,
    cld_by_metric: Optional[Dict[str, Dict[str, str]]] = None,
) -> Dict[str, Dict[str, str]]:
    """Generate all plots for all metrics.

    Returns nested dict: metric -> plot_type -> filepath.
    """
    os.makedirs(output_dir, exist_ok=True)

    if metrics is None:
        from stats_analysis import METRICS
        metrics = METRICS

    all_paths: Dict[str, Dict[str, str]] = {}

    for metric in metrics:
        safe_name = metric.replace(".", "_")
        paths = {}

        paths["hist"] = plot_histogram_per_variant(
            analysis_df, metric, variant_order,
            os.path.join(output_dir, f"{safe_name}_hist.png"),
        )
        paths["kde"] = plot_kde_per_variant(
            analysis_df, metric, variant_order,
            os.path.join(output_dir, f"{safe_name}_kde.png"),
        )
        paths["boxplot"] = plot_boxplots_per_variant(
            analysis_df, metric, variant_order,
            os.path.join(output_dir, f"{safe_name}_boxplot.png"),
        )
        paths["boxplot_replicate"] = plot_boxplots_per_replicate(
            analysis_df, metric, variant_order,
            os.path.join(output_dir, f"{safe_name}_boxplot_rep.png"),
        )

        if cld_by_metric is not None and metric in cld_by_metric:
            paths["summary_cld"] = plot_summary_with_cld(
                analysis_df, metric, variant_order, cld_by_metric[metric],
                os.path.join(output_dir, f"{safe_name}_summary_cld.png"),
            )

        all_paths[metric] = paths

    return all_paths
