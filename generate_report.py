#!/usr/bin/env python3
"""Main entry point: generate a seed experiment report from raw data."""

import argparse
import os
import sys

from data_parser import parse_all, build_analysis_dataframe, sorted_variants, germination_stats
from stats_analysis import compute_all_statistics
from plotting import generate_all_plots
from report_builder import build_report


def main():
    parser = argparse.ArgumentParser(description="Generate seed experiment report")
    parser.add_argument(
        "--raw-dir", default="raw_data",
        help="Directory containing raw .txt experiment files (default: raw_data)",
    )
    parser.add_argument(
        "--output", default="output/report.pdf",
        help="Output PDF path (default: output/report.pdf)",
    )
    parser.add_argument(
        "--title", default="Отчёт по эксперименту",
        help="Report title",
    )
    args = parser.parse_args()

    raw_dir = args.raw_dir
    output_path = args.output
    title = args.title

    if not os.path.isdir(raw_dir):
        print(f"Error: raw data directory '{raw_dir}' not found.", file=sys.stderr)
        sys.exit(1)

    print(f"Parsing files from {raw_dir}...")
    file_datas = parse_all(raw_dir)
    if not file_datas:
        print("Error: no .txt files found in raw data directory.", file=sys.stderr)
        sys.exit(1)

    for fd in file_datas:
        print(f"  {fd.filename}: variant={fd.variant}, replicate={fd.replicate}, "
              f"seeds={len(fd.seeds)}, failed={fd.failed_count}")

    print("\nBuilding analysis DataFrame...")
    analysis_df = build_analysis_dataframe(file_datas)
    variant_order = sorted_variants(analysis_df["variant"].unique().tolist())
    print(f"  {len(analysis_df)} seeds across {len(variant_order)} variants: {variant_order}")

    print("\nComputing statistics...")
    all_stats = compute_all_statistics(analysis_df, variant_order)
    for metric, ms in all_stats.items():
        desc = ms.descriptive
        if desc:
            means = ", ".join(f"{d.variant}={d.mean}" for d in desc)
            print(f"  {metric}: {means}")

    output_dir = os.path.dirname(output_path) or "."
    plots_dir = os.path.join(output_dir, "plots")

    print(f"\nGenerating plots in {plots_dir}...")
    cld_by_metric = {
        metric: (ms.anova.cld_letters if ms.anova and ms.anova.available else {})
        for metric, ms in all_stats.items()
    }
    all_plots = generate_all_plots(
        analysis_df, variant_order, plots_dir, cld_by_metric=cld_by_metric,
    )
    total_plots = sum(len(p) for p in all_plots.values())
    print(f"  {total_plots} plots generated")

    print("\nВсхожесть по файлам:")
    for fd in file_datas:
        g = germination_stats(fd)
        pct = 100.0 * g.rate if g.total > 0 else float("nan")
        pct_s = f"{pct:.2f}%" if g.total > 0 else "—"
        print(f"  {fd.filename}: {g.alive}/{g.total} ({pct_s})")

    print(f"\nBuilding PDF report: {output_path}...")
    build_report(file_datas, all_stats, all_plots, output_path, experiment_title=title)
    print(f"Report saved to: {output_path}")


if __name__ == "__main__":
    main()
