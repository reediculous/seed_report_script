"""PDF report builder using fpdf2 with Cyrillic support."""

import os
import re
import numpy as np
import pandas as pd
from fpdf import FPDF
from typing import List, Dict, Optional

from data_parser import (
    FileData,
    GerminationStats,
    build_seed_dataframe,
    germination_stats,
    sorted_variants,
    variant_display_sort_key,
)
from stats_analysis import (
    MetricStatistics, METRICS, METRIC_TITLES,
    WilcoxonResult, NormalityResult, DescriptiveStats, AnovaResult,
)


FONT_PATH = "/System/Library/Fonts/Supplemental/Arial Unicode.ttf"
FONT_BOLD_PATH = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"

# Figure captions under embedded charts (display only)
FIGURE_CAPTION_FONT_PT = 11
FIGURE_CAPTION_LINE_HEIGHT_MM = 5

TABLE_NUM = 0


def _next_table_num() -> int:
    global TABLE_NUM
    TABLE_NUM += 1
    return TABLE_NUM


class ReportPDF(FPDF):
    def __init__(self):
        super().__init__(orientation="L", unit="mm", format="A4")
        self.set_auto_page_break(auto=True, margin=15)

        if os.path.exists(FONT_PATH):
            self.add_font("ArialUni", "", FONT_PATH, uni=True)
        if os.path.exists(FONT_BOLD_PATH):
            self.add_font("ArialUni", "B", FONT_BOLD_PATH, uni=True)

        self._font_name = "ArialUni" if os.path.exists(FONT_PATH) else "Helvetica"
        self._fig_num = 0

    def header(self):
        pass

    def footer(self):
        self.set_y(-10)
        self.set_font(self._font_name, "", 8)
        self.cell(0, 10, f"Стр. {self.page_no()}", align="C")

    def section_title(self, title: str, level: int = 1):
        sizes = {1: 14, 2: 12, 3: 10}
        size = sizes.get(level, 10)
        self.set_font(self._font_name, "B", size)
        self.ln(4)
        self.multi_cell(0, 7, title)
        self.ln(2)

    def body_text(self, text: str, size: int = 9):
        self.set_font(self._font_name, "", size)
        self.multi_cell(0, 5, text)

    def add_image_full_width(self, image_path: str, caption: str = ""):
        if not os.path.exists(image_path):
            return
        page_w = self.w - self.l_margin - self.r_margin

        # Check if we need a new page
        if self.get_y() > self.h - 100:
            self.add_page()

        self.image(image_path, x=self.l_margin, w=page_w)
        if caption:
            self._fig_num += 1
            body = re.sub(r"^\s*Рис\.\s*:?\s*", "", caption)
            numbered = f"Рис. {self._fig_num}. {body}" if body else f"Рис. {self._fig_num}."
            self.set_font(self._font_name, "", FIGURE_CAPTION_FONT_PT)
            self.ln(2)
            self.multi_cell(
                0,
                FIGURE_CAPTION_LINE_HEIGHT_MM,
                numbered,
                align="C",
            )
        self.ln(4)


def _format_val(val, decimals=2) -> str:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return ""
    if isinstance(val, float):
        return f"{val:.{decimals}f}"
    return str(val)


def build_initial_data_table(pdf: ReportPDF, file_data: FileData, start_id: int = 1):
    """Render the initial data table for one file."""
    df = build_seed_dataframe(file_data, start_id=start_id)

    tnum = _next_table_num()
    pdf.section_title(
        f"Таблица {tnum}: Исходные данные — {file_data.variant}, "
        f"повторность {file_data.replicate} — {file_data.filename}",
        level=3,
    )

    pdf.set_font(pdf._font_name, "", 7)

    kor_cols = [c for c in df.columns if c.startswith("kor")]
    cols = ["id", "zar", "ver"] + kor_cols + ["sm.kor", "mn.kor", "rs.rat"]
    headers = cols[:]

    col_widths = []
    page_w = pdf.w - pdf.l_margin - pdf.r_margin
    n_cols = len(cols)
    # Give id, zar smaller widths
    base_w = page_w / n_cols
    for c in cols:
        if c in ("id", "zar"):
            col_widths.append(base_w * 0.7)
        elif c in ("sm.kor", "mn.kor", "rs.rat"):
            col_widths.append(base_w * 1.2)
        else:
            col_widths.append(base_w)

    # Normalize to fit page
    total = sum(col_widths)
    col_widths = [w * page_w / total for w in col_widths]

    row_h = 4

    # Header row
    pdf.set_font(pdf._font_name, "B", 7)
    for i, h in enumerate(headers):
        pdf.cell(col_widths[i], row_h, h, border=1, align="C")
    pdf.ln()

    # Data rows
    pdf.set_font(pdf._font_name, "", 6.5)
    for _, row in df.iterrows():
        if pdf.get_y() > pdf.h - 20:
            pdf.add_page()
            pdf.set_font(pdf._font_name, "B", 7)
            for i, h in enumerate(headers):
                pdf.cell(col_widths[i], row_h, h, border=1, align="C")
            pdf.ln()
            pdf.set_font(pdf._font_name, "", 6.5)

        for i, c in enumerate(cols):
            val = row.get(c)
            if c == "id":
                text = str(int(val)) if pd.notna(val) else ""
            elif c == "zar":
                text = str(int(val)) if pd.notna(val) else ""
            elif c == "rs.rat":
                text = _format_val(val, 2)
            elif c in ("sm.kor", "mn.kor", "ver"):
                text = str(int(val)) if pd.notna(val) else ""
            else:
                text = str(int(val)) if pd.notna(val) else ""
            pdf.cell(col_widths[i], row_h, text, border=1, align="C")
        pdf.ln()

    pdf.ln(3)


def build_wilcoxon_table(pdf: ReportPDF, results: List[WilcoxonResult],
                         title: str, loc_label: str):
    """Render Wilcoxon test results as a table."""
    tnum = _next_table_num()
    pdf.section_title(
        f"Таблица {tnum}: {title}",
        level=3,
    )

    pdf.set_font(pdf._font_name, "B", 8)
    headers = ["", "Вариант", loc_label, "E." + loc_label, "P", "Ci.l", "Ci.h", "Прошёл"]
    page_w = pdf.w - pdf.l_margin - pdf.r_margin
    col_w = page_w / len(headers)
    widths = [col_w * 0.5, col_w * 2.0, col_w, col_w, col_w, col_w, col_w, col_w]
    total = sum(widths)
    widths = [w * page_w / total for w in widths]
    row_h = 5

    for i, h in enumerate(headers):
        pdf.cell(widths[i], row_h, h, border=1, align="C")
    pdf.ln()

    pdf.set_font(pdf._font_name, "", 8)
    for idx, r in enumerate(results, 1):
        vals = [str(idx), r.variant, _format_val(r.location), _format_val(r.expected),
                _format_val(r.p_value), _format_val(r.ci_low), _format_val(r.ci_high),
                "да" if r.passed else "нет"]
        for i, v in enumerate(vals):
            pdf.cell(widths[i], row_h, v, border=1, align="C")
        pdf.ln()

    pdf.ln(3)


def build_normality_table(pdf: ReportPDF, results: List[NormalityResult], title: str):
    """Render normality test p-values as a table."""
    tnum = _next_table_num()
    pdf.section_title(f"Таблица {tnum}: {title}", level=3)

    pdf.set_font(pdf._font_name, "B", 8)
    headers = ["", "Вариант", "P", "Прошёл"]
    page_w = pdf.w - pdf.l_margin - pdf.r_margin
    widths = [page_w * 0.1, page_w * 0.5, page_w * 0.2, page_w * 0.2]
    row_h = 5

    for i, h in enumerate(headers):
        pdf.cell(widths[i], row_h, h, border=1, align="C")
    pdf.ln()

    pdf.set_font(pdf._font_name, "", 8)
    for idx, r in enumerate(results, 1):
        vals = [str(idx), r.variant, _format_val(r.p_value), "да" if r.passed else "нет"]
        for i, v in enumerate(vals):
            pdf.cell(widths[i], row_h, v, border=1, align="C")
        pdf.ln()

    pdf.ln(3)


def build_descriptive_table(pdf: ReportPDF, stats_list: List[DescriptiveStats], metric: str):
    """Render descriptive statistics summary table."""
    tnum = _next_table_num()
    metric_title = METRIC_TITLES.get(metric, metric)
    pdf.section_title(f"Таблица {tnum}: Итоговая статистика — {metric_title}", level=3)

    pdf.set_font(pdf._font_name, "B", 8)
    headers = ["Вт", "Mn", "Sd", "Se", "Ci", "N", "Min", "Max"]
    page_w = pdf.w - pdf.l_margin - pdf.r_margin
    widths = [page_w * 0.25] + [page_w * 0.75 / (len(headers) - 1)] * (len(headers) - 1)
    row_h = 5

    for i, h in enumerate(headers):
        pdf.cell(widths[i], row_h, h, border=1, align="C")
    pdf.ln()

    pdf.set_font(pdf._font_name, "", 8)
    for s in stats_list:
        vals = [
            s.variant,
            _format_val(s.mean),
            _format_val(s.std),
            _format_val(s.se),
            _format_val(s.ci),
            str(s.n),
            _format_val(s.min_val),
            _format_val(s.max_val),
        ]
        for i, v in enumerate(vals):
            pdf.cell(widths[i], row_h, v, border=1, align="C")
        pdf.ln()

    pdf.ln(3)


def _table_header(pdf: ReportPDF, headers: List[str], widths: List[float], row_h: float = 5):
    pdf.set_font(pdf._font_name, "B", 8)
    for i, h in enumerate(headers):
        pdf.cell(widths[i], row_h, h, border=1, align="C")
    pdf.ln()
    pdf.set_font(pdf._font_name, "", 8)


def build_anova_section(
    pdf: ReportPDF,
    section_num: int,
    metric: str,
    anova: AnovaResult,
    summary_plot_path: Optional[str] = None,
):
    """Build the two-factor ANOVA + Tukey HSD subsection."""
    title = METRIC_TITLES.get(metric, metric)
    pdf.section_title(f"{section_num}.4 Дисперсионный анализ — {title}", level=2)

    if not anova or not anova.available:
        reason = (anova.error if anova else "нет данных") or "нет данных"
        pdf.body_text(f"Дисперсионный анализ не выполнен: {reason}")
        return

    # OLS coefficients table
    tnum = _next_table_num()
    pdf.section_title(
        f"Таблица {tnum}: Коэффициенты двухфакторной линейной модели",
        level=3,
    )
    page_w = pdf.w - pdf.l_margin - pdf.r_margin
    headers = ["Член", "Оценка", "Ст. ошибка", "t", "P(>|t|)"]
    widths = [page_w * 0.44, page_w * 0.14, page_w * 0.14, page_w * 0.14, page_w * 0.14]
    _table_header(pdf, headers, widths)
    for c in anova.ols_coefficients:
        vals = [c.term, _format_val(c.estimate, 3), _format_val(c.std_err, 3),
                _format_val(c.t_value, 2), _format_val(c.p_value, 4)]
        for i, v in enumerate(vals):
            pdf.cell(widths[i], 5, v, border=1, align="C")
        pdf.ln()
    pdf.ln(1)
    pdf.body_text(
        f"R² = {_format_val(anova.r_squared, 3)}, "
        f"R²_adj = {_format_val(anova.adj_r_squared, 3)}, "
        f"F = {_format_val(anova.f_statistic, 2)}, "
        f"P = {_format_val(anova.f_p_value, 4)}"
    )
    pdf.ln(3)

    # ANOVA table
    tnum = _next_table_num()
    pdf.section_title(f"Таблица {tnum}: Таблица двухфакторного ANOVA", level=3)
    headers = ["Источник", "df", "Sum Sq", "Mean Sq", "F", "P(>F)"]
    widths = [page_w * 0.35, page_w * 0.09, page_w * 0.14, page_w * 0.14, page_w * 0.14, page_w * 0.14]
    _table_header(pdf, headers, widths)
    for r in anova.anova_table:
        vals = [r.term, _format_val(r.df, 0), _format_val(r.sum_sq, 2),
                _format_val(r.mean_sq, 2), _format_val(r.f_value, 2),
                _format_val(r.p_value, 4)]
        for i, v in enumerate(vals):
            pdf.cell(widths[i], 5, v, border=1, align="C")
        pdf.ln()
    pdf.ln(3)

    # Tukey HSD — global parameters (R agricolae HSD.test style)
    if anova.tukey_hsd_meta is not None:
        meta = anova.tukey_hsd_meta
        tnum = _next_table_num()
        pdf.section_title(
            f"Таблица {tnum}: Tukey HSD — параметры модели (Mean, MSerror, HSD, …)",
            level=3,
        )
        param_rows = [
            ("Mean", _format_val(meta.grand_mean, 4)),
            ("CV", _format_val(meta.cv, 4)),
            ("MSerror", _format_val(meta.ms_error, 6)),
            ("HSD", _format_val(meta.hsd, 4)),
            ("r.harmonic", _format_val(meta.r_harmonic, 4)),
            ("Df", _format_val(meta.df_error, 2)),
            ("ntr", str(meta.n_treatments)),
            ("StudentizedRange", _format_val(meta.studentized_range_q, 4)),
            ("alpha", _format_val(meta.alpha, 4)),
            ("test", meta.test),
            ("name.t", meta.name_t),
        ]
        wk, wv = page_w * 0.34, page_w * 0.66
        pdf.set_font(pdf._font_name, "B", 8)
        pdf.cell(wk, 5, "Параметр", border=1, align="L")
        pdf.cell(wv, 5, "Значение", border=1, align="C")
        pdf.ln()
        pdf.set_font(pdf._font_name, "", 8)
        for pk, pv in param_rows:
            pdf.cell(wk, 5, pk, border=1, align="L")
            pdf.cell(wv, 5, str(pv), border=1, align="C")
            pdf.ln()
        pdf.ln(2)

    if anova.tukey_treatment_rows:
        tnum = _next_table_num()
        pdf.section_title(
            f"Таблица {tnum}: Tukey HSD — варианты (otv, std, r, Min, Max, trt, means, M)",
            level=3,
        )
        headers = ["otv", "std", "r", "Min", "Max", "trt", "means", "M"]
        widths = [
            page_w * 0.06,
            page_w * 0.11,
            page_w * 0.06,
            page_w * 0.09,
            page_w * 0.09,
            page_w * 0.27,
            page_w * 0.14,
            page_w * 0.18,
        ]
        _table_header(pdf, headers, widths)
        pdf.set_font(pdf._font_name, "", 7)
        for tr in anova.tukey_treatment_rows:
            vals = [
                str(tr.otv),
                _format_val(tr.std, 4),
                str(tr.r),
                _format_val(tr.min_y, 2),
                _format_val(tr.max_y, 2),
                tr.trt,
                _format_val(tr.means, 4),
                tr.m,
            ]
            for i, v in enumerate(vals):
                align = "L" if i == 5 else "C"
                pdf.cell(widths[i], 5, str(v), border=1, align=align)
            pdf.ln()
        pdf.ln(3)

    # Tukey HSD pairwise
    if anova.tukey_pairs:
        tnum = _next_table_num()
        pdf.section_title(
            f"Таблица {tnum}: Множественные сравнения Тьюки HSD (α = 0.05)",
            level=3,
        )
        headers = ["Группа 1", "Группа 2", "Разн. средних", "P", "CI низ", "CI верх", "Значимо"]
        widths = [page_w * 0.17, page_w * 0.17, page_w * 0.15, page_w * 0.13, page_w * 0.13, page_w * 0.13, page_w * 0.12]
        _table_header(pdf, headers, widths)
        for p in sorted(
            anova.tukey_pairs,
            key=lambda pr: (variant_display_sort_key(pr.group1), variant_display_sort_key(pr.group2)),
        ):
            vals = [p.group1, p.group2, _format_val(p.mean_diff, 2),
                    _format_val(p.p_value, 4), _format_val(p.ci_low, 2),
                    _format_val(p.ci_high, 2), "да" if p.reject else "нет"]
            for i, v in enumerate(vals):
                pdf.cell(widths[i], 5, v, border=1, align="C")
            pdf.ln()
        pdf.ln(3)

    # CLD summary only if the extended treatment table is absent
    if anova.cld_letters and not anova.tukey_treatment_rows:
        tnum = _next_table_num()
        pdf.section_title(
            f"Таблица {tnum}: Группы однородности (CLD) по Тьюки HSD",
            level=3,
        )
        headers = ["Вариант", "Среднее", "Группа"]
        widths = [page_w * 0.4, page_w * 0.3, page_w * 0.3]
        _table_header(pdf, headers, widths)
        ordered = sorted_variants(anova.cld_letters.keys())
        for v in ordered:
            vals = [v, _format_val(anova.variant_means.get(v, float("nan")), 2),
                    anova.cld_letters.get(v, "")]
            for i, val in enumerate(vals):
                pdf.cell(widths[i], 5, val, border=1, align="C")
            pdf.ln()
        pdf.ln(3)

    if summary_plot_path and os.path.exists(summary_plot_path):
        pdf.add_image_full_width(
            summary_plot_path,
            f"Рис.: Средние по вариантам с буквами групп Tukey HSD — {title}",
        )


def build_metric_section(
    pdf: ReportPDF,
    section_num: int,
    metric_stats: MetricStatistics,
    plot_paths: Dict[str, str],
):
    """Build a full section for one metric."""
    metric = metric_stats.metric
    title = METRIC_TITLES.get(metric, metric)

    pdf.add_page()
    pdf.section_title(f"{section_num} {title}", level=1)

    # Distribution plots
    pdf.section_title(f"{section_num}.1 Форма и характер распределения данных", level=2)

    if "hist" in plot_paths:
        pdf.add_image_full_width(
            plot_paths["hist"],
            f"Рис.: Гистограммы распределения — {title}",
        )

    if "kde" in plot_paths:
        pdf.add_image_full_width(
            plot_paths["kde"],
            f"Рис.: Кривые плотности (KDE) — {title}",
        )

    if "boxplot_replicate" in plot_paths:
        pdf.add_image_full_width(
            plot_paths["boxplot_replicate"],
            f"Рис.: Распределение по повторностям — {title}",
        )

    if "boxplot" in plot_paths:
        pdf.add_image_full_width(
            plot_paths["boxplot"],
            f"Рис.: Боксплоты по вариантам — {title}",
        )

    # Normality tests
    pdf.section_title(f"{section_num}.2 Проверка нормальности распределения данных", level=2)

    build_wilcoxon_table(
        pdf, metric_stats.wilcoxon_mean,
        f"Проверка нормальности (ранговый критерий Уилкокса, μ ≈ среднее) — {title}",
        "Mn",
    )

    build_wilcoxon_table(
        pdf, metric_stats.wilcoxon_median,
        f"Проверка нормальности (ранговый критерий Уилкокса, μ ≈ медиана) — {title}",
        "Md",
    )

    build_normality_table(
        pdf, metric_stats.anderson_darling,
        f"Проверка нормальности (тест Андерсона-Дарлинга) — {title}",
    )

    build_normality_table(
        pdf, metric_stats.shapiro_wilk,
        f"Проверка нормальности (тест Шапиро-Уилка) — {title}",
    )

    # Descriptive summary
    pdf.section_title(f"{section_num}.3 Итоги", level=2)

    build_descriptive_table(pdf, metric_stats.descriptive, metric)

    # ANOVA (section num.4)
    if metric_stats.anova is not None:
        build_anova_section(
            pdf, section_num, metric, metric_stats.anova,
            summary_plot_path=plot_paths.get("summary_cld"),
        )


def build_germination_section(
    pdf: ReportPDF,
    file_datas: List[FileData],
    section_num: int,
):
    """Всхожесть: share of alive seeds per file and pooled by variant."""
    pdf.add_page()
    pdf.section_title(f"{section_num} Всхожесть", level=1)

    tnum = _next_table_num()
    pdf.section_title(
        f"Таблица {tnum}: Всхожесть по исходным файлам",
        level=3,
    )

    page_w = pdf.w - pdf.l_margin - pdf.r_margin
    headers = ["№", "Вариант", "Повт.", "Жив.", "Нежив.", "Всего", "Всхожесть, %"]
    widths = [
        page_w * 0.06,
        page_w * 0.28,
        page_w * 0.10,
        page_w * 0.12,
        page_w * 0.12,
        page_w * 0.12,
        page_w * 0.20,
    ]
    row_h = 5
    _table_header(pdf, headers, widths, row_h=row_h)
    pdf.set_font(pdf._font_name, "", 7)

    for idx, fd in enumerate(file_datas, 1):
        g = germination_stats(fd)
        rate_pct = 100.0 * g.rate if g.total > 0 else float("nan")
        vals = [
            str(idx),
            fd.variant,
            str(fd.replicate),
            str(g.alive),
            str(g.dead),
            str(g.total),
            _format_val(rate_pct, 2) if g.total > 0 else "—",
        ]
        if pdf.get_y() > pdf.h - 20:
            pdf.add_page()
            _table_header(pdf, headers, widths, row_h=row_h)
            pdf.set_font(pdf._font_name, "", 7)

        for i, v in enumerate(vals):
            align = "L" if i == 1 else "C"
            pdf.cell(widths[i], row_h, str(v), border=1, align=align)
        pdf.ln()

    pdf.ln(3)

    # Pooled by variant (sum of alive / sum of total across replicates)
    by_var: Dict[str, List[GerminationStats]] = {}
    for fd in file_datas:
        by_var.setdefault(fd.variant, []).append(germination_stats(fd))

    tnum = _next_table_num()
    pdf.section_title(
        f"Таблица {tnum}: Всхожесть по вариантам (все повторности)",
        level=3,
    )
    headers2 = ["Вариант", "Жив.", "Нежив.", "Всего", "Всхожесть, %"]
    w2 = [page_w * 0.35, page_w * 0.15, page_w * 0.15, page_w * 0.15, page_w * 0.20]
    _table_header(pdf, headers2, w2, row_h=row_h)

    pdf.set_font(pdf._font_name, "", 8)
    for v in sorted_variants(by_var.keys()):
        alive = sum(g.alive for g in by_var[v])
        dead = sum(g.dead for g in by_var[v])
        tot = alive + dead
        rate_pct = 100.0 * alive / tot if tot > 0 else float("nan")
        row_vals = [
            v,
            str(alive),
            str(dead),
            str(tot),
            _format_val(rate_pct, 2) if tot > 0 else "—",
        ]
        for i, val in enumerate(row_vals):
            pdf.cell(w2[i], row_h, val, border=1, align="C")
        pdf.ln()

    pdf.ln(3)


def build_report(
    file_datas: List[FileData],
    all_stats: Dict[str, MetricStatistics],
    all_plots: Dict[str, Dict[str, str]],
    output_path: str,
    experiment_title: str = "Отчёт по эксперименту",
):
    """Build the full PDF report."""
    global TABLE_NUM
    TABLE_NUM = 0

    pdf = ReportPDF()
    pdf.add_page()

    # Title page
    pdf.section_title(experiment_title, level=1)
    pdf.ln(5)

    # Section 1: Initial data
    pdf.section_title("1 Исходные данные", level=1)

    pdf.body_text(f"Варианты: {', '.join(sorted_variants(fd.variant for fd in file_datas))}")
    pdf.ln(3)

    for fd in file_datas:
        # One table per file (repeat); seed ids restart at 1 (display only).
        build_initial_data_table(pdf, fd, start_id=1)

    # Sections 2…(1+len(METRICS)): Statistical analysis for each metric
    for section_idx, metric in enumerate(METRICS, 2):
        if metric in all_stats and metric in all_plots:
            build_metric_section(pdf, section_idx, all_stats[metric], all_plots[metric])

    build_germination_section(pdf, file_datas, section_num=2 + len(METRICS))

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    pdf.output(output_path)
    return output_path
