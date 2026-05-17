"""Statistical tests and descriptive statistics for seed experiment data."""

import re
import warnings

import numpy as np
import pandas as pd

from scipy import stats
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from data_parser import sorted_variants

try:
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    from statsmodels.stats.anova import anova_lm
    from statsmodels.stats.multicomp import pairwise_tukeyhsd
    _HAS_STATSMODELS = True
except ImportError:
    _HAS_STATSMODELS = False


NORMALITY_ALPHA = 0.05


METRICS = ["ver", "sm.kor", "mn.kor", "rs.rat"]

METRIC_TITLES = {
    "ver": "Длина колеоптилей",
    "sm.kor": "Сумма длин корней",
    "mn.kor": "Средняя длина корня",
    "rs.rat": "Отношение подземной/надземной частей растения",
}


@dataclass
class DescriptiveStats:
    variant: str
    mean: float
    std: float
    se: float
    ci: float  # 95% confidence interval half-width
    n: int
    min_val: float
    max_val: float
    median: float


def _passed(p_value: float) -> bool:
    """Return True if data can be regarded as normal at alpha=0.05."""
    if p_value is None or (isinstance(p_value, float) and np.isnan(p_value)):
        return False
    return p_value >= NORMALITY_ALPHA


@dataclass
class WilcoxonResult:
    """Result of Wilcoxon signed-rank test against a location parameter."""
    variant: str
    location: float       # sample mean or median
    expected: float       # Hodges–Lehmann estimate (median of Walsh averages)
    p_value: float
    ci_low: float
    ci_high: float
    passed: bool = False


@dataclass
class NormalityResult:
    variant: str
    p_value: float
    passed: bool = False


@dataclass
class OLSCoefficient:
    term: str
    estimate: float
    std_err: float
    t_value: float
    p_value: float


@dataclass
class AnovaRow:
    term: str
    df: float
    sum_sq: float
    mean_sq: float
    f_value: float
    p_value: float


@dataclass
class TukeyPair:
    group1: str
    group2: str
    mean_diff: float
    p_value: float
    ci_low: float
    ci_high: float
    reject: bool


@dataclass
class TukeyHSDMeta:
    """Global HSD / ANOVA-error statistics (R agricolae::HSD.test style)."""

    grand_mean: float
    cv: float  # 100 * sqrt(MSerror) / |grand_mean|
    ms_error: float
    hsd: float
    r_harmonic: float
    df_error: float
    n_treatments: int
    studentized_range_q: float
    alpha: float
    test: str
    name_t: str


@dataclass
class TukeyTreatmentSummary:
    """Per-treatment line for Tukey summary (means, spread, CLD)."""

    otv: int
    trt: str
    means: float
    std: float
    r: int
    min_y: float
    max_y: float
    m: str


@dataclass
class AnovaResult:
    ols_coefficients: List[OLSCoefficient]
    r_squared: float
    adj_r_squared: float
    f_statistic: float
    f_p_value: float
    anova_table: List[AnovaRow]
    tukey_pairs: List[TukeyPair]
    cld_letters: Dict[str, str]
    variant_means: Dict[str, float]
    available: bool = True
    error: str = ""
    tukey_hsd_meta: Optional[TukeyHSDMeta] = None
    tukey_treatment_rows: List[TukeyTreatmentSummary] = field(default_factory=list)


@dataclass
class MetricStatistics:
    """All statistics for one metric across all variants."""
    metric: str
    descriptive: List[DescriptiveStats]
    wilcoxon_mean: List[WilcoxonResult]
    wilcoxon_median: List[WilcoxonResult]
    anderson_darling: List[NormalityResult]
    shapiro_wilk: List[NormalityResult]
    anova: Optional[AnovaResult] = None


def _wilcoxon_positive_rank_sum_pmf(n: int) -> np.ndarray:
    """P(W+ = k) for k = 0 … n(n+1)/2 under H0 (continuous symmetric, no ties).

    W+ is the sum of ranks of positive differences. Recurrence matches
    ``scipy.stats._hypotests._get_wilcoxon_distr``.
    """
    c = np.ones(1, dtype=np.float64)
    for k in range(1, n + 1):
        prev_c = c
        c = np.zeros(k * (k + 1) // 2 + 1, dtype=np.float64)
        m = len(prev_c)
        c[:m] = prev_c * 0.5
        c[-m:] += prev_c * 0.5
    return c


def _hodges_lehmann_pseudo_median_ci(
    sorted_walsh: np.ndarray,
    n: int,
    alpha: float,
) -> Tuple[float, float]:
    """Exact (no ties) Hodges–Lehmann CI for the one-sample pseudo-median.

    ``sorted_walsh`` must be the order statistics of all (X_i+X_j)/2, i <= j.
    Endpoints are Walsh order statistics indexed from the Wilcoxon signed-rank
    null at ``alpha/2`` (same construction as R ``wilcox.test`` / ``qsignrank``).
    """
    m_w = sorted_walsh.size
    pmf = _wilcoxon_positive_rank_sum_pmf(n)
    cdf = np.cumsum(pmf)
    alpha2 = alpha / 2.0
    q = int(np.searchsorted(cdf, alpha2, side="left"))
    lo = float(sorted_walsh[q])
    hi = float(sorted_walsh[m_w - q - 1])
    return lo, hi


def _wilcoxon_location_test(data: np.ndarray, location: float) -> dict:
    """Wilcoxon signed-rank test for H0: pseudo-median = location.

    Returns p-value and Hodges–Lehmann point/interval for the pseudo-median of
    ``data`` (interval does not depend on ``location``).
    """
    data = np.asarray(data, dtype=float)
    n = len(data)
    alpha = NORMALITY_ALPHA

    if n == 0:
        return {"p_value": np.nan, "ci_low": np.nan, "ci_high": np.nan,
                "expected": float(location)}

    walsh = [(data[i] + data[j]) / 2.0 for i in range(n) for j in range(i, n)]
    walsh_sorted = np.sort(np.asarray(walsh, dtype=float))
    expected = float(np.median(walsh_sorted))

    if n >= 2:
        ci_low, ci_high = _hodges_lehmann_pseudo_median_ci(walsh_sorted, n, alpha)
    else:
        ci_low = ci_high = float(walsh_sorted[0])

    shifted = data - location
    nonzero = shifted[shifted != 0]

    if len(nonzero) < 10:
        return {"p_value": np.nan, "ci_low": round(ci_low, 2),
                "ci_high": round(ci_high, 2), "expected": round(expected, 2)}

    try:
        _, p_value = stats.wilcoxon(nonzero, alternative="two-sided")
    except ValueError:
        p_value = np.nan

    return {"p_value": round(p_value, 2), "ci_low": round(ci_low, 2),
            "ci_high": round(ci_high, 2), "expected": round(expected, 2)}


def compute_descriptive(data: np.ndarray, variant: str) -> DescriptiveStats:
    n = len(data)
    mean = np.mean(data)
    std = np.std(data, ddof=1)
    se = std / np.sqrt(n)
    ci = 1.96 * se

    return DescriptiveStats(
        variant=variant,
        mean=round(mean, 2),
        std=round(std, 2),
        se=round(se, 2),
        ci=round(ci, 2),
        n=n,
        min_val=round(np.min(data), 2),
        max_val=round(np.max(data), 2),
        median=round(np.median(data), 2),
    )


def compute_wilcoxon_mean(data: np.ndarray, variant: str) -> WilcoxonResult:
    mean_val = round(np.mean(data), 2)
    result = _wilcoxon_location_test(data, mean_val)
    return WilcoxonResult(
        variant=variant,
        location=mean_val,
        expected=result["expected"],
        p_value=result["p_value"],
        ci_low=result["ci_low"],
        ci_high=result["ci_high"],
        passed=_passed(result["p_value"]),
    )


def compute_wilcoxon_median(data: np.ndarray, variant: str) -> WilcoxonResult:
    median_val = round(np.median(data), 2)
    result = _wilcoxon_location_test(data, median_val)
    return WilcoxonResult(
        variant=variant,
        location=median_val,
        expected=result["expected"],
        p_value=result["p_value"],
        ci_low=result["ci_low"],
        ci_high=result["ci_high"],
        passed=_passed(result["p_value"]),
    )


def compute_anderson_darling(data: np.ndarray, variant: str) -> NormalityResult:
    try:
        result = stats.anderson(data, dist="norm", method="interpolate")
        p = float(result.pvalue)
    except TypeError:
        # Fallback for older scipy without method parameter
        result = stats.anderson(data, dist="norm")
        stat_val = result.statistic
        crit = result.critical_values
        sig = result.significance_level
        if stat_val < crit[0]:
            p = sig[0] / 100.0
        elif stat_val > crit[-1]:
            p = sig[-1] / 100.0
        else:
            p = np.nan
            for i in range(len(crit) - 1):
                if crit[i] <= stat_val <= crit[i + 1]:
                    p = sig[i] / 100.0
                    break

    p_rounded = round(p, 2) if not np.isnan(p) else float("nan")
    return NormalityResult(variant=variant, p_value=p_rounded, passed=_passed(p))


def compute_shapiro_wilk(data: np.ndarray, variant: str) -> NormalityResult:
    if len(data) < 3:
        return NormalityResult(variant=variant, p_value=np.nan, passed=False)
    stat, p = stats.shapiro(data)
    return NormalityResult(variant=variant, p_value=round(p, 2), passed=_passed(p))


def _bron_kerbosch(
    r: Set[str],
    p: Set[str],
    x: Set[str],
    neighbors: Dict[str, Set[str]],
    out: List[Set[str]],
) -> None:
    """Enumerate maximal cliques (Bron–Kerbosch with pivot). Mutates p and x."""
    if not p and not x:
        out.append(set(r))
        return
    px = p | x
    if px:
        u = max(px, key=lambda v: len(neighbors[v] & p))
        candidates = set(p - neighbors[u])
    else:
        candidates = set(p)
    for v in list(candidates):
        nv = neighbors[v]
        _bron_kerbosch(r | {v}, set(p & nv), set(x & nv), neighbors, out)
        p.discard(v)
        x.add(v)


def _assign_cld_letters(
    variants: List[str],
    tukey_pairs: List[TukeyPair],
    variant_means: Dict[str, float],
) -> Dict[str, str]:
    """Assign compact letter display (CLD) letters based on Tukey HSD results.

    Each letter labels a maximal homogeneous subset: a maximal clique in the graph
    whose edges are pairwise non-significant Tukey comparisons. A variant receives
    every letter for each maximal clique it belongs to (e.g. ``ab`` when it is
    non-significant with disjoint subsets that differ from each other).

    Variants sharing at least one letter are not significantly different.
    """
    if not variants:
        return {}

    variant_set = list(dict.fromkeys(variants))  # preserve order hint; uniqued

    not_sig: Set[tuple] = set()
    for pr in tukey_pairs:
        if not pr.reject:
            not_sig.add((pr.group1, pr.group2))
            not_sig.add((pr.group2, pr.group1))

    neighbors: Dict[str, Set[str]] = {v: set() for v in variant_set}
    for i, u in enumerate(variant_set):
        for v in variant_set[i + 1 :]:
            if (u, v) in not_sig:
                neighbors[u].add(v)
                neighbors[v].add(u)

    maximal_cliques: List[Set[str]] = []
    p_init = set(variant_set)
    _bron_kerbosch(set(), p_init, set(), neighbors, maximal_cliques)

    def _clique_sort_key(clique: Set[str]) -> tuple:
        mx = max((variant_means.get(v, 0.0) for v in clique), default=0.0)
        return (-mx, tuple(sorted(clique)))

    maximal_cliques.sort(key=_clique_sort_key)

    letters_src = "abcdefghijklmnopqrstuvwxyz"
    result_lists: Dict[str, List[str]] = {v: [] for v in variant_set}
    for idx, clique in enumerate(maximal_cliques):
        letter = letters_src[idx] if idx < len(letters_src) else f"g{idx}"
        for v in clique:
            if letter not in result_lists[v]:
                result_lists[v].append(letter)

    return {
        v: "".join(sorted(result_lists[v])) if result_lists[v] else ""
        for v in variant_set
    }


_RE_OLS_VR = re.compile(
    r"^C\(variant\)\[T\.(.+)\]:C\(replicate\)\[T\.(.+)\]$"
)
_RE_OLS_RV = re.compile(
    r"^C\(replicate\)\[T\.(.+)\]:C\(variant\)\[T\.(.+)\]$"
)
_RE_OLS_V = re.compile(r"^C\(variant\)\[T\.(.+)\]$")
_RE_OLS_R = re.compile(r"^C\(replicate\)\[T\.(.+)\]$")


def display_ols_term(term: str) -> str:
    """Map patsy OLS coefficient names to vart_/povt labels (report display)."""
    if term == "Intercept":
        return term
    m = _RE_OLS_VR.match(term)
    if m:
        return f"vart_{m.group(1)}:povt{m.group(2)}"
    m = _RE_OLS_RV.match(term)
    if m:
        return f"vart_{m.group(2)}:povt{m.group(1)}"
    m = _RE_OLS_V.match(term)
    if m:
        return f"vart_{m.group(1)}"
    m = _RE_OLS_R.match(term)
    if m:
        return f"povt{m.group(1)}"
    return term


def display_anova_term(term: str) -> str:
    """Map statsmodels Type II ANOVA row names to vart/povt labels."""
    if term == "C(variant)":
        return "vart"
    if term == "C(replicate)":
        return "povt"
    if term == "C(variant):C(replicate)":
        return "vart:povt"
    return term


def _tukey_hsd_extras(
    model,
    sub: pd.DataFrame,
    variant_order: List[str],
    cld_letters: Dict[str, str],
    alpha: float,
) -> Tuple[Optional[TukeyHSDMeta], List[TukeyTreatmentSummary]]:
    """HSD.test-style global stats and per-treatment summaries (MS from OLS residuals)."""
    present = [v for v in variant_order if (sub["variant"] == v).any()]
    ntr = len(present)
    ms_err = float(getattr(model, "mse_resid", float("nan")) or float("nan"))
    df_e = float(model.df_resid) if model.df_resid is not None else float("nan")
    if ntr < 2 or not np.isfinite(ms_err) or ms_err <= 0 or not np.isfinite(df_e) or df_e <= 0:
        return None, []

    grand_mean = float(sub["y"].mean())
    treatment_rows: List[TukeyTreatmentSummary] = []
    counts: List[int] = []
    for otv, v in enumerate(sorted_variants(present), start=1):
        yv = sub.loc[sub["variant"] == v, "y"].astype(float)
        n = int(yv.shape[0])
        if n == 0:
            continue
        counts.append(n)
        std_v = float(yv.std(ddof=1)) if n > 1 else 0.0
        treatment_rows.append(
            TukeyTreatmentSummary(
                otv=otv,
                trt=v,
                means=float(yv.mean()),
                std=std_v,
                r=n,
                min_y=float(yv.min()),
                max_y=float(yv.max()),
                m=cld_letters.get(v, ""),
            )
        )

    inv_sum = sum(1.0 / c for c in counts if c > 0)
    r_harm = (len(counts) / inv_sum) if inv_sum > 0 else float("nan")

    try:
        q = float(stats.studentized_range.ppf(1.0 - alpha, ntr, df_e))
    except Exception:
        q = float("nan")

    hsd = (
        float(q * np.sqrt(ms_err / r_harm))
        if np.isfinite(q) and np.isfinite(r_harm) and r_harm > 0
        else float("nan")
    )
    cv = (
        float(100.0 * np.sqrt(ms_err) / abs(grand_mean))
        if abs(grand_mean) > 1e-15
        else float("nan")
    )

    meta = TukeyHSDMeta(
        grand_mean=grand_mean,
        cv=cv,
        ms_error=ms_err,
        hsd=hsd,
        r_harmonic=r_harm,
        df_error=df_e,
        n_treatments=ntr,
        studentized_range_q=q,
        alpha=alpha,
        test="Tukey HSD",
        name_t="variant",
    )
    return meta, treatment_rows


def compute_anova(
    analysis_df: pd.DataFrame,
    metric: str,
    variant_order: List[str],
) -> Optional[AnovaResult]:
    """Two-factor OLS model and ANOVA for metric ~ variant + replicate + variant:replicate.

    Falls back to a one-factor model if replicate data is insufficient.
    """
    if not _HAS_STATSMODELS:
        return AnovaResult(
            ols_coefficients=[], r_squared=float("nan"), adj_r_squared=float("nan"),
            f_statistic=float("nan"), f_p_value=float("nan"),
            anova_table=[], tukey_pairs=[], cld_letters={}, variant_means={},
            available=False, error="statsmodels not available",
        )

    sub = analysis_df.loc[:, ["variant", "replicate", metric]].copy()
    sub = sub.dropna(subset=[metric])
    sub = sub[sub["variant"].isin(variant_order)]
    sub["variant"] = sub["variant"].astype(str)
    sub["replicate"] = sub["replicate"].astype(str)
    sub = sub.rename(columns={metric: "y"})

    if len(sub) < 3 or sub["variant"].nunique() < 2:
        return AnovaResult(
            ols_coefficients=[], r_squared=float("nan"), adj_r_squared=float("nan"),
            f_statistic=float("nan"), f_p_value=float("nan"),
            anova_table=[], tukey_pairs=[], cld_letters={}, variant_means={},
            available=False, error="insufficient data for ANOVA",
        )

    has_replicate_factor = sub["replicate"].nunique() > 1
    variant_rep_counts = sub.groupby("variant")["replicate"].nunique()
    can_interact = has_replicate_factor and (variant_rep_counts >= 2).all()

    if can_interact:
        formula = "y ~ C(variant) + C(replicate) + C(variant):C(replicate)"
    elif has_replicate_factor:
        formula = "y ~ C(variant) + C(replicate)"
    else:
        formula = "y ~ C(variant)"

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = smf.ols(formula, data=sub).fit()
            anova_df = anova_lm(model, typ=2)
    except Exception as e:
        return AnovaResult(
            ols_coefficients=[], r_squared=float("nan"), adj_r_squared=float("nan"),
            f_statistic=float("nan"), f_p_value=float("nan"),
            anova_table=[], tukey_pairs=[], cld_letters={}, variant_means={},
            available=False, error=f"model error: {e}",
        )

    ols_coeffs = [
        OLSCoefficient(
            term=display_ols_term(str(name)),
            estimate=float(model.params[name]),
            std_err=float(model.bse[name]),
            t_value=float(model.tvalues[name]),
            p_value=float(model.pvalues[name]),
        )
        for name in model.params.index
    ]

    anova_rows = []
    for term, row in anova_df.iterrows():
        df_val = float(row.get("df", float("nan")))
        ss = float(row.get("sum_sq", float("nan")))
        ms = ss / df_val if df_val and not np.isnan(df_val) and df_val > 0 else float("nan")
        anova_rows.append(AnovaRow(
            term=display_anova_term(str(term)),
            df=df_val,
            sum_sq=ss,
            mean_sq=ms,
            f_value=float(row.get("F", float("nan"))) if "F" in row else float("nan"),
            p_value=float(row.get("PR(>F)", float("nan"))) if "PR(>F)" in row else float("nan"),
        ))

    # Tukey HSD on variant
    tukey_pairs: List[TukeyPair] = []
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tukey = pairwise_tukeyhsd(endog=sub["y"].values, groups=sub["variant"].values, alpha=0.05)
        summary_data = tukey._results_table.data[1:]
        for row in summary_data:
            tukey_pairs.append(TukeyPair(
                group1=str(row[0]),
                group2=str(row[1]),
                mean_diff=float(row[2]),
                p_value=float(row[3]),
                ci_low=float(row[4]),
                ci_high=float(row[5]),
                reject=bool(row[6]),
            ))
    except Exception as e:
        # Tukey failed but keep rest of ANOVA
        pass

    variant_means = {
        v: float(sub.loc[sub["variant"] == v, "y"].mean())
        for v in variant_order if (sub["variant"] == v).any()
    }
    cld_letters = _assign_cld_letters(list(variant_means.keys()), tukey_pairs, variant_means)

    tukey_meta, tukey_summ = _tukey_hsd_extras(
        model, sub, variant_order, cld_letters, alpha=0.05,
    )

    return AnovaResult(
        ols_coefficients=ols_coeffs,
        r_squared=float(model.rsquared),
        adj_r_squared=float(model.rsquared_adj),
        f_statistic=float(model.fvalue) if model.fvalue is not None else float("nan"),
        f_p_value=float(model.f_pvalue) if model.f_pvalue is not None else float("nan"),
        anova_table=anova_rows,
        tukey_pairs=tukey_pairs,
        cld_letters=cld_letters,
        variant_means=variant_means,
        available=True,
        tukey_hsd_meta=tukey_meta,
        tukey_treatment_rows=tukey_summ,
    )


def compute_metric_statistics(
    analysis_df: pd.DataFrame,
    metric: str,
    variant_order: List[str],
) -> MetricStatistics:
    """Compute all statistics for one metric across all variants."""
    descriptive = []
    wilcoxon_mean = []
    wilcoxon_median = []
    anderson_darling = []
    shapiro_wilk = []

    for variant in variant_order:
        mask = analysis_df["variant"] == variant
        data = analysis_df.loc[mask, metric].dropna().values.astype(float)

        if len(data) == 0:
            continue

        descriptive.append(compute_descriptive(data, variant))
        wilcoxon_mean.append(compute_wilcoxon_mean(data, variant))
        wilcoxon_median.append(compute_wilcoxon_median(data, variant))
        anderson_darling.append(compute_anderson_darling(data, variant))
        shapiro_wilk.append(compute_shapiro_wilk(data, variant))

    anova_result = compute_anova(analysis_df, metric, variant_order)

    return MetricStatistics(
        metric=metric,
        descriptive=descriptive,
        wilcoxon_mean=wilcoxon_mean,
        wilcoxon_median=wilcoxon_median,
        anderson_darling=anderson_darling,
        shapiro_wilk=shapiro_wilk,
        anova=anova_result,
    )


def compute_all_statistics(
    analysis_df: pd.DataFrame,
    variant_order: Optional[List[str]] = None,
) -> Dict[str, MetricStatistics]:
    """Compute statistics for all metrics."""
    if variant_order is None:
        variant_order = sorted_variants(analysis_df["variant"].unique().tolist())

    results = {}
    for metric in METRICS:
        results[metric] = compute_metric_statistics(analysis_df, metric, variant_order)

    return results
