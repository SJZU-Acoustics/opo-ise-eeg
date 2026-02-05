"""
分析2：行为时域趋势（Cr_t）混合效应模型 - 全样本（n=24）
Analysis 2: Behavioral Time Trend (Cr_t) via Linear Mixed-Effects Model - Full sample (n=24)

动机 / Motivation
原稿的 Table 1 使用“按 20s 分段后的跨被试均值序列”做一元回归（仅约 11 个点），
会抹平被试内变异并低估标准误，存在聚合偏误/伪显著风险。
本脚本改用个体层级 LMM 直接检验 time-on-task 斜率及其在 SPL 条件间的差异（time×SPL 交互）。

模型 / Model
Fixed effects:
  Cr ~ time * C(SPL) + noise_sensitivity_z
Random effects:
  (1 + time | subject)  （若随机斜率不收敛则回退为随机截距）

输出 / Outputs
  - 交互项 LRT: χ²(df), p
  - 各 SPL 条件的简单斜率（Cr / min）+ 95%CI + p
  - 图：均值±SEM + LMM 固定效应拟合线

运行 / Run:
  cd 投稿 && ../.venv/bin/python code/analysis_02_behavior_time_regression.py
"""

# Headless-safe matplotlib backend (prevents macOS GUI backend crashes in sandbox)
import os
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/xdg-cache")

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import norm
from statsmodels.formula.api import mixedlm
import matplotlib.pyplot as plt

from config import (
    DATA_CLEANED, COLORS, SPL_LABELS_SHORT,
    set_plot_style, ensure_output_dir
)


def load_data():
    """Load time-segment data (all subjects)."""
    df = pd.read_excel(DATA_CLEANED, sheet_name="time_series_behavior")
    df = df.dropna(subset=["subject_id", "spl_level", "time_bin", "Cr"]).copy()
    df["subject_id"] = df["subject_id"].astype(int)
    df["spl_level"] = df["spl_level"].astype(int)

    # Time midpoint (minutes)
    def parse_time(t):
        a, b = str(t).split("-")
        return (int(a) + int(b)) / 2 / 60

    df["time_min"] = df["time_bin"].apply(parse_time)
    df["time_c"] = df["time_min"] - df["time_min"].mean()

    # Noise sensitivity (subject-level; constant within subject)
    if "噪声" in df.columns and "noise_sensitivity" not in df.columns:
        df = df.rename(columns={"噪声": "noise_sensitivity"})

    if "noise_sensitivity" in df.columns:
        ns = df[["subject_id", "noise_sensitivity"]].dropna().drop_duplicates("subject_id").copy()
        ns["noise_z"] = (ns["noise_sensitivity"] - ns["noise_sensitivity"].mean()) / ns["noise_sensitivity"].std(ddof=0)
        df = df.merge(ns[["subject_id", "noise_z"]], on="subject_id", how="left")
        df["noise_z"] = df["noise_z"].fillna(0.0)
    else:
        df["noise_z"] = 0.0

    df["spl_level_cat"] = pd.Categorical(df["spl_level"], categories=[0, 1, 2, 3], ordered=True)
    return df


def _fit_mixedlm_with_fallback(formula, df):
    """Fit MixedLM with random slope; fallback to random intercept if needed."""
    try:
        return mixedlm(formula, df, groups=df["subject_id"], re_formula="1+time_c").fit(
            reml=False, method="lbfgs"
        )
    except Exception as e:
        print(f"[WARN] MixedLM random-slope failed; falling back to random-intercept model: {e}")
        return mixedlm(formula, df, groups=df["subject_id"], re_formula="1").fit(
            reml=False, method="lbfgs"
        )


def lrt(full, reduced):
    """Likelihood ratio test between nested models."""
    lr = 2 * (full.llf - reduced.llf)
    df_diff = len(full.fe_params) - len(reduced.fe_params)
    p = stats.chi2.sf(lr, df_diff)
    return float(lr), int(df_diff), float(p)


def _simple_slope(model, level):
    """Simple slope of time at a given SPL level (Wald z-test)."""
    fe = model.params
    cov = model.cov_params()

    base = float(fe["time_c"])
    if level == 0:
        coef = base
        var = float(cov.loc["time_c", "time_c"])
    else:
        term = f"time_c:C(spl_level_cat)[T.{level}]"
        coef = base + float(fe.get(term, 0.0))
        var = float(cov.loc["time_c", "time_c"] + cov.loc[term, term] + 2 * cov.loc["time_c", term])

    se = float(np.sqrt(var)) if var >= 0 else float("nan")
    z = float(coef / se) if (np.isfinite(se) and se > 0) else float("nan")
    p = float(2 * norm.sf(abs(z))) if np.isfinite(z) else float("nan")
    ci_low = coef - 1.96 * se if np.isfinite(se) else float("nan")
    ci_high = coef + 1.96 * se if np.isfinite(se) else float("nan")
    return coef, se, ci_low, ci_high, p


def run_time_trend_lmm(df):
    """Fit LMM, LRT for interaction, and output simple slopes by SPL."""
    m0 = _fit_mixedlm_with_fallback("Cr ~ time_c + C(spl_level_cat) + noise_z", df)
    m1 = _fit_mixedlm_with_fallback("Cr ~ time_c * C(spl_level_cat) + noise_z", df)

    chi2, df_diff, p_int = lrt(m1, m0)

    rows = []
    for spl in [0, 1, 2, 3]:
        slope, se, ci_low, ci_high, p = _simple_slope(m1, spl)
        rows.append({
            "spl": spl,
            "spl_label": ["Quiet(35-40)", "Lower(40-45)", "Medium(45-50)", "Higher(50-55)"][spl],
            "slope_per_min": slope,
            "se": se,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "p_slope": p,
            "n_subjects": int(df["subject_id"].nunique()),
            "n_rows": int(len(df)),
            "lrt_chi2_time_x_spl": chi2,
            "lrt_df": df_diff,
            "lrt_p": p_int,
        })

    return m1, pd.DataFrame(rows)


def plot_time_trends(df, model, slopes_table, output_dir):
    """Plot mean±SEM by time bin with LMM fixed-effect fitted lines."""
    fig, ax = plt.subplots(figsize=(8, 5))

    mean_time = float(df["time_min"].mean())
    fe = model.params

    for spl in [0, 1, 2, 3]:
        subset = df[df["spl_level"] == spl]
        agg = subset.groupby("time_min")["Cr"].agg(["mean", "sem"]).sort_index()

        ax.errorbar(
            agg.index, agg["mean"],
            yerr=agg["sem"],
            fmt="o", markersize=6,
            color=COLORS[spl], alpha=0.8,
            label=SPL_LABELS_SHORT[spl],
            capsize=2,
        )

        slope = float(slopes_table.loc[slopes_table["spl"] == spl, "slope_per_min"].iloc[0])
        p_slope = float(slopes_table.loc[slopes_table["spl"] == spl, "p_slope"].iloc[0])
        linestyle = "-" if p_slope < 0.05 else "--"

        intercept = float(fe["Intercept"])
        if spl != 0:
            intercept += float(fe.get(f"C(spl_level_cat)[T.{spl}]", 0.0))

        x_line = np.linspace(float(agg.index.min()), float(agg.index.max()), 50)
        y_line = intercept + slope * (x_line - mean_time)
        ax.plot(x_line, y_line, color=COLORS[spl], linestyle=linestyle, linewidth=2, alpha=0.7)

    chi2 = float(slopes_table["lrt_chi2_time_x_spl"].iloc[0])
    df_lrt = int(slopes_table["lrt_df"].iloc[0])
    p_lrt = float(slopes_table["lrt_p"].iloc[0])

    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Reaction Capacity (Cr)")
    ax.set_title(f"Cr Time Trend by SPL Level (n={df['subject_id'].nunique()})\nTime×SPL LRT: χ²({df_lrt})={chi2:.2f}, p={p_lrt:.3f}")
    ax.legend(loc="upper right", framealpha=0.9)

    plt.tight_layout()
    plt.savefig(output_dir / "fig_cr_time_trend_lmm.png")
    plt.savefig(output_dir / "fig_cr_time_trend_lmm.svg")
    plt.close()
    print(f"Saved: {output_dir / 'fig_cr_time_trend_lmm.png'}")


def main():
    set_plot_style()
    output_dir = ensure_output_dir()

    print("=" * 60)
    print("Analysis 2: Cr Time Trend via LMM (Full sample)")
    print("=" * 60)

    df = load_data()
    print(f"\nData: {len(df)} rows, {df['subject_id'].nunique()} subjects")

    model, table = run_time_trend_lmm(df)

    chi2 = float(table["lrt_chi2_time_x_spl"].iloc[0])
    df_lrt = int(table["lrt_df"].iloc[0])
    p_lrt = float(table["lrt_p"].iloc[0])

    print("\n--- Table 1 (Revised): Cr Time Trend (LMM) ---")
    print(f"Time×SPL LRT: χ²({df_lrt})={chi2:.3f}, p={p_lrt:.4f}")
    print(f"{'SPL':<14} {'Slope/min':<12} {'95% CI':<26} {'p':<10}")
    print("-" * 70)
    for _, r in table.iterrows():
        ci = f"[{r['ci_low']:.5f}, {r['ci_high']:.5f}]"
        p_txt = f"{r['p_slope']:.4f}" if r["p_slope"] >= 0.001 else "<0.001"
        print(f"{r['spl_label']:<14} {r['slope_per_min']:<12.6f} {ci:<26} {p_txt:<10}")

    plot_time_trends(df, model, table, output_dir)

    table.to_csv(output_dir / "table1_cr_time_lmm.csv", index=False)
    print(f"Saved: {output_dir / 'table1_cr_time_lmm.csv'}")

    return {"model": model, "table": table}


if __name__ == "__main__":
    main()
