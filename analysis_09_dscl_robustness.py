"""
分析9（补充）：△SCL 关键对比的稳健性检验
Analysis 9 (Supplementary): Robustness checks for key △SCL contrasts

针对 Fig.6a 中 △SCL 右偏/离群点的疑虑，本脚本在被试内配对差值层面提供稳健推断：
- 参数检验：配对差值对 0 的 t 检验（与主文一致口径）
- 非参数检验：Wilcoxon signed-rank
- 符号检验：binomial sign test
- 自助法置信区间：mean/median 的 bootstrap 95%CI

输出保存到 figures/ 目录，便于写入 Appendix 或稳健性段落。

Run:
  cd 投稿 && ../.venv/bin/python code/analysis_09_dscl_robustness.py
"""

import numpy as np
import pandas as pd
from scipy import stats

from config import DATA_CLEANED, ensure_output_dir


def _bootstrap_ci(x, stat_fn, n_boot=20000, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    boots = stat_fn(x[idx])
    lo, hi = np.quantile(boots, [0.025, 0.975])
    return float(lo), float(hi)


def _contrast_summary(wide, a, b, label):
    """
    wide: subject_id × spl_level
    contrast: b - a
    """
    d = (wide[b] - wide[a]).dropna()
    n = int(len(d))

    mean = float(d.mean())
    median = float(d.median())
    sd = float(d.std(ddof=1))
    dz = mean / sd if sd > 0 else float("nan")

    t_res = stats.ttest_1samp(d, 0.0)

    # Wilcoxon: needs at least one non-zero difference
    try:
        w_res = stats.wilcoxon(d)
        w_p = float(w_res.pvalue)
        w_stat = float(w_res.statistic)
    except ValueError:
        w_p = float("nan")
        w_stat = float("nan")

    # Sign test (ignore exact zeros)
    pos = int((d > 0).sum())
    neg = int((d < 0).sum())
    n_sign = pos + neg
    sign_p = float(stats.binomtest(pos, n_sign, 0.5, alternative="two-sided").pvalue) if n_sign > 0 else float("nan")

    mean_ci = _bootstrap_ci(d.values, lambda z: z.mean(axis=1))
    med_ci = _bootstrap_ci(d.values, lambda z: np.median(z, axis=1))

    return {
        "contrast": label,
        "a": a,
        "b": b,
        "n": n,
        "mean_diff": mean,
        "mean_ci_low": mean_ci[0],
        "mean_ci_high": mean_ci[1],
        "median_diff": median,
        "median_ci_low": med_ci[0],
        "median_ci_high": med_ci[1],
        "t_stat": float(t_res.statistic),
        "t_p": float(t_res.pvalue),
        "wilcoxon_stat": w_stat,
        "wilcoxon_p": w_p,
        "sign_pos": pos,
        "sign_neg": neg,
        "sign_p": sign_p,
        "cohen_dz": float(dz),
    }


def main():
    output_dir = ensure_output_dir()

    df = pd.read_excel(DATA_CLEANED, sheet_name="overall_behavior_eda")
    df = df.dropna(subset=["subject_id", "spl_level", "dSCL"]).copy()
    df["subject_id"] = df["subject_id"].astype(int)
    df["spl_level"] = df["spl_level"].astype(int)

    wide = df.pivot(index="subject_id", columns="spl_level", values="dSCL")

    contrasts = [
        (1, 3, "Higher - Lower (3-1)"),
        (1, 0, "Quiet - Lower (0-1)"),
        (0, 3, "Higher - Quiet (3-0)"),
    ]

    rows = []
    print("=" * 60)
    print("Analysis 9: dSCL Robustness Checks")
    print("=" * 60)

    for a, b, label in contrasts:
        res = _contrast_summary(wide, a=a, b=b, label=label)
        rows.append(res)

        mean_ci = f"[{res['mean_ci_low']:.3f}, {res['mean_ci_high']:.3f}]"
        med_ci = f"[{res['median_ci_low']:.3f}, {res['median_ci_high']:.3f}]"
        print(f"\n{label} (n={res['n']}):")
        print(f"  Mean diff = {res['mean_diff']:.3f}, bootstrap 95%CI {mean_ci}")
        print(f"  Median diff = {res['median_diff']:.3f}, bootstrap 95%CI {med_ci}")
        print(f"  t-test p = {res['t_p']:.4f}")
        print(f"  Wilcoxon p = {res['wilcoxon_p']:.4f}")
        print(f"  Sign test p = {res['sign_p']:.4f} (pos={res['sign_pos']}, neg={res['sign_neg']})")

    out = pd.DataFrame(rows)
    out.to_csv(output_dir / "table_dscl_robustness.csv", index=False)
    print(f"\nSaved: {output_dir / 'table_dscl_robustness.csv'}")

    return out


if __name__ == "__main__":
    main()

