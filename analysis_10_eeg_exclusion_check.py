"""
分析10（补充）：EEG 剔除样本的选择偏差检查
Analysis 10 (Supplementary): Selection-bias check for EEG exclusions

24 名被试中 EEG 有效样本为 16 名（剔除 8 名）。为回应“剔除是否导致系统性偏差”的疑虑，
本脚本比较保留 vs 剔除被试在行为/EDA/噪声敏感性上的差异（以被试平均值为单位）。

Run:
  cd 投稿 && ../.venv/bin/python code/analysis_10_eeg_exclusion_check.py
"""

import numpy as np
import pandas as pd
from scipy import stats

from config import DATA_CLEANED, EEG_EXCLUDED_SUBJECTS, ensure_output_dir


def hedges_g(x, y):
    """Hedges' g (bias-corrected standardized mean difference)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return float("nan")
    vx = x.var(ddof=1)
    vy = y.var(ddof=1)
    sp = np.sqrt(((nx - 1) * vx + (ny - 1) * vy) / (nx + ny - 2))
    if sp == 0:
        return float("nan")
    d = (x.mean() - y.mean()) / sp
    j = 1 - (3 / (4 * (nx + ny) - 9))
    return float(j * d)


def main():
    output_dir = ensure_output_dir()

    df = pd.read_excel(DATA_CLEANED, sheet_name="overall_behavior_eda")
    df = df.dropna(subset=["subject_id", "spl_level"]).copy()
    df["subject_id"] = df["subject_id"].astype(int)

    metrics = ["Cr", "Rc", "P", "dSCL", "noise_sensitivity"]
    subj = df.groupby("subject_id")[metrics].mean().reset_index()
    subj["eeg_included"] = ~subj["subject_id"].isin(set(EEG_EXCLUDED_SUBJECTS))

    included = subj[subj["eeg_included"]].copy()
    excluded = subj[~subj["eeg_included"]].copy()

    rows = []
    print("=" * 60)
    print("Analysis 10: EEG Exclusion Selection-Bias Check")
    print("=" * 60)
    print(f"\nIncluded n={len(included)}, Excluded n={len(excluded)}")

    for m in metrics:
        x = included[m].dropna().values
        y = excluded[m].dropna().values

        t_res = stats.ttest_ind(x, y, equal_var=False)
        u_res = stats.mannwhitneyu(x, y, alternative="two-sided")
        g = hedges_g(x, y)

        rows.append({
            "metric": m,
            "included_n": int(len(x)),
            "excluded_n": int(len(y)),
            "included_mean": float(np.mean(x)),
            "excluded_mean": float(np.mean(y)),
            "diff_included_minus_excluded": float(np.mean(x) - np.mean(y)),
            "welch_t_p": float(t_res.pvalue),
            "mw_u_p": float(u_res.pvalue),
            "hedges_g": g,
        })

        print(f"\n{m}:")
        print(f"  included mean={np.mean(x):.3f}, excluded mean={np.mean(y):.3f}, diff={np.mean(x)-np.mean(y):.3f}")
        print(f"  Welch t-test p={t_res.pvalue:.4f}, Mann-Whitney p={u_res.pvalue:.4f}, Hedges g={g:.2f}")

    out = pd.DataFrame(rows)
    out.to_csv(output_dir / "table_eeg_exclusion_check.csv", index=False)
    print(f"\nSaved: {output_dir / 'table_eeg_exclusion_check.csv'}")

    return out


if __name__ == "__main__":
    main()

