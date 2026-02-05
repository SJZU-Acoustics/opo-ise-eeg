"""
分析6：效应量与95% CI
Analysis 6: Effect Sizes and 95% Confidence Intervals

为Results 3.2提供△SCL的效应量(Cohen's dz)和置信区间。
Provides effect sizes (Cohen's dz) and CIs for △SCL results.

运行 / Run:
  cd 投稿 && ../.venv/bin/python code/analysis_06_effect_sizes.py
"""

import pandas as pd
import numpy as np
from scipy import stats

from config import DATA_CLEANED, set_plot_style, ensure_output_dir


def load_data():
    """Load dSCL data"""
    df = pd.read_excel(DATA_CLEANED, sheet_name="overall_behavior_eda")
    # Columns are English: subject_id, spl_level, dSCL
    
    df = df.dropna(subset=["subject_id", "spl_level", "dSCL"])
    df["subject_id"] = df["subject_id"].astype(int)
    df["spl_level"] = df["spl_level"].astype(int)
    return df


def mean_ci_by_condition(df, dv):
    """计算各条件的均值和95%CI"""
    out = df.groupby("spl_level")[dv].agg(["mean", "std", "count"]).reset_index()
    out["se"] = out["std"] / np.sqrt(out["count"])
    tcrit = stats.t.ppf(0.975, out["count"] - 1)
    out["ci_low"] = out["mean"] - tcrit * out["se"]
    out["ci_high"] = out["mean"] + tcrit * out["se"]
    return out


def paired_diff(df, dv, a, b):
    """计算被试内配对差异的效应量"""
    wide = df.pivot(index="subject_id", columns="spl_level", values=dv)
    d = (wide[a] - wide[b]).dropna()
    n = len(d)
    md = float(d.mean())
    sd = float(d.std(ddof=1))
    se = sd / np.sqrt(n)
    tcrit = stats.t.ppf(0.975, n - 1)
    ci_low = md - tcrit * se
    ci_high = md + tcrit * se
    t, p = stats.ttest_1samp(d, 0.0)
    dz = md / sd if sd > 0 else float("nan")
    return {
        "contrast": f"{a}-{b}",
        "n": n,
        "mean_diff": round(md, 3),
        "ci_low": round(ci_low, 3),
        "ci_high": round(ci_high, 3),
        "p": round(p, 4),
        "cohen_dz": round(dz, 2),
    }


def main():
    set_plot_style()
    output_dir = ensure_output_dir()
    
    print("=" * 60)
    print("Analysis 6: Effect Sizes and 95% CI")
    print("=" * 60)
    
    df = load_data()
    print(f"\nData: {len(df)} rows, {df['subject_id'].nunique()} subjects")
    
    # 各条件均值与CI
    print("\n--- dSCL Means and 95% CI by Condition ---")
    means = mean_ci_by_condition(df, "dSCL")
    print(means[["spl_level", "mean", "ci_low", "ci_high"]].to_string(index=False))
    
    # 关键对比
    print("\n--- dSCL Within-Subject Contrasts (Effect Sizes) ---")
    contrasts = [(3, 1), (3, 0), (2, 1), (3, 2)]
    results = []
    for a, b in contrasts:
        res = paired_diff(df, "dSCL", a, b)
        results.append(res)
        sig = "*" if res["p"] < 0.05 else ""
        print(f"{res['contrast']}: Δ={res['mean_diff']}, 95%CI [{res['ci_low']}, {res['ci_high']}], p={res['p']}{sig}, dz={res['cohen_dz']}")
    
    # 保存CSV
    pd.DataFrame(results).to_csv(output_dir / "table_effect_sizes.csv", index=False)
    print(f"\nSaved: {output_dir / 'table_effect_sizes.csv'}")
    
    # 返回关键对比供稿件使用
    key_contrast = [r for r in results if r["contrast"] == "3-1"][0]
    print(f"\nManuscript formatted text:")
    print(f"  higher(3) vs lower(1): delta dSCL = +{key_contrast['mean_diff']}, 95%CI [{key_contrast['ci_low']}, {key_contrast['ci_high']}], Cohen's dz = {key_contrast['cohen_dz']} (medium effect)")
    
    return results


if __name__ == "__main__":
    main()
