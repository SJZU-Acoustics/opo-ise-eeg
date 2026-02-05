"""
分析3：△SCL GLMM + 事后比较
Analysis 3: △SCL GLMM with Post-hoc Comparisons

检验皮肤电导水平变化率在不同SPL条件下的差异。
Tests whether △SCL differs significantly across SPL conditions.

模型 / Model:
  △SCL ~ C(SPL) + noise_sensitivity_z + (1|subject)

预期结果 / Expected results:
  - SPL 主效应（LRT, χ²）显著（约 p≈0.04）
  - planned contrast (higher vs lower): p≈0.014

运行 / Run:
  cd 投稿 && ../.venv/bin/python code/analysis_03_dscl_glmm.py
"""

import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.formula.api import mixedlm

# Headless-safe matplotlib backend (prevents macOS GUI backend crashes in sandbox)
import os
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib.pyplot as plt

from config import (
    DATA_CLEANED, COLORS, SPL_LABELS,
    set_plot_style, ensure_output_dir
)


def load_data():
    """Load dSCL data"""
    df = pd.read_excel(DATA_CLEANED, sheet_name="overall_behavior_eda")
    # Columns: subject_id, spl_level, dSCL, noise_sensitivity
    
    df = df.dropna(subset=["subject_id", "spl_level", "dSCL"])
    df["subject_id"] = df["subject_id"].astype(int)
    df["spl_level"] = df["spl_level"].astype(int)
    df["spl_level_cat"] = pd.Categorical(df["spl_level"], categories=[0, 1, 2, 3], ordered=True)
    df["noise_z"] = (df["noise_sensitivity"] - df["noise_sensitivity"].mean()) / df["noise_sensitivity"].std()
    return df


def lrt(full, reduced, df_diff):
    """似然比检验 / Likelihood ratio test"""
    lr = 2 * (full.llf - reduced.llf)
    p = 1 - stats.chi2.cdf(lr, df_diff)
    return lr, p


def run_glmm(df):
    """运行GLMM分析"""
    full = mixedlm(
        "dSCL ~ C(spl_level_cat) + noise_z",
        df, groups=df["subject_id"], re_formula="1"
    ).fit(reml=False, method="lbfgs")
    
    reduced = mixedlm(
        "dSCL ~ noise_z",
        df, groups=df["subject_id"], re_formula="1"
    ).fit(reml=False, method="lbfgs")
    
    lr, p = lrt(full, reduced, 3)
    return {"chi2": float(lr), "df": 3, "p": float(p)}


def run_posthoc(df):
    """事后两两比较（paired t-test）"""
    wide = df.pivot(index="subject_id", columns="spl_level", values="dSCL")
    
    comparisons = []
    pairs = [(1, 3), (0, 3), (1, 2), (0, 2), (2, 3), (0, 1)]
    
    for a, b in pairs:
        diff = wide[b] - wide[a]
        diff = diff.dropna()
        t, p = stats.ttest_1samp(diff, 0)
        mean_diff = diff.mean()
        se = diff.std() / np.sqrt(len(diff))
        ci_low = mean_diff - 1.96 * se
        ci_high = mean_diff + 1.96 * se
        dz = mean_diff / diff.std() if diff.std() > 0 else 0
        
        comparisons.append({
            "contrast": f"{b}-{a}",
            "mean_diff": round(mean_diff, 3),
            "ci_low": round(ci_low, 3),
            "ci_high": round(ci_high, 3),
            "t": round(t, 3),
            "p": round(p, 4),
            "dz": round(dz, 3),
            "n": len(diff),
        })
    
    return comparisons


def plot_dscl_boxplot(df, posthoc_results, output_dir):
    """绘制△SCL箱线图"""
    fig, ax = plt.subplots(figsize=(6, 5))
    
    bp_data = [df[df["spl_level"] == i]["dSCL"].values for i in range(4)]
    bp = ax.boxplot(bp_data, patch_artist=True, positions=[0, 1, 2, 3], widths=0.6)
    
    for i, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(COLORS[i])
        patch.set_alpha(0.75)
        patch.set_edgecolor("black")
    
    for element in ["whiskers", "caps"]:
        for line in bp[element]:
            line.set_color("black")
    
    ax.set_xticks(range(4))
    ax.set_xticklabels([SPL_LABELS[i] for i in range(4)])
    ax.set_ylabel(r"$\Delta$SCL")
    ax.set_xlabel("SPL Level (dB(A))")
    ax.set_title(r"$\Delta$SCL by SPL Level (n=24)")
    
    # 添加显著性标记 (lower vs higher)
    # 查找该对比的p值
    for comp in posthoc_results:
        if comp["contrast"] == "3-1" and comp["p"] < 0.05:
            y_max = max([max(d) for d in bp_data]) + 0.1
            ax.plot([1, 1, 3, 3], [y_max, y_max + 0.08, y_max + 0.08, y_max], 'k-', linewidth=1)
            ax.text(2, y_max + 0.12, '*', ha='center', va='bottom', fontsize=14)
            break
    
    plt.tight_layout()
    plt.savefig(output_dir / "fig_dscl_boxplot.png")
    plt.savefig(output_dir / "fig_dscl_boxplot.svg")
    plt.close()
    print(f"Saved: {output_dir / 'fig_dscl_boxplot.png'}")


def main():
    set_plot_style()
    output_dir = ensure_output_dir()
    
    print("=" * 60)
    print("Analysis 3: dSCL GLMM with Post-hoc")
    print("=" * 60)
    
    df = load_data()
    print(f"\nData: {len(df)} rows, {df['subject_id'].nunique()} subjects")
    
    # 描述统计
    print("\n--- Descriptive Stats ---")
    desc = df.groupby("spl_level")["dSCL"].agg(["mean", "std"]).round(3)
    print(desc)
    
    # GLMM
    glmm_result = run_glmm(df)
    print(f"\n--- GLMM Results ---")
    p_txt = f"{glmm_result['p']:.4f}" if glmm_result["p"] >= 0.001 else "<0.001"
    print(f"LRT χ²({glmm_result['df']}) = {glmm_result['chi2']:.3f}, p = {p_txt}")
    
    # Post-hoc
    posthoc = run_posthoc(df)
    print("\n--- Post-hoc Comparisons (paired t-test) ---")
    print(f"{'Contrast':<10} {'Diff':<10} {'95%CI':<20} {'p':<10} {'dz'}")
    print("-" * 60)
    for comp in posthoc:
        ci = f"[{comp['ci_low']}, {comp['ci_high']}]"
        sig = "*" if comp["p"] < 0.05 else ""
        print(f"{comp['contrast']:<10} {comp['mean_diff']:<10} {ci:<20} {comp['p']:<10} {comp['dz']}{sig}")
    
    # 绘图
    plot_dscl_boxplot(df, posthoc, output_dir)
    
    # 保存CSV
    pd.DataFrame(posthoc).to_csv(output_dir / "table_dscl_posthoc.csv", index=False)
    print(f"Saved: {output_dir / 'table_dscl_posthoc.csv'}")
    
    return {"glmm": glmm_result, "posthoc": posthoc}


if __name__ == "__main__":
    main()
