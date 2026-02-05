"""
分析1：行为GLMM（Cr/Rc/P）
Analysis 1: Behavioral GLMM (Cr/Rc/P)

检验不同无关言语声压级条件下，认知绩效指标是否存在显著差异。
Tests whether cognitive performance indices differ significantly across SPL conditions.

模型 / Model:
  DV ~ C(SPL) + noise_sensitivity_z + (1|subject)

预期结果 / Expected results:
  - SPL 主效应（LRT, χ²）在 Cr/Rc/P 上均不显著（ns）

运行 / Run:
  cd 投稿 && ../.venv/bin/python code/analysis_01_behavior_glmm.py
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
    """Load and preprocess data"""
    # Use English data file directly
    df = pd.read_excel(DATA_CLEANED, sheet_name="overall_behavior_eda")
    
    # Columns are already English in translation, but let's be explicit/safe or just use them
    # translate_data.py sets: spl_level, subject_id, noise_sensitivity, Cr, Rc, P
    # But wait, translate_data kept "noise_sensitivity" as "noise_sensitivity" or "noise"?
    # translate_data.py: "噪声": "noise_sensitivity" -> So col name is "noise_sensitivity"
    
    df = df.dropna(subset=["subject_id", "spl_level"])
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


def run_glmm(df, dv):
    """对单个因变量运行GLMM"""
    data = df.dropna(subset=[dv])
    
    full = mixedlm(
        f"{dv} ~ C(spl_level_cat) + noise_z",
        data, groups=data["subject_id"], re_formula="1"
    ).fit(reml=False, method="lbfgs")
    
    reduced = mixedlm(
        f"{dv} ~ noise_z",
        data, groups=data["subject_id"], re_formula="1"
    ).fit(reml=False, method="lbfgs")
    
    lr, p = lrt(full, reduced, 3)
    
    # 各条件均值与SE
    means = data.groupby("spl_level")[dv].mean()
    sems = data.groupby("spl_level")[dv].sem()
    
    return {
        "dv": dv,
        "chi2": float(lr),
        "df": 3,
        "p": float(p),
        "means": means.to_dict(),
        "sems": sems.to_dict(),
        "n": len(data),
    }


def plot_behavior_means(results, output_dir):
    """绘制行为指标各条件均值柱状图"""
    fig, axes = plt.subplots(1, 3, figsize=(10, 4))
    
    for ax, res in zip(axes, results):
        dv = res["dv"]
        means = [res["means"][i] for i in range(4)]
        sems = [res["sems"][i] for i in range(4)]
        
        bars = ax.bar(
            range(4), means,
            yerr=sems, capsize=4,
            color=[COLORS[i] for i in range(4)],
            edgecolor="black", linewidth=0.8,
            alpha=0.85,
        )
        ax.set_xticks(range(4))
        ax.set_xticklabels([SPL_LABELS[i] for i in range(4)], fontsize=9)
        ax.set_ylabel(dv)
        ax.set_xlabel("SPL Level (dB(A))")
        ax.set_title(f"{dv} (LRT: χ²({res['df']})={res['chi2']:.2f}, p={res['p']:.3f})")
    
    plt.tight_layout()
    plt.savefig(output_dir / "fig_behavior_means.png")
    plt.savefig(output_dir / "fig_behavior_means.svg")
    plt.close()
    print(f"Saved: {output_dir / 'fig_behavior_means.png'}")


def main():
    set_plot_style()
    output_dir = ensure_output_dir()
    
    print("=" * 60)
    print("Analysis 1: Behavioral GLMM (Cr/Rc/P)")
    print("=" * 60)
    
    df = load_data()
    print(f"\nData: {len(df)} rows, {df['subject_id'].nunique()} subjects")
    print(f"Data: {len(df)} rows, {df['subject_id'].nunique()} subjects\n")
    
    results = []
    for dv in ["Cr", "Rc", "P"]:
        res = run_glmm(df, dv)
        results.append(res)
        p_txt = f"{res['p']:.4f}" if res["p"] >= 0.001 else "<0.001"
        print(f"{dv}: LRT χ²({res['df']})={res['chi2']:.3f}, p={p_txt}")
    
    # 输出表格
    print(f"\n--- Results Table ---")
    print(f"{'Metric':<8} {'Chi2(df=3)':<14} {'p':<10} {'Significance'}")
    print("-" * 40)
    for res in results:
        sig = "Sig*" if res["p"] < 0.05 else "ns"
        p_txt = f"{res['p']:.4f}" if res["p"] >= 0.001 else "<0.001"
        print(f"{res['dv']:<8} {res['chi2']:<14.3f} {p_txt:<10} {sig}")
    
    # 绘图
    plot_behavior_means(results, output_dir)
    
    # 保存CSV
    pd.DataFrame(results).to_csv(output_dir / "table_behavior_glmm.csv", index=False)
    print(f"Saved: {output_dir / 'table_behavior_glmm.csv'}")
    
    return results


if __name__ == "__main__":
    main()
