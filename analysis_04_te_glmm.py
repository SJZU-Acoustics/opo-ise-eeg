"""
分析4：TE GLMM（16人脑电子集）
Analysis 4: Task Engagement GLMM (16-person EEG subset)

检验脑电任务投入指标在不同SPL条件下是否存在显著差异。
Tests whether Task Engagement (β/(θ+α)) differs across SPL conditions.

TE = β / (θ + α)，使用前额叶数据

模型 / Model:
  TE ~ C(SPL) + noise_sensitivity_z + (1|subject)

预期结果 / Expected results:
  SPL 主效应（LRT, χ²）不显著（约 p≈0.23）

运行 / Run:
  cd 投稿 && ../.venv/bin/python code/analysis_04_te_glmm.py
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
    DATA_SPSS, COLORS, SPL_LABELS,
    set_plot_style, ensure_output_dir
)


def load_data():
    """Load TE data (16-person EEG subset)"""
    df = pd.read_excel(DATA_SPSS, sheet_name="aggregated_eeg")
    # Columns are already English: subject_id, spl_level, noise_sensitivity, TE
    
    # Ensure correct types
    df = df.dropna(subset=["subject_id", "spl_level", "TE"])
    df["subject_id"] = df["subject_id"].astype(int)
    df["spl_level"] = df["spl_level"].astype(int)
    df["spl_level_cat"] = pd.Categorical(df["spl_level"], categories=[0, 1, 2, 3], ordered=True)

    # Noise sensitivity covariate (z-scored)
    if "noise_sensitivity" in df.columns:
        df["noise_z"] = (df["noise_sensitivity"] - df["noise_sensitivity"].mean()) / df["noise_sensitivity"].std(ddof=0)
    else:
        df["noise_z"] = 0.0
    
    return df


def lrt(full, reduced, df_diff):
    """似然比检验"""
    lr = 2 * (full.llf - reduced.llf)
    p = 1 - stats.chi2.cdf(lr, df_diff)
    return lr, p


def run_glmm(df):
    """运行TE GLMM分析"""
    full = mixedlm(
        "TE ~ C(spl_level_cat) + noise_z",
        df, groups=df["subject_id"], re_formula="1"
    ).fit(reml=False, method="lbfgs")
    
    reduced = mixedlm(
        "TE ~ noise_z",
        df, groups=df["subject_id"], re_formula="1"
    ).fit(reml=False, method="lbfgs")
    
    lr, p = lrt(full, reduced, 3)
    
    # 各条件均值
    means = df.groupby("spl_level")["TE"].mean().to_dict()
    sems = df.groupby("spl_level")["TE"].sem().to_dict()
    
    return {
        "chi2": float(lr),
        "df": 3,
        "p": float(p),
        "means": means,
        "sems": sems,
    }


def plot_te_by_condition(df, result, output_dir):
    """绘制TE各条件柱状图"""
    fig, ax = plt.subplots(figsize=(6, 5))
    
    means = [result["means"][i] for i in range(4)]
    sems = [result["sems"][i] for i in range(4)]
    
    bars = ax.bar(
        range(4), means,
        yerr=sems, capsize=4,
        color=[COLORS[i] for i in range(4)],
        edgecolor="black", linewidth=0.8,
        alpha=0.85,
    )
    
    ax.set_xticks(range(4))
    ax.set_xticklabels([SPL_LABELS[i] for i in range(4)])
    ax.set_ylabel("Task Engagement (β/(θ+α))")
    ax.set_xlabel("SPL Level (dB(A))")
    p_txt = f"{result['p']:.3f}" if result["p"] >= 0.001 else "<0.001"
    ax.set_title(f"Task Engagement by SPL Level (n=16)\nLRT: χ²({result['df']})={result['chi2']:.2f}, p={p_txt}")
    
    plt.tight_layout()
    plt.savefig(output_dir / "fig_te_means.png")
    plt.savefig(output_dir / "fig_te_means.svg")
    plt.close()
    print(f"Saved: {output_dir / 'fig_te_means.png'}")


def main():
    set_plot_style()
    output_dir = ensure_output_dir()
    
    print("=" * 60)
    print("Analysis 4: Task Engagement GLMM (16-person EEG subset)")
    print("=" * 60)
    
    df = load_data()
    print(f"\nData: {len(df)} rows, {df['subject_id'].nunique()} subjects")
    
    # 描述统计
    print("\n--- Descriptive Stats ---")
    desc = df.groupby("spl_level")["TE"].agg(["mean", "std"]).round(4)
    print(desc)
    
    # GLMM
    result = run_glmm(df)
    print(f"\n--- GLMM Results ---")
    p_txt = f"{result['p']:.4f}" if result["p"] >= 0.001 else "<0.001"
    print(f"LRT χ²({result['df']}) = {result['chi2']:.3f}, p = {p_txt}")
    sig = "Sig*" if result["p"] < 0.05 else "ns"
    print(f"Conclusion: {sig}")
    
    # 绘图
    plot_te_by_condition(df, result, output_dir)
    
    return result


if __name__ == "__main__":
    main()
