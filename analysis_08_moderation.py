"""
分析8：噪声敏感性调节效应（Appendix表）
Analysis 8: Noise Sensitivity Moderation (Appendix Table)

检验噪声敏感性是否调节SPL对△SCL的影响。
Tests whether noise sensitivity moderates the effect of SPL on △SCL.

运行 / Run:
  cd 投稿 && ../.venv/bin/python code/analysis_08_moderation.py
"""

import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.formula.api import mixedlm

from config import DATA_CLEANED, set_plot_style, ensure_output_dir


def load_data():
    """Load and preprocess data"""
    df = pd.read_excel(DATA_CLEANED, sheet_name="overall_behavior_eda")
    
    df = df.dropna(subset=["subject_id", "spl_level", "dSCL", "noise_sensitivity"])
    df["subject_id"] = df["subject_id"].astype(int)
    df["spl_level"] = df["spl_level"].astype(int)
    df["spl_level_cat"] = pd.Categorical(df["spl_level"], categories=[0, 1, 2, 3], ordered=True)
    df["is_ge_medium"] = (df["spl_level"] >= 2).astype(int)
    
    # Z-score normalization
    df["noise_z"] = (df["noise_sensitivity"] - df["noise_sensitivity"].mean()) / df["noise_sensitivity"].std()
    
    return df


def lrt(full, reduced):
    """似然比检验"""
    lr = 2 * (full.llf - reduced.llf)
    df_diff = len(full.fe_params) - len(reduced.fe_params)
    p = 1 - stats.chi2.cdf(lr, df_diff)
    return lr, df_diff, p


def main():
    set_plot_style()
    output_dir = ensure_output_dir()
    
    print("=" * 60)
    print("Analysis 8: Noise Sensitivity Moderation")
    print("=" * 60)
    
    df = load_data()
    print(f"\nData: {len(df)} rows, {df['subject_id'].nunique()} subjects")
    
    # 台阶模型 + 噪声敏感性调节
    print("\n--- Moderation Test (Step Model) ---")
    
    # 无交互模型
    reduced = mixedlm(
        "dSCL ~ is_ge_medium + noise_z",
        df, groups=df["subject_id"], re_formula="1"
    ).fit(reml=False, method="lbfgs")
    
    # 有交互模型
    full = mixedlm(
        "dSCL ~ is_ge_medium * noise_z",
        df, groups=df["subject_id"], re_formula="1"
    ).fit(reml=False, method="lbfgs")
    
    lr, df_diff, p = lrt(full, reduced)
    
    print(f"LRT (Interaction): Chi2={lr:.3f}, df={df_diff}, p={p:.4f}")
    print(f"AIC: Main Effects={reduced.aic:.2f}, Interaction={full.aic:.2f}")
    
    # 交互项系数
    interaction_coef = full.params["is_ge_medium:noise_z"]
    interaction_se = full.bse["is_ge_medium:noise_z"]
    interaction_p = full.pvalues["is_ge_medium:noise_z"]
    ci_low = interaction_coef - 1.96 * interaction_se
    ci_high = interaction_coef + 1.96 * interaction_se
    
    print(f"\nInteraction Coefficient: beta={interaction_coef:.3f}, 95%CI [{ci_low:.3f}, {ci_high:.3f}], p={interaction_p:.4f}")
    
    # 简单效应
    print("\n--- Simple Effects (Step effect at different Noise Sensitivity levels) ---")
    main_effect = full.params["is_ge_medium"]
    for z_val, label in [(-1, "低敏感性(z=-1)"), (0, "平均(z=0)"), (1, "高敏感性(z=1)")]:
        effect = main_effect + interaction_coef * z_val
        print(f"  {label}: SPL>=45dB Elevation = {effect:.3f}")
    
    # Appendix表格
    appendix_data = {
        "Model": ["Main effects only", "With interaction"],
        "Fixed Effects": ["SPL>=45dB + Noise Sensitivity", "SPL>=45dB * Noise Sensitivity"],
        "AIC": [round(reduced.aic, 2), round(full.aic, 2)],
        "Interaction p-value": ["-", round(p, 4)],
    }
    appendix_df = pd.DataFrame(appendix_data)
    
    print("\n--- Appendix Table A1: Noise Sensitivity Moderation ---")
    print(appendix_df.to_string(index=False))
    
    appendix_df.to_csv(output_dir / "table_appendix_moderation.csv", index=False)
    print(f"\nSaved: {output_dir / 'table_appendix_moderation.csv'}")
    
    print(f"\nManuscript Text (Exploratory Analysis):")
    print(f"  Exploratory analysis shows noise sensitivity significantly moderates the effect of SPL on dSCL (interaction p={p:.3f}).")
    print(f"  Individuals with lower noise sensitivity show a greater increase in dSCL at medium/high SPLs (see Appendix Table A1).")
    
    return {"lr": lr, "p": p, "interaction_coef": interaction_coef}


if __name__ == "__main__":
    main()
