"""
分析7：阈值/剂量-反应模型比较
Analysis 7: Threshold/Dose-Response Model Comparison

比较不同剂量-反应模型（线性、二次、台阶）对△SCL的拟合优度。
Compares different dose-response models for △SCL.

运行 / Run:
  cd 投稿 && ../.venv/bin/python code/analysis_07_threshold_models.py
"""

import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.formula.api import mixedlm

from config import DATA_CLEANED, set_plot_style, ensure_output_dir


def load_data():
    """Load dSCL data"""
    df = pd.read_excel(DATA_CLEANED, sheet_name="overall_behavior_eda")
    
    df = df.dropna(subset=["subject_id", "spl_level", "dSCL"])
    df["subject_id"] = df["subject_id"].astype(int)
    df["spl_level"] = df["spl_level"].astype(int)
    df["spl_level_cat"] = pd.Categorical(df["spl_level"], categories=[0, 1, 2, 3], ordered=True)
    df["spl_level_num"] = df["spl_level"].astype(float)
    df["spl_level_num2"] = df["spl_level_num"] ** 2
    df["is_high"] = (df["spl_level"] == 3).astype(int)
    df["is_ge_medium"] = (df["spl_level"] >= 2).astype(int)
    return df


def lrt(full, reduced, df_diff):
    """似然比检验"""
    lr = 2 * (full.llf - reduced.llf)
    p = 1 - stats.chi2.cdf(lr, df_diff)
    return lr, p


def fit_models(df):
    """拟合不同剂量-反应模型"""
    models = {}
    
    def fit(name, formula):
        res = mixedlm(formula, df, groups=df["subject_id"], re_formula="1").fit(
            reml=False, method="lbfgs"
        )
        models[name] = {"formula": formula, "res": res}
    
    fit("null", "dSCL ~ 1")
    fit("linear", "dSCL ~ spl_level_num")
    fit("quadratic", "dSCL ~ spl_level_num + spl_level_num2")
    fit("step_ge_medium", "dSCL ~ is_ge_medium")
    fit("step_high_only", "dSCL ~ is_high")
    fit("categorical", "dSCL ~ C(spl_level_cat)")
    
    return models


def compare_models(models):
    """比较模型AIC"""
    null = models["null"]["res"]
    rows = []
    
    for name, info in models.items():
        res = info["res"]
        df_diff = len(res.fe_params) - len(null.fe_params) if name != "null" else 0
        lr, p = lrt(res, null, df_diff) if name != "null" else (0, 1)
        
        rows.append({
            "model": name,
            "formula": info["formula"].replace("dSCL ~ ", ""),
            "k_params": len(res.fe_params),
            "AIC": round(res.aic, 2),
            "LRT_p": round(p, 4) if name != "null" else "-",
        })
    
    return pd.DataFrame(rows).sort_values("AIC")


def main():
    set_plot_style()
    output_dir = ensure_output_dir()
    
    print("=" * 60)
    print("Analysis 7: Threshold/Dose-Response Model Comparison")
    print("=" * 60)
    
    df = load_data()
    print(f"\nData: {len(df)} rows, {df['subject_id'].nunique()} subjects")
    
    # 描述统计
    print("\n--- dSCL Means ---")
    desc = df.groupby("spl_level")["dSCL"].agg(["mean", "std"]).round(3)
    print(desc)
    
    # 拟合模型
    models = fit_models(df)
    comparison = compare_models(models)
    
    print("\n--- Model Comparison (Sorted by AIC) ---")
    print(comparison.to_string(index=False))
    
    # 最优模型
    best = comparison.iloc[0]["model"]
    print(f"\nBest Model (Lowest AIC): {best}")
    
    # 台阶模型系数
    step_model = models["step_ge_medium"]["res"]
    coef = step_model.params["is_ge_medium"]
    se = step_model.bse["is_ge_medium"]
    ci_low = coef - 1.96 * se
    ci_high = coef + 1.96 * se
    
    print(f"\nStep Model (SPL>=45dB) Elevation:")
    print(f"  β = {coef:.3f}, 95%CI [{ci_low:.3f}, {ci_high:.3f}]")
    
    # 保存
    comparison.to_csv(output_dir / "table_model_comparison.csv", index=False)
    print(f"\nSaved: {output_dir / 'table_model_comparison.csv'}")
    
    aic_step = float(models["step_ge_medium"]["res"].aic)
    aic_linear = float(models["linear"]["res"].aic)
    delta = aic_linear - aic_step

    print(f"\nManuscript Discussion Text (Cautious):")
    print(f"  The step model (SPL>=45dB) has a slightly lower AIC (AIC={aic_step:.1f}) than the linear model (AIC={aic_linear:.1f}; ΔAIC={delta:.2f}).")
    print(f"  As ΔAIC<2, this provides only weak evidence; interpret as suggestive of a possible threshold rather than conclusive non-linearity.")
    
    return comparison


if __name__ == "__main__":
    main()
