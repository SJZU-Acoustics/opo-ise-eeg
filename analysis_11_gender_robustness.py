"""
分析11：性别稳健性检验（△SCL, TE）
Analysis 11: Gender robustness checks (dSCL, TE)

目的 / Goal
  在不改变主分析框架的前提下，探索性检验性别是否调节 SPL 对主要生理指标的影响：
    1) dSCL（n=24）
    2) TE（n=16，EEG子集）

模型 / Models（均以 ML 拟合，reml=False）
  M0: DV ~ C(SPL) + noise_z + (1|subject)
  M1: DV ~ C(SPL) + noise_z + C(gender) + (1|subject)
  M2: DV ~ C(SPL) + noise_z + C(gender) + C(SPL):C(gender) + (1|subject)

检验 / Tests（LRT）
  - Gender main effect: M1 vs M0 (df=1)
  - Gender × SPL interaction: M2 vs M1 (df=3)

输出 / Output
  - figures/table_gender_robustness.csv

运行 / Run
  cd 投稿 && ../.venv/bin/python code/analysis_11_gender_robustness.py
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.formula.api import mixedlm

from config import DATA_CLEANED, DATA_SPSS, ensure_output_dir, set_plot_style


def _find_gender_col(columns) -> str:
    candidates = [c for c in columns if str(c).strip().startswith("性别")]
    if not candidates:
        raise KeyError("Gender column not found (expected header starting with '性别').")
    return candidates[0]


def _gender_map_from_timeseries(path: str, sheet: str) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name=sheet)
    gender_col = _find_gender_col(df.columns)
    df = df.dropna(subset=["subject_id", gender_col])
    df = df[["subject_id", gender_col]].copy()
    df["subject_id"] = df["subject_id"].astype(int)
    df["gender"] = df[gender_col].astype(int)
    # Sanity: single gender code per subject
    chk = df.groupby("subject_id")["gender"].nunique()
    bad = chk[chk > 1]
    if len(bad) > 0:
        raise ValueError(f"Inconsistent gender coding for subjects: {bad.index.tolist()}")
    return df.drop_duplicates(subset=["subject_id"], keep="first")[["subject_id", "gender"]]


def load_dscl() -> pd.DataFrame:
    df = pd.read_excel(DATA_CLEANED, sheet_name="overall_behavior_eda")
    df = df.dropna(subset=["subject_id", "spl_level", "dSCL", "noise_sensitivity"]).copy()
    df["subject_id"] = df["subject_id"].astype(int)
    df["spl_level"] = df["spl_level"].astype(int)
    df["spl_level_cat"] = pd.Categorical(df["spl_level"], categories=[0, 1, 2, 3], ordered=True)
    df["noise_z"] = (df["noise_sensitivity"] - df["noise_sensitivity"].mean()) / df["noise_sensitivity"].std(ddof=0)

    gender_map = _gender_map_from_timeseries(DATA_CLEANED, sheet="time_series_behavior")
    df = df.merge(gender_map, on="subject_id", how="left")
    if df["gender"].isna().any():
        missing = sorted(df.loc[df["gender"].isna(), "subject_id"].unique().tolist())
        raise ValueError(f"Missing gender for subjects (dSCL): {missing}")
    df["gender_cat"] = pd.Categorical(df["gender"], categories=[0, 1], ordered=False)
    return df


def load_te() -> pd.DataFrame:
    df = pd.read_excel(DATA_SPSS, sheet_name="aggregated_eeg")
    df = df.dropna(subset=["subject_id", "spl_level", "TE", "noise_sensitivity"]).copy()
    df["subject_id"] = df["subject_id"].astype(int)
    df["spl_level"] = df["spl_level"].astype(int)
    df["spl_level_cat"] = pd.Categorical(df["spl_level"], categories=[0, 1, 2, 3], ordered=True)
    df["noise_z"] = (df["noise_sensitivity"] - df["noise_sensitivity"].mean()) / df["noise_sensitivity"].std(ddof=0)

    gender_map = _gender_map_from_timeseries(DATA_SPSS, sheet="time_series_eeg")
    df = df.merge(gender_map, on="subject_id", how="left")
    if df["gender"].isna().any():
        missing = sorted(df.loc[df["gender"].isna(), "subject_id"].unique().tolist())
        raise ValueError(f"Missing gender for subjects (TE): {missing}")
    df["gender_cat"] = pd.Categorical(df["gender"], categories=[0, 1], ordered=False)
    return df


def _lrt(full, reduced, df_diff: int) -> tuple[float, float]:
    lr = 2 * (full.llf - reduced.llf)
    p = 1 - stats.chi2.cdf(lr, df_diff)
    return float(lr), float(p)


def _fit_mixedlm(formula: str, df: pd.DataFrame):
    return mixedlm(formula, df, groups=df["subject_id"], re_formula="1").fit(
        reml=False, method="lbfgs"
    )


@dataclass(frozen=True)
class GenderRobustnessResult:
    dv: str
    n_subjects: int
    n_rows: int
    n_female: int
    n_male: int
    test: str
    chi2: float
    df: int
    p: float


def run_gender_robustness(df: pd.DataFrame, dv: str) -> list[GenderRobustnessResult]:
    n_subjects = int(df["subject_id"].nunique())
    n_rows = int(len(df))
    n_female = int((df.drop_duplicates("subject_id")["gender"] == 0).sum())
    n_male = int((df.drop_duplicates("subject_id")["gender"] == 1).sum())

    m0 = _fit_mixedlm(f"{dv} ~ C(spl_level_cat) + noise_z", df)
    m1 = _fit_mixedlm(f"{dv} ~ C(spl_level_cat) + noise_z + C(gender_cat)", df)
    m2 = _fit_mixedlm(f"{dv} ~ C(spl_level_cat) + noise_z + C(gender_cat) + C(spl_level_cat):C(gender_cat)", df)

    chi2_gender, p_gender = _lrt(m1, m0, df_diff=1)
    chi2_int, p_int = _lrt(m2, m1, df_diff=3)

    return [
        GenderRobustnessResult(
            dv=dv,
            n_subjects=n_subjects,
            n_rows=n_rows,
            n_female=n_female,
            n_male=n_male,
            test="Gender main effect (M1 vs M0)",
            chi2=chi2_gender,
            df=1,
            p=p_gender,
        ),
        GenderRobustnessResult(
            dv=dv,
            n_subjects=n_subjects,
            n_rows=n_rows,
            n_female=n_female,
            n_male=n_male,
            test="Gender×SPL interaction (M2 vs M1)",
            chi2=chi2_int,
            df=3,
            p=p_int,
        ),
    ]


def main():
    set_plot_style()
    output_dir = ensure_output_dir()

    print("=" * 60)
    print("Analysis 11: Gender robustness checks (dSCL, TE)")
    print("=" * 60)

    results: list[GenderRobustnessResult] = []

    df_dscl = load_dscl()
    results.extend(run_gender_robustness(df_dscl, dv="dSCL"))

    df_te = load_te()
    results.extend(run_gender_robustness(df_te, dv="TE"))

    out = pd.DataFrame([r.__dict__ for r in results])
    out["p"] = out["p"].astype(float)
    out.to_csv(output_dir / "table_gender_robustness.csv", index=False)
    print(f"Saved: {output_dir / 'table_gender_robustness.csv'}")

    print("\n--- LRT Results ---")
    for r in results:
        p_txt = f"{r.p:.4f}" if r.p >= 0.001 else "<0.001"
        print(f"{r.dv:<4} | {r.test:<32} | χ²({r.df})={r.chi2:.3f}, p={p_txt} | n={r.n_subjects} (F={r.n_female}, M={r.n_male})")

    return out


if __name__ == "__main__":
    main()

