"""
Master Script: Run all analyses for the manuscript
Run:
  cd submission && ../.venv/bin/python code/run_all.py

Output:
  - figures/*.png/svg - Figures
  - figures/*.csv - Tables
"""

from pathlib import Path
import sys

# 确保可以导入同目录模块
sys.path.insert(0, str(Path(__file__).parent))

from config import set_plot_style, ensure_output_dir


def main():
    set_plot_style()
    output_dir = ensure_output_dir()
    
    print("=" * 70)
    print("Building and Environment Submission Analysis")
    print("Running all 11 analyses for manuscript submission")
    print("=" * 70)
    
    # Analysis 1: Behavior GLMM
    print("\n>>> Running Analysis 1...")
    from analysis_01_behavior_glmm import main as run_1
    run_1()
    
    # Analysis 2: Cr Time Regression
    print("\n>>> Running Analysis 2...")
    from analysis_02_behavior_time_regression import main as run_2
    run_2()
    
    # Analysis 3: dSCL GLMM
    print("\n>>> Running Analysis 3...")
    from analysis_03_dscl_glmm import main as run_3
    run_3()
    
    # Analysis 4: TE GLMM
    print("\n>>> Running Analysis 4...")
    from analysis_04_te_glmm import main as run_4
    run_4()
    
    # Analysis 5: TE Time Regression
    print("\n>>> Running Analysis 5...")
    from analysis_05_te_time_regression import main as run_5
    run_5()
    
    # Analysis 6: Effect Sizes
    print("\n>>> Running Analysis 6...")
    from analysis_06_effect_sizes import main as run_6
    run_6()
    
    # Analysis 7: Threshold Models
    print("\n>>> Running Analysis 7...")
    from analysis_07_threshold_models import main as run_7
    run_7()
    
    # Analysis 8: Moderation
    print("\n>>> Running Analysis 8...")
    from analysis_08_moderation import main as run_8
    run_8()

    # Analysis 9: dSCL robustness checks
    print("\n>>> Running Analysis 9...")
    from analysis_09_dscl_robustness import main as run_9
    run_9()

    # Analysis 10: EEG exclusion selection-bias check
    print("\n>>> Running Analysis 10...")
    from analysis_10_eeg_exclusion_check import main as run_10
    run_10()

    # Analysis 11: Gender robustness checks
    print("\n>>> Running Analysis 11...")
    from analysis_11_gender_robustness import main as run_11
    run_11()
    
    print("\n" + "=" * 70)
    print("All analyses completed!")
    print(f"Figures and tables saved to: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
