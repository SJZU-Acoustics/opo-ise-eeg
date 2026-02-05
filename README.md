Python scripts for reproducing the statistical analyses and figures for the manuscript "Effects of irrelevant speech on cognitive and behavioural outcomes in open-plan offices".

## Requirements

- Python 3.11+

- Dependencies: `pandas`, `numpy`, `scipy`, `statsmodels`, `matplotlib`, `openpyxl`

- Install: 

  ```bash
  pip install -r requirements.txt
  ```

## File Structure

- `config.py`: Shared configuration (paths, color schemes, plot styles).
- `run_all.py`: Master script to run all analyses sequentially.
- **Core Analyses (Manuscript)**:
  - `analysis_01_behavior_glmm.py`: Analysis 1 - Behavioral GLMM (Cr/Rc/P).
  - `analysis_02_behavior_time_regression.py`: Analysis 2 - Behavioral Time Trend via LMM (Revised Table 1).
  - `analysis_03_dscl_glmm.py`: Analysis 3 - EDA (dSCL) GLMM & Post-hoc (Fig 6).
  - `analysis_04_te_glmm.py`: Analysis 4 - EEG (TE) LMM (includes noise sensitivity covariate).
  - `analysis_05_te_time_regression.py`: Analysis 5 - EEG Time Trend via LMM (Revised Table 2).
- **Supplementary Analyses (Appendix)**:
  - `analysis_06_effect_sizes.py`: Analysis 6 - Effect sizes (Cohen's dz) & CIs.
  - `analysis_07_threshold_models.py`: Analysis 7 - Threshold/Dose-response model comparison (Table A3).
  - `analysis_08_moderation.py`: Analysis 8 - Noise Sensitivity Moderation (Table A4).
  - `analysis_09_dscl_robustness.py`: Analysis 9 - Robustness checks for key △SCL contrasts.
  - `analysis_10_eeg_exclusion_check.py`: Analysis 10 - Selection-bias check for EEG exclusions.
  - `analysis_11_gender_robustness.py`: Analysis 11 - Gender robustness checks (Table A6).

## Usage

1. Ensure data files (`data_behavior_eda.xlsx`, `data_eeg.xlsx`) are in the `../data/` directory (relative to this code folder).

2. Run the master script:

   ```bash
   python run_all.py
   ```

3. Outputs/Figures will be saved to `../figures/`.
