"""
共享配置：颜色方案、标签、路径等
Shared configuration: color scheme, labels, paths
"""

from pathlib import Path

# Headless-safe matplotlib setup:
# - macOS default GUI backends can crash in sandboxed/non-GUI runs
# - Matplotlib default cache dirs may be non-writable under sandbox restrictions
import os
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

# ========== 路径配置 / Paths ==========
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "figures"


DATA_CLEANED = DATA_DIR / "data_behavior_eda.xlsx"
DATA_SPSS = DATA_DIR / "data_eeg.xlsx"

# ========== 被试子集 / Subject subsets ==========
# 16人脑电子集（排除采样率低或数据异常的8人）
# 16-person EEG subset (excluded 8 with low sampling rate or abnormal data)
EEG_EXCLUDED_SUBJECTS = [1, 2, 3, 7, 9, 15, 20, 21]

# ========== 颜色方案 / Color scheme ==========
COLORS = {
    0: "#66BB6A",  # quiet - 绿色 green
    1: "#42A5F5",  # lower - 蓝色 blue
    2: "#FFA726",  # medium - 橙色 orange
    3: "#EF5350",  # higher - 红色 red
}

# 用于连续图的渐变色
COLORS_GRADIENT = {
    0: "#A5D6A7",
    1: "#90CAF9",
    2: "#FFCC80",
    3: "#EF9A9A",
}

# ========== 标签 / Labels ==========
SPL_LABELS = {
    0: "Quiet\n(35-40)",
    1: "Lower\n(40-45)",
    2: "Medium\n(45-50)",
    3: "Higher\n(50-55)",
}

SPL_LABELS_SHORT = {
    0: "Quiet",
    1: "Lower",
    2: "Medium",
    3: "Higher",
}

SPL_LABELS_DB = {
    0: "35-40 dB(A)",
    1: "40-45 dB(A)",
    2: "45-50 dB(A)",
    3: "50-55 dB(A)",
}

# ========== 绘图风格 / Plot style ==========
import matplotlib.pyplot as plt

def set_plot_style():
    """设置统一的绘图风格"""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'font.family': 'Arial',
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1,
    })


def ensure_output_dir():
    """确保输出目录存在"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR
