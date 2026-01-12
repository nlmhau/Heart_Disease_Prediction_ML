# ============================================================
# EDA.PY - PHÂN TÍCH DỮ LIỆU KHÁM PHÁ (SMART VERSION)
# Cải tiến:
#   1. Load dữ liệu từ .pkl (Độc lập)
#   2. Phân loại biến nhị phân chặt chẽ
#   3. Tự động kết luận feature tốt/xấu
# ============================================================

import os
import sys
import pandas as pd
import numpy as np

# Đặt backend matplotlib trước khi import pyplot (tránh lỗi tkinter)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings

warnings.filterwarnings('ignore')

# ------------------------------------------------------------
# 1. CẤU HÌNH
# ------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EDA_DATA_PATH = os.path.join(BASE_DIR, "../saved_models/eda_dataset.pkl")
FIGURES_DIR = os.path.join(BASE_DIR, "../outputs/figures")
OUTPUTS_DIR = os.path.join(BASE_DIR, "../outputs")
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)

# Logger: In console + ghi file
class Logger:
    def __init__(self, log_file):
        self.log_file = open(log_file, 'w', encoding='utf-8')
    
    def log(self, message=""):
        print(message)
        self.log_file.write(message + '\n')
        self.log_file.flush()
    
    def close(self):
        self.log_file.close()

logger = Logger(os.path.join(OUTPUTS_DIR, "eda_log.txt"))

sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['figure.dpi'] = 150
PALETTE = {"0": "#2ecc71", "1": "#e74c3c"}  # String keys cho seaborn

def save_plot(filename):
    path = os.path.join(FIGURES_DIR, filename)
    plt.tight_layout()
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    logger.log(f"  [SAVED PLOT] {filename}")

# ------------------------------------------------------------
# 2. LOAD DATA (KIẾN TRÚC MỚI)
# ------------------------------------------------------------
def load_and_classify_data():
    logger.log("=" * 70)
    logger.log("I. TẢI DỮ LIỆU TỪ FILE ĐÃ XỬ LÝ (DECOUPLED)")
    logger.log("=" * 70)
    
    if not os.path.exists(EDA_DATA_PATH):
        logger.log(f"❌ Lỗi: Không tìm thấy file '{EDA_DATA_PATH}'")
        logger.log("👉 Vui lòng chạy 'python src/preprocessing.py' trước để tạo dữ liệu!")
        sys.exit(1)

    df = joblib.load(EDA_DATA_PATH)
    target = 'Bệnh_Tim'
    
    # --- PHÂN LOẠI BIẾN (LOGIC CHẶT CHẼ HƠN) ---
    num_cols = []
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    # Quét qua các cột số để xem cái nào là nhị phân thực sự
    potential_nums = df.select_dtypes(include=['number']).columns.tolist()
    if target in potential_nums: potential_nums.remove(target)

    for col in potential_nums:
        # Nếu chỉ chứa {0, 1} -> Biến phân loại nhị phân
        unique_vals = set(df[col].dropna().unique())
        if unique_vals.issubset({0, 1}):
            cat_cols.append(col)
        else:
            num_cols.append(col)

    logger.log(f" [OK] Dataset: {df.shape}")
    logger.log(f" [OK] Biến số ({len(num_cols)}): {num_cols}")
    logger.log(f" [OK] Biến phân loại ({len(cat_cols)}): {cat_cols}")
    
    return df, num_cols, cat_cols, target


# ------------------------------------------------------------
# 3. PHÂN TÍCH THỐNG KÊ & TỰ ĐỘNG KẾT LUẬN
# ------------------------------------------------------------
def report_statistical_significance(df, num_cols, cat_cols, target):
    logger.log("\n" + "=" * 70)
    logger.log("II. PHÂN TÍCH THỐNG KÊ & ĐÁNH GIÁ FEATURE")
    logger.log("=" * 70)

    # 3.1. Biến số
    logger.log("\n1. ĐÁNH GIÁ BIẾN SỐ (MEAN DIFFERENCE)")
    logger.log("-" * 65)
    logger.log(f"{'Biến số':<20} | {'Khỏe':<10} | {'Bệnh':<10} | {'Đánh giá'}")
    logger.log("-" * 65)
    
    for col in num_cols:
        mean_0 = df[df[target]==0][col].mean()
        mean_1 = df[df[target]==1][col].mean()
        
        # Tự động kết luận
        diff_pct = abs(mean_1 - mean_0) / mean_0 * 100
        evaluation = ""
        if diff_pct > 15:
            evaluation = " Tốt (Khác biệt lớn)"
        elif diff_pct > 5:
            evaluation = " Có tiềm năng"
        else:
            evaluation = " Ít khác biệt"

        logger.log(f"{col:<20} | {mean_0:<10.1f} | {mean_1:<10.1f} | {evaluation}")

    # 3.2. Biến phân loại
    logger.log("\n2. ĐÁNH GIÁ BIẾN PHÂN LOẠI (RISK RATIO)")
    logger.log("-" * 60)
    for col in cat_cols:
        logger.log(f"\n Phân tích: {col}")
        ct = pd.crosstab(df[col], df[target], normalize='index') * 100
        logger.log(ct.round(1).to_string())
        
        # Tự động phát hiện nhóm nguy cơ cao
        high_risk_groups = ct[ct[1] > 60].index.tolist()
        if high_risk_groups:
            logger.log(f"  => PHÁT HIỆN: Nhóm {high_risk_groups} có tỷ lệ bệnh > 60% (Nguy cơ cao)")


# ------------------------------------------------------------
# 4. TRỰC QUAN HÓA DỮ LIỆU
# ------------------------------------------------------------
def visualize_all(df, num_cols, cat_cols, target):
    logger.log("\n" + "=" * 70)
    logger.log("III. TRỰC QUAN HÓA DỮ LIỆU")
    logger.log("=" * 70)

    df_plot = df.copy()
    df_plot[target] = df_plot[target].astype(str)

    # 4.1 Biến mục tiêu
    plt.figure(figsize=(6, 4))
    sns.countplot(data=df_plot, x=target, hue=target, palette=PALETTE, legend=False)
    plt.title("Sự mất cân bằng dữ liệu")
    save_plot("1_Target.png")

    # 4.2 Biến số
    logger.log("  Dang ve bieu do bien so...")
    for col in num_cols:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        sns.kdeplot(data=df_plot, x=col, hue=target, fill=True, palette=PALETTE, ax=axes[0], warn_singular=False)
        axes[0].set_title(f'Phân phối: {col}')
        sns.boxplot(data=df_plot, x=target, y=col, palette=PALETTE, ax=axes[1])
        axes[1].set_title(f'Khác biệt giá trị: {col}')
        save_plot(f"Num_{col}.png")

    # 4.3 Biến phân loại
    logger.log("  Dang ve bieu do bien phan loai...")
    for col in cat_cols:
        plt.figure(figsize=(8, 5))
        sns.countplot(data=df_plot, x=col, hue=target, palette=PALETTE)
        plt.title(f'Tỷ lệ bệnh theo: {col}')
        plt.legend(title='Bệnh Tim', loc='upper right')
        save_plot(f"Cat_{col}.png")

    # 4.4 Heatmap
    logger.log("  Dang ve Heatmap...")
    plt.figure(figsize=(14, 12))
    numeric_df = df_plot.select_dtypes(include=['number'])
    sns.heatmap(numeric_df.corr(), annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5, center=0)
    plt.title('Ma trận tương quan')
    save_plot("Correlation_Matrix.png")


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------
def main():
    # 1. Load (Kiến trúc Decoupled)
    df, num_cols, cat_cols, target = load_and_classify_data()
    
    # 2. Báo cáo thông minh (Smart Report)
    report_statistical_significance(df, num_cols, cat_cols, target)
    
    # 3. Vẽ hình
    visualize_all(df, num_cols, cat_cols, target)
    
    logger.log("\n" + "=" * 70)
    logger.log(f" [HOAN THANH] Anh luu tai: {FIGURES_DIR}")
    logger.log(f" [LOG FILE] File log da luu tai: {os.path.join(OUTPUTS_DIR, 'eda_log.txt')}")
    logger.log("=" * 70)
    
    logger.close()

if __name__ == "__main__":
    main()