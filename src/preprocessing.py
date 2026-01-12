# ============================================================
# PREPROCESSING.PY
# Dự án: Dự đoán Bệnh Tim
# Mục tiêu:
#   - Phát hiện & xử lý dữ liệu lỗi
#   - Chuẩn bị dữ liệu cho EDA
#   - Xây dựng pipeline tiền xử lý cho mô hình ML
#   - Có giải thích y khoa & chứng minh Before/After
# ============================================================

import os
import sys
import numpy as np
import pandas as pd
import joblib
import warnings

from sklearn.model_selection import train_test_split
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import RobustScaler

from feature_engineering import create_medical_features

warnings.filterwarnings("ignore")  # Giữ output sạch cho báo cáo

# ============================================================
# I. CẤU HÌNH & LOGGER
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "../data/heart.csv")
SAVED_MODELS_DIR = os.path.join(BASE_DIR, "../saved_models")
OUTPUTS_DIR = os.path.join(BASE_DIR, "../outputs")
os.makedirs(SAVED_MODELS_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)

VIETNAMESE_COLUMNS = {
    "Age": "Tuổi",
    "Sex": "Giới_tính",
    "ChestPainType": "Loại_Đau_Ngực",
    "RestingBP": "Huyết_Áp_Nghỉ",
    "Cholesterol": "Cholesterol",
    "FastingBS": "Đường_Huyết_Đói",
    "RestingECG": "Điện_Tâm_Đồ",
    "MaxHR": "Nhịp_Tim_Tối_Đa",
    "ExerciseAngina": "Đau_Thắt_Vận_Động",
    "Oldpeak": "Độ_Chênh_ST",
    "ST_Slope": "Độ_Dốc_ST",
    "HeartDisease": "Bệnh_Tim"
}

# ---------- LOGGER ----------
class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout
        self.log = open(
            os.path.join(OUTPUTS_DIR, "preprocessing_log.txt"),
            "w",
            encoding="utf-8"
        )

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

sys.stdout = Logger()

# ============================================================
# II. CÁC HÀM PHÂN TÍCH & LOAD DỮ LIỆU
# ============================================================

def load_data():
    print("\n" + "=" * 60)
    print("II. PHÂN TÍCH & TẢI DỮ LIỆU")
    print("=" * 60)

    if not os.path.exists(DATA_PATH):
        print(f"[LOI] Không tìm thấy file {DATA_PATH}")
        sys.exit(1)

    df = pd.read_csv(DATA_PATH)
    df = df.rename(columns=VIETNAMESE_COLUMNS)

    print("\n2.1 Thông tin tổng quan:")
    print(df.info())

    print("\n2.2 Thống kê mô tả:")
    print(df.describe())

    return df


def inspect_data_quality(df):
    print("\n2.3 Kiểm tra chất lượng dữ liệu (0 & âm)")
    num_cols = df.select_dtypes(include="number").columns

    print(f"{'Cột':<20} | {'giá trị = 0':<12} | {'giá trị < 0':<12}")
    print("-" * 44)
    for col in num_cols:
        print(
            f"{col:<20} | {(df[col] == 0).sum():<12} | {(df[col] < 0).sum():<8}"
        )


def analyze_negative_oldpeak(df):
    print("\n2.4 Phân tích Oldpeak (Độ_Chênh_ST) âm")

    neg = df[df["Độ_Chênh_ST"] < 0]
    n_total = len(neg)
    print(f"  - Số giá trị âm: {n_total}")

    if n_total > 0:
        counts = neg["Bệnh_Tim"].value_counts()
        n_sick = counts.get(1, 0)     # Số người bệnh
        n_healthy = counts.get(0, 0)  # Số người khỏe
        local_rate = (n_sick / n_total) * 100
        global_rate = df["Bệnh_Tim"].mean() * 100 

        print(f"  - Chi tiết nhóm này: {n_sick} người Bệnh (1) | {n_healthy} người Khỏe (0)")
        print(f"  - Tỷ lệ bệnh (Nhóm Oldpeak âm): {local_rate:.2f}%")
        print(f"  - Tỷ lệ bệnh (Trung bình cả tập): {global_rate:.2f}%")
        
        print(
            "  => NHẬN XÉT: Nhóm có Oldpeak âm có tỷ lệ mắc bệnh CAO HƠN mức trung bình.\n"
            "     (Y khoa: Đây là dấu hiệu ST Elevation - Nhồi máu cơ tim cấp).\n"
            "     => QUYẾT ĐỊNH: Dữ liệu có giá trị dự báo cao → GIỮ NGUYÊN."
        )


# ============================================================
# III. LÀM SẠCH DỮ LIỆU CƠ BẢN
# ============================================================

def basic_cleaning(df):
    print("\n" + "=" * 60)
    print("III. LÀM SẠCH DỮ LIỆU CƠ BẢN")
    print("=" * 60)

    for col in ["Cholesterol", "Huyết_Áp_Nghỉ"]:
        count = (df[col] == 0).sum()
        if count > 0:
            df[col] = df[col].replace(0, np.nan)
            print(f"  - {col}: Đã thay {count} giá trị 0 thành NaN")

    return df


# ============================================================
# IV. DATA PROVIDER CHO EDA
# ============================================================

def get_data_for_eda():
    print("\n" + "=" * 60)
    print("IV. CHUẨN BỊ DỮ LIỆU CHO EDA")
    print("=" * 60)

    df = load_data()
    inspect_data_quality(df)
    analyze_negative_oldpeak(df)

    df = basic_cleaning(df)
    
    # Feature Engineering cho EDA
    df = create_medical_features(df)

    cols_impute = [
        "Cholesterol",
        "Huyết_Áp_Nghỉ",
        "Tuổi",
        "Nhịp_Tim_Tối_Đa",
        "Cholesterol_Tuoi"
    ]

    print("\n  Trước impute:")
    for col in cols_impute:
        print(f"   - {col}: {df[col].isna().sum()} NaN")

    imputer = IterativeImputer(random_state=2026)
    df[cols_impute] = imputer.fit_transform(df[cols_impute])

    print("\n  Sau impute:")
    for col in cols_impute:
        print(f"   - {col}: {df[col].isna().sum()} NaN")

    print("\n Dataset EDA sẵn sàng (đã impute, chưa encode & scale)")

    joblib.dump(df, f"{SAVED_MODELS_DIR}/eda_dataset.pkl")
    print(f" [OK] Đã lưu file: {SAVED_MODELS_DIR}/eda_dataset.pkl")

    return df


# ============================================================
# V. PIPELINE TIỀN XỬ LÝ CHO MÔ HÌNH ML
# ============================================================

def process_and_save_data():
    print("\n" + "=" * 60)
    print("V. PIPELINE TIỀN XỬ LÝ CHO MÔ HÌNH")
    print("=" * 60)

    # 1. Load & phân tích
    df = load_data()
    inspect_data_quality(df)
    analyze_negative_oldpeak(df)

    # 2. Clean
    df = basic_cleaning(df)

    # 3. Train/Test Split
    print("\n5.1. Chia Train/Test (80/20)")
    X = df.drop("Bệnh_Tim", axis=1)
    y = df["Bệnh_Tim"]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=2026, stratify=y
    )
    print(f"  Train: {X_train.shape} | Test: {X_test.shape}")

    # 4. Encoding
    print("\n5.2. MÃ HÓA BIẾN PHÂN LOẠI")
    print("  Trước mã hóa:")
    print(f"   - Số cột: {X_train.shape[1]}")
    
    binary_map = {"M": 1, "F": 0, "Y": 1, "N": 0}
    for col in ["Giới_tính", "Đau_Thắt_Vận_Động"]:
        X_train[col] = X_train[col].map(binary_map)
        X_test[col] = X_test[col].map(binary_map)

    X_train = pd.get_dummies(X_train, drop_first=True)
    X_test = pd.get_dummies(X_test, drop_first=True)
    X_train, X_test = X_train.align(X_test, join="left", axis=1, fill_value=0)

    print("  Sau mã hóa:")
    print(f"   - Số cột: {X_train.shape[1]}")
    print(f"   - Ví dụ cột mới: {[c for c in X_train.columns if '_' in c][:8]}")

    # 5. Imputation (CHỈ CÁC CỘT GỐC TRƯỚC)
    print("\n6. ĐIỀN KHUYẾT (Iterative Imputer – FIT TRÊN TRAIN)")
    
    # [LƯU Ý] Bỏ Cholesterol_Tuoi ra vì chưa tạo.
    cols_impute_base = ["Cholesterol", "Huyết_Áp_Nghỉ", "Tuổi", "Nhịp_Tim_Tối_Đa"]

    imputer = IterativeImputer(random_state=2026, max_iter=20)

    print("  - FIT imputer CHỈ trên X_train (Cột gốc)")
    imputer.fit(X_train[cols_impute_base])
    
    # Kiểm tra số lượng NaN
    missing_train = X_train[cols_impute_base].isna().sum()
    missing_test = X_test[cols_impute_base].isna().sum()

    print("  - Transform X_train & X_test")
    X_train[cols_impute_base] = imputer.transform(X_train[cols_impute_base])
    X_test[cols_impute_base] = imputer.transform(X_test[cols_impute_base])

    # In báo cáo chi tiết NaN
    print(" Thống kê số lượng giá trị đã được điền:")
    print(f"  {'Tên cột':<20} | {'Tổng cộng':<10} | {'Train':<8} | {'Test':<8}")
    print("  " + "-" * 55)
    for col in cols_impute_base:
        total = missing_train[col] + missing_test[col]
        if total > 0:
            print(f"  {col:<20} | {total:<10} | {missing_train[col]:<8} | {missing_test[col]:<8}")

    # 7. Feature Engineering (SAU KHI ĐÃ CÓ DỮ LIỆU SẠCH)
    print("\n7. TẠO ĐẶC TRƯNG Y KHOA")

    before_cols = set(X_train.columns)
    X_train = create_medical_features(X_train)
    X_test = create_medical_features(X_test)
    after_cols = set(X_train.columns)
    
    new_features = sorted(list(after_cols - before_cols))
    print("  - Các đặc trưng y khoa mới được tạo:")
    for f in new_features:
        print(f"    + {f}")

    print("  - Tổng số feature sau engineering:", X_train.shape[1])

    # 8. Scaling
    print("\n8. CHUẨN HÓA (RobustScaler – chống outlier)")
    
    # Lúc này mới có Cholesterol_Tuoi để scale
    cols_scale = cols_impute_base + ["Cholesterol_Tuoi", "Độ_Chênh_ST"]

    print(f"  Median Cholesterol trước scale: {X_train['Cholesterol'].median():.2f}")

    scaler = RobustScaler()
    X_train[cols_scale] = scaler.fit_transform(X_train[cols_scale])
    X_test[cols_scale] = scaler.transform(X_test[cols_scale])

    print(f"  Median Cholesterol sau scale: {X_train['Cholesterol'].median():.2f}")

    # 9. Save
    print("\n9. LƯU DỮ LIỆU & MÔ HÌNH TIỀN XỬ LÝ")
    
    joblib.dump(X_train, f"{SAVED_MODELS_DIR}/X_train.pkl")
    joblib.dump(X_test, f"{SAVED_MODELS_DIR}/X_test.pkl")
    joblib.dump(y_train, f"{SAVED_MODELS_DIR}/y_train.pkl")
    joblib.dump(y_test, f"{SAVED_MODELS_DIR}/y_test.pkl")
    joblib.dump(imputer, f"{SAVED_MODELS_DIR}/imputer.pkl")
    joblib.dump(scaler, f"{SAVED_MODELS_DIR}/scaler.pkl")
    joblib.dump(X_train.columns.tolist(), f"{SAVED_MODELS_DIR}/feature_columns.pkl")

    print("\n HOÀN TẤT PIPELINE – SẴN SÀNG CHO MODEL")

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    # Bước 1: Tạo dữ liệu cho EDA
    get_data_for_eda()
    
    # Bước 2: Tạo dữ liệu cho Model
    process_and_save_data()