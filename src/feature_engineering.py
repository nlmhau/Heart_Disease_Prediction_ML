# ============================================================
# FEATURE_ENGINEERING.PY
# Dự án: Dự đoán Bệnh Tim
# Mục tiêu:
#   - Tạo đặc trưng y khoa có ý nghĩa
#   - Phục vụ EDA và mô hình học máy
# ============================================================

import pandas as pd

# ============================================================
# I. TẠO ĐẶC TRƯNG Y KHOA
# ============================================================

def create_medical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tạo các đặc trưng dựa trên kiến thức y khoa
    """

    # --------------------------------------------------------
    # 1. Nguy cơ tim mạch RẤT CAO
    # Điều kiện:
    #   - Huyết áp tâm thu ≥ 140 mmHg (THA độ 2)
    #   - Cholesterol toàn phần ≥ 240 mg/dL
    # Ý nghĩa:
    #   - Nhóm bệnh nhân có nguy cơ tim mạch nghiêm trọng
    # --------------------------------------------------------
    df["NguyCo_TimMach_RatCao"] = (
        (df["Huyết_Áp_Nghỉ"] >= 140) &
        (df["Cholesterol"] >= 240)
    ).astype(int)

    # --------------------------------------------------------
    # 2. Tỷ lệ Cholesterol / Tuổi
    # Giả thuyết:
    #   - Cholesterol cao ở người trẻ nguy hiểm hơn
    # --------------------------------------------------------
    df["Cholesterol_Tuoi"] = df["Cholesterol"] / df["Tuổi"]


    return df
