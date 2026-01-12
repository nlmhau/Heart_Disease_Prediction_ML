# ============================================================
# APP.PY – DASHBOARD DỰ ĐOÁN BỆNH TIM
# Nhóm 7 - S26-65TTNT
# Mục tiêu:
#   - Tải dữ liệu hoặc chọn file mẫu
#   - Hiển thị biểu đồ phân tích (EDA)
#   - Cho phép nhập input và hiển thị dự đoán
#   - So sánh 3 mô hình: Random Forest, XGBoost, Neural Network
#   - Hiển thị Hidden Patterns được phát hiện
# ============================================================

import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from PIL import Image

# ============================================================
# 0. CẤU HÌNH GIAO DIỆN
# ============================================================

st.set_page_config(
    page_title="Dashboard Dự Đoán Bệnh Tim",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS tùy chỉnh
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #e74c3c;
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #f8f9fa 0%, #ffffff 100%);
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #3498db;
    }
    .warning-box {
        background-color: #fff3cd;
        border-left: 5px solid #ffc107;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">🫀 DASHBOARD DỰ ĐOÁN NGUY CƠ BỆNH TIM</div>', unsafe_allow_html=True)

st.markdown("""
<div class="warning-box">
    <strong>⚠️ LƯU Ý:</strong> Ứng dụng này chỉ mang tính chất hỗ trợ nghiên cứu và minh họa khoa học. 
    <strong>KHÔNG THAY THẾ</strong> việc chẩn đoán và điều trị y khoa chuyên nghiệp.
</div>
""", unsafe_allow_html=True)

# ============================================================
# 1. LOAD PIPELINE & MODELS (Cache để tối ưu performance)
# ============================================================

@st.cache_resource
def load_all_resources():
    """
    Load tất cả mô hình, scaler, imputer và dữ liệu
    Cache để tránh load lại mỗi lần user tương tác
    """
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    SAVED_MODELS_DIR = os.path.join(BASE_DIR, "../saved_models")
    FIGURES_DIR = os.path.join(BASE_DIR, "../outputs/figures")
    
    # Load 3 mô hình
    rf_model = joblib.load(os.path.join(SAVED_MODELS_DIR, "random_forest.pkl"))
    xgb_model = joblib.load(os.path.join(SAVED_MODELS_DIR, "xgboost.pkl"))
    nn_model = joblib.load(os.path.join(SAVED_MODELS_DIR, "neural_network.pkl"))
    
    # Load metadata
    rf_meta = joblib.load(os.path.join(SAVED_MODELS_DIR, "rf_metadata.pkl"))
    xgb_meta = joblib.load(os.path.join(SAVED_MODELS_DIR, "xgboost_metadata.pkl"))
    nn_meta = joblib.load(os.path.join(SAVED_MODELS_DIR, "nn_metadata.pkl"))
    
    # Load preprocessing tools
    scaler = joblib.load(os.path.join(SAVED_MODELS_DIR, "scaler.pkl"))
    imputer = joblib.load(os.path.join(SAVED_MODELS_DIR, "imputer.pkl"))
    feature_cols = joblib.load(os.path.join(SAVED_MODELS_DIR, "feature_columns.pkl"))
    
    # Load datasets
    eda_df = joblib.load(os.path.join(SAVED_MODELS_DIR, "eda_dataset.pkl"))
    X_test = joblib.load(os.path.join(SAVED_MODELS_DIR, "X_test.pkl"))
    y_test = joblib.load(os.path.join(SAVED_MODELS_DIR, "y_test.pkl"))
    
    return {
        'models': {'rf': rf_model, 'xgb': xgb_model, 'nn': nn_model},
        'metadata': {'rf': rf_meta, 'xgb': xgb_meta, 'nn': nn_meta},
        'preprocessing': {'scaler': scaler, 'imputer': imputer, 'feature_cols': feature_cols},
        'data': {'eda_df': eda_df, 'X_test': X_test, 'y_test': y_test},
        'dirs': {'figures': FIGURES_DIR}
    }

# Load tất cả resources
try:
    resources = load_all_resources()
    models = resources['models']
    metadata = resources['metadata']
    scaler = resources['preprocessing']['scaler']
    imputer = resources['preprocessing']['imputer']
    feature_cols = resources['preprocessing']['feature_cols']
    eda_df = resources['data']['eda_df']
    X_test = resources['data']['X_test']
    y_test = resources['data']['y_test']
    FIGURES_DIR = resources['dirs']['figures']
except Exception as e:
    st.error(f"❌ Lỗi khi load mô hình: {e}")
    st.info("💡 Hãy chạy các file preprocessing.py → models → evaluation.py trước!")
    st.stop()

# ============================================================
# 2. SIDEBAR - MENU ĐIỀU HƯỚNG
# ============================================================

st.sidebar.title("📋 MENU ĐIỀU HƯỚNG")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Chọn chức năng:",
    [
        "🏠 Trang chủ",
        "📂 Dữ liệu & Mô tả",
        "📊 Phân tích EDA",
        "🔬 So sánh mô hình",
        "🩺 Dự đoán nguy cơ",
        "🔍 Hidden Patterns",
        "📖 Hướng dẫn sử dụng"
    ]
)

st.sidebar.markdown("---")
st.sidebar.info("""
**Nhóm 7 - S26-65TTNT**
- Nguyễn Lê Minh Hậu
- Nguyễn Đức Huy

**Đồ án môn:** Machine Learning
""")

# ============================================================
# 3. TRANG CHỦ
# ============================================================

if page == "🏠 Trang chủ":
    st.header("🏠 TỔNG QUAN DỰ ÁN")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>📊 Dữ liệu</h3>
            <p><strong>918</strong> bệnh nhân</p>
            <p><strong>17</strong> đặc trưng</p>
            <p><strong>55.3%</strong> tỷ lệ bệnh</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>🤖 Mô hình</h3>
            <p><strong>3</strong> mô hình ML</p>
            <p><strong>96.08%</strong> Recall tốt nhất</p>
            <p><strong>XGBoost</strong> hiệu quả nhất</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>🎯 Mục tiêu</h3>
            <p>Phát hiện sớm</p>
            <p>Giảm bỏ sót</p>
            <p>Hỗ trợ quyết định</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.subheader("📌 Quy trình xử lý dữ liệu")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### Pipeline xử lý:
        
        1. **Preprocessing (preprocessing.py)**
           - Xử lý missing values (Cholesterol: 172, BP: 1)
           - Giữ nguyên Oldpeak âm (ý nghĩa y khoa)
           - Chia Train/Test (80/20 stratified)
           - Encoding (Label + One-Hot)
           - Iterative Imputer
           - Feature Engineering (Cholesterol_Tuoi, NguyCo_TimMach_RatCao)
           - RobustScaler
        
        2. **EDA (eda.py)**
           - Phân tích biến số (Mean Difference)
           - Phân tích biến phân loại (Risk Ratio)
           - Visualization (15 biểu đồ)
           - Phát hiện nhóm nguy cơ cao
        
        3. **Modeling**
           - **Random Forest:** GridSearch 36 tổ hợp, Threshold Tuning
           - **XGBoost:** Gradient Boosting, scale_pos_weight=2
           - **Neural Network:** MLP (64,32), Early Stopping
        
        4. **Evaluation (evaluation.py)**
           - So sánh 3 mô hình
           - Phân tích Hidden Patterns
           - Báo cáo chi tiết
        """)
    
    with col2:
        st.markdown("""
        ### Kết quả chính:
        
        **🏆 Mô hình tốt nhất:** XGBoost
        - ✅ Recall: 96.08%
        - ✅ FN: 4 (ít nhất)
        - ✅ 5 nhóm nguy cơ
        - ✅ 102 BN phát hiện
        
        **📊 So sánh:**
        - Random Forest: 95.10% Recall
        - Neural Network: 94.12% Recall
        
        **🔍 Hidden Patterns:**
        - Độ_Chênh_ST > 0.67 → 92%
        - Combo patterns: 2-3 triệu chứng
        - Threshold cụ thể cho từng mức
        """)
    
    st.markdown("---")
    
    st.subheader("🎯 Tính năng Dashboard")
    
    features_col1, features_col2 = st.columns(2)
    
    with features_col1:
        st.markdown("""
        ✅ **Tải & Xem dữ liệu:**
        - Upload file CSV hoặc dùng file mẫu
        - Hiển thị thống kê mô tả
        - Giải thích ý nghĩa các cột
        
        ✅ **Phân tích EDA:**
        - Biểu đồ phân bố
        - Ma trận tương quan
        - Phân tích theo nhóm nguy cơ
        """)
    
    with features_col2:
        st.markdown("""
        ✅ **Dự đoán nguy cơ:**
        - Nhập thông tin bệnh nhân
        - Dự đoán bằng 3 mô hình
        - Hiển thị xác suất chi tiết
        
        ✅ **Hidden Patterns:**
        - 15 mẫu ẩn từ 3 mô hình
        - Threshold cụ thể
        - Giải thích y khoa
        """)

# ============================================================
# 4. DỮ LIỆU & MÔ TẢ
# ============================================================

elif page == "📂 Dữ liệu & Mô tả":
    st.header("📂 DỮ LIỆU VÀ MÔ TẢ")
    
    # Option để chọn dữ liệu
    data_option = st.radio(
        "Chọn nguồn dữ liệu:",
        ["📊 Sử dụng dữ liệu mẫu (đã load sẵn)", "📁 Tải file CSV của bạn"]
    )
    
    if data_option == "📁 Tải file CSV của bạn":
        uploaded_file = st.file_uploader("Chọn file CSV", type=['csv'])
        
        if uploaded_file is not None:
            try:
                df_display = pd.read_csv(uploaded_file)
                st.success(f"✅ Đã tải file thành công! Số dòng: {len(df_display)}")
            except Exception as e:
                st.error(f"❌ Lỗi khi đọc file: {e}")
                df_display = eda_df
        else:
            st.info("💡 Vui lòng chọn file CSV để tiếp tục")
            df_display = eda_df
    else:
        df_display = eda_df
    
    # Hiển thị dữ liệu
    st.subheader("🔎 Xem trước dữ liệu")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Tổng số bệnh nhân", len(df_display))
    with col2:
        st.metric("Số features", len(df_display.columns))
    with col3:
        if 'Bệnh_Tim' in df_display.columns:
            benh_rate = df_display['Bệnh_Tim'].mean() * 100
            st.metric("Tỷ lệ bệnh", f"{benh_rate:.1f}%")
    
    # Hiển thị mẫu dữ liệu
    st.dataframe(df_display.head(20), use_container_width=True, height=400)
    
    # Thống kê mô tả
    st.subheader("📈 Thống kê mô tả")
    st.dataframe(df_display.describe(), use_container_width=True)
    
    st.markdown("---")
    
    # Mô tả các cột
    st.subheader("📘 Ý nghĩa các cột dữ liệu")

    data_desc = pd.DataFrame([
        ["Tuổi", "Tuổi bệnh nhân", "28-77 tuổi", "Nguy cơ tăng theo tuổi"],
        ["Giới_tính", "Giới tính (0=Nữ, 1=Nam)", "Binary", "Nam có nguy cơ cao hơn (63.2%)"],
        ["Loại_Đau_Ngực", "Phân loại đau ngực", "ATA/NAP/ASY/TA", "ASY (không triệu chứng) nguy hiểm nhất (79%)"],
        ["Huyết_Áp_Nghỉ", "Huyết áp tâm thu lúc nghỉ", "mmHg", "Cao huyết áp → tăng nguy cơ"],
        ["Cholesterol", "Cholesterol toàn phần", "mg/dL", "Xơ vữa động mạch nếu cao"],
        ["Đường_Huyết_Đói", "Đường huyết lúc đói (>120mg/dL)", "0/1", "Liên quan tiểu đường (79.4% nếu cao)"],
        ["Điện_Tâm_Đồ", "Kết quả ECG lúc nghỉ", "Normal/ST/LVH", "Bất thường ECG → nguy cơ"],
        ["Nhịp_Tim_Tối_Đa", "Nhịp tim khi gắng sức", "nhịp/phút", "Thấp → khả năng tim kém"],
        ["Đau_Thắt_Vận_Động", "Đau ngực khi vận động", "0=Không, 1=Có", "Có đau → 85.2% nguy cơ bệnh"],
        ["Độ_Chênh_ST", "ST depression (Oldpeak)", "Giá trị thực", "Âm = ST Elevation (nhồi máu cấp)"],
        ["Độ_Dốc_ST", "Hình dạng đoạn ST", "Up/Flat/Down", "Flat/Down → 78-83% nguy cơ"],
        ["Cholesterol_Tuoi", "Cholesterol chia cho tuổi", "Tính toán", "Feature engineering - đánh giá theo tuổi"],
        ["NguyCo_TimMach_RatCao", "Kết hợp nhiều yếu tố nguy cơ", "0/1", "Feature engineering - cảnh báo tổng hợp"],
        ["Bệnh_Tim", "Chẩn đoán bệnh tim", "0=Khỏe, 1=Bệnh", "Biến mục tiêu (target)"]
    ], columns=["Tên cột", "Mô tả", "Giá trị", "Ý nghĩa y khoa"])

    st.dataframe(data_desc, use_container_width=True, height=500)
    
    # Download dữ liệu mẫu
    st.markdown("---")
    st.subheader("💾 Tải dữ liệu mẫu")
    
    csv = df_display.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download CSV",
        data=csv,
        file_name="heart_disease_sample.csv",
        mime="text/csv"
    )

# ============================================================
# 5. PHÂN TÍCH EDA
# ============================================================

elif page == "📊 Phân tích EDA":
    st.header("📊 PHÂN TÍCH DỮ LIỆU KHÁM PHÁ (EDA)")
    
    st.info("💡 Các biểu đồ dưới đây được tạo tự động từ file eda.py trong quá trình training")
    
    # Phân bố biến mục tiêu
    st.subheader("🎯 Phân bố biến mục tiêu (Bệnh Tim)")
    
    fig_path = os.path.join(FIGURES_DIR, "1_Target.png")
    if os.path.exists(fig_path):
        img = Image.open(fig_path)
        st.image(img, use_column_width=True)
    else:
        # Vẽ lại nếu không có file
        fig, ax = plt.subplots(figsize=(10, 5))
        eda_df['Bệnh_Tim'].value_counts().plot(kind='bar', ax=ax, color=['#2ecc71', '#e74c3c'])
        ax.set_title("Phân bố Bệnh Tim", fontsize=16)
        ax.set_xlabel("Tình trạng (0=Khỏe, 1=Bệnh)")
        ax.set_ylabel("Số lượng")
        st.pyplot(fig)
    
    st.markdown("---")
    
    # Chọn biến để phân tích
    st.subheader("📈 Phân tích theo biến")
    
    analysis_type = st.selectbox(
        "Chọn loại phân tích:",
        ["Biến số (Numerical)", "Biến phân loại (Categorical)", "Ma trận tương quan"]
    )
    
    if analysis_type == "Biến số (Numerical)":
        num_vars = ['Tuổi', 'Huyết_Áp_Nghỉ', 'Cholesterol', 'Nhịp_Tim_Tối_Đa', 'Độ_Chênh_ST', 'Cholesterol_Tuoi']
        selected_var = st.selectbox("Chọn biến số:", num_vars)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Histogram + KDE
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.histplot(data=eda_df, x=selected_var, kde=True, hue='Bệnh_Tim', ax=ax, palette=['#2ecc71', '#e74c3c'])
            ax.set_title(f"Phân bố {selected_var}", fontsize=14)
            st.pyplot(fig)
        
        with col2:
            # Boxplot
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.boxplot(data=eda_df, x='Bệnh_Tim', y=selected_var, ax=ax, palette=['#2ecc71', '#e74c3c'])
            ax.set_title(f"{selected_var} theo tình trạng bệnh", fontsize=14)
            ax.set_xticklabels(['Khỏe', 'Bệnh'])
            st.pyplot(fig)
        
        # Thống kê so sánh
        st.subheader(f"📊 Thống kê {selected_var}")
        stats_comparison = eda_df.groupby('Bệnh_Tim')[selected_var].agg(['mean', 'median', 'std', 'min', 'max']).round(2)
        stats_comparison.index = ['Khỏe (0)', 'Bệnh (1)']
        st.dataframe(stats_comparison, use_container_width=True)
    
    elif analysis_type == "Biến phân loại (Categorical)":
        cat_vars = ['Giới_tính', 'Loại_Đau_Ngực', 'Điện_Tâm_Đồ', 'Đau_Thắt_Vận_Động', 'Độ_Dốc_ST', 'Đường_Huyết_Đói']
        selected_var = st.selectbox("Chọn biến phân loại:", cat_vars)
        
        # Crosstab
        st.subheader(f"📊 Phân tích {selected_var}")
        
        crosstab = pd.crosstab(eda_df[selected_var], eda_df['Bệnh_Tim'], normalize='index') * 100
        crosstab.columns = ['Khỏe (%)', 'Bệnh (%)']
        st.dataframe(crosstab.round(2), use_container_width=True)
        
        # Countplot
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.countplot(data=eda_df, x=selected_var, hue='Bệnh_Tim', ax=ax, palette=['#2ecc71', '#e74c3c'])
        ax.set_title(f"Phân bố {selected_var} theo tình trạng bệnh", fontsize=14)
        ax.legend(title='Bệnh Tim', labels=['Khỏe', 'Bệnh'])
        st.pyplot(fig)
        
        # Phát hiện nhóm nguy cơ cao
        risk_rates = crosstab['Bệnh (%)']
        high_risk = risk_rates[risk_rates > 60].sort_values(ascending=False)
        
        if len(high_risk) > 0:
            st.warning(f"⚠️ **NHÓM NGUY CƠ CAO (>60%):**")
            for idx, val in high_risk.items():
                st.write(f"- **{selected_var} = {idx}:** {val:.1f}% nguy cơ bệnh")
    
    else:  # Ma trận tương quan
        st.subheader("🔗 Ma trận tương quan")
        
        # Chọn cột số
        numeric_cols = eda_df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) > 0:
            corr_matrix = eda_df[numeric_cols].corr()
            
            fig, ax = plt.subplots(figsize=(14, 10))
            sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0, 
                       square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax)
            ax.set_title("Ma trận tương quan giữa các biến số", fontsize=16)
            st.pyplot(fig)
            
            # Top correlations with target
            if 'Bệnh_Tim' in numeric_cols:
                st.subheader("🎯 Tương quan với biến mục tiêu (Bệnh_Tim)")
                target_corr = corr_matrix['Bệnh_Tim'].drop('Bệnh_Tim').sort_values(ascending=False)
                
                fig, ax = plt.subplots(figsize=(10, 6))
                target_corr.plot(kind='barh', ax=ax, color=['#e74c3c' if x > 0 else '#3498db' for x in target_corr])
                ax.set_title("Tương quan với Bệnh_Tim", fontsize=14)
                ax.set_xlabel("Hệ số tương quan")
                st.pyplot(fig)
                
                st.dataframe(target_corr.to_frame('Correlation'), use_container_width=True)

# ============================================================
# 6. SO SÁNH MÔ HÌNH
# ============================================================


elif page == "🔬 So sánh mô hình":
    st.header("🔬 SO SÁNH 3 MÔ HÌNH MACHINE LEARNING")
    
    st.info("💡 Kết quả được lấy từ evaluation.py đã chạy trước đó")
    
    # Bảng so sánh metrics
    st.subheader("📊 Bảng so sánh hiệu suất")
    
    comparison_df = pd.DataFrame({
        'Mô hình': ['Random Forest', 'XGBoost 🏆', 'Neural Network'],
        'Accuracy': [0.8804, 0.8207, 0.8696],
        'Precision': [0.8509, 0.7717, 0.8421],
        'Recall': [0.9510, 0.9608, 0.9412],
        'F1-Score': [0.8981, 0.8559, 0.8889],
        'AUC-ROC': [0.9232, 0.9064, 0.9449],
        'FN (Bỏ sót)': [5, 4, 6],
        'Ngưỡng tối ưu': [0.5491, 0.6711, 0.6184]
    })
    
    # Highlight best values
    st.dataframe(
        comparison_df.style.highlight_max(subset=['Recall', 'AUC-ROC'], color='lightgreen')
                          .highlight_min(subset=['FN (Bỏ sót)'], color='lightgreen'),
        use_container_width=True
    )
    
    st.markdown("---")
    
    # Visualize comparison
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 So sánh Metrics chính")
        
        metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x = np.arange(len(comparison_df['Mô hình']))
        width = 0.2
        
        for i, metric in enumerate(metrics_to_plot):
            ax.bar(x + i*width, comparison_df[metric], width, label=metric)
        
        ax.set_xlabel('Mô hình')
        ax.set_ylabel('Giá trị')
        ax.set_title('So sánh Metrics')
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(comparison_df['Mô hình'])
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        st.pyplot(fig)
    
    with col2:
        st.subheader("🎯 False Negatives (Bỏ sót)")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = ['#3498db', '#2ecc71', '#e74c3c']
        bars = ax.barh(comparison_df['Mô hình'], comparison_df['FN (Bỏ sót)'], color=colors)
        ax.set_xlabel('Số bệnh nhân bị bỏ sót')
        ax.set_title('False Negatives - Càng thấp càng tốt')
        
        # Add value labels
        for bar in bars:
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height()/2, 
                   f'{int(width)}', ha='left', va='center', fontweight='bold')
        
        ax.grid(axis='x', alpha=0.3)
        st.pyplot(fig)
    
    st.markdown("---")
    
    # So sánh Hidden Patterns
    st.subheader("🔍 So sánh khả năng phát hiện Hidden Patterns")
    
    patterns_df = pd.DataFrame({
        'Model': ['Random Forest', 'XGBoost', 'Neural Network'],
        'Nhóm_xuat_hien': [4, 5, 4],
        'Benh_nhan_phat_hien': [97, 102, 99],
        'Combo_patterns': ['0/5 (0%)', '3/5 (60%)', '4/5 (80%)']
    })
    
    st.dataframe(patterns_df, use_container_width=True)
    
    st.markdown("""
    **Giải thích:**
    - **Nhóm_xuat_hien:** Số nhóm xác suất có bệnh nhân (càng nhiều càng đa dạng)
    - **Benh_nhan_phat_hien:** Tổng bệnh nhân phát hiện với xác suất ≥ 25%
    - **Combo_patterns:** Tỷ lệ patterns kết hợp nhiều triệu chứng
    
    ⭐ **XGBoost thắng vì:**
    - Phát hiện 5 nhóm nguy cơ (đa dạng nhất)
    - Phát hiện 102 bệnh nhân (nhạy nhất)
    - Recall cao nhất (96.08%)
    - Bỏ sót ít nhất (4 FN)
    """)
    
    st.markdown("---")
    
    # Hiển thị ROC Curves
    st.subheader("📉 Đường cong ROC")
    
    roc_img_path = os.path.join(FIGURES_DIR, "Comparison_ROC.png")
    if os.path.exists(roc_img_path):
        img = Image.open(roc_img_path)
        st.image(img, use_column_width=True)
    else:
        st.warning("⚠️ Chưa có biểu đồ ROC. Hãy chạy evaluation.py trước!")
    
    # Confusion Matrices
    st.subheader("📊 Ma trận nhầm lẫn (Confusion Matrix)")
    
    cm_img_path = os.path.join(FIGURES_DIR, "Comparison_ConfusionMatrix.png")
    if os.path.exists(cm_img_path):
        img = Image.open(cm_img_path)
        st.image(img, use_column_width=True)
    else:
        st.warning("⚠️ Chưa có Confusion Matrix. Hãy chạy evaluation.py trước!")
    
    st.markdown("---")
    
    # Khuyến nghị
    st.subheader("💡 Khuyến nghị sử dụng")
    
    rec_col1, rec_col2, rec_col3 = st.columns(3)
    
    with rec_col1:
        st.markdown("""
        <div style='background-color:#d4edda; padding:15px; border-radius:10px; border-left:5px solid #28a745;'>
            <h4>🏥 Ứng dụng y tế</h4>
            <p><strong>→ XGBoost</strong></p>
            <ul>
                <li>Recall cao nhất (96.08%)</li>
                <li>Bỏ sót ít nhất (4 FN)</li>
                <li>An toàn cho screening</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with rec_col2:
        st.markdown("""
        <div style='background-color:#d1ecf1; padding:15px; border-radius:10px; border-left:5px solid #0c5460;'>
            <h4>🎯 Cần chính xác cao</h4>
            <p><strong>→ Random Forest</strong></p>
            <ul>
                <li>Accuracy cao nhất (88.04%)</li>
                <li>F1-Score cao nhất (0.8981)</li>
                <li>Cân bằng tốt</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with rec_col3:
        st.markdown("""
        <div style='background-color:#fff3cd; padding:15px; border-radius:10px; border-left:5px solid #856404;'>
            <h4>🔬 Nghiên cứu khoa học</h4>
            <p><strong>→ Cả 3 mô hình</strong></p>
            <ul>
                <li>So sánh ensemble</li>
                <li>Phát hiện đa dạng</li>
                <li>AUC-ROC cao nhất: NN</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

# ============================================================
# 7. DỰ ĐOÁN NGUY CƠ
# ============================================================

elif page == "🩺 Dự đoán nguy cơ":
    st.header("🩺 DỰ ĐOÁN NGUY CƠ BỆNH TIM")
    
    st.markdown("""
    <div class="warning-box">
        <strong>⚠️ CHÚ Ý QUAN TRỌNG:</strong> Kết quả dự đoán chỉ mang tính tham khảo. 
        Vui lòng tham khảo ý kiến bác sĩ chuyên khoa để có chẩn đoán chính xác!
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Form nhập liệu
    st.subheader("📝 Nhập thông tin bệnh nhân")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Thông tin cơ bản:**")
        age = st.number_input("Tuổi", min_value=20, max_value=100, value=50, step=1)
        sex = st.selectbox("Giới tính", ["Nam", "Nữ"])
        chest_pain = st.selectbox("Loại đau ngực", ["ATA (Không điển hình)", "NAP (Không đau ngực)", "ASY (Không triệu chứng)", "TA (Điển hình)"])
    
    with col2:
        st.markdown("**Chỉ số sinh học:**")
        bp = st.number_input("Huyết áp nghỉ (mmHg)", min_value=80, max_value=200, value=120, step=5)
        chol = st.number_input("Cholesterol (mg/dL)", min_value=100, max_value=600, value=200, step=10)
        fbs = st.selectbox("Đường huyết đói (>120 mg/dL)", ["Không (0)", "Có (1)"])
    
    with col3:
        st.markdown("**Điện tâm đồ & vận động:**")
        ecg = st.selectbox("Kết quả ECG", ["Normal", "ST", "LVH"])
        max_hr = st.number_input("Nhịp tim tối đa", min_value=60, max_value=220, value=150, step=5)
        angina = st.selectbox("Đau thắt vận động", ["Không (N)", "Có (Y)"])
        oldpeak = st.number_input("Độ chênh ST (Oldpeak)", min_value=-3.0, max_value=7.0, value=0.0, step=0.1)
        st_slope = st.selectbox("Độ dốc ST", ["Up", "Flat", "Down"])
    
    st.markdown("---")
    
    # Chọn mô hình
    model_choice = st.multiselect(
        "Chọn mô hình dự đoán (có thể chọn nhiều):",
        ["Random Forest", "XGBoost", "Neural Network"],
        default=["XGBoost"]
    )
    
    if st.button("🔍 DỰ ĐOÁN NGUY CƠ", type="primary"):
        if len(model_choice) == 0:
            st.error("❌ Vui lòng chọn ít nhất 1 mô hình!")
        else:
            with st.spinner("Đang xử lý dữ liệu và dự đoán..."):
                try:
                    # ===== BƯỚC 2: Input dict (15 features, CHƯA engineered) =====
                    input_dict = {
                        'Tuổi': age,
                        'Giới_tính': 1 if sex == "Nam" else 0,
                        'Huyết_Áp_Nghỉ': bp,
                        'Cholesterol': chol,
                        'Đường_Huyết_Đói': 1 if fbs == "Có (1)" else 0,
                        'Nhịp_Tim_Tối_Đa': max_hr,
                        'Đau_Thắt_Vận_Động': 1 if angina == "Có (Y)" else 0,
                        'Độ_Chênh_ST': oldpeak,
                    }
                    
                    # ===== BƯỚC 3: IMPUTATION - Numpy array thuần túy =====
                    impute_values = np.array([[
                        input_dict['Cholesterol'],
                        input_dict['Huyết_Áp_Nghỉ'],
                        input_dict['Tuổi'],
                        input_dict['Nhịp_Tim_Tối_Đa']
                    ]])
                    
                    imputed_values = imputer.transform(impute_values)
                    
                    # Gán lại vào dict
                    input_dict['Cholesterol'] = float(imputed_values[0, 0])
                    input_dict['Huyết_Áp_Nghỉ'] = float(imputed_values[0, 1])
                    input_dict['Tuổi'] = float(imputed_values[0, 2])
                    input_dict['Nhịp_Tim_Tối_Đa'] = float(imputed_values[0, 3])
                    
                    # ===== BƯỚC 4: FEATURE ENGINEERING (SAU imputation) =====
                    input_dict['Cholesterol_Tuoi'] = input_dict['Cholesterol'] / input_dict['Tuổi']
                    input_dict['NguyCo_TimMach_RatCao'] = int(
                        (input_dict['Huyết_Áp_Nghỉ'] >= 140) and (input_dict['Cholesterol'] >= 240)
                    )
                    
                    # ===== BƯỚC 5: Encoding categorical =====
                    chest_pain_map = {"ATA (Không điển hình)": "ATA", "NAP (Không đau ngực)": "NAP", 
                                    "ASY (Không triệu chứng)": "ASY", "TA (Điển hình)": "TA"}
                    cp_code = chest_pain_map[chest_pain]
                    
                    input_dict['Loại_Đau_Ngực_ATA'] = 1 if cp_code == 'ATA' else 0
                    input_dict['Loại_Đau_Ngực_NAP'] = 1 if cp_code == 'NAP' else 0
                    input_dict['Loại_Đau_Ngực_TA'] = 1 if cp_code == 'TA' else 0
                    
                    input_dict['Điện_Tâm_Đồ_Normal'] = 1 if ecg == 'Normal' else 0
                    input_dict['Điện_Tâm_Đồ_ST'] = 1 if ecg == 'ST' else 0
                    
                    input_dict['Độ_Dốc_ST_Flat'] = 1 if st_slope == 'Flat' else 0
                    input_dict['Độ_Dốc_ST_Up'] = 1 if st_slope == 'Up' else 0
                    
                    # ===== BƯỚC 6: Create DataFrame (1 lần duy nhất với 17 features) =====
                    input_data = pd.DataFrame([input_dict])
                    
                    # ===== BƯỚC 7: Align với feature_cols =====
                    X = input_data[feature_cols].copy()
                    
                    # ===== BƯỚC 8: SCALING =====
                    cols_scale = ['Tuổi', 'Huyết_Áp_Nghỉ', 'Cholesterol', 'Nhịp_Tim_Tối_Đa', 
                                 'Độ_Chênh_ST', 'Cholesterol_Tuoi']
                    X[cols_scale] = scaler.transform(X[cols_scale].values)
                    
                    # Dự đoán
                    st.success("✅ Dữ liệu đã được xử lý thành công!")
                    st.markdown("---")
                    
                    st.subheader("📊 KẾT QUẢ DỰ ĐOÁN")
                    
                    results = []
                    
                    for model_name in model_choice:
                        if model_name == "Random Forest":
                            model = models['rf']
                            threshold = metadata['rf']['threshold']
                        elif model_name == "XGBoost":
                            model = models['xgb']
                            threshold = metadata['xgb']['threshold']
                        else:  # Neural Network
                            model = models['nn']
                            threshold = metadata['nn']['threshold']
                        
                        # Predict probability
                        prob = model.predict_proba(X)[0][1]
                        prediction = 1 if prob >= threshold else 0
                        
                        results.append({
                            'Mô hình': model_name,
                            'Xác suất bệnh': f"{prob*100:.2f}%",
                            'Dự đoán': 'Có nguy cơ bệnh tim ⚠️' if prediction == 1 else 'Không có nguy cơ ✅',
                            'Ngưỡng': f"{threshold:.4f}",
                            'Prob_value': prob
                        })
                    
                    # Hiển thị kết quả
                    result_cols = st.columns(len(results))
                    
                    for i, result in enumerate(results):
                        with result_cols[i]:
                            prob_val = result['Prob_value']
                            color = '#e74c3c' if prob_val >= 0.5 else '#2ecc71'
                            
                            st.markdown(f"""
                            <div style='background-color:{color}15; padding:20px; border-radius:10px; border-left:5px solid {color};'>
                                <h3 style='color:{color};'>{result['Mô hình']}</h3>
                                <h1 style='color:{color}; margin:10px 0;'>{result['Xác suất bệnh']}</h1>
                                <p style='font-size:16px; margin:5px 0;'><strong>{result['Dự đoán']}</strong></p>
                                <p style='font-size:12px; color:#666;'>Ngưỡng: {result['Ngưỡng']}</p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Progress bar
                            st.progress(float(prob_val))
                    
                    st.markdown("---")
                    
                    # Bảng tổng hợp
                    st.subheader("📋 Tổng hợp kết quả")
                    results_df = pd.DataFrame(results)[['Mô hình', 'Xác suất bệnh', 'Dự đoán', 'Ngưỡng']]
                    st.dataframe(results_df, use_container_width=True)
                    
                    # Giải thích
                    st.markdown("---")
                    st.subheader("💡 Giải thích kết quả")
                    
                    avg_prob = np.mean([r['Prob_value'] for r in results])
                    
                    if avg_prob >= 0.7:
                        st.error("""
                        **🚨 NGUY CƠ CAO:** Xác suất trung bình ≥ 70%
                        
                        **Khuyến nghị:**
                        - ⚠️ Cần khám tim mạch NGAY
                        - Làm các xét nghiệm chuyên sâu (ECG, Holter, siêu âm tim)
                        - Tham khảo bác sĩ tim mạch
                        """)
                    elif avg_prob >= 0.5:
                        st.warning("""
                        **⚠️ NGUY CƠ TRUNG BÌNH:** Xác suất 50-70%
                        
                        **Khuyến nghị:**
                        - Nên đi khám để kiểm tra
                        - Theo dõi các triệu chứng
                        - Điều chỉnh lối sống (ăn uống, tập luyện)
                        """)
                    else:
                        st.success("""
                        **✅ NGUY CƠ THẤP:** Xác suất < 50%
                        
                        **Khuyến nghị:**
                        - Duy trì lối sống lành mạnh
                        - Khám định kỳ hàng năm
                        - Kiểm soát cholesterol, huyết áp
                        """)
                    
                    # Thông tin đầu vào
                    with st.expander("📋 Xem thông tin đầu vào đã xử lý"):
                        st.write("**Dữ liệu gốc:**")
                        st.json(input_data)
                        st.write("**Features sau preprocessing:**")
                        st.dataframe(X.head(), use_container_width=True)
                
                except Exception as e:
                    st.error(f"❌ Lỗi khi dự đoán: {e}")
                    st.exception(e)

# ============================================================
# 8. HIDDEN PATTERNS
# ============================================================

elif page == "🔍 Hidden Patterns":
    st.header("🔍 CÁC MẪU ẨN ĐƯỢC PHÁT HIỆN (HIDDEN PATTERNS)")
    
    st.info("""
    💡 **Hidden Patterns** là các tổ hợp triệu chứng cụ thể dẫn đến các mức độ nguy cơ bệnh tim khác nhau.
    Các mẫu này được trích xuất tự động từ 3 mô hình ML trong quá trình training.
    """)
    
    # Chọn mô hình
    pattern_model = st.selectbox(
        "Chọn mô hình để xem patterns:",
        ["Random Forest", "XGBoost (Khuyến nghị)", "Neural Network"]
    )
    
    st.markdown("---")
    
    if pattern_model == "Random Forest":
        st.subheader("🌳 Hidden Patterns từ Random Forest")
        
        st.markdown("""
        **Phương pháp:** Trích xuất rules từ 50 cây quyết định
        **Kết quả:** 5 patterns (4 nhóm nguy cơ), 97 bệnh nhân phát hiện
        """)
        
        patterns = [
            {
                'no': 1,
                'conditions': ['Độ_Dốc_ST_Up ≤ 0.50', 'Giới_tính > 0.50 (Nam)', 'Độ_Chênh_ST ≤ -0.30'],
                'risk': '100%',
                'patients': 51,
                'level': 'Rất cao'
            },
            {
                'no': 2,
                'conditions': ['Độ_Chênh_ST > -0.03', 'Độ_Chênh_ST > 1.23', 'Huyết_Áp_Nghỉ > -1.27'],
                'risk': '100%',
                'patients': 44,
                'level': 'Rất cao'
            },
            {
                'no': 3,
                'conditions': ['Độ_Dốc_ST_Flat > 0.50', 'Loại_Đau_Ngực_NAP ≤ 0.50', 'Đường_Huyết_Đói > 0.50', 'Độ_Chênh_ST ≤ 0.93'],
                'risk': '100%',
                'patients': 44,
                'level': 'Rất cao'
            },
            {
                'no': 4,
                'conditions': ['Loại_Đau_Ngực_NAP > 0.50', 'Đau_Thắt_Vận_Động > 0.50', 'Nhịp_Tim_Tối_Đa ≤ 0.36', 'Huyết_Áp_Nghỉ ≤ -0.88'],
                'risk': '33%',
                'patients': 3,
                'level': 'Trung bình'
            },
            {
                'no': 5,
                'conditions': ['Đau_Thắt_Vận_Động > 0.50', 'Loại_Đau_Ngực_TA > 0.50'],
                'risk': '40%',
                'patients': 3,
                'level': 'Trung bình'
            }
        ]
        
        for p in patterns:
            risk_color = '#e74c3c' if p['level'] == 'Rất cao' else ('#ff9800' if p['level'] == 'Trung bình' else '#2ecc71')
            
            st.markdown(f"""
            <div style='background-color:{risk_color}15; padding:15px; margin:10px 0; border-radius:10px; border-left:5px solid {risk_color};'>
                <h4 style='color:{risk_color};'>Pattern #{p['no']} - Nguy cơ {p['risk']} ({p['level']})</h4>
                <p><strong>NẾU:</strong></p>
                <ul>
                    {''.join([f'<li>{cond}</li>' for cond in p['conditions']])}
                </ul>
                <p><strong>→ Căn cứ:</strong> {p['patients']} bệnh nhân trong tập huấn luyện</p>
            </div>
            """, unsafe_allow_html=True)
    
    elif pattern_model == "XGBoost (Khuyến nghị)":
        st.subheader("🚀 Hidden Patterns từ XGBoost")
        
        st.markdown("""
        **Phương pháp:** Phân tích theo probability ranges (6 ranges)
        **Kết quả:** 5 patterns (5 nhóm nguy cơ), **102 bệnh nhân phát hiện** ⭐
        **Đặc biệt:** 3/5 patterns là combo (kết hợp 2 triệu chứng)
        """)
        
        patterns = [
            {
                'no': 1,
                'conditions': ['Độ_Chênh_ST > 0.67', 'Đường_Huyết_Đói > 0.00'],
                'risk': '92%',
                'patients': 57,
                'level': 'Rất cao',
                'is_combo': True
            },
            {
                'no': 2,
                'conditions': ['Độ_Chênh_ST > 0.75'],
                'risk': '92%',
                'patients': 57,
                'level': 'Rất cao',
                'is_combo': False
            },
            {
                'no': 3,
                'conditions': ['Độ_Chênh_ST > 0.67', 'Đường_Huyết_Đói > 0.00'],
                'risk': '77%',
                'patients': 36,
                'level': 'Cao',
                'is_combo': True
            },
            {
                'no': 4,
                'conditions': ['Độ_Chênh_ST > 0.53', 'Đau_Thắt_Vận_Động > 0.00'],
                'risk': '62%',
                'patients': 4,
                'level': 'Trung bình',
                'is_combo': True
            },
            {
                'no': 5,
                'conditions': ['Độ_Chênh_ST > -0.60'],
                'risk': '48%',
                'patients': 2,
                'level': 'Vừa phải',
                'is_combo': False
            }
        ]
        
        for p in patterns:
            risk_color = '#e74c3c' if p['level'] == 'Rất cao' else ('#ff5722' if p['level'] == 'Cao' else ('#ff9800' if p['level'] == 'Trung bình' else '#ffc107'))
            combo_badge = '🔗 COMBO' if p['is_combo'] else '📌 SINGLE'
            
            st.markdown(f"""
            <div style='background-color:{risk_color}15; padding:15px; margin:10px 0; border-radius:10px; border-left:5px solid {risk_color};'>
                <h4 style='color:{risk_color};'>Pattern #{p['no']} - Nguy cơ {p['risk']} ({p['level']}) {combo_badge}</h4>
                <p><strong>NẾU:</strong></p>
                <ul>
                    {''.join([f'<li>{cond}</li>' for cond in p['conditions']])}
                </ul>
                <p><strong>→ Căn cứ:</strong> XGBoost phân tích từ {p['patients']} bệnh nhân</p>
                {'<p style="font-size:12px; color:#666;"><em>Ghi chú: Mức độ triệu chứng KHÁC NHAU giữa các nhóm</em></p>' if p['is_combo'] else ''}
            </div>
            """, unsafe_allow_html=True)
        
        st.success("""
        ⭐ **TẠI SAO XGBOOST TỐT NHẤT?**
        - Phát hiện 5 nhóm nguy cơ (đa dạng nhất)
        - Phát hiện 102 bệnh nhân (nhiều nhất)
        - 60% patterns là combo (phát hiện tương tác)
        - Có threshold cụ thể cho từng mức độ triệu chứng
        """)
    
    else:  # Neural Network
        st.subheader("🧠 Hidden Patterns từ Neural Network")
        
        st.markdown("""
        **Phương pháp:** Permutation Importance + Probability ranges (5 ranges)
        **Kết quả:** 5 patterns (4 nhóm nguy cơ), 99 bệnh nhân phát hiện
        **Đặc biệt:** 4/5 patterns là combo (80% - cao nhất!)
        """)
        
        patterns = [
            {
                'no': 1,
                'conditions': ['Độ_Chênh_ST > 0.67', 'Đường_Huyết_Đói > 0.00'],
                'risk': '92%',
                'patients': 72,
                'level': 'Rất cao',
                'is_combo': True
            },
            {
                'no': 2,
                'conditions': ['Độ_Chênh_ST > 0.95'],
                'risk': '92%',
                'patients': 72,
                'level': 'Rất cao',
                'is_combo': False
            },
            {
                'no': 3,
                'conditions': ['Độ_Chênh_ST > 0.33', 'Đường_Huyết_Đói > 0.00'],
                'risk': '77%',
                'patients': 14,
                'level': 'Cao',
                'is_combo': True
            },
            {
                'no': 4,
                'conditions': ['Độ_Chênh_ST > 0.37', 'Đường_Huyết_Đói > 0.00'],
                'risk': '62%',
                'patients': 10,
                'level': 'Trung bình',
                'is_combo': True
            },
            {
                'no': 5,
                'conditions': ['Độ_Chênh_ST > 0.33', 'Đường_Huyết_Đói > 0.00'],
                'risk': '32%',
                'patients': 3,
                'level': 'Thấp',
                'is_combo': True
            }
        ]
        
        for p in patterns:
            risk_color = '#e74c3c' if p['level'] == 'Rất cao' else ('#ff5722' if p['level'] == 'Cao' else ('#ff9800' if p['level'] == 'Trung bình' else '#2ecc71'))
            combo_badge = '🔗 COMBO' if p['is_combo'] else '📌 SINGLE'
            
            st.markdown(f"""
            <div style='background-color:{risk_color}15; padding:15px; margin:10px 0; border-radius:10px; border-left:5px solid {risk_color};'>
                <h4 style='color:{risk_color};'>Pattern #{p['no']} - Nguy cơ {p['risk']} ({p['level']}) {combo_badge}</h4>
                <p><strong>NẾU:</strong></p>
                <ul>
                    {''.join([f'<li>{cond}</li>' for cond in p['conditions']])}
                </ul>
                <p><strong>→ Căn cứ:</strong> Neural Network phân tích từ {p['patients']} bệnh nhân</p>
                {'<p style="font-size:12px; color:#666;"><em>Ghi chú: Mức độ triệu chứng KHÁC NHAU giữa các nhóm</em></p>' if p['is_combo'] else ''}
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Giải thích tại sao cùng triệu chứng khác % nguy cơ
    st.subheader("❓ Tại sao cùng triệu chứng mà khác % nguy cơ?")
    
    st.info("""
    **VÍ DỤ:** Độ_Chênh_ST xuất hiện ở nhiều patterns với % khác nhau:
    - Pattern 1: Độ_Chênh_ST > 0.67 → 92% nguy cơ
    - Pattern 3: Độ_Chênh_ST > 0.67 → 77% nguy cơ
    - Pattern 5: Độ_Chênh_ST > 0.33 → 32% nguy cơ
    
    **GIẢI THÍCH:**
    1. **MỨC ĐỘ triệu chứng khác nhau:** > 0.67 vs > 0.33 (cao hơn = nguy hiểm hơn)
    2. **TỔ HỢP với triệu chứng khác:** Cùng với Đường_Huyết_Đói hay không
    3. **NHÓM bệnh nhân khác nhau:** Độ tuổi, giới tính, tiền sử khác nhau
    
    → Đây chính là ưu điểm của **Hidden Patterns**: Phát hiện được sự phức tạp và tương tác giữa các triệu chứng!
    """)

# ============================================================
# 9. HƯỚNG DẪN SỬ DỤNG
# ============================================================

elif page == "📖 Hướng dẫn sử dụng":
    st.header("📖 HƯỚNG DẪN SỬ DỤNG DASHBOARD")
    
    st.markdown("""
    ## 🎯 Mục đích Dashboard
    
    Dashboard này được xây dựng để minh họa một hệ thống Machine Learning hoàn chỉnh trong lĩnh vực y tế,
    từ xử lý dữ liệu, phân tích, training model cho đến triển khai dự đoán.
    
    ---
    
    ## 📋 Các trang chức năng
    
    ### 1. 🏠 Trang chủ
    - Tổng quan về dự án
    - Thống kê cơ bản (918 bệnh nhân, 17 features, 55.3% tỷ lệ bệnh)
    - Quy trình xử lý (Pipeline)
    - Tính năng chính của dashboard
    
    ### 2. 📂 Dữ liệu & Mô tả
    **Chức năng:**
    - ✅ Tải file CSV của bạn HOẶC dùng dữ liệu mẫu
    - ✅ Xem trước dữ liệu (20 dòng đầu)
    - ✅ Thống kê mô tả (mean, std, min, max, ...)
    - ✅ Giải thích ý nghĩa từng cột với context y khoa
    - ✅ Download dữ liệu mẫu về máy
    
    **Cách sử dụng:**
    1. Chọn "Tải file CSV của bạn"
    2. Click "Browse files" và chọn file .csv
    3. Dashboard sẽ tự động load và hiển thị
    
    ### 3. 📊 Phân tích EDA
    **Chức năng:**
    - Phân bố biến mục tiêu (Bệnh Tim)
    - Phân tích biến số: Histogram + Boxplot
    - Phân tích biến phân loại: Countplot + Crosstab
    - Ma trận tương quan
    - Phát hiện nhóm nguy cơ cao (>60%)
    
    **Cách sử dụng:**
    1. Chọn loại phân tích (Numerical/Categorical/Correlation)
    2. Chọn biến cần phân tích
    3. Xem biểu đồ và thống kê
    
    ### 4. 🔬 So sánh mô hình
    **Chức năng:**
    - Bảng so sánh 3 mô hình (RF, XGBoost, NN)
    - 8 metrics: Accuracy, Precision, Recall, F1, AUC-ROC, FN, Threshold
    - Biểu đồ so sánh metrics
    - Biểu đồ False Negatives (bỏ sót)
    - So sánh Hidden Patterns
    - ROC Curves
    - Confusion Matrices
    - Khuyến nghị sử dụng theo từng mục đích
    
    ### 5. 🩺 Dự đoán nguy cơ
    **Chức năng:**
    - Nhập thông tin bệnh nhân (11 trường)
    - Chọn 1 hoặc nhiều mô hình để dự đoán
    - Hiển thị xác suất bệnh tim (%)
    - Dự đoán: Có/Không nguy cơ
    - Khuyến nghị y khoa dựa trên kết quả
    
    **Cách sử dụng:**
    1. Điền đầy đủ thông tin bệnh nhân:
       - **Cơ bản:** Tuổi, Giới tính, Loại đau ngực
       - **Sinh học:** Huyết áp, Cholesterol, Đường huyết đói
       - **ECG & Vận động:** Kết quả ECG, Nhịp tim, Đau thắt, Oldpeak, ST Slope
    2. Chọn mô hình (mặc định: XGBoost)
    3. Click "DỰ ĐOÁN NGUY CƠ"
    4. Xem kết quả và khuyến nghị
    
    **Giải thích kết quả:**
    - 🚨 **Nguy cơ cao (≥70%):** Cần khám ngay
    - ⚠️ **Nguy cơ trung bình (50-70%):** Nên đi khám kiểm tra
    - ✅ **Nguy cơ thấp (<50%):** Duy trì lối sống lành mạnh
    
    ### 6. 🔍 Hidden Patterns
    **Chức năng:**
    - Xem 5 patterns từ mỗi mô hình
    - Giải thích từng pattern: Điều kiện → Nguy cơ → Số bệnh nhân
    - Phân biệt Single vs Combo patterns
    - Giải thích tại sao cùng triệu chứng khác % nguy cơ
    
    **Các loại patterns:**
    - 📌 **SINGLE:** 1 triệu chứng
    - 🔗 **COMBO:** 2-3 triệu chứng kết hợp
    
    ---
    
    ## ⚠️ Lưu ý quan trọng
    
    ### 🔴 Về mặt y khoa:
    1. **KHÔNG tự chẩn đoán** dựa trên kết quả dashboard
    2. **KHÔNG thay đổi thuốc** hoặc liệu trình điều trị
    3. **PHẢI tham khảo bác sĩ** để có chẩn đoán chính xác
    4. Dashboard chỉ mang tính chất **hỗ trợ** và **minh họa khoa học**
    
    ### 📊 Về mặt kỹ thuật:
    1. Mô hình được train trên 918 bệnh nhân → **Chưa đủ lớn** cho production
    2. Recall 96% nghĩa là vẫn **bỏ sót 4%** (4/102 bệnh nhân)
    3. False Positive cao (19-29) → Nhiều người khỏe bị dự đoán nhầm
    4. Cần **external validation** trên dataset khác
    5. Cần **clinical trials** và **FDA approval** trước khi triển khai thực tế
    
    ---
    
    ## 🚀 Hướng phát triển
    
    ### Cải thiện mô hình:
    - Ensemble 3 mô hình (Voting)
    - Thêm Deep Learning (CNN, RNN)
    - SHAP values cho explainability
    
    ### Tăng dữ liệu:
    - Tích hợp thêm datasets (>10,000 BN)
    - Real-time data từ bệnh viện
    - Multi-center validation
    
    ### Triển khai thực tế:
    - API RESTful (FastAPI)
    - Mobile app (Flutter)
    - Tích hợp hệ thống bệnh viện (FHIR)
    - HIPAA compliance
    
    ---
    
    ## 📞 Liên hệ & Hỗ trợ
    
    **Nhóm phát triển:**
    - Nhóm 7 - S26-65TTNT
    - Nguyễn Lê Minh Hậu
    - Nguyễn Đức Huy
    
    **Đồ án môn:** Machine Learning
    
    **GitHub:** [Link repository]
    
    ---
    
    ## 📚 Tài liệu tham khảo
    
    1. UCI Heart Disease Dataset
    2. Kaggle Heart Disease Dataset
    3. scikit-learn Documentation
    4. XGBoost Documentation
    5. Streamlit Documentation
    
    ---
    
    <div style='background-color:#d4edda; padding:20px; border-radius:10px; border-left:5px solid #28a745; margin-top:30px;'>
        <h3 style='color:#155724;'>✅ Checklist sử dụng Dashboard</h3>
        <ul style='color:#155724;'>
            <li>☑️ Đã đọc hướng dẫn và hiểu rõ mục đích</li>
            <li>☑️ Đã chạy preprocessing.py → models → evaluation.py</li>
            <li>☑️ Đã có đủ 10 file .pkl trong saved_models/</li>
            <li>☑️ Đã có các biểu đồ trong outputs/figures/</li>
            <li>☑️ Hiểu rằng kết quả chỉ mang tính tham khảo</li>
            <li>☑️ Không tự chẩn đoán dựa trên dashboard</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# FOOTER
# ============================================================

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p><strong>🫀 Dashboard Dự Đoán Bệnh Tim</strong></p>
    <p>Nhóm 7 - S26-65TTNT | Đồ án môn Machine Learning</p>
    <p style='font-size: 12px;'>⚠️ Chỉ mang tính nghiên cứu và minh họa - Không thay thế chẩn đoán y khoa</p>
</div>
""", unsafe_allow_html=True)
