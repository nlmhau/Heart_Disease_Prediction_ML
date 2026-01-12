> **Đồ án môn học Machine Learning (Học Máy)**  
> **Lớp:** S26-65TTNT - Nhóm 7  
> **Sử dụng Ensemble Learning và Deep Learning với Phân tích Mẫu Ẩn (Hidden Pattern Analysis)**

---

##  Giới Thiệu

Dự án này xây dựng một mô hình Machine Learning để dự đoán khả năng mắc bệnh tim dựa trên các chỉ số lâm sàng. Mục tiêu là tối ưu hóa quy trình xử lý dữ liệu và đạt độ chính xác cao trên tập kiểm thử.

- **Random Forest** - Ensemble Learning với cây quyết định
- **XGBoost** - Gradient Boosting tối ưu hóa
- **Neural Network** - Deep Learning với MLPClassifier

Hệ thống không chỉ dự đoán nguy cơ mà còn **phát hiện các mẫu ẩn (Hidden Patterns)** - những tổ hợp triệu chứng đặc biệt dẫn đến bệnh tim, giúp hỗ trợ quyết định y khoa.

###  Mục Tiêu Học Tập

- Áp dụng kỹ thuật tiền xử lý dữ liệu y tế (xử lý missing values, outliers)
- Thực hiện Feature Engineering dựa trên kiến thức y khoa
- So sánh hiệu năng của các thuật toán ML khác nhau
- Tối ưu hóa ngưỡng dự đoán (Threshold Tuning) cho bài toán y tế
- Giải thích mô hình (Model Interpretability) với Permutation Importance
- Xây dựng Web Application để demo sản phẩm

---

##  Đặc Điểm Nổi Bật

###  Kỹ Thuật Cao

1. **Pipeline Tiền Xử Lý**
   - Iterative Imputer (MICE) cho missing values
   - RobustScaler chống outlier
   - Feature Engineering dựa y khoa (Cholesterol/Tuổi, Nguy cơ tim mạch rất cao)

2. **Hyperparameter Tuning**
   - Grid Search CV với 5-fold cross-validation
   - Tối ưu hóa theo Recall (ưu tiên phát hiện bệnh nhân)

3. **Threshold Tuning**
   - Tìm ngưỡng cắt tối ưu (thay vì 0.5 mặc định)
   - Tối đa hóa F1-Score để cân bằng Precision và Recall

4. **Hidden Pattern Analysis** 
   - Trích xuất quy luật từ Random Forest (89 patterns)
   - Phân tích tương tác đặc trưng từ XGBoost
   - Giải mã "hộp đen" Neural Network

5. **Model Interpretability**
   - Permutation Importance
   - Feature Importance Visualization
   - Decision Tree Extraction

###  Đánh Giá Toàn Diện

- Confusion Matrix chi tiết
- ROC Curve và AUC-ROC Score
- Classification Report đầy đủ
- So sánh 3 mô hình trên cùng test set

---

##  Dataset

### Thông Tin Chung

- **Tên Dataset:** Heart Disease Dataset
- **Nguồn:** [Kaggle](https://www.kaggle.com/datasets/tan5577/heart-failure-dataset)
- **Số Mẫu:** 918 bệnh nhân
- **Số Đặc Trưng:** 11 features + 1 target
- **Phân Bố Lớp:**
  - Khỏe mạnh (0): 410 người (44.7%)
  - Bệnh tim (1): 508 người (55.3%)

### Các Đặc Trưng (Features)

| Tên Gốc | Tên Tiếng Việt | Loại | Mô Tả |
|----------|----------------|------|-------|
| Age | Tuổi | Số | Tuổi của bệnh nhân (28-77) |
| Sex | Giới_tính | Phân loại | M (Nam), F (Nữ) |
| ChestPainType | Loại_Đau_Ngực | Phân loại | TA, ATA, NAP, ASY |
| RestingBP | Huyết_Áp_Nghỉ | Số | Huyết áp tâm thu (mmHg) |
| Cholesterol | Cholesterol | Số | Cholesterol toàn phần (mg/dL) |
| FastingBS | Đường_Huyết_Đói | Nhị phân | 1 nếu > 120 mg/dL |
| RestingECG | Điện_Tâm_Đồ | Phân loại | Normal, ST, LVH |
| MaxHR | Nhịp_Tim_Tối_Đa | Số | Nhịp tim tối đa (60-202) |
| ExerciseAngina | Đau_Thắt_Vận_Động | Nhị phân | Y (Có), N (Không) |
| Oldpeak | Độ_Chênh_ST | Số | ST depression (-2.6 đến 6.2) |
| ST_Slope | Độ_Dốc_ST | Phân loại | Up, Flat, Down |
| **HeartDisease** | **Bệnh_Tim** | **Target** | **0 (Khỏe), 1 (Bệnh)** |

### Phân Tích Chất Lượng Dữ Liệu

- **Missing Values:**
  - Cholesterol = 0: 172 giá trị (18.7%) → Thay bằng NaN và impute
  - Huyết_Áp_Nghỉ = 0: 1 giá trị → Thay bằng NaN và impute
  
- **Giá Trị Âm:**
  - Độ_Chênh_ST < 0: 13 giá trị
  - **Quyết định:** GIỮ NGUYÊN (có ý nghĩa y khoa - ST Elevation trong nhồi máu cơ tim cấp)
  - Nhóm này có tỷ lệ bệnh **69.23%** (cao hơn trung bình 55.34%)

---

##  Cấu Trúc Dự Án

```
S26-65TTNT_Nhom7_DuDoanBenhTim/
│
├── data/
│   └── heart.csv                          # Dataset gốc
│
├── reports/ 
│  └── N7_report.pdf                       # Báo cáo
│
├── src/
│   ├── preprocessing.py                   # Tiền xử lý dữ liệu & Pipeline
│   ├── feature_engineering.py             # Tạo đặc trưng y khoa
│   ├── eda.py                             # Phân tích khám phá dữ liệu
│   ├── RandomForest_NguyenLeMinhHau.py    # Mô hình Random Forest
│   ├── XGBoost_NguyenLeMinhHau.py         # Mô hình XGBoost
│   ├── NeuralNetwork_NguyenDucHuy.py      # Mô hình Neural Network
│   ├── evaluation.py                      # So sánh & đánh giá mô hình
│   └── app.py                             # Streamlit Web Application
│
├── saved_models/
│   ├── random_forest.pkl                  # Mô hình RF đã train
│   ├── xgboost.pkl                        # Mô hình XGB đã train
│   ├── neural_network.pkl                 # Mô hình NN đã train
│   ├── scaler.pkl                         # RobustScaler
│   ├── imputer.pkl                        # Iterative Imputer
│   ├── feature_columns.pkl                # Danh sách features
│   ├── X_train.pkl, X_test.pkl            # Dữ liệu train/test
│   └── *.metadata.pkl                     # Metadata của mô hình
│
├── outputs/
│   ├── figures/                           # Các biểu đồ (ROC, Confusion Matrix, etc.)
│   | 
│   ├── preprocessing_log.txt              # Log tiền xử lý
│   ├── eda_log.txt                        # Log phân tích EDA
│   ├── RandomForest_log.txt               # Log huấn luyện RF
│   ├── XGBoost_log.txt                    # Log huấn luyện XGB
│   ├── NeuralNetwork_log.txt              # Log huấn luyện NN
│   ├── Evaluation_log.txt                 # Log đánh giá
│   └── Evaluation_Report.txt              # Báo cáo tổng hợp
│ 
├── generate_report.py                     # Script tạo báo cáo Word
├── requirements.txt                       # Dependencies
├── README.md                              # File này
└── .venv/                                 # Virtual environment
```

---

##  Yêu Cầu Hệ Thống

### Phần Cứng

- **RAM:** Tối thiểu 4GB (Khuyến nghị 8GB+)
- **CPU:** Multi-core (Grid Search sử dụng đa luồng)
- **Ổ Đĩa:** ~500MB cho dependencies + models

### Phần Mềm

- **Python:** 3.9 - 3.11
- **Hệ Điều Hành:** Windows 10/11, macOS, Linux

### Thư Viện Chính

| Thư viện | Version | Mục đích |
|----------|---------|----------|
| pandas | ≥2.0.0 | Xử lý dữ liệu |
| numpy | ≥1.24.0 | Tính toán số học |
| scikit-learn | ≥1.3.0 | ML algorithms |
| xgboost | ≥2.0.0 | XGBoost model |
| matplotlib | ≥3.7.0 | Visualization |
| seaborn | ≥0.13.0 | Statistical plots |
| streamlit | ≥1.29.0 | Web application |
| joblib | ≥1.3.0 | Model serialization |

---

##  Hướng Dẫn Cài Đặt

### Bước 1: Clone Repository

```bash
git clone https://github.com/your-username/heart-disease-prediction.git
cd S26-65TTNT_Nhom7_DuDoanBenhTim
```

### Bước 2: Tạo Virtual Environment

**Windows:**
```powershell
python -m venv .venv
.venv\Scripts\activate
```

**macOS/Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Bước 3: Cài Đặt Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Bước 4: Kiểm Tra Cài Đặt

```python
python -c "import sklearn, xgboost, streamlit; print('All packages installed successfully!')"
```

---

## Hướng Dẫn Sử Dụng

### Workflow Đầy Đủ (Chạy Lần Đầu)

Chạy các script theo thứ tự sau:

#### 1️ Tiền Xử Lý Dữ Liệu

```bash
python src/preprocessing.py
```

**Output:**
- `saved_models/eda_dataset.pkl` (Dữ liệu cho EDA)
- `saved_models/X_train.pkl`, `X_test.pkl`, `y_train.pkl`, `y_test.pkl`
- `saved_models/scaler.pkl`, `imputer.pkl`, `feature_columns.pkl`
- `outputs/preprocessing_log.txt`

**Kết quả:**
- Train: 734 samples, Test: 184 samples
- 11 → 17 features sau feature engineering
- Xử lý 172 missing values (Cholesterol) và 1 missing value (Huyết Áp)

#### 2️ Phân Tích Khám Phá Dữ Liệu (EDA)

```bash
python src/eda.py
```

**Output:**
- `outputs/figures/1_Target.png` (Phân bố target)
- `outputs/figures/Num_*.png` (Biểu đồ biến số)
- `outputs/figures/Cat_*.png` (Biểu đồ biến phân loại)
- `outputs/figures/Correlation_Matrix.png`
- `outputs/eda_log.txt`

**Phát hiện chính:**
- Độ_Chênh_ST có khác biệt lớn giữa Khỏe (0.4) và Bệnh (1.3)
- Nam giới có nguy cơ cao hơn: 63.2% vs Nữ 25.9%
- Đau ngực ASY có nguy cơ rất cao: 79.0%

#### 3️ Huấn Luyện Mô Hình

**Random Forest (Nguyễn Lê Minh Hậu):**
```bash
python src/RandomForest_NguyenLeMinhHau.py
```
- Grid Search: 180 fits (36 combinations × 5 folds)
- Best Params: `n_estimators=200, max_depth=15, min_samples_leaf=2`
- Threshold: 0.5491
- Output: `saved_models/random_forest.pkl`, `outputs/figures/RF_*.png`

**XGBoost (Nguyễn Lê Minh Hậu):**
```bash
python src/XGBoost_NguyenLeMinhHau.py
```
- Grid Search: 180 fits (36 combinations × 5 folds)
- Best Params: `learning_rate=0.01, max_depth=3, n_estimators=100`
- Threshold: 0.6711
- Output: `saved_models/xgboost.pkl`, `outputs/figures/XGBoost_*.png`

**Neural Network (Nguyễn Đức Huy):**
```bash
python src/NeuralNetwork_NguyenDucHuy.py
```
- Grid Search: 90 fits (18 combinations × 5 folds)
- Best Params: `hidden_layers=(64,32), alpha=0.0001, lr=0.001`
- Epochs: 39 (với Early Stopping)
- Threshold: 0.6184
- Output: `saved_models/neural_network.pkl`, `outputs/figures/NN_*.png`

#### 4️ Đánh Giá & So Sánh

```bash
python src/evaluation.py
```

**Output:**
- `outputs/figures/Comparison_*.png` (Biểu đồ so sánh)
- `outputs/Evaluation_Report.txt` (Báo cáo chi tiết)
- `outputs/Evaluation_log.txt`

---

## Kết Quả Đạt Được

### Bảng So Sánh Hiệu Năng (Test Set: 184 samples)

| Mô Hình | Accuracy | Precision | Recall | F1-Score | AUC-ROC | FN (Bỏ Sót) |
|---------|----------|-----------|--------|----------|---------|--------------|
| **Random Forest** | **88.04%** | **85.09%** | 95.10% | **89.81%** | 92.32% | 5 |
| **XGBoost** | 82.07% | 77.17% | **96.08%** | 85.59% | 90.64% | **4**  |
| **Neural Network** | 86.96% | 84.21% | 94.12% | 88.89% | **94.49%** | 6 |

### Xếp Hạng Mô Hình

**Tiêu chí xếp hạng:** Recall (ưu tiên) > F1-Score > Accuracy

#### Hạng 1: **XGBoost** (Khuyến nghị sử dụng)
- **Recall: 96.08%** - Phát hiện được 98/102 bệnh nhân
- **Bỏ sót: 4 người** (thấp nhất)
- **Ý nghĩa:** Tốt nhất cho ứng dụng y tế (ưu tiên tránh bỏ sót)

#### Hạng 2: **Random Forest**
- **Recall: 95.10%** - Phát hiện được 97/102 bệnh nhân
- **F1-Score: 89.81%** (cao nhất) - Cân bằng tốt
- **Bỏ sót: 5 người**

#### Hạng 3: **Neural Network**
- **AUC-ROC: 94.49%** (cao nhất) - Khả năng phân loại tổng quát tốt
- **Recall: 94.12%** - Phát hiện được 96/102 bệnh nhân
- **Bỏ sót: 6 người**

### Confusion Matrix Chi Tiết

#### XGBoost (Mô Hình Tốt Nhất)
```
                Dự Đoán
              Khỏe  Bệnh
Thực  Khỏe     53    29
Tế    Bệnh      4    98
```
- True Negatives (TN): 53 (28.8%)
- False Positives (FP): 29 (15.8%) - Dương tính giả
- False Negatives (FN): **4 (2.2%)** - Âm tính giả 
- True Positives (TP): 98 (53.3%)

### Hidden Patterns (Top 3)

#### Pattern 1 (Random Forest - Nguy cơ 100%)
```
NEU:
  + Độ_Dốc_ST_Up ≤ 0.50
  + Giới_tính > 0.50 (Nam)
  + Độ_Chênh_ST ≤ -0.30 (ST Depression)
→ Nguy cơ bệnh tim RẤT CAO
→ Dựa trên 51 bệnh nhân
```

#### Pattern 2 (XGBoost - Nguy cơ 92%)
```
NEU:
  + Độ_Chênh_ST > 0.67
  + Đường_Huyết_Đói > 0.00 (Có tiểu đường)
→ Nguy cơ bệnh tim RẤT CAO
→ Dựa trên 57 bệnh nhân
```

#### Pattern 3 (Neural Network - Nguy cơ 92%)
```
NEU:
  + Độ_Chênh_ST > 0.95
→ Nguy cơ bệnh tim RẤT CAO
→ Dựa trên 72 bệnh nhân
```

### Feature Importance (Top 5)

**Random Forest (Permutation):**
1. Độ_Dốc_ST_Up: 0.0467
2. Độ_Chênh_ST: 0.0283
3. Độ_Dốc_ST_Flat: 0.0152
4. Đau_Thắt_Vận_Động: 0.0141
5. Đường_Huyết_Đói: 0.0125

**XGBoost (Gain):**
1. Độ_Dốc_ST_Up: 0.4431
2. Độ_Chênh_ST: 0.0890
3. Đau_Thắt_Vận_Động: 0.0779
4. Đường_Huyết_Đói: 0.0518
5. Loại_Đau_Ngực_NAP: 0.0482

---

## 🌐 Demo Web Application

### Khởi Chạy Streamlit App

```bash
streamlit run src/app.py
```

Ứng dụng sẽ mở tại: `http://localhost:8501`

### Các Tính Năng Chính

1. **🏠 Trang Chủ**
   - Giới thiệu dự án
   - Workflow tổng quan

2. **📂 Dữ Liệu & Mô Tả**
   - Thông tin dataset
   - Mô tả các đặc trưng

3. **📊 Phân Tích EDA**
   - Biểu đồ phân bố
   - Correlation matrix
   - Thống kê mô tả

4. **🔬 So Sánh Mô Hình**
   - Bảng so sánh metrics
   - ROC curves
   - Confusion matrices

5. **🩺 Dự Đoán Nguy Cơ**
   - Nhập thông tin bệnh nhân
   - Dự đoán bằng 3 mô hình
   - Hiển thị xác suất chi tiết

6. **🔍 Hidden Patterns**
   - Các mẫu ẩn đã phát hiện
   - Quy luật triệu chứng

7. **📖 Hướng Dẫn Sử Dụng**
   - Cách sử dụng ứng dụng
   - Giải thích kết quả

### Screenshot Demo

```
[Hình ảnh sẽ được thêm sau khi chạy app và chụp màn hình]
```

---

## 👥 Thành Viên Nhóm

| STT | Họ Tên | MSSV | Nhiệm Vụ | Đóng Góp |
|-----|--------|------|----------|----------|
| 1 | **Nguyễn Lê Minh Hậu** | [2351267261] | **Team Lead & ML Engineer** | Random Forest, XGBoost, Pipeline Integration, Hidden Patterns Extraction |
| 2 | **Nguyễn Đức Huy** | [2351267265] | **ML Engineer & Evaluator** | Neural Network, Model Evaluation, Comparison Analysis, Documentation |

### Phân Công Chi Tiết

#### Nguyễn Lê Minh Hậu
-  Random Forest: Hyperparameter tuning, Threshold optimization, Permutation importance
-  XGBoost: Grid search, Feature importance, Interaction analysis
-  Pipeline Integration: Kết nối các module, workflow automation
-  Hidden Patterns: Trích xuất quy luật từ Random Forest và XGBoost
-  Code Review: Kiểm tra chất lượng code

#### Nguyễn Đức Huy
-  Neural Network: Architecture design, Early stopping, Learning curve analysis
-  Evaluation Module: So sánh 3 mô hình, metrics visualization
-  Model Interpretability: Permutation importance cho NN
-  Hidden Patterns: Phân tích patterns từ Neural Network
-  Documentation: README, reports, code comments

#### Chung (Cả Nhóm)
-  Preprocessing: Feature engineering, data cleaning (collaborative)
-  EDA: Exploratory data analysis, visualization
-  Streamlit App: Web application development
-  Testing & Debugging: Kiểm tra và sửa lỗi

---

## Kiến Thức Áp Dụng

### Machine Learning
- Supervised Learning (Classification)
- Ensemble Methods (Random Forest, XGBoost)
- Neural Networks (MLPClassifier)
- Cross-Validation (K-Fold)
- Hyperparameter Tuning (Grid Search)
- Model Evaluation (Confusion Matrix, ROC, AUC)

### Data Science
- Data Preprocessing (Imputation, Scaling)
- Feature Engineering
- Exploratory Data Analysis (EDA)
- Data Visualization (Matplotlib, Seaborn)
- Statistical Analysis

### Software Engineering
- Modular Code Design
- Logging & Documentation
- Version Control (Git)
- Virtual Environments
- Web Application Development (Streamlit)

---


##  Lưu Ý Quan Trọng

### Giới Hạn Sử Dụng

 **Hệ thống này CHỈ MANG TÍNH CHẤT HỌC TẬP VÀ NGHIÊN CỨU.**

- **KHÔNG thay thế** việc khám, chẩn đoán và điều trị y khoa chuyên nghiệp
- **KHÔNG tự ý** thay đổi phác đồ điều trị dựa trên kết quả dự đoán
- Luôn tham khảo ý kiến bác sĩ chuyên khoa tim mạch
- Kết quả dự đoán có thể sai lệch (False Negatives vẫn tồn tại)

### Khuyến Nghị Phát Triển

 **Các cải tiến trong tương lai:**

1. **Dataset lớn hơn:** Thu thập thêm dữ liệu từ bệnh viện tại Việt Nam
2. **Ensemble Voting:** Kết hợp 3 mô hình bằng Soft Voting
3. **SHAP Values:** Giải thích chi tiết hơn cho từng dự đoán
4. **Production Deployment:** Dockerize và deploy lên cloud (AWS, GCP)
5. **Mobile App:** Phát triển ứng dụng di động (React Native / Flutter)
6. **Real-time Monitoring:** Tích hợp với thiết bị đo y tế (wearable devices)

---


##  Đóng Góp & Liên Hệ

### Báo Lỗi (Bug Report)

Nếu phát hiện lỗi, vui lòng tạo [Issue](https://github.com/your-repo/issues) với thông tin:
- Mô tả lỗi chi tiết
- Các bước tái hiện
- Screenshot (nếu có)
- Environment (OS, Python version)



## 📎 Phụ Lục

### A. Các Lệnh Nhanh (Quick Commands)

```bash
# Chạy toàn bộ pipeline
python src/preprocessing.py && \
python src/eda.py && \
python src/RandomForest_NguyenLeMinhHau.py && \
python src/XGBoost_NguyenLeMinhHau.py && \
python src/NeuralNetwork_NguyenDucHuy.py && \
python src/evaluation.py

# Khởi động web app
streamlit run src/app.py

# Tạo báo cáo Word (nếu có)
python generate_report.py
```

### B. Troubleshooting

**Lỗi: "No module named 'sklearn'"**
```bash
pip install scikit-learn
```

**Lỗi: "FileNotFoundError: [Errno 2] No such file or directory: 'data/heart.csv'"**
- Kiểm tra file `heart.csv` có tồn tại trong thư mục `data/`
- Đảm bảo đang chạy lệnh từ thư mục gốc của dự án

**Streamlit App không khởi động:**
```bash
# Kiểm tra port 8501 đã được sử dụng chưa
streamlit run src/app.py --server.port 8502
```

### C. Environment Variables

Nếu cần tùy chỉnh, tạo file `.env`:

```env
# Đường dẫn dataset
DATA_PATH=data/heart.csv

# Random seed
RANDOM_STATE=2026

# Test size
TEST_SIZE=0.2
```

---

** Cảm ơn bạn đã quan tâm đến dự án của chúng tôi! **
