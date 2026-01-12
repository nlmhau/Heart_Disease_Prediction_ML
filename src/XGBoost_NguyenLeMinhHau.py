# ============================================================
# MODEL: XGBOOST 
# Nguyễn Lê Minh Hậu
# Mục tiêu:
#   1. Threshold Tuning (Tìm ngưỡng cắt tối ưu cho Y tế)
#   2. Visualization chuyên sâu (ROC, Confusion Matrix)
#   3. Phân tích tương tác đặc trưng (Interaction Analysis)
# ============================================================

import os
import joblib
import numpy as np
import pandas as pd

# Đặt backend matplotlib trước khi import pyplot (tránh lỗi tkinter)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from datetime import datetime

# Tắt cảnh báo
warnings.filterwarnings("ignore")

from xgboost import XGBClassifier, plot_importance
from sklearn.metrics import (accuracy_score, confusion_matrix, classification_report, 
                             roc_auc_score, roc_curve, precision_recall_curve, f1_score)
from sklearn.model_selection import GridSearchCV

# ------------------------------------------------------------
# 1. CẤU HÌNH & LOAD DATA
# ------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAVED_MODELS_DIR = os.path.join(BASE_DIR, "../saved_models")
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

logger = Logger(os.path.join(OUTPUTS_DIR, "XGBoost_log.txt"))

logger.log("="*60)
logger.log("I. LOAD DỮ LIỆU & CHUẨN BỊ")
logger.log("="*60)

try:
    X_train = joblib.load(os.path.join(SAVED_MODELS_DIR, "X_train.pkl"))
    X_test  = joblib.load(os.path.join(SAVED_MODELS_DIR, "X_test.pkl"))
    y_train = joblib.load(os.path.join(SAVED_MODELS_DIR, "y_train.pkl"))
    y_test  = joblib.load(os.path.join(SAVED_MODELS_DIR, "y_test.pkl"))
    feature_names = X_train.columns.tolist()
    logger.log(f" Da load du lieu: Train {X_train.shape}, Test {X_test.shape}")
    logger.log(f" - So dac trung: {len(feature_names)}")
    logger.log(f" - Phan bo lop (train): Khoe={sum(y_train==0)}, Benh={sum(y_train==1)}")
    logger.log(f" - Phan bo lop (test): Khoe={sum(y_test==0)}, Benh={sum(y_test==1)}")
except FileNotFoundError:
    logger.log(" Lỗi: Không tìm thấy file dữ liệu. Hãy chạy preprocessing.py trước!")
    logger.close()

    exit()

# ------------------------------------------------------------
# 2. HUẤN LUYỆN & TUNING (GRID SEARCH)
# ------------------------------------------------------------
logger.log("\n" + "="*60)
logger.log("II. HUẤN LUYỆN & TỐI ƯU HÓA (GRID SEARCH)")
logger.log("="*60)

# Không gian tham số (đã được tinh chỉnh cho dataset nhỏ < 1000 dòng)
param_grid = {
    "n_estimators": [100, 200],         # Số lượng cây
    "max_depth": [3, 4, 5],             # Độ sâu (thấp để tránh overfitting)
    "learning_rate": [0.01, 0.05, 0.1], # Tốc độ học
    "subsample": [0.8],                 # Chọn mẫu ngẫu nhiên
    "scale_pos_weight": [1, 2]          # Quan trọng: Tăng trọng số cho lớp Bệnh (1)
}


logger.log(" Dang tim tham so toi uu (GridSearchCV)...")
logger.log(f" - Tong so to hop tham so: {len(param_grid['n_estimators']) * len(param_grid['max_depth']) * len(param_grid['learning_rate']) * len(param_grid['subsample']) * len(param_grid['scale_pos_weight'])}")
logger.log(f" - So fold cross-validation: 5")
logger.log(f" - Tieu chi danh gia: Recall (uu tien phat hien benh nhan)")
xgb = XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    use_label_encoder=False,
    random_state=2026,
    n_jobs=1
)

# Ưu tiên Recall (ưu tiên phát hiện bệnh nhân)
grid_search = GridSearchCV(
    estimator=xgb,
    param_grid=param_grid,
    scoring='recall', 
    cv=5,
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_

logger.log(f"\n Hoan thanh Grid Search!")
logger.log(f" - Best Params: {grid_search.best_params_}")
logger.log(f" - Best CV Recall: {grid_search.best_score_:.4f}")
logger.log(f" - Thoi gian train: Da hoan thanh")

# ------------------------------------------------------------
# 3. THRESHOLD TUNING (KỸ THUẬT SENIOR)
# ------------------------------------------------------------
logger.log("\n" + "="*60)
logger.log("III. TỐI ƯU HÓA NGƯỠNG DỰ ĐOÁN (THRESHOLD TUNING)")
logger.log("="*60)
logger.log(" Tìm ngưỡng cắt tối ưu (thay vì 0.5) để tối đa hóa F1-Score.")

# Dự đoán xác suất
y_probs = best_model.predict_proba(X_test)[:, 1]

# Tìm ngưỡng tốt nhất
precisions, recalls, thresholds = precision_recall_curve(y_test, y_probs)
f1_scores = 2 * (precisions * recalls) / (precisions + recalls)
best_idx = np.argmax(f1_scores)
best_threshold = thresholds[best_idx]

logger.log(f" Nguong toi uu: {best_threshold:.4f}")
logger.log(f"   -> Precision={precisions[best_idx]:.4f} | Recall={recalls[best_idx]:.4f}")

# Áp dụng ngưỡng mới
y_pred_new = (y_probs >= best_threshold).astype(int)

# Đánh giá chi tiết
logger.log("\n" + "="*60)
logger.log("KET QUA TREN TAP TEST")
logger.log("="*60)
logger.log(classification_report(y_test, y_pred_new, target_names=['Khoe', 'Benh'], digits=4))
logger.log(f"\n AUC-ROC Score: {roc_auc_score(y_test, y_probs):.4f}")

# Confusion Matrix chi tiết
cm = confusion_matrix(y_test, y_pred_new)
tn, fp, fn, tp = cm.ravel()
logger.log(f"\n Confusion Matrix:")
logger.log(f"   - True Negatives (TN): {tn} (Du doan dung nguoi khoe)")
logger.log(f"   - False Positives (FP): {fp} (Du doan nham nguoi khoe thanh benh)")
logger.log(f"   - False Negatives (FN): {fn} (BO SOT nguoi benh - quan trong!)")
logger.log(f"   - True Positives (TP): {tp} (Du doan dung nguoi benh)")

# Phần visualization
logger.log("\n" + "="*60)
logger.log("IV. TAO BIEU DO DANH GIA")
logger.log("="*60)

# Vẽ ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_probs)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc_score(y_test, y_probs):.4f}", color="#e74c3c", lw=2)
plt.plot([0, 1], [0, 1], linestyle="--", color="gray", alpha=0.3)
plt.xlabel("False Positive Rate (Ty le duong tinh gia)", fontsize=11)
plt.ylabel("True Positive Rate (Ty le duong tinh that)", fontsize=11)
plt.title("ROC Curve - XGBoost", fontsize=13, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
save_path_roc = os.path.join(FIGURES_DIR, "XGBoost_ROC.png")
plt.savefig(save_path_roc, dpi=150)
plt.close()
logger.log("   Da luu: XGBoost_ROC.png")

# Vẽ Confusion Matrix
plt.figure(figsize=(7, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', cbar=True,
            xticklabels=['Khoe', 'Benh'], yticklabels=['Khoe', 'Benh'],
            annot_kws={"fontsize": 14})
plt.title(f"Confusion Matrix (Nguong = {best_threshold:.3f})", fontsize=13, fontweight='bold')
plt.ylabel("Thuc te", fontsize=11)
plt.xlabel("Du doan", fontsize=11)
save_path_cm = os.path.join(FIGURES_DIR, "XGBoost_Confusion_Matrix.png")
plt.savefig(save_path_cm, dpi=150)
plt.close()
logger.log("   Da luu: XGBoost_Confusion_Matrix.png")

# ------------------------------------------------------------
# 5. FEATURE IMPORTANCE & INTERACTION
# ------------------------------------------------------------
logger.log("\n" + "="*60)
logger.log("V. PHAN TICH DO QUAN TRONG DAC TRUNG")
logger.log("="*60)
logger.log(" Dang tinh toan Feature Importance...")

# Lấy Feature Importance
importances = best_model.feature_importances_
indices = np.argsort(importances)[::-1]
top_features = [feature_names[i] for i in indices]

# In Top 10
df_imp = pd.DataFrame({'Dac trung': top_features, 'Do quan trong': importances[indices]})
logger.log("\n Top 10 dac trung anh huong nhat:")
for idx, row in df_imp.head(10).iterrows():
    logger.log(f"   {idx+1}. {row['Dac trung']:<25} -> {row['Do quan trong']:.4f}")

# Vẽ biểu đồ
plt.figure(figsize=(10, 8))
plt.barh(df_imp["Dac trung"][:10], df_imp["Do quan trong"][:10], color='#e74c3c')
plt.gca().invert_yaxis()
plt.title("Top 10 Dac Trung Quan Trong Nhat (XGBoost)", fontsize=13, fontweight='bold')
plt.xlabel("Muc do quan trong", fontsize=11)
plt.ylabel("Dac trung", fontsize=11)
plt.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "XGBoost_Feature_Importance.png"), dpi=150)
plt.close()
logger.log("   Da luu: XGBoost_Feature_Importance.png")

# Bỏ phần VI cũ - đã được thay thế bằng phần VII mới

# ------------------------------------------------------------
# 7. PHAT HIEN MAU AN (HIDDEN PATTERNS)
# ------------------------------------------------------------
logger.log("\n" + "="*60)
logger.log("VII. PHAT HIEN MAU AN (HIDDEN PATTERNS)")
logger.log("="*60)
logger.log(" Trich xuat cac quy luat: Trieu chung A + B + C → Benh Tim")

def extract_xgboost_rules(model, X_test, y_test, y_probs, feature_names, max_rules=30):
    """Trích xuất quy luật từ XGBoost bằng cách phân tích các case dự đoán đúng"""
    import numpy as np
    import pandas as pd
    
    # Lấy các trường hợp dự đoán bệnh với các mức độ xác suất khác nhau
    patterns = []
    
    # Chia thành các nhóm xác suất
    prob_ranges = [
        (0.85, 1.0, "Rat cao"),
        (0.70, 0.85, "Cao"),
        (0.55, 0.70, "Trung binh-cao"),
        (0.40, 0.55, "Trung binh"),
        (0.25, 0.40, "Vua"),
        (0.10, 0.25, "Thap"),
    ]
    
    # Lấy feature importance
    importance = model.feature_importances_
    top_features_idx = np.argsort(importance)[-10:][::-1]
    top_features = [feature_names[i] for i in top_features_idx]
    
    for min_prob, max_prob, level in prob_ranges:
        indices = np.where((y_test == 1) & (y_probs >= min_prob) & (y_probs < max_prob))[0]
        
        if len(indices) < 1:  # Chỉ cần >=1 bệnh nhân
            continue
            
        X_subset = X_test.iloc[indices]
        
        # Tìm đặc trưng phân biệt nhất trong nhóm này
        patterns_for_range = 0
        max_patterns_for_range = 2 if len(indices) >= 50 else 1  # Nếu range có >= 50 bệnh nhân, lấy 2 patterns
        
        # ƯU TIÊN: Pattern Combo TRƯỚC (2 triệu chứng kết hợp)
        if len(top_features) >= 2 and patterns_for_range < max_patterns_for_range:
            # Thử NHIỀU cặp combo, không chỉ các cặp liền kề
            combo_pairs = []
            for i in range(min(6, len(top_features))):
                for j in range(i+1, min(6, len(top_features))):
                    combo_pairs.append((i, j))
            
            for i, j in combo_pairs[:12]:  # Check tối đa 12 cặp
                if patterns_for_range >= max_patterns_for_range:
                    break
                    
                feat1, feat2 = top_features[i], top_features[j]
                
                if X_subset[feat1].dtype == 'bool' or X_subset[feat2].dtype == 'bool':
                    continue
                    
                if X_subset[feat1].std() > 0 and X_subset[feat2].std() > 0:
                    t1 = X_subset[feat1].quantile(0.50)  # Giảm xuống 0.50
                    t2 = X_subset[feat2].quantile(0.50)
                    
                    mask = (X_subset[feat1] > t1) & (X_subset[feat2] > t2)
                    ratio = mask.mean()
                    
                    if ratio >= 0.03:  # Giảm XUỐNG 3% để lấy nhiều combo hơn
                        patterns.append({
                            'features': [feat1, feat2],
                            'thresholds': [t1, t2],
                            'confidence': (min_prob + max_prob) / 2,
                            'sample_count': len(indices),
                            'type': 'combo'
                        })
                        patterns_for_range += 1
        
        # Nếu không tìm được combo, lấy Single
        if patterns_for_range < max_patterns_for_range:
            for feat in top_features[:8]:
                if patterns_for_range >= max_patterns_for_range:
                    break
                    
                if X_subset[feat].dtype == 'bool':
                    continue
                    
                if X_subset[feat].std() > 0:
                    threshold = X_subset[feat].quantile(0.6)
                    high_ratio = (X_subset[feat] > threshold).mean()
                    
                    if high_ratio >= 0.10:  # HẠ xuống 10% để bắt được range thấp
                        patterns.append({
                            'feature': feat,
                            'condition': '>',
                            'threshold': threshold,
                            'confidence': (min_prob + max_prob) / 2,
                            'sample_count': len(indices),
                            'type': 'single'
                        })
                        patterns_for_range += 1
    
    return patterns[:max_rules]

def clean_feature_name(feat):
    """Chuyển tên feature thành tên tiếng Việt"""
    mapping = {
        'Tuoi': 'Tuổi',
        'Gioi_tinh': 'Giới_tính',
        'Do_Chenh_ST': 'Độ_Chênh_ST',
        'Do_Doc_ST_Flat': 'Độ_Dốc_ST_Flat',
        'Do_Doc_ST_Up': 'Độ_Dốc_ST_Up',
        'Dau_That_Van_Dong': 'Đau_Thắt_Vận_Động',
        'Nhip_Tim_Toi_Da': 'Nhịp_Tim_Tối_Đa',
        'Cholesterol': 'Cholesterol',
        'Huyet_Ap_Nghi': 'Huyết_Áp_Nghỉ',
        'Duong_Huyet_Doi': 'Đường_Huyết_Đói'
    }
    return mapping.get(feat, feat)

patterns = extract_xgboost_rules(best_model, X_test, y_test, y_probs, feature_names)

# Tính số bệnh nhân với các mức xác suất khác nhau
n_total = np.sum(y_test == 1)

if patterns:
    # Sắp xếp theo confidence
    patterns_sorted = sorted(patterns, key=lambda x: x['confidence'], reverse=True)
    
    # Hiển thị top 5 patterns
    display_patterns = patterns_sorted[:5]
    logger.log(f"\n Trich xuat duoc {len(patterns_sorted)} patterns tu XGBoost")
    logger.log(f" Hien thi top {len(display_patterns)} mau dai dien theo cac muc do rui ro:")
    logger.log()
    
    for i, pattern in enumerate(display_patterns, 1):
        if pattern['type'] == 'single':
            feat_text = clean_feature_name(pattern['feature'])
            threshold_val = pattern['threshold']
            logger.log(f"{i}. NEU:")
            logger.log(f"      + {feat_text} > {threshold_val:.2f}")
            logger.log(f"   → KET LUAN: Nguy co benh tim {pattern['confidence']*100:.0f}%")
            logger.log(f"   → Can cu: XGBoost phan tich tu {pattern['sample_count']} benh nhan")
            logger.log()
        else:  # combo
            logger.log(f"{i}. NEU:")
            for j, feat in enumerate(pattern['features']):
                feat_text = clean_feature_name(feat)
                threshold_val = pattern['thresholds'][j]
                logger.log(f"      + {feat_text} > {threshold_val:.2f}")
            logger.log(f"   → KET LUAN: Nguy co benh tim {pattern['confidence']*100:.0f}%")
            logger.log(f"   → Can cu: XGBoost phan tich tu {pattern['sample_count']} benh nhan")
            logger.log(f"   → Ghi chu: Muc do trieu chung KHAC NHAU giua cac nhom")
            logger.log()
else:
    logger.log("\n Khong du du lieu de trich xuat pattern.")

# ------------------------------------------------------------
# 8. SAVE MODEL
# ------------------------------------------------------------
logger.log("\n" + "="*60)
logger.log("VIII. LUU MO HINH")
logger.log("="*60)

joblib.dump(best_model, os.path.join(SAVED_MODELS_DIR, "xgboost.pkl"))
logger.log(" Da luu mo hinh XGBoost: xgboost.pkl")

# Lưu metadata
metadata = {
    'threshold': best_threshold,
    'best_params': grid_search.best_params_,
    'cv_f1': grid_search.best_score_,
    'test_accuracy': (y_pred_new == y_test).mean(),
    'test_recall': recalls[best_idx],
    'auc_roc': roc_auc_score(y_test, y_probs)
}
joblib.dump(metadata, os.path.join(SAVED_MODELS_DIR, "xgboost_metadata.pkl"))
logger.log(" Da luu metadata: xgboost_metadata.pkl")

logger.log("\n" + "="*60)
logger.log(" HOAN THANH!")
logger.log("="*60)
logger.log(f" Tom tat ket qua:")
logger.log(f"   - Nguong toi uu: {best_threshold:.4f}")
logger.log(f"   - Test Accuracy: {metadata['test_accuracy']:.4f}")
logger.log(f"   - Test Recall: {metadata['test_recall']:.4f}")
logger.log(f"   - AUC-ROC: {metadata['auc_roc']:.4f}")
logger.log(f"   - False Negatives: {fn} (Bo sot {fn} benh nhan)")
logger.log(f"\n Cac file output da duoc luu tai: {FIGURES_DIR}")
logger.log(f" File log da duoc luu tai: {os.path.join(OUTPUTS_DIR, 'XGBoost_log.txt')}")

logger.close()
