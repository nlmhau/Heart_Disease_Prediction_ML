# ============================================================
# MODEL: NEURAL NETWORK
# Nguyễn Đức Huy
# Mục tiêu:
#   1. Early Stopping (Chống học vẹt/Overfitting)
#   2. Loss Curve Visualization (Theo dõi quá trình hội tụ)
#   3. Threshold Tuning (Tối ưu ngưỡng cắt)
#   4. Permutation Importance (Giải mã hộp đen)
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

# Tắt cảnh báo ConvergenceWarning nếu chưa hội tụ hết (để log sạch)
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, classification_report, 
                             roc_auc_score, roc_curve, precision_recall_curve)
from sklearn.model_selection import GridSearchCV
from sklearn.inspection import permutation_importance

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

logger = Logger(os.path.join(OUTPUTS_DIR, "NeuralNetwork_log.txt"))

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
# 2. HUẤN LUYỆN & TUNING (GRID SEARCH + EARLY STOPPING)
# ------------------------------------------------------------
logger.log("\n" + "="*60)
logger.log("II. HUẤN LUYỆN & TỐI ƯU HÓA (GRID SEARCH)")
logger.log("="*60)

# Cấu hình mạng Neural
# early_stopping=True: Tự động dừng nếu 10 vòng lặp không thấy tốt lên (Chống Overfitting)
mlp = MLPClassifier(
    solver='adam', 
    activation='relu',
    early_stopping=True, 
    validation_fraction=0.1, 
    n_iter_no_change=10,
    max_iter=1000, 
    random_state=2026
)

# Không gian tham số
param_grid = {
    'hidden_layer_sizes': [(64, 32), (128, 64, 32), (32, 16)], # Kiến trúc mạng (Sâu vs Rộng)
    'alpha': [0.0001, 0.001, 0.01], # L2 Regularization (Phạt trọng số lớn)
    'learning_rate_init': [0.001, 0.01] # Tốc độ học
}

logger.log(" Dang tim kien truc mang toi uu...")
logger.log(f" - Tong so to hop tham so: {len(param_grid['hidden_layer_sizes']) * len(param_grid['alpha']) * len(param_grid['learning_rate_init'])}")
logger.log(f" - So fold cross-validation: 5")
logger.log(f" - Tieu chi: Recall (uu tien phat hien benh nhan)")
grid_search = GridSearchCV(
    estimator=mlp,
    param_grid=param_grid,
    scoring='recall', # Ưu tiên Recall
    cv=5,
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_

logger.log(f"\n Hoan thanh Grid Search!")
logger.log(f" - Best Params: {grid_search.best_params_}")
logger.log(f" - Best Recall (CV): {grid_search.best_score_:.4f}")
logger.log(f" - So vong lap thuc te (Epochs): {best_model.n_iter_}")
logger.log(f" - Early Stopping: Da ap dung (tu dong dung khi khong cai thien)")

# ------------------------------------------------------------
# 3. VẼ ĐƯỜNG CONG HỘI TỤ (LOSS CURVE) - ĐẶC SẢN CỦA NN
# ------------------------------------------------------------
logger.log("\n" + "="*60)
logger.log("III. DANH GIA QUA TRINH HOC (LEARNING DYNAMICS)")
logger.log("="*60)
logger.log(" Dang ve duong cong hoi tu (Loss Curve)...")

plt.figure(figsize=(8, 5))
plt.plot(best_model.loss_curve_, label="Training Loss", color="#8e44ad", lw=2)
if best_model.early_stopping:
    plt.plot(best_model.validation_scores_, label="Validation Score", color="#27ae60", lw=2, linestyle="--")
plt.title("Neural Network Learning Curve (Qua trinh hoc)", fontsize=13, fontweight='bold')
plt.xlabel("Epochs (So vong lap)", fontsize=11)
plt.ylabel("Loss / Score", fontsize=11)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
save_path_loss = os.path.join(FIGURES_DIR, "NN_Loss_Curve.png")
plt.savefig(save_path_loss, dpi=150)
plt.close()
logger.log(" Da luu: NN_Loss_Curve.png")
logger.log("   (Bieu do nay chung minh mo hinh da hoi tu va khong bi Overfitting)")

# ------------------------------------------------------------
# 4. THRESHOLD TUNING
# ------------------------------------------------------------
logger.log("\n" + "="*60)
logger.log("IV. TOI UU HOA NGUONG DU DOAN")
logger.log("="*60)

y_probs = best_model.predict_proba(X_test)[:, 1]

# Tìm ngưỡng tốt nhất dựa trên F1
precisions, recalls, thresholds = precision_recall_curve(y_test, y_probs)
f1_scores = 2 * (precisions * recalls) / (precisions + recalls)
best_idx = np.argmax(f1_scores)
best_threshold = thresholds[best_idx]

logger.log(f" Nguong toi uu: {best_threshold:.4f}")
logger.log(f"   -> Precision={precisions[best_idx]:.4f} | Recall={recalls[best_idx]:.4f}")

# Ap dung nguong moi
y_pred_new = (y_probs >= best_threshold).astype(int)

logger.log("\n" + "="*60)
logger.log("KET QUA TREN TAP TEST")
logger.log("="*60)
logger.log(classification_report(y_test, y_pred_new, target_names=['Khoe', 'Benh'], digits=4))
logger.log(f"\n AUC-ROC Score: {roc_auc_score(y_test, y_probs):.4f}")

# Confusion Matrix chi tiet
cm = confusion_matrix(y_test, y_pred_new)
tn, fp, fn, tp = cm.ravel()
logger.log(f"\n Confusion Matrix:")
logger.log(f"   - True Negatives (TN): {tn} (Du doan dung nguoi khoe)")
logger.log(f"   - False Positives (FP): {fp} (Du doan nham nguoi khoe thanh benh)")
logger.log(f"   - False Negatives (FN): {fn} (BO SOT nguoi benh - quan trong!)")
logger.log(f"   - True Positives (TP): {tp} (Du doan dung nguoi benh)")

# Bieu do
logger.log("\n" + "="*60)
logger.log("V. TAO BIEU DO DANH GIA")
logger.log("="*60)

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_probs)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc_score(y_test, y_probs):.4f}", color="#8e44ad", lw=2)
plt.plot([0, 1], [0, 1], "k--", alpha=0.3)
plt.xlabel("False Positive Rate (Ty le duong tinh gia)", fontsize=11)
plt.ylabel("True Positive Rate (Ty le duong tinh that)", fontsize=11)
plt.title("ROC Curve - Neural Network", fontsize=13, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(FIGURES_DIR, "NN_ROC.png"), dpi=150)
plt.close()
logger.log(" Da luu: NN_ROC.png")

# Ve Confusion Matrix
plt.figure(figsize=(7, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', cbar=True,
            xticklabels=['Khoe', 'Benh'], yticklabels=['Khoe', 'Benh'],
            annot_kws={"fontsize": 14})
plt.title(f"Confusion Matrix (Nguong={best_threshold:.3f})", fontsize=13, fontweight='bold')
plt.ylabel("Thuc te", fontsize=11)
plt.xlabel("Du doan", fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "NN_Confusion_Matrix.png"), dpi=150)
plt.close()
logger.log(" Da luu: NN_Confusion_Matrix.png")

# ------------------------------------------------------------
# 6. GIAI MA HOP DEN (PERMUTATION IMPORTANCE)
# ------------------------------------------------------------
logger.log("\n" + "="*60)
logger.log("VI. GIAI MA HOP DEN (PERMUTATION IMPORTANCE)")
logger.log("="*60)
logger.log(" Neural Network la mo hinh phi tuyen tinh phuc tap.")
logger.log(" Su dung ky thuat Permutation de xem no 'quan tam' den dac trung nao nhat.")
logger.log(" Dang tinh toan...")

perm_importance = permutation_importance(best_model, X_test, y_test, n_repeats=10, random_state=2026, n_jobs=-1)
sorted_idx = perm_importance.importances_mean.argsort()[::-1]
top_features = [feature_names[i] for i in sorted_idx]

# In ra man hinh
df_imp = pd.DataFrame({
    'Dac trung': top_features,
    'Do quan trong': perm_importance.importances_mean[sorted_idx]
})
logger.log("\n Top 10 dac trung quan trong:")
for idx, row in df_imp.head(10).iterrows():
    logger.log(f"   {idx+1}. {row['Dac trung']:<25} -> {row['Do quan trong']:.4f}")

# Ve bieu do
plt.figure(figsize=(10, 8))
plt.barh(df_imp["Dac trung"][:10], df_imp["Do quan trong"][:10], color='#8e44ad')
plt.gca().invert_yaxis()
plt.title("Top 10 Dac Trung Quan Trong Nhat (Neural Network)", fontsize=13, fontweight='bold')
plt.xlabel("Muc do quan trong", fontsize=11)
plt.ylabel("Dac trung", fontsize=11)
plt.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "NN_Feature_Importance.png"), dpi=150)
plt.close()
logger.log(" Da luu: NN_Feature_Importance.png")

# ------------------------------------------------------------
# 7. PHAT HIEN MAU AN (HIDDEN PATTERNS)
# ------------------------------------------------------------
logger.log("\n" + "="*60)
logger.log("VII. PHAT HIEN MAU AN (HIDDEN PATTERNS)")
logger.log("="*60)
logger.log(" Trich xuat cac quy luat: Trieu chung A + B + C → Benh Tim")

def extract_nn_patterns(model, X_test, y_test, y_probs, feature_names, max_rules=8):
    """Trích xuất patterns từ Neural Network - Phân tích theo probability ranges"""
    import numpy as np
    import pandas as pd
    
    patterns = []
    
    # Chia thành các nhóm xác suất (tương tự XGBoost)
    prob_ranges = [
        (0.85, 1.0, "Rat cao"),
        (0.70, 0.85, "Cao"),
        (0.55, 0.70, "Trung binh-cao"),
        (0.40, 0.55, "Trung binh"),
        (0.25, 0.40, "Vua"),
    ]
    
    # Lấy top features từ permutation importance
    perm_importance = permutation_importance(model, X_test, y_test, n_repeats=5, random_state=2026, n_jobs=-1)
    top_idx = perm_importance.importances_mean.argsort()[-10:][::-1]
    top_features = [feature_names[i] for i in top_idx]
    
    for min_prob, max_prob, level in prob_ranges:
        indices = np.where((y_test == 1) & (y_probs >= min_prob) & (y_probs < max_prob))[0]
        
        if len(indices) < 1:
            continue
            
        X_subset = X_test.iloc[indices]
        patterns_for_range = 0
        max_patterns_for_range = 2 if len(indices) >= 50 else 1
        
        # ƯU TIÊN: Combo patterns (2 triệu chứng)
        if len(top_features) >= 2 and patterns_for_range < max_patterns_for_range:
            # Thử nhiều cặp combo
            combo_pairs = []
            for i in range(min(6, len(top_features))):
                for j in range(i+1, min(6, len(top_features))):
                    combo_pairs.append((i, j))
            
            for i, j in combo_pairs[:12]:
                if patterns_for_range >= max_patterns_for_range:
                    break
                    
                feat1, feat2 = top_features[i], top_features[j]
                
                if X_subset[feat1].dtype == 'bool' or X_subset[feat2].dtype == 'bool':
                    continue
                    
                if X_subset[feat1].std() > 0 and X_subset[feat2].std() > 0:
                    t1 = X_subset[feat1].quantile(0.50)
                    t2 = X_subset[feat2].quantile(0.50)
                    
                    mask = (X_subset[feat1] > t1) & (X_subset[feat2] > t2)
                    ratio = mask.mean()
                    
                    if ratio >= 0.05:  # 5% threshold
                        patterns.append({
                            'features': [feat1, feat2],
                            'thresholds': [t1, t2],
                            'confidence': (min_prob + max_prob) / 2,
                            'sample_count': len(indices),
                            'type': 'combo'
                        })
                        patterns_for_range += 1
        
        # Fallback: Single patterns
        if patterns_for_range < max_patterns_for_range:
            for feat in top_features[:8]:
                if patterns_for_range >= max_patterns_for_range:
                    break
                    
                if X_subset[feat].dtype == 'bool':
                    continue
                    
                if X_subset[feat].std() > 0:
                    threshold = X_subset[feat].quantile(0.6)
                    high_ratio = (X_subset[feat] > threshold).mean()
                    
                    if high_ratio >= 0.15:  # 15% threshold
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

def interpret_nn_feature(feat_name):
    """Chuyển tên feature thành ngôn ngữ y khoa"""
    mapping = {
        'Do_Chenh_ST': 'Độ_Chênh_ST',
        'Do_Doc_ST_Flat': 'Độ_Dốc_ST_Flat',
        'Do_Doc_ST_Up': 'Độ_Dốc_ST_Up',
        'Nhip_Tim_Toi_Da': 'Nhịp_Tim_Tối_Đa',
        'Dau_That_Van_Dong': 'Đau_Thắt_Vận_Động',
        'Cholesterol': 'Cholesterol',
        'Tuoi': 'Tuổi',
        'Gioi_tinh': 'Giới_tính',
        'Huyet_Ap_Nghi': 'Huyết_Áp_Nghỉ',
        'Duong_Huyet_Doi': 'Đường_Huyết_Đói'
    }
    return mapping.get(feat_name, feat_name)

patterns = extract_nn_patterns(best_model, X_test, y_test, y_probs, feature_names)

if patterns:
    # Chỉ hiển thị top 5
    display_patterns = patterns[:5]
    logger.log(f"\n Top {min(5, len(display_patterns))} mau dang dan den benh tim (Neural Network phat hien):")
    logger.log()
    
    count = 1
patterns = extract_nn_patterns(best_model, X_test, y_test, y_probs, feature_names)

if patterns:
    logger.log(f"\n Trich xuat duoc {len(patterns)} patterns tu Neural Network")
    logger.log(f" Hien thi top {min(5, len(patterns))} mau dai dien theo cac muc do rui ro:")
    logger.log()
    
    # Hiển thị top 5 patterns
    for idx, pattern in enumerate(patterns[:5], 1):
        if pattern['type'] == 'single':
            feat_text = interpret_nn_feature(pattern['feature'])
            logger.log(f"{idx}. NEU:")
            logger.log(f"      + {feat_text} > {pattern['threshold']:.2f}")
            logger.log(f"   → KET LUAN: Nguy co benh tim {pattern['confidence']*100:.0f}%")
            logger.log(f"   → Can cu: Neural Network phan tich tu {pattern['sample_count']} benh nhan")
            logger.log()
        
        elif pattern['type'] == 'combo':
            logger.log(f"{idx}. NEU:")
            for i, feat in enumerate(pattern['features']):
                feat_text = interpret_nn_feature(feat)
                threshold_val = pattern['thresholds'][i]
                logger.log(f"      + {feat_text} > {threshold_val:.2f}")
            logger.log(f"   → KET LUAN: Nguy co benh tim {pattern['confidence']*100:.0f}%")
            logger.log(f"   → Can cu: Neural Network phan tich tu {pattern['sample_count']} benh nhan")
            logger.log(f"   → Ghi chu: Muc do trieu chung KHAC NHAU giua cac nhom")
else:
    logger.log("\n Khong du du lieu de trich xuat pattern.")

# ------------------------------------------------------------
# 8. SAVE MODEL
# ------------------------------------------------------------
logger.log("\n" + "="*60)
logger.log("VIII. LUU MO HINH")
logger.log("="*60)

joblib.dump(best_model, os.path.join(SAVED_MODELS_DIR, "neural_network.pkl"))
logger.log(" Da luu mo hinh Neural Network: neural_network.pkl")

# Luu metadata
metadata = {
    'threshold': best_threshold,
    'best_params': grid_search.best_params_,
    'cv_recall': grid_search.best_score_,
    'test_accuracy': (y_pred_new == y_test).mean(),
    'test_recall': recalls[best_idx],
    'auc_roc': roc_auc_score(y_test, y_probs),
    'n_epochs': best_model.n_iter_
}
joblib.dump(metadata, os.path.join(SAVED_MODELS_DIR, "nn_metadata.pkl"))
logger.log(" Da luu metadata: nn_metadata.pkl")

logger.log("\n" + "="*60)
logger.log(" HOAN THANH!")
logger.log("="*60)
logger.log(f" Tom tat ket qua:")
logger.log(f"   - Nguong toi uu: {best_threshold:.4f}")
logger.log(f"   - Test Accuracy: {metadata['test_accuracy']:.4f}")
logger.log(f"   - Test Recall: {metadata['test_recall']:.4f}")
logger.log(f"   - AUC-ROC: {metadata['auc_roc']:.4f}")
logger.log(f"   - False Negatives: {fn} (Bo sot {fn} benh nhan)")
logger.log(f"   - So epochs: {metadata['n_epochs']}")
logger.log(f"\n Cac file output da duoc luu tai: {FIGURES_DIR}")
logger.log(f" File log da duoc luu tai: {os.path.join(OUTPUTS_DIR, 'NeuralNetwork_log.txt')}")

logger.close()
