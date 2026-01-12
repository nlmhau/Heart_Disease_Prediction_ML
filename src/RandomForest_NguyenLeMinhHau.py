# ============================================================
# RANDOM_FOREST.PY
# Nguyễn Lê Minh Hậu
# Mục tiêu:
#   - Huấn luyện Random Forest với Hyperparameter Tuning
#   - Tối ưu ngưỡng dự đoán (Threshold Tuning)
#   - Đánh giá mô hình theo hướng y tế (ưu tiên Recall)
#   - Giải thích mô hình bằng Permutation Importance
#   - Trích xuất & minh họa luật từ cây quyết định đại diện
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

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    precision_recall_curve
)
from sklearn.inspection import permutation_importance
from sklearn.tree import export_text, plot_tree

warnings.filterwarnings("ignore")


# ============================================================
# I. CẤU HÌNH & LOAD DỮ LIỆU
# ============================================================

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

logger = Logger(os.path.join(OUTPUTS_DIR, "RandomForest_log.txt"))

logger.log("=" * 60)
logger.log("I. LOAD DỮ LIỆU & CHUẨN BỊ")
logger.log("=" * 60)

try:
    X_train = joblib.load(os.path.join(SAVED_MODELS_DIR, "X_train.pkl"))
    X_test  = joblib.load(os.path.join(SAVED_MODELS_DIR, "X_test.pkl"))
    y_train = joblib.load(os.path.join(SAVED_MODELS_DIR, "y_train.pkl"))
    y_test  = joblib.load(os.path.join(SAVED_MODELS_DIR, "y_test.pkl"))
    feature_names = X_train.columns.tolist()
    logger.log(f" Đã load dữ liệu: Train {X_train.shape}, Test {X_test.shape}")
except FileNotFoundError:
    logger.log(" Lỗi: Chưa có dữ liệu tiền xử lý. Hãy chạy preprocessing.py trước.")
    logger.close()
    exit()

# ============================================================
# II. HUẤN LUYỆN & HYPERPARAMETER TUNING
# ============================================================

logger.log("\n" + "=" * 60)
logger.log("II. HUẤN LUYỆN & HYPERPARAMETER TUNING (GRID SEARCH)")
logger.log("=" * 60)

param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [10, 15, 20, None],
    "min_samples_leaf": [1, 2, 4]
}

rf = RandomForestClassifier(
    random_state=2026,
    n_jobs=-1
)

grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    scoring="recall",     # Ưu tiên phát hiện người bệnh
    cv=5,
    n_jobs=1,
    verbose=1
)

logger.log(" Đang tìm kiếm hyperparameters tốt nhất...")
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_

logger.log(f"\n Hoàn thành Grid Search!")
logger.log(f"   Best Params: {grid_search.best_params_}")
logger.log(f"   Best Recall (CV): {grid_search.best_score_:.4f}")

# ============================================================
# III. THRESHOLD TUNING (OPTIMIZE DECISION BOUNDARY)
# ============================================================

logger.log("\n" + "=" * 60)
logger.log("III. TỐI ƯU NGƯỠNG DỰ ĐOÁN (THRESHOLD TUNING)")
logger.log("=" * 60)

y_probs = best_model.predict_proba(X_test)[:, 1]

precisions, recalls, thresholds = precision_recall_curve(y_test, y_probs)
f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)

best_idx = np.argmax(f1_scores)
best_threshold = thresholds[best_idx]

logger.log(f" Ngưỡng tối ưu: {best_threshold:.4f}")
logger.log(f"   → Precision={precisions[best_idx]:.4f} | Recall={recalls[best_idx]:.4f}")

y_pred = (y_probs >= best_threshold).astype(int)

logger.log("\n" + "=" * 60)
logger.log("KẾT QUẢ TRÊN TẬP TEST")
logger.log("=" * 60)
logger.log(classification_report(y_test, y_pred, target_names=["Khỏe", "Bệnh"], digits=4))
logger.log(f"\n AUC-ROC Score: {roc_auc_score(y_test, y_probs):.4f}")

# Confusion Matrix chi tiết
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()
logger.log(f"\n Confusion Matrix:")
logger.log(f"   - True Negatives (TN): {tn} (Dự đoán đúng người khỏe)")
logger.log(f"   - False Positives (FP): {fp} (Dự đoán nhầm người khỏe thành bệnh)")
logger.log(f"   - False Negatives (FN): {fn}  (BỎ SÓT người bệnh - quan trọng!)")
logger.log(f"   - True Positives (TP): {tp} (Dự đoán đúng người bệnh)")

# ============================================================
# IV. BIỂU ĐỒ ĐÁNH GIÁ
# ============================================================

logger.log("\n" + "=" * 60)
logger.log("IV. TẠO BIỂU ĐỒ ĐÁNH GIÁ")
logger.log("=" * 60)

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_probs)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc_score(y_test, y_probs):.4f}", linewidth=2)
plt.plot([0, 1], [0, 1], "k--", alpha=0.3)
plt.xlabel("False Positive Rate (Tỷ lệ dương tính giả)", fontsize=11)
plt.ylabel("True Positive Rate (Tỷ lệ dương tính thật)", fontsize=11)
plt.title("ROC Curve – Random Forest", fontsize=13, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(FIGURES_DIR, "RF_ROC.png"), dpi=150)
plt.close()
logger.log("   Đã lưu: RF_ROC.png")

# Confusion Matrix
plt.figure(figsize=(7, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=True,
            xticklabels=["Khỏe", "Bệnh"],
            yticklabels=["Khỏe", "Bệnh"],
            annot_kws={"fontsize": 14})
plt.title(f"Confusion Matrix (Ngưỡng = {best_threshold:.3f})", fontsize=13, fontweight='bold')
plt.ylabel("Thực tế", fontsize=11)
plt.xlabel("Dự đoán", fontsize=11)
plt.savefig(os.path.join(FIGURES_DIR, "RF_Confusion_Matrix.png"), dpi=150)
plt.close()
logger.log("   Đã lưu: RF_Confusion_Matrix.png")

# ============================================================
# V. GIẢI THÍCH MÔ HÌNH (PERMUTATION IMPORTANCE)
# ============================================================

logger.log("\n" + "=" * 60)
logger.log("V. PHÂN TÍCH ĐỘ QUAN TRỌNG ĐẶC TRƯNG (PERMUTATION)")
logger.log("=" * 60)

logger.log("🔬 Đang tính toán Permutation Importance...")
perm = permutation_importance(
    best_model, X_test, y_test,
    n_repeats=10,
    random_state=2026,
    n_jobs=-1
)

sorted_idx = perm.importances_mean.argsort()[::-1]

df_imp = pd.DataFrame({
    "Đặc trưng": np.array(feature_names)[sorted_idx],
    "Độ quan trọng": perm.importances_mean[sorted_idx]
})

logger.log("\n Top 10 đặc trưng ảnh hưởng nhất:")
for idx, row in df_imp.head(10).iterrows():
    logger.log(f"   {idx+1}. {row['Đặc trưng']:<25} → {row['Độ quan trọng']:.4f}")

plt.figure(figsize=(10, 8))
plt.barh(df_imp["Đặc trưng"][:10], df_imp["Độ quan trọng"][:10], color='#3498db')
plt.gca().invert_yaxis()
plt.title("Top 10 Đặc Trưng Quan Trọng Nhất (Permutation Importance)", 
          fontsize=13, fontweight='bold')
plt.xlabel("Mức độ giảm độ chính xác khi xáo trộn", fontsize=11)
plt.ylabel("Đặc trưng", fontsize=11)
plt.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "RF_Permutation_Importance.png"), dpi=150)
plt.close()
logger.log("    Đã lưu: RF_Permutation_Importance.png")

# ============================================================
# VI. TRÍCH XUẤT & MINH HỌA LUẬT
# ============================================================

logger.log("\n" + "=" * 60)
logger.log("VI. TRÍCH XUẤT LUẬT TỪ CÂY QUYẾT ĐỊNH ĐẠI DIỆN")
logger.log("=" * 60)

one_tree = best_model.estimators_[0]

tree_rules = export_text(
    one_tree,
    feature_names=feature_names,
    max_depth=5  # Chỉ hiển thị 5 tầng đầu cho dễ đọc
)
logger.log("\n Luật quyết định (5 tầng đầu):")
logger.log(tree_rules)

logger.log("\n Đang tạo biểu đồ cây quyết định...")
plt.figure(figsize=(24, 14))
plot_tree(
    one_tree,
    feature_names=feature_names,
    class_names=["Khỏe", "Bệnh"],
    filled=True,
    rounded=True,
    max_depth=3,  # Giới hạn depth để dễ nhìn
    fontsize=10
)
plt.title("Cây Quyết Định Đại Diện (Random Forest - Độ sâu 3 tầng)", 
          fontsize=14, fontweight='bold')
plt.savefig(os.path.join(FIGURES_DIR, "RF_Decision_Tree.png"),
            dpi=150, bbox_inches="tight")
plt.close()
logger.log("    Đã lưu: RF_Decision_Tree.png")

# ============================================================
# VII. PHÁT HIỆN MẪU ẨN (HIDDEN PATTERNS)
# ============================================================

logger.log("\n" + "=" * 60)
logger.log("VII. PHÁT HIỆN MẪU ẨN (HIDDEN PATTERNS)")
logger.log("=" * 60)
logger.log(" Trích xuất các quy luật: Triệu chứng A + B + C → Bệnh Tim")

def extract_rules_from_tree(tree, feature_names, max_rules=10):
    """Trích xuất quy luật IF-THEN dễ hiểu từ cây quyết định"""
    tree_ = tree.tree_
    feature_name = [feature_names[i] if i != -2 else "undefined" 
                   for i in tree_.feature]
    
    rules = []
    
    def recurse(node, conditions, depth=0):
        if depth > 4:  # Giới hạn độ sâu
            return
        
        if tree_.feature[node] != -2:  # Không phải lá
            feature = feature_name[node]
            threshold = tree_.threshold[node]
            
            # Nhánh trái (<=)
            left_conditions = conditions + [(feature, "<=", threshold)]
            recurse(tree_.children_left[node], left_conditions, depth + 1)
            
            # Nhánh phải (>)
            right_conditions = conditions + [(feature, ">", threshold)]
            recurse(tree_.children_right[node], right_conditions, depth + 1)
        else:
            # Lá: tính xác suất
            samples = tree_.n_node_samples[node]
            value = tree_.value[node][0]
            disease_prob = value[1] / (value[0] + value[1]) if (value[0] + value[1]) > 0 else 0
            
            # Lấy quy luật dự đoán bệnh với nhiều mức độ rủi ro khác nhau
            if disease_prob >= 0.30 and samples >= 3:
                rules.append({
                    'conditions': conditions,
                    'disease_prob': disease_prob,
                    'samples': samples,
                    'depth': len(conditions)
                })
    
    recurse(0, [])
    
    # Sắp xếp theo: độ tin cậy → số mẫu → độ sâu
    rules.sort(key=lambda x: (-x['disease_prob'], -x['samples'], x['depth']))
    return rules[:max_rules]

def interpret_condition(feature, operator, threshold):
    """Chuyển điều kiện kỹ thuật thành ngôn ngữ y khoa"""
    # Ánh xạ tên đặc trưng sang giải thích
    interpretations = {
        "Tuoi": "Tuoi",
        "Gioi_tinh": "Nam gioi" if operator == ">" else "Nu gioi",
        "Cholesterol": "Cholesterol",
        "Huyet_Ap_Nghi": "Huyet ap nghi",
        "Duong_Huyet_Doi": "Duong huyet doi cao" if operator == ">" else "Duong huyet doi thap",
        "Nhip_Tim_Toi_Da": "Nhip tim toi da",
        "Dau_That_Van_Dong": "Co dau that vung nguc khi van dong" if operator == ">" else "Khong dau that khi van dong",
        "Do_Chenh_ST": "Do chenh ST",
        "Do_Doc_ST_Up": "Do doc ST len" if operator == ">" else "Do doc ST khong len",
        "Do_Doc_ST_Flat": "Do doc ST phang" if operator == ">" else "Do doc ST khong phang",
        "Cholesterol_Tuoi": "Ty le Cholesterol/Tuoi",
    }
    
    # Xử lý đặc biệt cho một số biến
    if feature == "Gioi_tinh":
        return "Nam gioi" if operator == ">" else "Nu gioi"
    elif feature == "Duong_Huyet_Doi":
        return "Duong huyet doi cao" if operator == ">" else "Duong huyet doi binh thuong"
    elif feature == "Dau_That_Van_Dong":
        return "Dau that vung nguc khi van dong" if operator == ">" else "Khong dau that vung nguc"
    elif feature in ["Do_Doc_ST_Up", "Do_Doc_ST_Flat"]:
        return interpretations.get(feature, feature)
    else:
        # Các biến liên tục: hiển thị ngưỡng
        op_str = ">" if operator == ">" else "<="
        return f"{feature} {op_str} {threshold:.2f}"

# Trích xuất từ NHIỀU cây trong Random Forest (không chỉ 1 cây)
all_rules = []
num_trees_to_check = min(50, len(best_model.estimators_))  # Kiểm tra 50 cây

logger.log(f"\n Dang trich xuat quy luat tu {num_trees_to_check} cay trong Random Forest...")

for tree_idx in range(num_trees_to_check):
    tree = best_model.estimators_[tree_idx]
    tree_rules = extract_rules_from_tree(tree, feature_names, max_rules=10)
    all_rules.extend(tree_rules)

# Loại bỏ quy luật trùng lặp và sắp xếp lại
unique_rules = []
seen_conditions = set()

for rule in all_rules:
    # Tạo signature từ điều kiện
    cond_str = str(sorted(rule['conditions']))
    if cond_str not in seen_conditions:
        seen_conditions.add(cond_str)
        unique_rules.append(rule)

# Sắp xếp theo độ tin cậy và số mẫu
unique_rules.sort(key=lambda x: (-x['disease_prob'], -x['samples'], x['depth']))

logger.log(f" Trich xuat duoc {len(unique_rules)} quy luat benh tim tu {num_trees_to_check} cay")

# Hiển thị 5 mẫu: Top 3 cao nhất + 2 thấp nhất (để thấy sự đa dạng)
display_rules = []

# Lấy top 3 mẫu rủi ro CAO NHẤT
top_high = unique_rules[:3]
for rule in top_high:
    display_rules.append((rule, f"Rat cao ({rule['disease_prob']*100:.0f}%)"))

# Lấy 2 mẫu rủi ro THẤP NHẤT (nhưng vẫn >30%)
bottom_2 = sorted(unique_rules, key=lambda x: x['disease_prob'])[:2]
for rule in bottom_2:
    display_rules.append((rule, f"Trung binh ({rule['disease_prob']*100:.0f}%)"))

logger.log(f" Hien thi {len(display_rules)} mau dai dien (3 cao nhat + 2 thap nhat):")
logger.log()

for i, (rule, risk_label) in enumerate(display_rules, 1):
    conditions_text = []
    for feat, op, thresh in rule['conditions']:
        cond_str = interpret_condition(feat, op, thresh)
        conditions_text.append(cond_str)
    
    # Hiển thị tất cả điều kiện quan trọng
    logger.log(f"{i}. NEU:")
    for cond in conditions_text:
        logger.log(f"      + {cond}")
    logger.log(f"   → KET LUAN: Nguy co benh tim {risk_label}")
    logger.log(f"   → Can cu: {rule['samples']} benh nhan trong tap huan luyen")
    logger.log()
    logger.log()

# ============================================================
# VIII. LƯU MÔ HÌNH
# ============================================================

logger.log("\n" + "=" * 60)
logger.log("VIII. LUU MO HINH")
logger.log("=" * 60)

joblib.dump(best_model, os.path.join(SAVED_MODELS_DIR, "random_forest.pkl"))
logger.log(" Đã lưu mô hình Random Forest: random_forest.pkl")

# Lưu metadata
metadata = {
    'threshold': best_threshold,
    'best_params': grid_search.best_params_,
    'cv_recall': grid_search.best_score_,
    'test_accuracy': (y_pred == y_test).mean(),
    'test_recall': recalls[best_idx],
    'auc_roc': roc_auc_score(y_test, y_probs)
}
joblib.dump(metadata, os.path.join(SAVED_MODELS_DIR, "rf_metadata.pkl"))
logger.log(" Đã lưu metadata: rf_metadata.pkl")

logger.log("\n" + "=" * 60)
logger.log(" HOÀN THÀNH!")
logger.log("=" * 60)
logger.log(f" Tóm tắt kết quả:")
logger.log(f"   - Ngưỡng tối ưu: {best_threshold:.4f}")
logger.log(f"   - Test Accuracy: {metadata['test_accuracy']:.4f}")
logger.log(f"   - Test Recall: {metadata['test_recall']:.4f}")
logger.log(f"   - AUC-ROC: {metadata['auc_roc']:.4f}")
logger.log(f"   - False Negatives: {fn} (Bỏ sót {fn} bệnh nhân)")
logger.log(f"\n Các file output đã được lưu tại: {FIGURES_DIR}")
logger.log(f" File log đã được lưu tại: {os.path.join(OUTPUTS_DIR, 'RandomForest_log.txt')}")

logger.close()
