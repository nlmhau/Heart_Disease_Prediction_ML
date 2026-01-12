# ============================================================
# EVALUATION.PY
# Đánh giá & so sánh các mô hình dự đoán Bệnh Tim
# ============================================================

import os
import joblib
import pandas as pd
import numpy as np

# Đặt backend matplotlib trước khi import pyplot (tránh lỗi tkinter)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve
)

# ------------------------------------------------------------
# 1. CẤU HÌNH ĐƯỜNG DẪN
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

# Khởi tạo logger
logger = Logger(os.path.join(OUTPUTS_DIR, "Evaluation_log.txt"))

# ------------------------------------------------------------
# 2. LOAD DỮ LIỆU TEST
# ------------------------------------------------------------
def load_test_data():
    logger.log("\n" + "="*60)
    logger.log("I. LOAD DU LIEU TEST")
    logger.log("="*60)
    
    X_test = joblib.load(os.path.join(SAVED_MODELS_DIR, "X_test.pkl"))
    y_test = joblib.load(os.path.join(SAVED_MODELS_DIR, "y_test.pkl"))
    
    logger.log(f" Da load du lieu test: {X_test.shape}")
    logger.log(f" - Phan bo lop: Khoe={sum(y_test==0)}, Benh={sum(y_test==1)}")
    
    return X_test, y_test

# ------------------------------------------------------------
# 3. LOAD CÁC MÔ HÌNH & METADATA
# ------------------------------------------------------------
def load_models_and_metadata():
    logger.log("\n" + "="*60)
    logger.log("II. LOAD CAC MO HINH & METADATA")
    logger.log("="*60)
    
    models_info = {}
    
    # Random Forest
    try:
        rf_model = joblib.load(os.path.join(SAVED_MODELS_DIR, "random_forest.pkl"))
        rf_metadata = joblib.load(os.path.join(SAVED_MODELS_DIR, "rf_metadata.pkl"))
        models_info["Random Forest"] = {
            "model": rf_model,
            "metadata": rf_metadata,
            "color": "#3498db"
        }
        logger.log(" [OK] Da load Random Forest")
    except FileNotFoundError:
        logger.log(" [LOI] Khong tim thay Random Forest")
    
    # XGBoost
    try:
        xgb_model = joblib.load(os.path.join(SAVED_MODELS_DIR, "xgboost.pkl"))
        xgb_metadata = joblib.load(os.path.join(SAVED_MODELS_DIR, "xgboost_metadata.pkl"))
        models_info["XGBoost"] = {
            "model": xgb_model,
            "metadata": xgb_metadata,
            "color": "#e74c3c"
        }
        logger.log(" [OK] Da load XGBoost")
    except FileNotFoundError:
        logger.log(" [LOI] Khong tim thay XGBoost")
    
    # Neural Network
    try:
        nn_model = joblib.load(os.path.join(SAVED_MODELS_DIR, "neural_network.pkl"))
        nn_metadata = joblib.load(os.path.join(SAVED_MODELS_DIR, "nn_metadata.pkl"))
        models_info["Neural Network"] = {
            "model": nn_model,
            "metadata": nn_metadata,
            "color": "#2ecc71"
        }
        logger.log(" [OK] Da load Neural Network")
    except FileNotFoundError:
        logger.log(" [LOI] Khong tim thay Neural Network")
    
    return models_info

# ------------------------------------------------------------
# 4. ĐÁNH GIÁ TỪNG MÔ HÌNH CHI TIẾT
# ------------------------------------------------------------
def evaluate_single_model(name, model_info, X_test, y_test):
    """Đánh giá chi tiết một mô hình"""
    
    logger.log("\n" + "="*60)
    logger.log(f"III. DANH GIA MO HINH: {name}")
    logger.log("="*60)
    
    model = model_info["model"]
    metadata = model_info["metadata"]
    
    # Dự đoán
    y_pred = model.predict(X_test)
    y_probs = model.predict_proba(X_test)[:, 1]
    
    # Tính các metric
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc_roc = roc_auc_score(y_test, y_probs)
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    # In kết quả
    logger.log("\n CHI TIEU HIEU SUAT:")
    logger.log(f"   - Accuracy (Do chinh xac):     {accuracy:.4f} ({accuracy*100:.2f}%)")
    logger.log(f"   - Precision (Do chinh xac DT): {precision:.4f} ({precision*100:.2f}%)")
    logger.log(f"   - Recall (Ty le phat hien):    {recall:.4f} ({recall*100:.2f}%)")
    logger.log(f"   - F1-Score (Diem tong hop):    {f1:.4f}")
    logger.log(f"   - AUC-ROC:                      {auc_roc:.4f}")
    
    logger.log("\n CONFUSION MATRIX (MA TRAN NHAM LAN):")
    logger.log(f"   - True Negatives (TN):  {tn:3d} (Du doan dung nguoi khoe)")
    logger.log(f"   - False Positives (FP): {fp:3d} (Du doan nham nguoi khoe thanh benh)")
    logger.log(f"   - False Negatives (FN): {fn:3d} (BO SOT nguoi benh - quan trong!)")
    logger.log(f"   - True Positives (TP):  {tp:3d} (Du doan dung nguoi benh)")
    
    # Tỷ lệ phần trăm
    total = tn + fp + fn + tp
    logger.log(f"\n TY LE PHAN TRAM:")
    logger.log(f"   - TN: {tn/total*100:5.1f}% | FP: {fp/total*100:5.1f}%")
    logger.log(f"   - FN: {fn/total*100:5.1f}% | TP: {tp/total*100:5.1f}%")
    
    # Metadata từ training
    if metadata:
        logger.log("\n THONG TIN HUAN LUYEN:")
        if 'threshold' in metadata:
            logger.log(f"   - Nguong toi uu: {metadata['threshold']:.4f}")
        if 'best_params' in metadata:
            logger.log(f"   - Tham so tot nhat: {metadata['best_params']}")
        if 'cv_recall' in metadata:
            logger.log(f"   - CV Recall: {metadata['cv_recall']:.4f}")
    
    # Trả về kết quả
    return {
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1_Score": f1,
        "AUC_ROC": auc_roc,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
        "y_probs": y_probs,
        "color": model_info["color"]
    }

# ------------------------------------------------------------
# 5. SO SÁNH TẤT CẢ MÔ HÌNH
# ------------------------------------------------------------
def compare_all_models(results):
    logger.log("\n" + "="*60)
    logger.log("IV. BANG SO SANH TAT CA MO HINH")
    logger.log("="*60)
    
    # Tạo DataFrame để so sánh
    df_comparison = pd.DataFrame([
        {
            "Mo hinh": r["Model"],
            "Accuracy": f"{r['Accuracy']:.4f}",
            "Precision": f"{r['Precision']:.4f}",
            "Recall": f"{r['Recall']:.4f}",
            "F1-Score": f"{r['F1_Score']:.4f}",
            "AUC-ROC": f"{r['AUC_ROC']:.4f}",
            "FN (Bo sot)": r["FN"]
        }
        for r in results
    ])
    
    logger.log("\n" + df_comparison.to_string(index=False))
    
    # Xếp hạng theo Recall (quan trọng nhất cho y tế)
    sorted_results = sorted(results, key=lambda x: (x["Recall"], x["F1_Score"]), reverse=True)
    
    logger.log("\n" + "="*60)
    logger.log("V. XEP HANG MO HINH")
    logger.log("="*60)
    logger.log("\n Tieu chi xep hang: Recall (uu tien) > F1-Score > Accuracy")
    logger.log()
    
    for rank, result in enumerate(sorted_results, 1):
        medal = ["Hang 1 (TOT NHAT)", "Hang 2", "Hang 3"][rank-1] if rank <= 3 else f"Hang {rank}"
        logger.log(f" {medal}: {result['Model']}")
        logger.log(f"   -> Recall: {result['Recall']:.4f} | F1: {result['F1_Score']:.4f} | AUC: {result['AUC_ROC']:.4f}")
        logger.log(f"   -> Bo sot (FN): {result['FN']} benh nhan")
        logger.log()
    
    return sorted_results[0]

# ------------------------------------------------------------
# 6. VẼ BIỂU ĐỒ SO SÁNH
# ------------------------------------------------------------
def plot_comparison_charts(results, y_test):
    logger.log("\n" + "="*60)
    logger.log("VI. TAO BIEU DO SO SANH")
    logger.log("="*60)
    
    # 1. Biểu đồ so sánh metrics
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    metrics = ["Accuracy", "Precision", "Recall", "F1_Score"]
    metric_names = ["Accuracy", "Precision", "Recall", "F1-Score"]
    
    for idx, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
        ax = axes[idx // 2, idx % 2]
        
        models = [r["Model"] for r in results]
        values = [r[metric] for r in results]
        colors = [r["color"] for r in results]
        
        bars = ax.bar(models, values, color=colors, alpha=0.7, edgecolor='black')
        ax.set_ylabel(metric_name, fontsize=11)
        ax.set_ylim([0.5, 1.0])
        ax.set_title(f'So sanh {metric_name}', fontsize=12, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        # Thêm giá trị trên cột
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "Comparison_Metrics.png"), dpi=150)
    plt.close()
    logger.log(" [OK] Da luu: Comparison_Metrics.png")
    
    # 2. Biểu đồ ROC Curves cho tất cả models
    plt.figure(figsize=(10, 8))
    
    for result in results:
        fpr, tpr, _ = roc_curve(y_test, result["y_probs"])
        auc = result["AUC_ROC"]
        plt.plot(fpr, tpr, label=f'{result["Model"]} (AUC = {auc:.4f})',
                linewidth=2, color=result["color"])
    
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Random Classifier')
    plt.xlabel('False Positive Rate', fontsize=11)
    plt.ylabel('True Positive Rate', fontsize=11)
    plt.title('So sanh ROC Curves - Tat ca mo hinh', fontsize=13, fontweight='bold')
    plt.legend(fontsize=10, loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "Comparison_ROC.png"), dpi=150)
    plt.close()
    logger.log(" [OK] Da luu: Comparison_ROC.png")
    
    # 3. Biểu đồ Confusion Matrix comparison
    fig, axes = plt.subplots(1, len(results), figsize=(5*len(results), 4))
    if len(results) == 1:
        axes = [axes]
    
    for ax, result in zip(axes, results):
        cm = np.array([[result["TN"], result["FP"]], 
                       [result["FN"], result["TP"]]])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                   xticklabels=['Khoe', 'Benh'], yticklabels=['Khoe', 'Benh'],
                   ax=ax, annot_kws={"fontsize": 12})
        ax.set_title(f'{result["Model"]}\nFN={result["FN"]}', 
                    fontsize=11, fontweight='bold')
        ax.set_ylabel('Thuc te', fontsize=10)
        ax.set_xlabel('Du doan', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "Comparison_ConfusionMatrix.png"), dpi=150)
    plt.close()
    logger.log(" [OK] Da luu: Comparison_ConfusionMatrix.png")

# ------------------------------------------------------------
# 6.5. SO SÁNH HIDDEN PATTERNS
# ------------------------------------------------------------
def compare_hidden_patterns(models_info, X_test, y_test):
    """So sánh khả năng phát hiện patterns của các models"""
    logger.log("\n" + "="*60)
    logger.log("VI.5. SO SANH PHAT HIEN HIDDEN PATTERNS")
    logger.log("="*60)
    
    logger.log("\n PHAN TICH KHA NANG PHAT HIEN MAU AN:")
    logger.log()
    
    pattern_summary = []
    
    for name, info in models_info.items():
        model = info["model"]
        
        # Dự đoán xác suất
        y_probs = model.predict_proba(X_test)[:, 1]
        
        # Phân nhóm theo xác suất
        ranges = [
            (0.85, 1.0, "Rat cao (85-100%)"),
            (0.70, 0.85, "Cao (70-85%)"),
            (0.55, 0.70, "Trung binh (55-70%)"),
            (0.40, 0.55, "Vua phai (40-55%)"),
            (0.25, 0.40, "Thap (25-40%)"),
        ]
        
        logger.log(f" {name}:")
        total_patterns = 0
        
        for min_prob, max_prob, label in ranges:
            indices = np.where((y_test == 1) & (y_probs >= min_prob) & (y_probs < max_prob))[0]
            if len(indices) > 0:
                logger.log(f"   - {label}: {len(indices):2d} benh nhan")
                total_patterns += 1
        
        pattern_summary.append({
            "Model": name,
            "Nhom_xuat_hien": total_patterns,
            "Benh_nhan_phat_hien": np.sum((y_test == 1) & (y_probs >= 0.25))
        })
        logger.log()
    
    # Tạo bảng tổng hợp
    logger.log(" BANG TONG HOP KHA NANG PHAT HIEN:")
    df_patterns = pd.DataFrame(pattern_summary)
    logger.log("\n" + df_patterns.to_string(index=False))
    
    logger.log("\n GIAI THICH:")
    logger.log("   - Nhom_xuat_hien: So nhom xac suat co benh nhan")
    logger.log("   - Benh_nhan_phat_hien: Tong benh nhan phat hien (xac suat >= 25%)")
    logger.log()
    logger.log(" => Model nao co nhieu nhom hon = phat hien da dang hon")
    logger.log(" => Model nao phat hien nhieu benh nhan hon = nhay hon")

# ------------------------------------------------------------
# 7. KẾT LUẬN CUỐI CÙNG
# ------------------------------------------------------------
def final_conclusion(best_model):
    logger.log("\n" + "="*60)
    logger.log("VIII. KET LUAN & DE XUAT")
    logger.log("="*60)
    
    logger.log(f"\n MO HINH TOT NHAT: {best_model['Model']}")
    logger.log(f"\n LY DO LUA CHON:")
    logger.log(f"   1. Recall cao nhat: {best_model['Recall']:.4f} ({best_model['Recall']*100:.1f}%)")
    logger.log(f"      => Phat hien duoc nhieu benh nhan nhat")
    logger.log(f"   2. So benh nhan bo sot (FN): {best_model['FN']} nguoi")
    logger.log(f"      => It bo sot nhat trong cac mo hinh")
    logger.log(f"   3. F1-Score: {best_model['F1_Score']:.4f}")
    logger.log(f"      => Can bang tot giua Precision va Recall")
    logger.log(f"   4. AUC-ROC: {best_model['AUC_ROC']:.4f}")
    logger.log(f"      => Kha nang phan loai tong quat tot")
    
    logger.log("\n Y NGHIA THUC TE (Y KHOA):")
    logger.log(f"   - Trong {best_model['TP'] + best_model['FN']} benh nhan thuc su,")
    logger.log(f"   - Mo hinh phat hien duoc: {best_model['TP']} nguoi ({best_model['Recall']*100:.1f}%)")
    logger.log(f"   - Bo sot: {best_model['FN']} nguoi ({best_model['FN']/(best_model['TP']+best_model['FN'])*100:.1f}%)")
    logger.log(f"   => Uu tien Recall de tranh bo sot benh nhan!")
    
    logger.log("\n CAC FILE OUTPUT DA LUU:")
    logger.log(f"   - Bieu do so sanh metrics: {FIGURES_DIR}/Comparison_Metrics.png")
    logger.log(f"   - Bieu do ROC: {FIGURES_DIR}/Comparison_ROC.png")
    logger.log(f"   - Confusion matrices: {FIGURES_DIR}/Comparison_ConfusionMatrix.png")
    
    logger.log("\n DE XUAT SU DUNG:")
    logger.log("   1. Cho ket qua CHINH XAC cao nhat:")
    logger.log(f"      => Dung mo hinh co Precision cao nhat")
    logger.log("   2. Cho ket qua AN TOAN nhat (tranh bo sot):")
    logger.log(f"      => Dung mo hinh co Recall cao nhat (mo hinh tot nhat)")
    logger.log("   3. Cho ket qua CAN BANG:")
    logger.log(f"      => Dung mo hinh co F1-Score cao nhat")
    logger.log()
    logger.log("   KHUYẾN CÁO: Trong y khoa, ưu tiên Recall để tránh bỏ sót bệnh nhân!")

# ------------------------------------------------------------
# 8. LƯU BÁO CÁO
# ------------------------------------------------------------
def save_evaluation_report(results, best_model):
    """Lưu báo cáo đánh giá ra file text"""
    report_path = os.path.join(OUTPUTS_DIR, "Evaluation_Report.txt")
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write("    BAO CAO DANH GIA & SO SANH CAC MO HINH\n")
        f.write("="*60 + "\n\n")
        
        # Bảng so sánh
        f.write("BANG SO SANH CHI TIEU:\n")
        f.write("-"*60 + "\n")
        for r in results:
            f.write(f"\n{r['Model']}:\n")
            f.write(f"  - Accuracy:  {r['Accuracy']:.4f}\n")
            f.write(f"  - Precision: {r['Precision']:.4f}\n")
            f.write(f"  - Recall:    {r['Recall']:.4f}\n")
            f.write(f"  - F1-Score:  {r['F1_Score']:.4f}\n")
            f.write(f"  - AUC-ROC:   {r['AUC_ROC']:.4f}\n")
            f.write(f"  - Bo sot:    {r['FN']} benh nhan\n")
        
        # Model tốt nhất
        f.write("\n" + "="*60 + "\n")
        f.write(f"MO HINH TOT NHAT: {best_model['Model']}\n")
        f.write("="*60 + "\n")
        f.write(f"  - Recall: {best_model['Recall']:.4f}\n")
        f.write(f"  - F1-Score: {best_model['F1_Score']:.4f}\n")
        f.write(f"  - AUC-ROC: {best_model['AUC_ROC']:.4f}\n")
        f.write(f"  - Bo sot: {best_model['FN']} benh nhan\n")
    
    logger.log(f"\n [OK] Da luu bao cao: {report_path}")

# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------
def main():
    # Bắt đầu
    logger.log("\n" + "="*60)
    logger.log("    DANH GIA & SO SANH CAC MO HINH DU DOAN BENH TIM")
    logger.log("="*60)
    
    # Load data
    X_test, y_test = load_test_data()
    
    # Load models
    models_info = load_models_and_metadata()
    
    if not models_info:
        logger.log("\n [LOI] Khong co mo hinh nao de danh gia!")
        logger.log(" Hay chay cac file huan luyen truoc:")
        logger.log("   - RandomForest_NguyenLeMinhHau.py")
        logger.log("   - XGBoost_NguyenLeMinhHau.py")
        logger.log("   - NeuralNetwork_NguyenDucHuy.py")
        return
    
    # Đánh giá từng model
    results = []
    for name, model_info in models_info.items():
        result = evaluate_single_model(name, model_info, X_test, y_test)
        results.append(result)
    
    # So sánh tất cả
    best_model = compare_all_models(results)
    
    # Vẽ biểu đồ
    plot_comparison_charts(results, y_test)
    
    # So sánh Hidden Patterns
    compare_hidden_patterns(models_info, X_test, y_test)
    
    # Kết luận
    final_conclusion(best_model)
    
    # Lưu báo cáo
    save_evaluation_report(results, best_model)
    
    logger.log("\n" + "="*60)
    logger.log("   HOAN THANH DANH GIA!")
    logger.log("="*60)
    logger.log(f"\n File log da duoc luu tai: {os.path.join(OUTPUTS_DIR, 'Evaluation_log.txt')}")
    
    # Đóng logger
    logger.close()

if __name__ == "__main__":
    main()
