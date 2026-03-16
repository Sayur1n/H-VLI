import json
import collections
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report

# ==========================================
# 定义类型到难度的映射配置
# ==========================================
DIFFICULTY_MAP = {
    # Easy Group
    "000": "Easy", 
    "011": "Easy", 
    "101": "Easy", 
    "111": "Easy",
    # Normal Group
    "100": "Normal", 
    "010": "Normal",
    # Hard Group
    "001": "Hard", 
    "110": "Hard"
}

def calculate_metrics(results, ground_truth_data, mode='multiclass'):
    """
    计算评估指标，包括：
    1. Sklearn 标准指标 (F1, Precision, Recall)
    2. 按难度 (Easy/Normal/Hard) 的分组统计
    3. 按类型 (Type code) 的分组统计
    
    :param mode: 'multiclass' or 'binary'
    """
    # 初始化统计容器
    stats = {
        "global": {
            "total": 0, "strict_correct": 0, "binary_correct": 0, "refused_count": 0
        },
        "by_type": collections.defaultdict(lambda: {
            "total": 0, "strict_correct": 0, "binary_correct": 0, "refused_count": 0
        }),
        "by_difficulty": collections.defaultdict(lambda: {
            "total": 0, "strict_correct": 0, "binary_correct": 0, "refused_count": 0
        })
    }

    y_true = []
    y_pred = []

    for sid, item in ground_truth_data.items():
        # 如果该样本不在预测结果中，跳过
        if sid not in results:
            continue
            
        pred_info = results[sid]
        
        # 获取元数据
        item_type = str(item.get("type", "unknown"))
        difficulty = DIFFICULTY_MAP.get(item_type, "Unknown")
        
        # 1. 检查 Refusal
        is_refused = False
        if isinstance(pred_info, dict) and pred_info.get("refusal") is True:
            is_refused = True
        
        # 记录 Refusal (分母排除逻辑)
        if is_refused:
            stats["global"]["refused_count"] += 1
            stats["by_type"][item_type]["refused_count"] += 1
            stats["by_difficulty"][difficulty]["refused_count"] += 1
            if difficulty in ["Normal", "Hard"]:
                stats["by_difficulty"]["Normal + Hard"]["refused_count"] += 1
            continue

        # 2. 获取并处理 GT Label
        original_gt = item.get("final_label")
        if mode == 'binary':
            # 二分类模式：0->0, 1~5->1
            gt_label = 0 if original_gt == 0 else 1
        else:
            gt_label = original_gt

        # 3. 获取并处理 Pred Label
        raw_pred = pred_info.get("label")
        
        # 容错处理：确保转为 int
        try:
            # 如果是 -1 (不确定)，保持 -1，后续算错
            pred_label = int(raw_pred) if raw_pred is not None else -1
        except:
            pred_label = -1

        # 二分类模式下的 Pred 处理
        if mode == 'binary':
            if pred_label > 0: 
                pred_label = 1 # 任何大于0的预测都视为 Hate(1)
            elif pred_label == 0:
                pred_label = 0
            # 如果是 -1，保持 -1
        
        # 收集给 sklearn 计算用 (排除 -1 以避免混淆，或者保留看需求。通常 -1 视为错误)
        # 这里为了 F1 计算，我们将 -1 视为一个错误的类别(例如 -1)，
        # 但在 binary 模式下 sklearn 会报错如果 label 不在 [0,1]。
        # 策略：如果 pred 是 -1，在二分类下把它变成 1-gt_label (强制算错) 或者保持 -1 并设置 sklearn labels=[0,1]
        y_true.append(gt_label)
        y_pred.append(pred_label if pred_label != -1 else (1 if gt_label == 0 else 0)) 

        # 4. 判断正确性 (手动统计部分)
        # Strict Correct: 严格匹配 (Binary模式下即 0vs0, 1vs1)
        is_strict_correct = (pred_label == gt_label)
        
        # Binary Correct: 仅判断是否有害 (Binary模式下等同于 Strict)
        if mode == 'binary':
            is_binary_correct = is_strict_correct
        else:
            # Multiclass 下：GT>0 AND Pred>0 算对，或者 GT=0 AND Pred=0 算对
            is_gt_hate = gt_label > 0
            is_pred_hate = pred_label > 0
            if pred_label < 0:
                is_binary_correct = False
            else:
                is_binary_correct = (is_gt_hate == is_pred_hate)

        # 5. 更新计数
        # Global
        stats["global"]["total"] += 1
        if is_strict_correct: stats["global"]["strict_correct"] += 1
        if is_binary_correct: stats["global"]["binary_correct"] += 1

        # By Type
        stats["by_type"][item_type]["total"] += 1
        if is_strict_correct: stats["by_type"][item_type]["strict_correct"] += 1
        if is_binary_correct: stats["by_type"][item_type]["binary_correct"] += 1
        
        # By Difficulty
        stats["by_difficulty"][difficulty]["total"] += 1
        if is_strict_correct: stats["by_difficulty"][difficulty]["strict_correct"] += 1
        if is_binary_correct: stats["by_difficulty"][difficulty]["binary_correct"] += 1
        
        # By Difficulty Group (Normal + Hard)
        if difficulty in ["Normal", "Hard"]:
            stats["by_difficulty"]["Normal + Hard"]["total"] += 1
            if is_strict_correct: stats["by_difficulty"]["Normal + Hard"]["strict_correct"] += 1
            if is_binary_correct: stats["by_difficulty"]["Normal + Hard"]["binary_correct"] += 1

    # ==========================================
    # 计算 Sklearn 指标 (Global)
    # ==========================================
    sklearn_metrics = {}
    if len(y_true) > 0:
        if mode == 'binary':
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_true, y_pred, average='binary', pos_label=1, zero_division=0
            )
            sklearn_metrics = {
                "f1": f1, "precision": precision, "recall": recall,
                "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()
            }
        else:
            p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
            p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)
            sklearn_metrics = {
                "f1_macro": f1_macro, "f1_weighted": f1_weighted,
                "precision_macro": p_macro, "recall_macro": r_macro
            }
    
    # 合并结果
    final_metrics = stats
    final_metrics["sklearn"] = sklearn_metrics
    final_metrics["mode"] = mode # 记录当前模式以便 Report 使用
    
    return final_metrics

def get_metrics_report(model_name, strategy_name, metrics, class_mode='multiclass'):
    """
    生成包含 Sklearn 指标 + 难度 Breakdown 的综合报告
    根据 class_mode 输出不同的 Data Source 列表格式
    """
    lines = []
    lines.append(f"\n📊 Report: [{strategy_name}] - Model: {model_name} (Mode: {class_mode})")
    
    g = metrics["global"]
    sk = metrics.get("sklearn", {})
    
    refused_count = g['refused_count']
    valid_total = g['total']
    total_all = valid_total + refused_count
    
    lines.append(f"   🛡️ Safety Refused: {refused_count} samples (Excluded from stats)")

    strict_acc = 0.0
    
    if valid_total > 0:
        # 1. Global Summary
        strict_acc = (g["strict_correct"] / valid_total) * 100
        binary_acc = (g["binary_correct"] / valid_total) * 100
        
        lines.append("-" * 40)
        lines.append(f"   [Global Performance] (N={valid_total})")
        lines.append(f"   >> Strict Accuracy: {strict_acc:.2f}%")
        
        if class_mode == 'binary':
            # Binary 特有指标
            lines.append(f"   >> F1 Score (Binary): {sk.get('f1', 0):.4f}")
            lines.append(f"   >> Precision: {sk.get('precision', 0):.4f}")
            lines.append(f"   >> Recall:    {sk.get('recall', 0):.4f}")
            lines.append(f"   >> Confusion Matrix: {sk.get('confusion_matrix', [])}")
        else:
            # Multiclass 特有指标
            lines.append(f"   >> Binary Accuracy: {binary_acc:.2f}% (Hate vs Non-Hate)")
            lines.append(f"   >> F1 Macro:    {sk.get('f1_macro', 0):.4f}")
            lines.append(f"   >> F1 Weighted: {sk.get('f1_weighted', 0):.4f}")
        
        lines.append("-" * 40)

        # 2. Breakdown by Difficulty
        lines.append("\n   [Breakdown by Difficulty]")
        display_order = ["Easy", "Normal", "Hard", "Normal + Hard", "Unknown"]
        
        for diff in display_order:
            if diff in metrics["by_difficulty"]:
                m = metrics["by_difficulty"][diff]
                if m["total"] > 0 or m["refused_count"] > 0:
                    s_acc = 0.0
                    b_acc = 0.0
                    if m["total"] > 0:
                        s_acc = (m["strict_correct"] / m["total"]) * 100
                        b_acc = (m["binary_correct"] / m["total"]) * 100
                    
                    ref_info = f", Refused={m['refused_count']}" if m['refused_count'] > 0 else ""
                    
                    if class_mode == 'binary':
                        # Binary 模式只显示 Acc (因为 Strict == Binary)
                        lines.append(f"      - {diff:<13}: Acc={s_acc:6.2f}% (N={m['total']}{ref_info})")
                    else:
                        # Multiclass 模式显示 Strict 和 Binary 两种 Acc
                        lines.append(f"      - {diff:<13}: Strict={s_acc:6.2f}%, Binary={b_acc:6.2f}% (N={m['total']}{ref_info})")

        # 3. Breakdown by Type (Detailed)
        lines.append("\n   [Breakdown by Type]")
        for t, m in sorted(metrics["by_type"].items()):
            if m["total"] > 0:
                s_acc = (m["strict_correct"] / m["total"]) * 100
                lines.append(f"      - Type {t}: Acc={s_acc:5.1f}% (N={m['total']})")
                    
    else:
        lines.append("   >> No valid samples to calculate metrics.")

    # ==========================================
    # 4. 生成 Data Source 格式 (区分模式)
    # ==========================================
    lines.append("\n" + "="*20 + " [COPY DATA SOURCE] " + "="*20)
    
    # 辅助函数：获取 Strict Correct 准确率
    def _get_acc(diff_key):
        d = metrics["by_difficulty"].get(diff_key, {"total": 0})
        if d["total"] == 0: return 0.0
        return (d.get("strict_correct", 0) / d["total"]) * 100

    # Refusal Rate (相对于所有样本 Total + Refused)
    ref_rate = (refused_count / total_all * 100) if total_all > 0 else 0.0
    
    # 基础 Acc 数据
    acc_easy = _get_acc("Easy")
    acc_norm = _get_acc("Normal")
    acc_hard = _get_acc("Hard")
    acc_all  = strict_acc

    vals = []

    if class_mode == 'multiclass':
        # Multiclass 格式: [Easy, Normal, Hard, All, MacF1, WgtF1, Ref#, Ref%]
        f1_mac = sk.get("f1_macro", 0) * 100
        f1_wgt = sk.get("f1_weighted", 0) * 100
        vals = [acc_easy, acc_norm, acc_hard, acc_all, f1_mac, f1_wgt, refused_count, ref_rate]

    elif class_mode == 'binary':
        # Binary 格式: [Easy, Normal, Hard, All, Recall, F1, Ref#, Ref%]
        # 注意：这里用的是 Binary F1 和 Recall
        recall_bin = sk.get("recall", 0) * 100
        f1_bin     = sk.get("f1", 0) * 100
        vals = [acc_easy, acc_norm, acc_hard, acc_all, recall_bin, f1_bin, refused_count, ref_rate]

    if vals:
        # 格式化数值 (保留2位小数)
        val_str = ", ".join([f"{x:.2f}" for x in vals])
        
        # 生成 Python List 字符串
        strat_repr = f"'{strategy_name}'" if isinstance(strategy_name, str) else str(strategy_name)
        
        lines.append(f"[\"{model_name}\", {strat_repr}, \n [{val_str}]],")
        lines.append("="*60)

    return "\n".join(lines)