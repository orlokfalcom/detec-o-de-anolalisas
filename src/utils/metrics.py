import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

def calculate_fraud_metrics(y_true, y_pred, y_prob=None):
    """
    Calculates standard machine learning classification metrics.
    """
    report = classification_report(y_true, y_pred, output_dict=True)
    matrix = confusion_matrix(y_true, y_pred)
    
    metrics = {
        "precision": report["1"]["precision"],
        "recall": report["1"]["recall"],
        "f1_score": report["1"]["f1-score"],
        "accuracy": report["accuracy"],
        "confusion_matrix": {
            "tn": int(matrix[0, 0]),
            "fp": int(matrix[0, 1]),
            "fn": int(matrix[1, 0]),
            "tp": int(matrix[1, 1])
        }
    }
    
    if y_prob is not None:
        try:
            metrics["auc_roc"] = roc_auc_score(y_true, y_prob)
        except Exception:
            metrics["auc_roc"] = None
            
    return metrics

def calculate_financial_impact(y_true, y_pred, amounts):
    """
    Computes financial metrics:
    - Total amount saved (TP)
    - Total fraud occurred (FN)
    - False positives cost (FP * unit friction cost)
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    amounts = np.array(amounts)
    
    tp_mask = (y_true == 1) & (y_pred == 1)
    fn_mask = (y_true == 1) & (y_pred == 0)
    fp_mask = (y_true == 0) & (y_pred == 1)
    
    saved_amount = float(np.sum(amounts[tp_mask]))
    lost_amount = float(np.sum(amounts[fn_mask]))
    false_alarm_amount = float(np.sum(amounts[fp_mask]))
    
    # Assume static friction cost of $5 (USD/BRL equivalent) per false positive customer review
    friction_cost_per_fp = 5.0
    total_friction_cost = float(np.sum(fp_mask) * friction_cost_per_fp)
    
    return {
        "saved_fraud_value": saved_amount,
        "lost_fraud_value": lost_amount,
        "false_alarm_value": false_alarm_amount,
        "total_friction_cost": total_friction_cost,
        "net_savings": saved_amount - total_friction_cost
    }
