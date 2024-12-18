import numpy as np
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score

def calculate_auc(y_true, y_pred_proba):
    """
    Calculate Area Under the ROC Curve (AUC)
    
    :param y_true: True labels (0 or 1)
    :param y_pred_proba: Predicted probabilities for the positive class
    :return: AUC score
    """
    return roc_auc_score(y_true, y_pred_proba)

def calculate_precision(y_true, y_pred):
    """
    Calculate Precision
    
    :param y_true: True labels (0 or 1)
    :param y_pred: Predicted labels (0 or 1)
    :return: Precision score
    """
    return precision_score(y_true, y_pred)

def calculate_recall(y_true, y_pred):
    """
    Calculate Recall
    
    :param y_true: True labels (0 or 1)
    :param y_pred: Predicted labels (0 or 1)
    :return: Recall score
    """
    return recall_score(y_true, y_pred)

def calculate_f1(y_true, y_pred):
    """
    Calculate F1 Score
    
    :param y_true: True labels (0 or 1)
    :param y_pred: Predicted labels (0 or 1)
    :return: F1 score
    """
    return f1_score(y_true, y_pred)

# Example usage
if __name__ == "__main__":
    # Sample data
    y_true = np.array([0, 1, 1, 0, 1, 1, 0, 0, 1, 0])
    y_pred = np.array([0, 1, 1, 0, 0, 1, 1, 0, 1, 0])
    y_pred_proba = np.array([0.1, 0.9, 0.8, 0.3, 0.4, 0.8, 0.6, 0.2, 0.9, 0.3])

    # Calculate metrics
    auc = calculate_auc(y_true, y_pred_proba)
    precision = calculate_precision(y_true, y_pred)
    recall = calculate_recall(y_true, y_pred)
    f1 = calculate_f1(y_true, y_pred)

    # Print results
    print(f"AUC: {auc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
