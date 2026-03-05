"""
Evaluation metrics for complaint classification.

Based on ErrorAnalysis.ipynb and evaluation code from Specialist_Model.ipynb.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    matthews_corrcoef,
    confusion_matrix
)
from typing import List, Dict, Optional

from ..data.label_schema import ID2LABEL, LABEL_LIST


def compute_classification_metrics(
    y_true: List[int],
    y_pred: List[int],
    label_names: Optional[List[str]] = None
) -> Dict:
    """
    Compute comprehensive classification metrics.
    
    Based on metrics from Technical Appendix.
    
    Args:
        y_true: True label IDs
        y_pred: Predicted label IDs
        label_names: List of label names (uses LABEL_LIST if None)
        
    Returns:
        Dictionary with all metrics
    """
    if label_names is None:
        label_names = LABEL_LIST
    
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted"),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "per_class_f1": f1_score(y_true, y_pred, average=None).tolist()
    }
    
    return metrics


def generate_confusion_matrix(
    y_true: List[int],
    y_pred: List[int],
    label_names: Optional[List[str]] = None,
    normalize: Optional[str] = None
) -> pd.DataFrame:
    """
    Generate confusion matrix as DataFrame.
    
    Based on error_analysis function from ErrorAnalysis.ipynb.
    
    Args:
        y_true: True label IDs
        y_pred: Predicted label IDs
        label_names: List of label names
        normalize: {'true', 'pred', 'all', None}
        
    Returns:
        DataFrame with confusion matrix
    """
    if label_names is None:
        label_names = LABEL_LIST
    
    cm = confusion_matrix(y_true, y_pred, normalize=normalize)
    cm_df = pd.DataFrame(cm, index=label_names, columns=label_names)
    
    return cm_df


def get_classification_report(
    y_true: List[int],
    y_pred: List[int],
    label_names: Optional[List[str]] = None,
    output_dict: bool = False
) -> str:
    """
    Generate detailed classification report.
    
    Based on ErrorAnalysis.ipynb.
    
    Args:
        y_true: True label IDs
        y_pred: Predicted label IDs
        label_names: List of label names
        output_dict: Return as dictionary instead of string
        
    Returns:
        Classification report (string or dict)
    """
    if label_names is None:
        label_names = LABEL_LIST
    
    return classification_report(
        y_true,
        y_pred,
        target_names=label_names,
        digits=4,
        output_dict=output_dict
    )


def compare_models(
    y_true: List[int],
    predictions_dict: Dict[str, List[int]],
    label_names: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Compare multiple models on same test set.
    
    Args:
        y_true: True labels
        predictions_dict: Dictionary mapping model names to predictions
        label_names: List of label names
        
    Returns:
        DataFrame comparing models across metrics
    """
    results = []
    
    for model_name, y_pred in predictions_dict.items():
        metrics = compute_classification_metrics(y_true, y_pred, label_names)
        
        results.append({
            "Model": model_name,
            "Accuracy": metrics["accuracy"],
            "Macro F1": metrics["macro_f1"],
            "Weighted F1": metrics["weighted_f1"],
            "MCC": metrics["mcc"]
        })
    
    df = pd.DataFrame(results)
    df = df.sort_values("Macro F1", ascending=False).reset_index(drop=True)
    
    return df


def get_per_class_metrics(
    y_true: List[int],
    y_pred: List[int],
    label_names: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Get detailed metrics per class.
    
    Args:
        y_true: True label IDs
        y_pred: Predicted label IDs
        label_names: List of label names
        
    Returns:
        DataFrame with per-class metrics
    """
    if label_names is None:
        label_names = LABEL_LIST
    
    report_dict = classification_report(
        y_true,
        y_pred,
        target_names=label_names,
        digits=4,
        output_dict=True
    )
    
    # Extract per-class metrics
    metrics_list = []
    for label in label_names:
        if label in report_dict:
            metrics_list.append({
                "Label": label,
                "Precision": report_dict[label]["precision"],
                "Recall": report_dict[label]["recall"],
                "F1-Score": report_dict[label]["f1-score"],
                "Support": int(report_dict[label]["support"])
            })
    
    df = pd.DataFrame(metrics_list)
    return df


def calculate_krippendorffs_alpha(
    annotations: pd.DataFrame,
    value_column: str = "label",
    annotator_column: str = "annotator",
    item_column: str = "item_id"
) -> float:
    """
    Calculate Krippendorff's Alpha for inter-rater reliability.
    
    Based on Krippendorfs_Alpha_Computation.ipynb.
    
    Args:
        annotations: DataFrame with annotations
        value_column: Column with label values
        annotator_column: Column identifying annotators
        item_column: Column identifying items
        
    Returns:
        Krippendorff's Alpha value
    """
    try:
        import krippendorff
    except ImportError:
        raise ImportError("krippendorff package required. Install with: pip install krippendorff")
    
    # Pivot to reliability matrix format
    reliability_data = annotations.pivot(
        index=item_column,
        columns=annotator_column,
        values=value_column
    ).values.T
    
    alpha = krippendorff.alpha(reliability_data, level_of_measurement="nominal")
    
    return alpha


def get_misclassified_examples(
    df: pd.DataFrame,
    y_true: List[int],
    y_pred: List[int],
    label_names: Optional[List[str]] = None,
    n: int = 10,
    text_column: str = "sentence"
) -> pd.DataFrame:
    """
    Get sample of misclassified examples.
    
    Based on get_misclassified_examples from ErrorAnalysis.ipynb.
    
    Args:
        df: Original DataFrame with text
        y_true: True label IDs
        y_pred: Predicted label IDs
        label_names: List of label names
        n: Number of examples to return
        text_column: Column with complaint text
        
    Returns:
        DataFrame with misclassified examples
    """
    if label_names is None:
        label_names = LABEL_LIST
    
    df = df.copy()
    df["true_label"] = [label_names[i] for i in y_true]
    df["pred_label"] = [label_names[i] for i in y_pred]
    df["correct"] = df["true_label"] == df["pred_label"]
    
    errors = df[~df["correct"]]
    
    print(f"Total errors: {len(errors)} ({len(errors)/len(df)*100:.1f}%)")
    
    if len(errors) == 0:
        return pd.DataFrame()
    
    # Sample random errors
    sample_size = min(n, len(errors))
    return errors[[text_column, "true_label", "pred_label"]].sample(sample_size).reset_index(drop=True)


def get_error_distribution(
    y_true: List[int],
    y_pred: List[int],
    label_names: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Analyze distribution of errors by true label.
    
    Args:
        y_true: True label IDs
        y_pred: Predicted label IDs
        label_names: List of label names
        
    Returns:
        DataFrame with error counts per class
    """
    if label_names is None:
        label_names = LABEL_LIST
    
    df = pd.DataFrame({
        "true_id": y_true,
        "pred_id": y_pred
    })
    
    df["true_label"] = df["true_id"].map(lambda x: label_names[x])
    df["pred_label"] = df["pred_id"].map(lambda x: label_names[x])
    df["error"] = df["true_label"] != df["pred_label"]
    
    error_dist = df.groupby("true_label").agg({
        "error": ["count", "sum"]
    })
    
    error_dist.columns = ["Total", "Errors"]
    error_dist["Error_Rate_%"] = (error_dist["Errors"] / error_dist["Total"] * 100).round(2)
    error_dist = error_dist.sort_values("Error_Rate_%", ascending=False)
    
    return error_dist
