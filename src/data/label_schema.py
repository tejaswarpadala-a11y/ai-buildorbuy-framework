"""
Label schema and mapping for CFPB complaint classification.

7-category root cause taxonomy with decision rules.
"""

from typing import Dict, List

# Label list (order matters - this is the id2label mapping)
LABEL_LIST = [
    "Hidden Fees",
    "Fraud / Security Issue",
    "Credit Reporting Error",
    "Loan/Mortgage Servicing Issue",
    "Account Access / Administration",
    "Process Failure / Red Tape",
    "Other"
]

# Label to ID mapping
LABEL2ID = {label: idx for idx, label in enumerate(LABEL_LIST)}

# ID to Label mapping
ID2LABEL = {idx: label for idx, label in enumerate(LABEL_LIST)}

NUM_LABELS = len(LABEL_LIST)


def get_label_id(label_name: str) -> int:
    """
    Convert label name to integer ID.
    
    Args:
        label_name: One of the 7 category names
        
    Returns:
        Integer ID (0-6)
        
    Raises:
        ValueError: If label_name not in schema
    """
    if label_name not in LABEL2ID:
        raise ValueError(f"Unknown label: {label_name}. Must be one of {LABEL_LIST}")
    return LABEL2ID[label_name]


def get_label_name(label_id: int) -> str:
    """
    Convert integer ID to label name.
    
    Args:
        label_id: Integer 0-6
        
    Returns:
        Label name string
        
    Raises:
        ValueError: If label_id out of range
    """
    if label_id not in ID2LABEL:
        raise ValueError(f"Unknown label ID: {label_id}. Must be 0-{NUM_LABELS-1}")
    return ID2LABEL[label_id]


def convert_labels_to_ids(labels: List[str]) -> List[int]:
    """
    Convert list of label names to IDs.
    
    Args:
        labels: List of label name strings
        
    Returns:
        List of integer IDs
    """
    return [get_label_id(label) for label in labels]


def convert_ids_to_labels(ids: List[int]) -> List[str]:
    """
    Convert list of IDs to label names.
    
    Args:
        ids: List of integer IDs
        
    Returns:
        List of label name strings
    """
    return [get_label_name(id) for id in ids]


# Label definitions (from codebook)
LABEL_DEFINITIONS = {
    "Hidden Fees": {
        "definition": "Undisclosed or confusing charges",
        "examples": ["maintenance fee not disclosed", "duplicate charges", "surprise service fees"],
        "priority": "High"
    },
    "Fraud / Security Issue": {
        "definition": "Unauthorized transactions or identity theft",
        "examples": ["unauthorized charges", "account hacked", "identity theft"],
        "priority": "Critical",
        "special_rule": "Fraud overrides all other labels"
    },
    "Credit Reporting Error": {
        "definition": "Incorrect information sent to credit bureaus",
        "examples": ["late payment incorrectly reported", "wrong balance on credit report"],
        "priority": "High"
    },
    "Loan/Mortgage Servicing Issue": {
        "definition": "Payment application or mortgage servicing failures",
        "examples": ["payments not applied correctly", "escrow miscalculated"],
        "priority": "Medium"
    },
    "Account Access / Administration": {
        "definition": "Locked accounts, closures/freezes, login issues (non-fraud)",
        "examples": ["account frozen", "can't log in", "bank closed my account"],
        "priority": "Medium",
        "clarification": "Account locked due to suspected fraud → label as Fraud"
    },
    "Process Failure / Red Tape": {
        "definition": "Repeated transfers, delays, unresolved loops",
        "examples": ["called multiple times with no resolution", "endless department transfers"],
        "priority": "Low"
    },
    "Other": {
        "definition": "No clear operational failure / ambiguous root cause",
        "expected_frequency": "<5%",
        "priority": "Triage"
    }
}


def get_label_distribution_summary(labels: List[str]) -> Dict:
    """
    Generate distribution summary for a set of labels.
    
    Args:
        labels: List of label names
        
    Returns:
        Dictionary with count and percentage for each label
    """
    from collections import Counter
    
    counts = Counter(labels)
    total = len(labels)
    
    summary = {}
    for label in LABEL_LIST:
        count = counts.get(label, 0)
        pct = (count / total * 100) if total > 0 else 0
        summary[label] = {
            "count": count,
            "percentage": round(pct, 2)
        }
    
    return summary
