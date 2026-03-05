"""
Inference module for RoBERTa specialist model.

Based on prediction code from Specialist_Model.ipynb.
"""

import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, DataCollatorWithPadding
from datasets import Dataset
from typing import List, Dict, Tuple, Optional

from ..data.label_schema import ID2LABEL, LABEL_LIST


def load_model(model_path: str) -> Tuple[AutoModelForSequenceClassification, AutoTokenizer]:
    """
    Load trained RoBERTa model and tokenizer.
    
    Args:
        model_path: Path to saved model directory
        
    Returns:
        Tuple of (model, tokenizer)
    """
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    
    return model, tokenizer


def preprocess_for_inference(
    examples: Dict,
    tokenizer,
    max_length: int = 256
) -> Dict:
    """
    Tokenize examples for inference.
    
    Args:
        examples: Dictionary with 'sentence' key
        tokenizer: HuggingFace tokenizer
        max_length: Maximum sequence length
        
    Returns:
        Tokenized examples
    """
    return tokenizer(
        examples["sentence"],
        truncation=True,
        padding="max_length",
        max_length=max_length
    )


def predict_batch(
    model,
    tokenizer,
    texts: List[str],
    max_length: int = 256,
    return_probabilities: bool = False
) -> Tuple[List[int], List[str], Optional[np.ndarray]]:
    """
    Predict labels for a batch of texts.
    
    Based on inference code from Specialist_Model.ipynb.
    
    Args:
        model: Trained model
        tokenizer: Tokenizer
        texts: List of complaint texts
        max_length: Maximum sequence length
        return_probabilities: Whether to return class probabilities
        
    Returns:
        Tuple of (predicted_ids, predicted_labels, probabilities)
    """
    # Create dataset
    df = pd.DataFrame({"sentence": texts})
    dataset = Dataset.from_pandas(df)
    
    # Tokenize
    tokenized = dataset.map(
        lambda examples: preprocess_for_inference(examples, tokenizer, max_length),
        batched=True,
        remove_columns=dataset.column_names
    )
    
    # Create data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # Create trainer for prediction
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=data_collator
    )
    
    # Predict
    predictions = trainer.predict(tokenized)
    
    # Get predicted classes
    logits = predictions.predictions
    predicted_ids = np.argmax(logits, axis=-1).tolist()
    predicted_labels = [ID2LABEL[id] for id in predicted_ids]
    
    # Get probabilities if requested
    probabilities = None
    if return_probabilities:
        probabilities = torch.softmax(torch.tensor(logits), dim=-1).numpy()
    
    return predicted_ids, predicted_labels, probabilities


def predict_single(
    model,
    tokenizer,
    text: str,
    max_length: int = 256,
    return_confidence: bool = True
) -> Dict:
    """
    Predict label for a single complaint text.
    
    Args:
        model: Trained model
        tokenizer: Tokenizer
        text: Complaint text
        max_length: Maximum sequence length
        return_confidence: Whether to return confidence score
        
    Returns:
        Dictionary with prediction results
    """
    ids, labels, probs = predict_batch(
        model,
        tokenizer,
        [text],
        max_length,
        return_probabilities=return_confidence
    )
    
    result = {
        "predicted_id": ids[0],
        "predicted_label": labels[0]
    }
    
    if return_confidence and probs is not None:
        result["confidence"] = float(probs[0][ids[0]])
        result["all_probabilities"] = {
            label: float(prob)
            for label, prob in zip(LABEL_LIST, probs[0])
        }
    
    return result


def predict_dataframe(
    model,
    tokenizer,
    df: pd.DataFrame,
    text_column: str = "sentence",
    batch_size: int = 32,
    max_length: int = 256,
    return_confidence: bool = True
) -> pd.DataFrame:
    """
    Predict labels for all texts in a DataFrame.
    
    Args:
        model: Trained model
        tokenizer: Tokenizer
        df: Input DataFrame
        text_column: Column containing complaint text
        batch_size: Batch size for processing
        max_length: Maximum sequence length
        return_confidence: Whether to add confidence column
        
    Returns:
        DataFrame with added prediction columns
    """
    df = df.copy()
    texts = df[text_column].tolist()
    
    # Process in batches
    all_ids = []
    all_labels = []
    all_confidences = []
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        
        ids, labels, probs = predict_batch(
            model,
            tokenizer,
            batch_texts,
            max_length,
            return_probabilities=return_confidence
        )
        
        all_ids.extend(ids)
        all_labels.extend(labels)
        
        if return_confidence and probs is not None:
            confidences = [float(probs[j][id]) for j, id in enumerate(ids)]
            all_confidences.extend(confidences)
    
    # Add to DataFrame
    df["predicted_id"] = all_ids
    df["predicted_label"] = all_labels
    
    if return_confidence:
        df["confidence"] = all_confidences
    
    return df


def get_high_confidence_predictions(
    df: pd.DataFrame,
    confidence_threshold: float = 0.70,
    confidence_column: str = "confidence"
) -> pd.DataFrame:
    """
    Filter predictions by confidence threshold.
    
    Based on deployment strategy from Technical Appendix (95% specialist, 5% human review).
    
    Args:
        df: DataFrame with predictions and confidence scores
        confidence_threshold: Minimum confidence (default 0.70)
        confidence_column: Column name with confidence scores
        
    Returns:
        DataFrame with high-confidence predictions only
    """
    return df[df[confidence_column] >= confidence_threshold].copy()


def get_low_confidence_predictions(
    df: pd.DataFrame,
    confidence_threshold: float = 0.70,
    confidence_column: str = "confidence"
) -> pd.DataFrame:
    """
    Get predictions below confidence threshold (for human review).
    
    Args:
        df: DataFrame with predictions and confidence scores
        confidence_threshold: Minimum confidence (default 0.70)
        confidence_column: Column name with confidence scores
        
    Returns:
        DataFrame with low-confidence predictions requiring review
    """
    return df[df[confidence_column] < confidence_threshold].copy()
