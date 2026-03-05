"""
RoBERTa specialist model training module.

Based on Specialist_Model.ipynb with hyperparameter configurations from Technical Appendix.
"""

import pandas as pd
import numpy as np
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
import evaluate
from typing import Dict, Tuple, Optional

from ..data.label_schema import LABEL_LIST, LABEL2ID, NUM_LABELS


# Training configurations from TA-9
TRAINING_CONFIGS = [
    {"lr": 2e-5, "batch": 16, "epochs": 3, "name": "config-1-baseline"},
    {"lr": 3e-5, "batch": 16, "epochs": 3, "name": "config-2-higher-lr"},
    {"lr": 2e-5, "batch": 32, "epochs": 4, "name": "config-3-larger-batch"},
    {"lr": 5e-5, "batch": 16, "epochs": 3, "name": "config-4-highest-lr"}
]

# Best configuration (from your results)
BEST_CONFIG = TRAINING_CONFIGS[0]  # config-1: lr=2e-5, batch=16


def prepare_datasets(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    text_column: str = "sentence",
    label_column: str = "predicted_label"
) -> Tuple[Dataset, Dataset]:
    """
    Prepare HuggingFace datasets from DataFrames.
    
    Args:
        train_df: Training DataFrame
        val_df: Validation DataFrame
        text_column: Column with complaint text
        label_column: Column with label names or IDs
        
    Returns:
        Tuple of (train_dataset, val_dataset)
    """
    # Ensure labels are integers
    if train_df[label_column].dtype == 'object':
        # Labels are strings, convert to IDs
        train_df['label'] = train_df[label_column].map(LABEL2ID)
        val_df['label'] = val_df[label_column].map(LABEL2ID)
    else:
        # Labels are already integers
        train_df['label'] = train_df[label_column].astype(int)
        val_df['label'] = val_df[label_column].astype(int)
    
    # Create datasets
    train_ds = Dataset.from_pandas(train_df[[text_column, 'label']])
    val_ds = Dataset.from_pandas(val_df[[text_column, 'label']])
    
    return train_ds, val_ds


def preprocess_function(examples: Dict, tokenizer, max_length: int = 256) -> Dict:
    """
    Tokenize and preprocess examples.
    
    Based on preprocess function from Specialist_Model.ipynb.
    
    Args:
        examples: Batch of examples from dataset
        tokenizer: HuggingFace tokenizer
        max_length: Maximum sequence length
        
    Returns:
        Tokenized examples with labels
    """
    result = tokenizer(
        examples["sentence"],
        truncation=True,
        padding="max_length",
        max_length=max_length
    )
    
    # Ensure labels are integers
    result["labels"] = [int(l) for l in examples["label"]]
    
    return result


def compute_metrics(eval_pred) -> Dict:
    """
    Compute F1 metrics for evaluation.
    
    Based on compute_metrics from Specialist_Model.ipynb.
    
    Args:
        eval_pred: Tuple of (predictions, labels)
        
    Returns:
        Dictionary with macro F1 score
    """
    f1_metric = evaluate.load("f1")
    
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    return f1_metric.compute(
        predictions=predictions,
        references=labels,
        average="macro"
    )


def train_specialist_model(
    train_ds: Dataset,
    val_ds: Dataset,
    output_dir: str,
    model_name: str = "roberta-base",
    learning_rate: float = 2e-5,
    batch_size: int = 16,
    num_epochs: int = 3,
    max_length: int = 256,
    save_best_only: bool = True
) -> Tuple[Trainer, AutoModelForSequenceClassification]:
    """
    Train RoBERTa specialist model.
    
    Based on training loop from Specialist_Model.ipynb.
    
    Args:
        train_ds: Training dataset
        val_ds: Validation dataset
        output_dir: Directory to save model
        model_name: HuggingFace model name
        learning_rate: Learning rate (default: 2e-5 from best config)
        batch_size: Batch size (default: 16)
        num_epochs: Number of training epochs (default: 3)
        max_length: Maximum sequence length
        save_best_only: Only save best model checkpoint
        
    Returns:
        Tuple of (trainer, model)
    """
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Tokenize datasets
    train_tokenized = train_ds.map(
        lambda examples: preprocess_function(examples, tokenizer, max_length),
        batched=True,
        remove_columns=train_ds.column_names
    )
    
    val_tokenized = val_ds.map(
        lambda examples: preprocess_function(examples, tokenizer, max_length),
        batched=True,
        remove_columns=val_ds.column_names
    )
    
    # Initialize model
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=NUM_LABELS,
        id2label={i: label for i, label in enumerate(LABEL_LIST)},
        label2id=LABEL2ID
    )
    
    # Data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_epochs,
        weight_decay=0.01,
        load_best_model_at_end=save_best_only,
        metric_for_best_model="f1",
        greater_is_better=True,
        push_to_hub=False,
        logging_dir=f"{output_dir}/logs",
        logging_steps=100,
        save_total_limit=2 if save_best_only else None
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=val_tokenized,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )
    
    # Train
    trainer.train()
    
    return trainer, model


def save_best_model(trainer: Trainer, output_path: str):
    """
    Save the best model and tokenizer.
    
    Args:
        trainer: Trained Trainer object
        output_path: Path to save model
    """
    trainer.save_model(output_path)
    trainer.tokenizer.save_pretrained(output_path)
    print(f"✅ Model saved to {output_path}")


def run_multiple_configs(
    train_ds: Dataset,
    val_ds: Dataset,
    base_output_dir: str,
    configs: list = None
) -> Dict:
    """
    Train multiple configurations and track best F1.
    
    Args:
        train_ds: Training dataset
        val_ds: Validation dataset
        base_output_dir: Base directory for outputs
        configs: List of config dictionaries (uses TRAINING_CONFIGS if None)
        
    Returns:
        Dictionary with results for each config
    """
    if configs is None:
        configs = TRAINING_CONFIGS
    
    results = {}
    best_f1 = 0
    best_config_name = None
    
    for config in configs:
        print(f"\n{'='*60}")
        print(f"Training {config['name']}: LR={config['lr']}, Batch={config['batch']}, Epochs={config['epochs']}")
        print(f"{'='*60}")
        
        output_dir = f"{base_output_dir}/{config['name']}"
        
        trainer, model = train_specialist_model(
            train_ds=train_ds,
            val_ds=val_ds,
            output_dir=output_dir,
            learning_rate=config['lr'],
            batch_size=config['batch'],
            num_epochs=config['epochs']
        )
        
        # Get final metrics
        eval_results = trainer.evaluate()
        f1_score = eval_results.get('eval_f1', 0)
        
        results[config['name']] = {
            "config": config,
            "f1_score": f1_score,
            "eval_results": eval_results
        }
        
        # Track best
        if f1_score > best_f1:
            best_f1 = f1_score
            best_config_name = config['name']
            save_best_model(trainer, f"{base_output_dir}/specialist_roberta_best")
        
        print(f"\n{config['name']} F1: {f1_score:.4f}")
    
    print(f"\n{'='*60}")
    print(f"🏆 Best config: {best_config_name} with F1 = {best_f1:.4f}")
    print(f"{'='*60}")
    
    return results
