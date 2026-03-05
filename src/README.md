# Source Code (`src/`) Documentation

Production-ready Python modules extracted from project notebooks.

---

## Module Structure

```
src/
├── data/                    # Data preprocessing and schema
│   ├── preprocessing.py     # Data loading, cleaning, splitting
│   └── label_schema.py      # 7-category taxonomy and mappings
├── models/                  # Model training and inference
│   ├── specialist_trainer.py  # RoBERTa training functions
│   └── inference.py         # Prediction functions
├── evaluation/              # Metrics and error analysis
│   └── metrics.py           # F1, MCC, confusion matrix, error analysis
└── utils/                   # Configuration and constants
    └── config.py            # Hyperparameters, paths, constants
```

---

## Quick Start

### 1. Data Preprocessing

```python
from src.data import load_cfpb_data, clean_complaint_text, remove_duplicates

# Load data
df = load_cfpb_data("data/complaints.csv")

# Clean
df = clean_complaint_text(df)
df = remove_duplicates(df)

# Get summary statistics
from src.data import get_data_summary
summary = get_data_summary(df, name="My Dataset")
print(summary)
```

### 2. Label Schema

```python
from src.data import LABEL_LIST, LABEL2ID, ID2LABEL

# Show all labels
print(LABEL_LIST)
# ['Hidden Fees', 'Fraud / Security Issue', ...]

# Convert label names to IDs
from src.data import convert_labels_to_ids
ids = convert_labels_to_ids(['Hidden Fees', 'Fraud / Security Issue'])
print(ids)  # [0, 1]

# Convert IDs back to names
from src.data import convert_ids_to_labels
labels = convert_ids_to_labels([0, 1])
print(labels)  # ['Hidden Fees', 'Fraud / Security Issue']
```

### 3. Train Specialist Model

```python
from src.models import prepare_datasets, train_specialist_model, save_best_model
from src.data import LABEL2ID

# Prepare data
train_df['label'] = train_df['predicted_label'].map(LABEL2ID)
val_df['label'] = val_df['predicted_label'].map(LABEL2ID)

train_ds, val_ds = prepare_datasets(train_df, val_df)

# Train model (uses best config: lr=2e-5, batch=16)
trainer, model = train_specialist_model(
    train_ds=train_ds,
    val_ds=val_ds,
    output_dir="./models/specialist"
)

# Save best model
save_best_model(trainer, "./models/specialist_best")
```

### 4. Run Multiple Configurations

```python
from src.models import run_multiple_configs

# Try all 4 configurations from Technical Appendix
results = run_multiple_configs(
    train_ds=train_ds,
    val_ds=val_ds,
    base_output_dir="./experiments"
)

# Best model automatically saved to ./experiments/specialist_roberta_best
```

### 5. Make Predictions

```python
from src.models import load_model, predict_dataframe

# Load trained model
model, tokenizer = load_model("./models/specialist_best")

# Predict on new data
results_df = predict_dataframe(
    model=model,
    tokenizer=tokenizer,
    df=test_df,
    text_column="sentence",
    return_confidence=True
)

print(results_df[['sentence', 'predicted_label', 'confidence']].head())
```

### 6. Single Prediction

```python
from src.models import predict_single

result = predict_single(
    model=model,
    tokenizer=tokenizer,
    text="They charged me a fee I never agreed to!",
    return_confidence=True
)

print(result)
# {
#     'predicted_id': 0,
#     'predicted_label': 'Hidden Fees',
#     'confidence': 0.94,
#     'all_probabilities': {...}
# }
```

### 7. Confidence-Based Routing

```python
from src.models import get_high_confidence_predictions, get_low_confidence_predictions

# 95% auto-route (confidence > 0.70)
high_conf = get_high_confidence_predictions(results_df, confidence_threshold=0.70)
print(f"Auto-route: {len(high_conf)} complaints")

# 5% human review (confidence < 0.70)
low_conf = get_low_confidence_predictions(results_df, confidence_threshold=0.70)
print(f"Human review: {len(low_conf)} complaints")
```

### 8. Evaluate Model

```python
from src.evaluation import compute_classification_metrics, generate_confusion_matrix

# Compute metrics
metrics = compute_classification_metrics(
    y_true=test_df['true_label_id'].tolist(),
    y_pred=test_df['predicted_id'].tolist()
)

print(f"Accuracy: {metrics['accuracy']:.4f}")
print(f"Macro F1: {metrics['macro_f1']:.4f}")
print(f"MCC: {metrics['mcc']:.4f}")

# Confusion matrix
cm_df = generate_confusion_matrix(
    y_true=test_df['true_label_id'].tolist(),
    y_pred=test_df['predicted_id'].tolist()
)
print(cm_df)
```

### 9. Error Analysis

```python
from src.evaluation import get_misclassified_examples, get_error_distribution

# Get sample of errors
errors = get_misclassified_examples(
    df=test_df,
    y_true=test_df['true_label_id'].tolist(),
    y_pred=test_df['predicted_id'].tolist(),
    n=10
)
print(errors)

# Error distribution by class
error_dist = get_error_distribution(
    y_true=test_df['true_label_id'].tolist(),
    y_pred=test_df['predicted_id'].tolist()
)
print(error_dist)
```

### 10. Compare Multiple Models

```python
from src.evaluation import compare_models

predictions_dict = {
    "RoBERTa Specialist": specialist_predictions,
    "GPT-4o": gpt4o_predictions,
    "Claude Sonnet": claude_predictions
}

comparison = compare_models(
    y_true=test_df['true_label_id'].tolist(),
    predictions_dict=predictions_dict
)

print(comparison)
#        Model  Accuracy  Macro F1  Weighted F1      MCC
# 0  RoBERTa    0.9275    0.8492       0.9156   0.8634
# 1  GPT-4o     0.9060    0.7900       0.8890   0.8365
# 2  Claude     0.7955    0.6319       0.7420   0.6396
```

---

## Configuration

All hyperparameters and constants are in `src/utils/config.py`:

```python
from src.utils import BEST_LEARNING_RATE, BEST_BATCH_SIZE, CONFIDENCE_THRESHOLD

print(f"Best LR: {BEST_LEARNING_RATE}")  # 2e-5
print(f"Best Batch: {BEST_BATCH_SIZE}")   # 16
print(f"Confidence Threshold: {CONFIDENCE_THRESHOLD}")  # 0.70
```

---

## Import Patterns

```python
# Data modules
from src.data import (
    load_cfpb_data,
    clean_complaint_text,
    LABEL_LIST,
    LABEL2ID
)

# Model modules
from src.models import (
    train_specialist_model,
    load_model,
    predict_dataframe
)

# Evaluation modules
from src.evaluation import (
    compute_classification_metrics,
    generate_confusion_matrix
)

# Config
from src.utils import BEST_LEARNING_RATE, CONFIDENCE_THRESHOLD
```

---

## Notes

- All functions have type hints and docstrings
- Based on actual code from project notebooks
- Follows production best practices
- Imports work from repository root: `from src.data import ...`

---

## Testing

Run a quick test:

```python
# Test data loading
from src.data import LABEL_LIST, LABEL2ID
print(f"✅ {len(LABEL_LIST)} labels loaded")

# Test label conversion
from src.data import get_label_id, get_label_name
assert get_label_id("Hidden Fees") == 0
assert get_label_name(0) == "Hidden Fees"
print("✅ Label mapping works")

# Test config
from src.utils import BEST_LEARNING_RATE
assert BEST_LEARNING_RATE == 2e-5
print("✅ Config loaded")

print("\n🎉 All imports working!")
```

---

For notebook examples, see `/notebooks` folder.
