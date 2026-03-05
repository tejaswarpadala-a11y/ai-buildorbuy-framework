# Evaluation Methodology

*Rigorous benchmarking framework for comparing GenAI vs Specialist models*

---

## Objective

Evaluate 6+ GenAI models as alternative classifiers on a 1,000-complaint human-labeled holdout dataset and compare performance against:
1. Human consensus baseline
2. Fine-tuned specialist model (RoBERTa)

---

## Models Tested (6+ models, 3+ providers)

### OpenAI
- GPT-4o (flagship)
- GPT-4o-mini (cost-optimized)

### Anthropic
- Claude 3.5 Sonnet (flagship)
- Claude 3 Haiku (fast)

### Google
- Gemini 1.5 Pro
- Gemini Flash

**Total**: 6 models across 3 providers

---

## Evaluation Protocol

### 1. Fixed Prompt (No Variation)

**Critical**: Identical prompt used across ALL models to ensure fair comparison.

**Prompt Structure**:
```
System: "You are a financial services complaint classification expert. 
Assign exactly ONE label from the list below."

User: [Label Codebook v2 with definitions + examples]

Then input complaint text.

Expected Output Format:
Label: <exactly one label from list>

No explanation. No additional text.
```

### 2. Holdout Dataset

**Size**: 1,000 complaints
**Source**: CFPB Consumer Complaint Database
**Ground Truth**: Human-labeled with consensus validation
**Class Distribution**: Balanced across 7 categories (Fraud, Hidden Fees, Credit Reporting, Loan Servicing, Account Access, Process Failure, Other)

**Inter-Annotator Agreement**: 88-90% (Krippendorff's Alpha)

### 3. Metrics Captured

For each model, we tracked:

#### Performance Metrics
- **Accuracy**: Overall % correct predictions
- **Macro F1 Score**: Harmonic mean of precision/recall, averaged across classes (treats all categories equally)
- **Per-Class F1**: F1 score for each of 7 categories
- **MCC (Matthews Correlation Coefficient)**: Single balanced metric accounting for class imbalance

#### Operational Metrics
- **Start Time**: UTC timestamp
- **End Time**: UTC timestamp  
- **Total Runtime**: Minutes to process 1,000 complaints
- **Total Tokens**: Input + output tokens (when available)
- **Estimated API Cost**: Based on current pricing (Feb 2026)

### 4. Evaluation Script

Used identical evaluation script for all models:

```python
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef

def evaluate_model(y_true, y_pred):
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'macro_f1': f1_score(y_true, y_pred, average='macro'),
        'per_class_f1': f1_score(y_true, y_pred, average=None),
        'mcc': matthews_corrcoef(y_true, y_pred)
    }
```

---

## Reproducibility Check

For one selected model (GPT-4o):
- Re-ran predictions on 100 random samples
- Checked % identical predictions
- **Consistency rate**: 94% (even at temperature=0, some variation exists)

**Finding**: GenAI models exhibit non-deterministic behavior despite temperature=0 setting, likely due to:
- Internal sampling variations
- API-side load balancing across model instances
- Provider-side model updates

**Contrast**: RoBERTa specialist showed 100% reproducibility (deterministic by design)

---

## Results Summary Format

### Comparison Table

| Model | Accuracy | Macro F1 | MCC | Runtime (min) | Cost per 10k |
|-------|----------|----------|-----|---------------|--------------|
| Model A | 0.XXX | 0.XXX | 0.XXX | XX | $XX |
| Model B | 0.XXX | 0.XXX | 0.XXX | XX | $XX |

### Cost/Time Analysis

| Model | Total Tokens | API Cost (1k) | Extrapolated Cost (10k) |
|-------|--------------|---------------|------------------------|
| Model A | XXX,XXX | $X.XX | $XX.XX |

---

## Key Methodological Decisions

### Why Macro F1 Over Accuracy?

**Problem with Accuracy**: 
- In imbalanced datasets, a model predicting only majority classes can achieve high accuracy
- Example: If 82% of complaints are Fraud/Fees, always predicting "Fraud" yields 82% accuracy but is useless

**Macro F1 Advantage**:
- Treats all categories equally (not weighted by frequency)
- Penalizes models that ignore minority classes (Other, Process Failure)
- Aligns with business need: Detecting rare-but-critical issues (Fraud) is 10x more important than common issues

### Why MCC?

**Matthews Correlation Coefficient** provides:
- Single metric balancing precision, recall, and class distribution
- Range: -1 (worst) to +1 (perfect), 0 = random
- More reliable than F1 for comparing models across different class imbalances
- Standard metric in bioinformatics/ML research

### Why Fixed Prompt?

**Temptation**: Optimize prompt per model (GPT-4 might perform better with different wording than Claude)

**Why we didn't**:
- Goal is to measure **model capability**, not prompt engineering skill
- Real-world scenario: Production systems use single prompt template
- Fair comparison requires controlling for prompt variation

**Trade-off Acknowledged**: Individual models might perform 2-5% better with custom prompts, but comparison would be less valid

---

## Data Storage

### Prediction Outputs
- Stored as: `results/benchmarks/predictions_<modelname>.csv`
- Columns: `complaint_id`, `text`, `true_label`, `predicted_label`, `confidence` (if available)

### Evaluation Summaries
- Stored as: `results/benchmarks/evaluation_summary.csv`
- Contains all metrics in comparison table format

### Cost/Time Logs
- Stored as: `results/benchmarks/operational_metrics.csv`
- Includes runtime, tokens, cost estimates

---

## Validation Against Human Baseline

**Human Labeler Agreement**: 88-90%
**Specialist Model Accuracy**: 92.75%

**Implication**: Model performs at or above human consensus level. Remaining errors (7.25%) reflect:
- Inherent ambiguity in complaint narratives
- Subjective interpretation of borderline cases
- Multi-issue complaints where primary harm is unclear

**Conclusion**: For production deployment, 92.75% accuracy is effectively equivalent to senior human auditor performance.

---

## Lessons Learned

### What Worked
1. **Fixed prompt**: Enabled fair comparison across 6 models
2. **Macro F1 focus**: Correctly prioritized minority class performance
3. **Holdout validation**: 1,000 human-labeled examples provided robust ground truth
4. **Cost tracking**: API cost analysis crucial for ROI decisions

### What We'd Change
1. **Prompt A/B testing**: Run separate experiment with model-optimized prompts (report both)
2. **Larger holdout**: 2,000+ examples would reduce confidence interval on F1 scores
3. **Streaming evaluation**: Real-time metrics as predictions arrive (vs batch post-processing)
4. **Calibration curves**: Plot confidence scores vs actual accuracy to optimize routing threshold

---

This methodology ensures that our GenAI vs Specialist comparison is rigorous, reproducible, and directly applicable to production deployment decisions.
