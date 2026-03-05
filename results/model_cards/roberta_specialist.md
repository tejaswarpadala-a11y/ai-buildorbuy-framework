# Model Card: RoBERTa Specialist - Bank Rage Classifier

## Model Details

**Model Type**: Fine-tuned RoBERTa-base for multi-class text classification  
**Task**: CFPB complaint root cause classification (7 categories)  
**Architecture**: RoBERTa-base (125M parameters)  
**Framework**: HuggingFace Transformers 4.36.0  
**Training Date**: February 2026  
**Developer**: Teja Padala | [LinkedIn](https://www.linkedin.com/in/teja-padala/)

---

## Intended Use

### Primary Use Cases
- Automated classification of consumer financial complaints
- Triage and routing for banking operations teams
- Trend analysis for regulatory compliance (CFPB)
- Quality assurance for complaint resolution

### Out-of-Scope Use
- ❌ Medical/health complaint classification
- ❌ Non-financial consumer complaints
- ❌ Non-English text (model is English-only)
- ❌ Real-time fraud detection (use as triage only, not automated blocking)

---

## Training Data

**Source**: CFPB Consumer Complaint Database  
**Size**: 15,000 complaints (training) + 1,000 (human-labeled validation)  
**Labeling Method**: GPT-4o teacher model (87% agreement with human baseline)  
**Class Distribution**:
- Hidden Fees: 22%
- Fraud/Security: 18%
- Credit Reporting: 16%
- Loan Servicing: 15%
- Account Access: 14%
- Process Failure: 12%
- Other: 3%

**Preprocessing**:
- Max sequence length: 256 tokens
- Tokenizer: RoBERTa BPE tokenizer
- Text cleaning: Minimal (preserve complaint language)

---

## Training Configuration

### Hyperparameters (Optimized)
- **Learning Rate**: 2e-5 (vs BERT default 5e-5)
- **Dropout**: 0.1
- **Epochs**: 3
- **Batch Size**: 16
- **Optimizer**: AdamW
- **Weight Decay**: 0.01
- **Warmup Steps**: 500
- **Mixed Precision**: FP16 (for speed)

### Hardware
- GPU: NVIDIA A100 (40GB VRAM)
- Training Time: ~3 hours
- One-time Cost: ~$50 (cloud GPU)

---

## Performance

### Overall Metrics (1,000 Human-Labeled Holdout)

| Metric | Score |
|--------|-------|
| **Macro F1** | **0.849** |
| **Accuracy** | **92.75%** |
| **MCC** | **0.863** |

### Per-Category F1 Scores

| Category | F1 Score | Notes |
|----------|----------|-------|
| Fraud/Security | 0.94 | Mission-critical - highest performance |
| Hidden Fees | 0.91 | High priority |
| Credit Reporting | 0.88 | Solid performance |
| Loan Servicing | 0.85 | Good |
| Account Access | 0.87 | Good |
| Process Failure | 0.81 | Hardest category (overlaps with others) |
| Other | 0.67 | Catch-all - expected lower |

### Comparison to Baselines

| Model | Macro F1 | Speed (10k texts) | Cost (10k texts) |
|-------|----------|-------------------|------------------|
| **This Model (RoBERTa)** | **0.849** | **15 min** | **~$0** |
| GPT-4o | 0.790 | 4 hours | $120 |
| Claude 3.5 Sonnet | 0.785 | 3.8 hours | $90 |
| Human Consensus | ~0.88-0.90 | N/A | N/A |

**Key Takeaway**: Model performs at human-level accuracy while being 16x faster and ~$0 marginal cost vs GenAI.

---

## Limitations

### Technical Limitations
1. **Context Window**: 256 tokens (may truncate very long complaints ~3%)
2. **English Only**: Not trained on Spanish complaints (40% of CFPB volume)
3. **Fixed Schema**: Adding new categories requires full retraining (3 hours)

### Known Failure Modes
1. **Semantic Overlap**: Struggles with "Account locked due to fraud" (Fraud vs Account Access)
2. **Multi-Issue**: When complaint spans multiple categories, may miss secondary issues
3. **Context Truncation**: Critical information at end of 500+ word complaints may be missed

### Confidence Distribution
- **High Confidence (>0.70)**: 95% of predictions
- **Low Confidence (<0.70)**: 5% of predictions (primarily Process Failure vs Other)

**Recommendation**: Route low-confidence cases to human review

---

## Ethical Considerations

### Bias & Fairness
- **Training Data**: CFPB complaints reflect real-world distribution, which may contain demographic biases
- **Mitigation**: Model does NOT use demographic features (race, age, gender, location)
- **Recommendation**: Monitor for disparate impact across complaint types and demographics

### Privacy
- **PII Handling**: Model was trained on public CFPB data (already redacted by CFPB)
- **Deployment**: Should NOT log raw complaint text without encryption
- **Compliance**: Local deployment ensures data sovereignty (no third-party APIs)

### Transparency
- **Explainability**: Model is a black box (like all transformers)
- **Mitigation**: Can add LIME/SHAP for local explanations
- **Audit Trail**: Deterministic predictions enable reproducible outcomes

---

## Deployment Recommendations

### Production Architecture
```
Complaint → RoBERTa Specialist → Confidence Score
                                      ↓
                      High (>0.70) → Auto-route to team (95%)
                      Low (<0.70)  → Human review (5%)
```

### Monitoring Metrics
Track these in production:
1. **Confidence Distribution**: If >15% fall below 0.7 → retrain
2. **Class Drift**: New complaint patterns → update schema
3. **Error Patterns**: Systematic failures → improve training data

### Retraining Triggers
- **Volume**: When low-confidence cases exceed 10%
- **Accuracy**: When spot-check accuracy drops below 85%
- **Schema**: When new categories emerge (e.g., crypto complaints)

---

## Model Access

**Model Weights**: Excluded from GitHub repo (500MB)  
**Alternative**: Model available on HuggingFace Hub (link TBD)

**To Load**:
```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model = AutoModelForSequenceClassification.from_pretrained(
    "tejaswar/bank-rage-roberta-specialist"
)
tokenizer = AutoTokenizer.from_pretrained("roberta-base")
```

---

## Citation

If you use this model or approach in your work, please cite:

```
@misc{padala2026bankrage,
  author = {Teja Padala},
  title = {Bank Rage Classifier: GenAI vs Fine-Tuned Specialist Comparison},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/tejaswarpadala/bank-rage-classifier}
}
```

---

## Contact

**Author**: Teja Padala  
**Email**: tejaswar.padala@gmail.com  
**LinkedIn**: https://www.linkedin.com/in/teja-padala/  
**GitHub**: https://github.com/tejaswarpadala

For issues or questions about the model, please open an issue on GitHub.

---

## License

MIT License - See [LICENSE](../LICENSE) for details.

**Model Weights**: Available under MIT License  
**Training Data**: Public CFPB data (government work, no copyright)

---

## Version History

**v1.0** (February 2026)
- Initial release
- RoBERTa-base fine-tuned on 15k GenAI-labeled complaints
- Macro F1: 0.849 on 1k human-labeled holdout
