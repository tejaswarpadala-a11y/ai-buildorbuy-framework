# Results Analysis & Strategic Recommendation

*Comprehensive benchmark of Specialist vs GenAI models on 1,000 human-labeled holdout set*

---

## Performance Comparison: Specialist vs GenAI

All models tested on identical 1,000-complaint holdout set with human ground truth labels.

### Overall Performance Metrics

| Model | Provider | Macro F1 | Accuracy | MCC | Runtime | Cost per 10k |
|-------|----------|----------|----------|-----|---------|--------------|
| **RoBERTa Specialist** ✅ | HuggingFace | **0.8492** | **92.75%** | **0.8634** | **15 min** | **~$0** |
| GPT-4o | OpenAI | 0.7900 | 90.6% | 0.8364 | 4 hrs | $120 |
| Claude 3.5 Sonnet | Anthropic | 0.7850 | 90.3% | 0.8320 | 3.8 hrs | $90 |
| GPT-4o-mini | OpenAI | 0.7680 | 89.1% | 0.8190 | 2.1 hrs | $12 |
| Gemini 1.5 Pro | Google | 0.7520 | 88.4% | 0.8050 | 3.2 hrs | $25 |
| Claude 3 Haiku | Anthropic | 0.7410 | 87.6% | 0.7980 | 1.9 hrs | $8 |
| Gemini Flash | Google | 0.7290 | 86.8% | 0.7890 | 1.2 hrs | $5 |

---

## Key Findings

### 1. Performance Gap: +5.9 F1 Points

**RoBERTa Specialist** outperformed the best GenAI (GPT-4o) by:
- **+5.92 percentage points** in Macro F1 (0.849 vs 0.790)
- **+2.15 percentage points** in accuracy (92.75% vs 90.6%)
- **+0.027** in MCC (0.863 vs 0.836)

**Implication**: After learning from 15k GenAI-generated labels, the specialist model generalized patterns more effectively than the GenAI teacher itself.

---

### 2. Operational Efficiency

#### Speed
- **Specialist**: 15 minutes for 10k texts (**16x faster than GPT-4o**)
- **GPT-4o**: 4 hours for 10k texts
- **Fastest GenAI (Gemini Flash)**: 1.2 hours (still 4.8x slower)

#### Cost
- **Specialist**: ~$0 marginal cost (local inference)
- **GPT-4o Mini**: $12 per 10k texts
- **GPT-4o**: $120 per 10k texts

**Break-even Point**: 19,200 texts (one-time training cost $230 ÷ $0.012 per text savings)

**At 100k texts/month**:
- GenAI annual cost: $144,000 (GPT-4o) or $14,400 (GPT-4o-mini)
- Specialist annual cost: $230 (one-time training)
- **Savings**: $143,770/year (GPT-4o) or $14,170/year (GPT-4o-mini)

---

### 3. Consistency & Reproducibility

| Dimension | Specialist | GenAI |
|-----------|-----------|-------|
| **Determinism** | Exact same label for same text (frozen weights) | Non-deterministic even at temp=0 |
| **Version Stability** | Controlled (explicit model versioning) | Subject to provider API updates |
| **Audit Trail** | Complete (model weights + training data) | Limited (API black box) |
| **Data Privacy** | Full (local deployment) | Third-party API required |

**Critical for Financial Services**: Regulatory compliance and audit requirements favor deterministic, locally-controlled models.

---

## Per-Category Performance Analysis

### Where Specialist Excels

**Fraud Detection** (Highest Priority):
- Specialist F1: 0.94
- GPT-4o F1: 0.88
- **Gap**: +6 points on mission-critical category

**Hidden Fees** (High Priority):
- Specialist F1: 0.91
- GPT-4o F1: 0.85
- **Gap**: +6 points

### Where GenAI Struggled

**Common Failure Patterns**:
1. **Semantic Overlap** (31 errors): "Account locked due to fraud" → mislabeled as Account Access instead of Fraud
2. **Multi-Issue Confusion** (18 errors): Fee + process complaints → mislabeled as Other instead of prioritizing primary harm
3. **Hierarchy Violations** (10 errors): Failed to apply precedence rules (Fraud > Fees > Process)

**Root Cause**: GenAI over-indexes on keywords rather than learned decision hierarchy.

---

## Strategic Recommendation

### ✅ Deploy RoBERTa Specialist Model

**Justification**:

1. **Superior Accuracy**
   - 5.9 F1 point advantage reduces misclassification risk
   - Critical for Fraud and Hidden Fees (highest business impact)

2. **Economic ROI**
   - Zero marginal cost vs $12-$120 per 10k texts
   - Break-even at 19,200 texts (typically 2 weeks at enterprise scale)
   - 16x speed improvement enables real-time routing

3. **Data Sovereignty**
   - Sensitive consumer data stays in secure environment
   - Major compliance advantage for financial sector (CFPB, GDPR)

4. **Operational Stability**
   - Deterministic classification enables long-term trend analysis
   - No vendor lock-in to API providers
   - Controlled model versioning and audit trails

---

## Deployment Strategy

### Hybrid Confidence-Based Routing

**95% Automation** (Confidence >0.70):
- Route directly to appropriate team
- Specialist handles with 0.849 F1 accuracy
- Zero marginal cost

**5% Human Review** (Confidence <0.70):
- Primarily Process Failure vs Other ambiguity
- Optional: Use GenAI for context/explanation
- Combined cost: ~$600/month at 100k volume

**Net Outcome**:
- **Blended F1**: 0.846 (weighted by confidence routing)
- **Cost Savings**: 96% vs pure GenAI ($600/month vs $14,400/month)
- **Speed**: 95% instant routing vs 4-hour GenAI processing

---

## When to Use GenAI Instead

GenAI (GPT-4/Claude) is better when:
- **Low volume** (<1,000 texts/month) - training cost not justified
- **Schema evolves frequently** - no retraining overhead
- **Explanations required** - chain-of-thought reasoning
- **Bootstrapping phase** - generating initial training data
- **Multi-task needs** - same model for classification + summarization + Q&A

**Our approach**: Use both - GenAI as teacher (data creation), Specialist as student (production inference)

---

## Limitations & Future Improvements

### Current Limitations
1. **Context Window**: 256 tokens may truncate long complaints (~3% affected)
2. **Schema Rigidity**: Adding new categories requires full retraining (3 hours)
3. **Single Language**: English only (40% of CFPB complaints are Spanish)

### Roadmap
1. **Active Learning**: Reduce labeling cost from $180 → $72 per 15k examples (60% savings)
2. **Multi-Lingual**: Extend to Spanish complaints
3. **Dynamic Few-Shot**: Use vector DB for retrieval-augmented classification (+2-5% F1 expected)
4. **Confidence Calibration**: Improve 0.7 threshold via isotonic regression

---

## Conclusion

The RoBERTa specialist demonstrates that **well-scoped fine-tuning beats general-purpose GenAI** for high-volume, domain-specific classification tasks. The 5.9 F1 point advantage, combined with 16x speed improvement and zero marginal cost, makes this the clear choice for production deployment.

**The broader lesson**: GenAI is powerful for bootstrapping and low-volume tasks, but at enterprise scale, a hybrid approach (GenAI teacher → specialist student) delivers superior economics and performance.
