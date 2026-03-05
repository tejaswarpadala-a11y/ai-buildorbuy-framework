# Design Tradeoffs & Decision Framework

*How we evaluated alternatives and made principled choices*

---

## Executive Summary

This document captures the **why** behind every major technical decision in the Bank Rage Classifier project. For PM roles at companies like Amazon, Google, Anthropic, and Scale AI, understanding tradeoffs is more important than implementation details.

**Key Principle**: There is no universally "best" solution - only tradeoffs that align with specific constraints and objectives.

---

## 1. Build vs Buy: Fine-Tuning vs GenAI APIs

### The Core Question
*Should we use GPT-4/Claude out-of-the-box, or fine-tune our own model?*

### Decision Matrix

| Dimension | GenAI API (GPT-4o) | Fine-Tuned Specialist (RoBERTa) |
|-----------|-------------------|----------------------------------|
| **Initial Cost** | $0 (pay-per-use) | ~$230 (training + labeling) |
| **Marginal Cost** | $12 per 1k texts | ~$0 (local inference) |
| **Performance (F1)** | 0.79 | **0.849** (+7.5%) |
| **Latency** | 2-3 sec/text | **50ms/text** (40-60x faster) |
| **Consistency** | Probabilistic (temp≠0) | Deterministic |
| **Explainability** | Chain-of-thought | Feature attribution |
| **Schema Changes** | Zero-shot adaptable | Requires retraining |
| **Data Privacy** | Third-party API | **In-house** |
| **Vendor Lock-in** | High (API changes) | Low (open-source) |

### Break-Even Analysis

**Assumptions**:
- GenAI cost: $0.012/text
- Training cost (one-time): $230
- Monthly volume: V texts

**Break-even point**: V = $230 / $0.012 ≈ **19,200 texts**

**At 100k texts/month**:
- GenAI annual cost: $144,000
- Specialist annual cost: $230 (training) + $0 (inference) = $230
- **Savings**: $143,770/year

### Decision: Hybrid Approach ✅

**Why not pure GenAI?**
1. At enterprise scale (100k+ texts/month), costs become prohibitive
2. Latency requirements for real-time routing (need <100ms)
3. Regulatory compliance (financial data can't leave infrastructure)

**Why not pure specialist?**
1. Cold-start problem: No labeled data initially
2. Schema evolution: Adding new categories requires retraining

**Chosen Solution**: 
- Use GenAI to *create training data* (teacher model)
- Fine-tune specialist for *inference* (student model)
- Route low-confidence cases back to GenAI (5% of traffic)

**Expected Outcome**: 95% cost reduction vs pure GenAI while maintaining >90% automation

---

## 2. Data Labeling Strategy: Human vs GenAI vs Hybrid

### The Challenge
Need 15,000+ labeled examples to fine-tune a competitive model.

### Options Evaluated

| Approach | Cost | Time | Quality | Scalability |
|----------|------|------|---------|-------------|
| **Human Only** | $45k+ | 3-6 months | 90% agreement | Low (bottleneck) |
| **GenAI Only** | $180 | 4 hours | 87% agreement | **High** |
| **Crowdsourcing** | $3k | 2-4 weeks | 70-80% agreement | Medium |
| **Active Learning** | $15k | 6-8 weeks | 92% agreement | Medium |
| **Hybrid (Chosen)** ✅ | $2k | 2 weeks | 89% agreement | **High** |

### Decision: GenAI + Human Validation

**Strategy**:
1. Use GPT-4o to generate 15k labels (87% accurate)
2. Validate on 1k human-labeled holdout set
3. Identify systematic errors → refine prompt → regenerate
4. Use human labelers only for edge cases (confidence <0.7)

**Why this works**:
- **Quality**: GenAI mistakes are *systematic* (can be prompt-engineered away)
- **Speed**: 4 hours vs 3-6 months
- **Cost**: $180 vs $45,000
- **Iteration**: Can re-label entire dataset in hours if schema changes

**Risk Mitigation**: 
- Human holdout set (1k examples) as ground truth
- Monitor specialist performance against human baseline
- Budget 20% of labels for human verification if accuracy drops

---

## 3. Model Architecture: RoBERTa vs DeBERTa vs DistilBERT

### Evaluation Criteria
1. **Performance**: F1 score on 7-way classification
2. **Training Stability**: Gradient convergence, loss curves
3. **Inference Speed**: Latency for production deployment
4. **Model Size**: Memory footprint, deployment constraints

### Results

| Model | F1 Score | Training Time | Inference (ms) | Model Size | Stability |
|-------|----------|---------------|----------------|------------|-----------|
| **RoBERTa-base** ✅ | **0.849** | 3 hrs | **50ms** | 500MB | High |
| DeBERTa-v3 | 0.814 | 4 hrs | 72ms | 700MB | Medium (FP16 issues) |
| DistilBERT | 0.762 | 1.5 hrs | 32ms | 250MB | High |

### Deep Dive: Why Not DeBERTa?

**DeBERTa Advantages**:
- Disentangled attention (theoretically superior)
- State-of-the-art on many benchmarks

**In Practice (Our Task)**:
- Training instability with mixed-precision (FP16)
- Required custom gradient unscaling loop (technical debt)
- Only 3.5% F1 improvement over RoBERTa, but:
  - 33% slower inference
  - 40% larger model size
  - Harder to debug/maintain

**Decision**: RoBERTa hit the sweet spot for our constraints.

### Deep Dive: Why Not DistilBERT?

**DistilBERT Advantages**:
- 40% smaller, 60% faster than BERT/RoBERTa
- Great for edge/mobile deployment

**Our Context**:
- 8.7 percentage point F1 gap vs RoBERTa (0.762 vs 0.849)
- Categories like "Process Failure" and "Credit Reporting" require nuanced understanding
- DistilBERT's compression removed too much semantic capacity

**When DistilBERT would win**:
- Mobile app deployment (size matters more than accuracy)
- Latency <<50ms required (e.g., autocomplete)
- Binary classification (less nuance needed)

---

## 4. Training Configuration: Hyperparameters That Mattered

### Learning Rate: The 10x Variable

| Learning Rate | F1 Score | Convergence | Issue |
|---------------|----------|-------------|-------|
| 5e-5 (BERT default) | 0.762 | Fast (1 epoch) | Overshot minima; missed nuance |
| **2e-5** ✅ | **0.849** | Gradual (3 epochs) | Stable; learned teacher's logic |
| 1e-5 | 0.828 | Very slow | Underfit; needed 5+ epochs |

**Why 2e-5 won**:
- Slower updates allowed model to internalize GenAI's reasoning patterns
- Fast rates (5e-5) converged quickly but *memorized* training quirks
- Validated via learning curve analysis (see notebooks/03)

### Dropout: Finding the Signal-to-Noise Ratio

| Dropout Rate | F1 Score | Class Performance | Insight |
|--------------|----------|-------------------|---------|
| 0.2 (high) | 0.762 | Fraud: 0.88, Other: 0.43 | Too aggressive; "Other" collapsed |
| **0.1** ✅ | **0.849** | Fraud: 0.94, Other: 0.67 | Balanced regularization |
| 0.0 (none) | 0.841 | Fraud: 0.96, Other: 0.61 | Slight overfit on Fraud |

**Key Insight**: 
Low-frequency classes (Other, Process Failure) are most sensitive to dropout. High dropout (0.2) effectively erased signal for rare categories.

---

## 5. Evaluation Metrics: Why F1 Over Accuracy

### The Imbalance Problem

| Category | Frequency | Why It Matters |
|----------|-----------|----------------|
| Fraud | 18% | **Mission-critical**: False negatives = financial loss |
| Hidden Fees | 22% | **High-priority**: Customer retention impact |
| Process Failure | 12% | Low-priority: Doesn't justify urgent action |
| Other | 8% | Triage category |

**Problem with Accuracy**:
- Model that predicts "Fraud" for everything gets 82% accuracy (just by guessing majority classes)
- Masks catastrophic failures on rare-but-critical categories

**Why Macro F1**:
- Treats all classes equally (not weighted by frequency)
- Punishes models that ignore minority classes
- Aligns with business need (Fraud detection is 10x more important than Other)

### Confusion Matrix Insights

**Most Common Errors** (GenAI vs Specialist):

| True Label | GenAI Predicted | Specialist Predicted | Root Cause |
|-----------|-----------------|----------------------|-----------|
| Fraud | Account Access (31 cases) | Fraud ✅ | GenAI over-indexed on "locked account" keywords |
| Hidden Fees | Process Failure (18 cases) | Hidden Fees ✅ | GenAI confused fee complaints with poor service |
| Credit Reporting | Loan Servicing (14 cases) | Credit Reporting ✅ | Overlapping financial concepts |

**Why Specialist Won**:
- Learned to prioritize **decision hierarchy** from training data
- GenAI lacked consistent precedence logic despite prompt engineering

---

## 6. Deployment Strategy: Hybrid Confidence-Based Routing

### The 95/5 Rule

**Observation**: 95% of complaints have specialist confidence >0.70

**Design**:
```
┌─────────────┐
│  Complaint  │
└──────┬──────┘
       │
       ▼
┌──────────────────────┐
│ RoBERTa Confidence   │
└──────┬───────────────┘
       │
       ├─→ >0.70 (95%) ──→ Auto-route to team
       │
       └─→ <0.70 (5%)  ──→ GenAI + human review
```

**Economics**:
- Pure Specialist: 95% automation, 0.849 F1, $0/month
- 5% GenAI Fallback: 0.79 F1, ~$600/month (100k volume)
- **Net**: 0.846 blended F1, 96% cost savings vs pure GenAI

### Alternative Approaches Considered

| Approach | Pros | Cons | Why Not Chosen |
|----------|------|------|----------------|
| **Pure Specialist** | Cheapest, fastest | Rigid (no schema changes) | Need flexibility for evolving categories |
| **Pure GenAI** | Most flexible | Expensive, slow | Cost prohibitive at scale |
| **Ensemble (Specialist+GenAI)** | Highest accuracy | 2x latency + cost | Marginal 0.02 F1 gain not worth complexity |
| **Hybrid (Chosen)** ✅ | Balanced | Requires confidence calibration | Best ROI for our constraints |

---

## 7. Lessons for Future Projects

### What Worked
1. **Teacher-Student Pattern**: GenAI for data creation, specialist for inference
2. **Rigorous Benchmarking**: Tested 6+ models before committing
3. **Business-First Metrics**: Macro F1 over accuracy (aligned with priorities)
4. **Error-Driven Iteration**: Fixed systematic failures through prompt refinement

### What I'd Change
1. **Active Learning Earlier**: Could've reduced GenAI labeling from 15k → 5k
2. **Multi-Task Training**: Joint training for classification + confidence prediction
3. **Cost Instrumentation**: Track every API call from day 1 (not retroactively)
4. **A/B Testing Framework**: Deploy both models side-by-side for 2 weeks

### Broader Takeaways

**For PM Roles**:
- Technology choices are **constraint-driven**, not technology-driven
- The "best" model is the one that fits your specific cost/latency/accuracy triangle
- GenAI shines at **bootstrapping** (data creation), not always at **production inference**

**For ML Engineers**:
- Hyperparameters can swing F1 by 10+ points (never use defaults blindly)
- Error analysis >>> accuracy numbers (understand *why* models fail)
- Deployment constraints (latency, cost) should inform architecture choices upfront

---

## Appendix: Decision Trees for Future Work

### When to Retrain the Specialist

```
Confidence Distribution Shift?
  ├─→ Yes: >15% fall below 0.7 ──→ RETRAIN
  └─→ No
        │
        Class Drift Detected? (new complaint patterns)
          ├─→ Yes ──→ RETRAIN with updated schema
          └─→ No ──→ Continue monitoring
```

### When to Use GenAI vs Specialist (General Framework)

```
High Volume (>10k/month)?
  ├─→ Yes
  │     │
  │     Stable Schema?
  │       ├─→ Yes ──→ SPECIALIST
  │       └─→ No ──→ GENAI (fast iteration)
  └─→ No
        │
        Budget <$500/month?
          ├─→ Yes ──→ SPECIALIST (break-even at 1.9k texts)
          └─→ No ──→ GENAI (flexibility worth premium)
```

---

**This document demonstrates**:
- ✅ Systematic evaluation of alternatives
- ✅ Quantitative tradeoff analysis (not just intuition)
- ✅ Business context for technical decisions
- ✅ Willingness to choose "boring" tech when it fits constraints

*Exactly the signal senior PM roles at MAANG/AI companies want to see.*
