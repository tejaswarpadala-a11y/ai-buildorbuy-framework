# AI Build-or-Buy Framework: Quantifying GenAI vs Fine-Tuned Models

**A strategic decision framework for evaluating GenAI APIs vs specialist models, with quantified ROI analysis**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![HuggingFace](https://img.shields.io/badge/🤗-HuggingFace-yellow)](https://huggingface.co/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**Built by Teja Padala | Lead Engineer → Founder (TekWinn: $1.8M ARR Exit) → AI Strategy (EssilorLuxottica) → VC Associate (Excelerate Health Ventures) | MBA at UNC Kenan-Flagler 2026**

---

## The Problem

Financial institutions process millions of consumer complaints annually through the CFPB (Consumer Financial Protection Bureau). These complaints contain critical signals about operational failures - from hidden fees to fraud - but classifying them manually is:

- **Expensive**: $2-5 per complaint at scale (human labelers)
- **Slow**: Days to weeks for large batches
- **Inconsistent**: Inter-annotator agreement hovers around 88-90%

**The question**: Can GenAI models like GPT-4 or Claude replace this process entirely?

**The better question**: *When should you fine-tune a specialist model vs. use GenAI out-of-the-box?*

This project answers that question through rigorous benchmarking.

---

## Why I Built This

After scaling TekWinn to **$1.8M ARR and exit**, I wanted to deeply understand the GenAI vs traditional ML tradeoff that every product leader faces today. Rather than just read about it, I built a rigorous benchmark.

**The practical question**: When should you use GPT-4/Claude out-of-the-box vs invest in fine-tuning your own model?

**The typical answer**: "It depends."

**My answer**: "It depends on your constraints - and here's exactly how to evaluate them."

This project quantifies the build-vs-buy decision with real data:
- **Performance**: RoBERTa specialist beats GPT-4o by 5.9 F1 points
- **Cost**: $0 marginal cost vs $12/10k texts (GPT-4o Mini) to $120/10k (GPT-4o)
- **Speed**: 15 minutes vs 4 hours for 10k classifications
- **Break-even**: 19,200 texts (typically hit in first 2 weeks at enterprise scale)

As a product leader, I've learned that the "best" technology is rarely the newest one - it's the one that fits your specific cost/speed/accuracy constraints. This project proves that with numbers.

---

## 🔬 Research Collaboration

This project is being developed in collaboration with **Professor Daniel Ringel** (UNC Kenan-Flagler Business School) for potential academic publication.

**Current Status**: Expanding from proof-of-concept (F1: 0.849) to research-grade quality

**Next Phase Goals**:
- Upgrade to reasoning models (GPT-o1, Claude Opus) for higher-quality training data
- Expand balanced dataset to 100+ examples per category
- Target performance: F1 ≥ 0.94 (research-grade threshold)
- Explore multi-label classification for co-occurring complaint types

**What This Means**: This work bridges academic rigor with practical deployment - combining strategic PM thinking with research-quality methodology.

---

## The Approach: Hybrid Intelligence

Rather than choosing GenAI *or* traditional ML, I designed a three-stage system that leverages the strengths of both:

```
┌─────────────────┐      ┌──────────────────┐      ┌─────────────────┐
│   GenAI Teacher │ ───> │  15k Labels      │ ───> │ RoBERTa Student │
│   (GPT-4o)      │      │  Generated       │      │ (Specialist)    │
└─────────────────┘      └──────────────────┘      └─────────────────┘
     Cost: $180                                         Cost: ~$0
     Speed: ~4 hrs                                      Speed: ~15 min
                                                        
                         ↓
                         
              ┌─────────────────────────┐
              │  Human Validation       │
              │  (1k holdout set)       │
              └─────────────────────────┘
```

**Hypothesis**: GenAI can create high-quality training data, but a fine-tuned specialist model will outperform it on inference while dramatically reducing costs.

---

## Architecture & Design Decisions

### 1. Classification Schema Design

I developed a 7-category taxonomy optimized for **operational prioritization**, not academic accuracy:

| Category | Business Priority | Example Signal |
|----------|------------------|----------------|
| 🔴 Fraud / Security | **Critical** | Unauthorized transactions, identity theft |
| 🟠 Hidden Fees | **High** | Undisclosed charges, surprise fees |
| 🟡 Credit Reporting Error | **High** | Incorrect credit bureau data |
| 🟢 Loan/Mortgage Servicing | **Medium** | Payment application issues |
| 🔵 Account Access | **Medium** | Account freezes, login issues |
| 🟣 Process Failure | **Low** | Red tape, slow resolution |
| ⚪ Other | **Triage** | Ambiguous or multi-issue complaints |

**Why this schema?** Banks need to triage by risk, not just categorize. Fraud overrides all other labels. Fees take precedence over process complaints. This hierarchy ensures critical issues surface first.

### 2. Data Labeling Strategy

**Challenge**: Obtaining 15,000+ human labels would cost ~$45,000 and take months.

**Solution**: Use GPT-4o as a "teacher model" with a carefully engineered prompt:
- **Prompt Engineering**: 3-shot examples per category + explicit decision rules
- **Cost Optimization**: GPT-4o Mini for initial pass, GPT-4o for confidence <0.7
- **Quality Control**: Validated on 1,000 human-labeled holdout (87% agreement)

**Result**: 15,000 synthetic labels generated in ~4 hours for $180

### 3. Model Selection: RoBERTa vs DeBERTa vs DistilBERT

I evaluated three transformer architectures:

| Model | Parameters | Training Time | F1 Score | Why/Why Not |
|-------|-----------|---------------|----------|-------------|
| **RoBERTa-base** ✅ | 125M | ~3 hrs | **0.849** | Optimized pre-training, stable convergence |
| DeBERTa-v3 | 184M | ~4 hrs | 0.814 | Disentangled attention caused instability |
| DistilBERT | 66M | ~1.5 hrs | 0.762 | Too compressed; missed nuanced categories |

**Choice**: RoBERTa-base struck the best balance of performance, training stability, and deployment size.

### 4. Hyperparameter Optimization

Key finding: Small changes = huge swings in F1 score

| Experiment | Learning Rate | Dropout | F1 Score | Insight |
|-----------|---------------|---------|----------|---------|
| Baseline | 5e-5 | 0.2 | 0.762 | Too aggressive; overfit quickly |
| **Optimized** ✅ | **2e-5** | **0.1** | **0.849** | Slower convergence, better generalization |

**Why 2e-5?** Allowed the model to learn the GenAI teacher's logic without overfitting to labeling quirks.

---

## Results: Specialist vs GenAI Showdown

I benchmarked the fine-tuned RoBERTa specialist against 6 GenAI models on a 1,000-complaint human-labeled holdout set:

### Evaluation Metrics Explained

**Why these metrics matter**:
- **Macro F1 Score**: Harmonic mean of precision and recall, averaged across all classes. Treats rare categories (Fraud) equally with common ones (Fees). Range: 0-1, higher is better. *The gold standard for imbalanced classification.*
- **Accuracy**: Overall % of correct predictions. Can be misleading when classes are imbalanced (a model predicting "Fraud" for everything gets 82% accuracy!).
- **MCC (Matthews Correlation Coefficient)**: Single metric that balances precision, recall, and class imbalance. Range: -1 to +1, where +1 is perfect, 0 is random. *Most reliable for comparing models.*
- **Runtime**: Time to classify 10,000 texts (critical for production scalability)
- **Cost**: Price per 10,000 texts at current API rates

### Performance Comparison

| Model | Macro F1 | Accuracy | MCC | Runtime | Cost (10k) |
|-------|----------|----------|-----|---------|-----------|
| **RoBERTa Specialist** ✅ | **0.849** | **92.75%** | **0.863** | **15 min** | **~$0*** |
| GPT-4o | 0.790 | 90.6% | 0.836 | 4 hrs | $120 |
| Claude 3.5 Sonnet | 0.785 | 90.3% | 0.832 | 3.8 hrs | $90 |
| GPT-4o Mini | 0.768 | 89.1% | 0.819 | 2.1 hrs | $12 |
| Gemini 1.5 Pro | 0.752 | 88.4% | 0.805 | 3.2 hrs | $25 |
| Claude 3 Haiku | 0.741 | 87.6% | 0.798 | 1.9 hrs | $8 |
| Gemini Flash | 0.729 | 86.8% | 0.789 | 1.2 hrs | $5 |

*After one-time training cost of ~$50 (GPU compute)

### Key Insights

1. **Performance**: Specialist model beat the best GenAI by **+5.9 percentage points** in F1 score
2. **Speed**: 16x faster than GPT-4o (15 min vs 4 hours for 10k texts)
3. **Cost**: Effectively zero marginal cost after training
4. **Consistency**: Deterministic outputs vs GenAI's temperature-based variance

### Where GenAI Failed (and Why)

**Error Analysis** on the 73 cases where RoBERTa succeeded but GPT-4o failed:

| Error Type | Count | Root Cause | Example |
|-----------|-------|------------|---------|
| Semantic Overlap | 31 | GenAI over-indexed on keywords | "Account locked due to fraud" → labeled Account Access (should be Fraud) |
| Multi-Issue Confusion | 18 | GenAI tried to capture everything | Fee dispute + long process → labeled Other (should be Hidden Fees) |
| Context Window Limit | 14 | Truncated complaints lost key details | Critical fraud mention at end of 500-word complaint |
| Decision Hierarchy Violation | 10 | GenAI didn't respect precedence rules | Fraud mentioned but labeled as Credit Reporting |

**Conclusion**: GenAI is phenomenal at understanding language but struggles with *consistent business logic*.

---

## The Tradeoff Matrix: When to Use What

This is the PM thinking that matters most:

### Use Fine-Tuned Specialist When:
✅ **High volume** (>10k texts/month) - ROI on training cost  
✅ **Consistency critical** (regulatory/audit) - deterministic outputs  
✅ **Cost-sensitive** (startup/scale-up) - zero marginal cost  
✅ **Latency matters** (real-time) - 50ms vs 2-3s per text  
✅ **Schema is stable** (fixed categories) - no retraining overhead  

### Use GenAI (GPT-4/Claude) When:
✅ **Low volume** (<1k texts/month) - not worth training  
✅ **Schema evolves frequently** - no retraining needed  
✅ **Explanations required** - chain-of-thought reasoning  
✅ **Bootstrapping phase** - generate training data  
✅ **Multi-task system** - same model for classification + summarization  

### The Hybrid Approach (This Project):
1. **Phase 1**: Use GenAI to generate 15k training labels ($180)
2. **Phase 2**: Fine-tune specialist on synthetic data ($50)
3. **Phase 3**: Deploy specialist for inference (~$0/month)
4. **Phase 4**: Use GenAI for edge cases (confidence <0.7) → 5% of traffic

**Economic Outcome**: 
- **Monthly savings at 100k complaints/month**: ~$12,000 vs pure GenAI
- **Break-even point**: 1,900 complaints (achieved in first 2 days)
- **Payback period on training**: <48 hours

---

## Deployment Considerations

### Production Architecture

```
┌──────────────┐
│  Complaint   │
│   Stream     │
└──────┬───────┘
       │
       ▼
┌──────────────────────┐
│  RoBERTa Specialist  │  ← 95% of traffic
│  (Confidence >0.70)  │
└──────┬───────────────┘
       │
       ├─→ High Confidence ──→ [Auto-Route to Team]
       │
       └─→ Low Confidence ──→ [GenAI + Human Review]
```

### Key Metrics to Monitor
1. **Confidence Distribution**: If >10% fall below 0.7 → retrain specialist
2. **Class Drift**: New complaint patterns → update schema
3. **Error Patterns**: Systematic failures → improve training data

---

## Lessons Learned (PM Reflection)

### What Worked
1. **Teacher-Student Pattern**: GenAI for data creation, specialist for inference
2. **Business-First Schema**: Prioritized operational needs over academic purity
3. **Rigorous Benchmarking**: Tested 6+ models to justify decision
4. **Error Analysis**: Understood *why* models failed, not just *that* they failed

### What I'd Do Differently
1. **Active Learning**: Start with 5k GenAI labels, use specialist to find uncertain cases, get human labels for those
2. **Multi-Task Learning**: Train specialist to also predict confidence scores
3. **Prompt Optimization**: Spend more time on GenAI prompt engineering before scaling to 15k
4. **Cost Tracking**: Instrument every API call from day 1

### Broader Implications
This project reinforced a core PM principle: **The best solution is rarely the newest technology**. GenAI is powerful, but for well-scoped, high-volume tasks, a fine-tuned specialist often wins on:
- **Reliability** (consistency)
- **Economics** (cost)
- **Latency** (speed)
- **Control** (business logic)

The art is knowing *when* to use which tool.

---

## Technical Implementation

### Requirements
```bash
pip install -r requirements.txt
```

Key dependencies:
- `transformers==4.36.0` (HuggingFace)
- `torch==2.1.0` (PyTorch)
- `scikit-learn==1.3.2` (metrics)
- `pandas`, `numpy`, `matplotlib`

### Quick Start

**1. Train Specialist Model**
```python
from src.models.specialist_trainer import train_roberta_specialist

model = train_roberta_specialist(
    train_data="data/genai_labeled_15k.csv",
    val_data="data/human_labeled_1k.csv",
    learning_rate=2e-5,
    epochs=3
)
```

**2. Run Benchmarking**
```python
from src.evaluation.benchmark import compare_models

results = compare_models(
    test_data="data/human_labeled_1k.csv",
    models=["gpt-4o", "claude-3.5-sonnet", "specialist"]
)
```

**3. Inference**
```python
from src.models.inference import classify_complaint

prediction = classify_complaint(
    text="They charged me a fee I never agreed to!",
    model="specialist"
)
# Output: {'label': 'Hidden Fees', 'confidence': 0.94}
```

### Notebooks
Explore the full pipeline in `notebooks/`:
1. **Data Exploration**: CFPB complaint analysis
2. **GenAI Labeling**: Teacher model prompt engineering
3. **Specialist Training**: RoBERTa fine-tuning
4. **GenAI Benchmark**: Multi-model comparison
5. **Error Analysis**: Failure case deep dive

---

## Project Structure
```
bank-rage-classifier/
├── notebooks/           # Jupyter notebooks (exploration → deployment)
├── src/                # Production code
│   ├── data/          # Preprocessing & schema
│   ├── models/        # GenAI + Specialist training
│   ├── evaluation/    # Metrics & benchmarking
│   └── utils/
├── results/           # Benchmarks, visualizations, model cards
├── docs/              # Architecture, methodology, tradeoffs
└── data/              # Sample data (100 complaints)
```

---

## Future Work: Product Roadmap

This project demonstrates **v1 thinking**: ship the core value proposition fast, then iterate based on user needs. The roadmap below reflects both **production evolution** and **research collaboration goals** with UNC Kenan-Flagler faculty.

**Research Track** (Phases 2-3): Expanding to F1 ≥ 0.94 with reasoning models and balanced datasets  
**Production Track** (Phases 3-5): Scaling to enterprise deployment with RAG and active learning

### Phase 2: Enhanced Accuracy & Explainability

**1. Dynamic Few-Shot Learning (Vector DB)**
- **Problem**: Current GenAI labeling uses static 3-shot examples for all texts
- **Solution**: Use ChromaDB/Pinecone to retrieve the 3 most similar labeled examples per complaint
- **Expected Impact**: +2-5% F1 improvement on edge cases (semantic overlap, multi-issue)
- **When to build**: After 50k+ labeled examples (need critical mass for retrieval)

**2. RAG-Powered Explanations**
- **Problem**: Specialist model returns labels without reasoning (black box to end users)
- **Solution**: Vector search over historical complaints + resolutions to generate explanations
  ```
  "This complaint was classified as 'Hidden Fees' because it's similar to 
   3,241 past cases where customers reported undisclosed charges. 
   Typical resolution: Fee refund + policy update."
  ```
- **Expected Impact**: 30% reduction in human review time (context helps triaging)
- **When to build**: When compliance teams need audit trails (regulatory requirement)

### Phase 3: Intelligent Routing & Knowledge Management

**3. Context-Aware Escalation (RAG)**
- **Problem**: Low-confidence cases (5%) route to humans with no context
- **Solution**: RAG system that retrieves:
  - 5 similar historical complaints
  - Relevant policy/regulation excerpts
  - Past resolution outcomes
- **Expected Impact**: 50% faster human review (pre-populated context)
- **When to build**: When human review becomes bottleneck (>500 escalations/day)

**4. Multi-Issue Detection (Vector Similarity)**
- **Problem**: 8% of complaints span multiple categories (e.g., Fraud + Hidden Fees)
- **Solution**: Use vector embeddings to detect semantic overlap → flag for multi-label classification
- **Expected Impact**: Capture 40% more actionable signals (Fraud often co-occurs with other issues)
- **When to build**: When compliance needs comprehensive issue tracking

### Phase 4: Continuous Learning & Adaptation

**5. Active Learning Pipeline**
- **Problem**: GenAI labeling cost ($180 for 15k) doesn't scale to millions of complaints
- **Solution**: Specialist identifies uncertain cases → human labels only those → retrain
- **Expected Impact**: 60% reduction in labeling cost ($180 → $72 per 15k)
- **When to build**: When data volume exceeds 100k/month

**6. Policy Update Detection (RAG)**
- **Problem**: New regulations/products create categories we haven't seen before
- **Solution**: RAG over CFPB policy documents to detect emerging complaint types → trigger schema review
- **Expected Impact**: Proactive category evolution (vs reactive retraining)
- **When to build**: When schema drift detection shows >5% "Other" category growth

### Phase 5: Production Readiness

**7. Real-Time Deployment**: FastAPI + Docker for <100ms inference
**8. Multi-Lingual Support**: Spanish complaints (40% of CFPB submissions)
**9. Confidence Calibration**: Improve low-confidence routing thresholds
**10. Explainability**: Add LIME/SHAP for model interpretability

---

### Why This Roadmap Demonstrates Product Thinking

**v1 (Current)**: Prove the core hypothesis - specialist beats GenAI at lower cost
- ✅ Ships fast (3 weeks)
- ✅ Quantifies ROI ($144k/year savings)
- ✅ De-risks biggest assumption (can we match GenAI quality?)

**v2-v4**: Add complexity only when justified by user needs
- Vector DB when retrieval unlocks accuracy gains (>50k examples)
- RAG when humans need context (compliance/audit requirements)
- Active learning when volume justifies cost optimization (>100k/month)

**Key Principle**: Don't build for hypothetical scale - build for current constraints, then iterate.

This mirrors how I scaled TekWinn: ship v1 to validate demand, then invest in infrastructure only when growth justified it.

---

## About Me

**Teja Padala** | [LinkedIn](https://www.linkedin.com/in/teja-padala/) | [GitHub](https://github.com/tejaswarpadala)

I'm an MBA student at UNC Kenan-Flagler (graduating May 2026) with a unique background: **entrepreneur → engineer → product leader**.

**Entrepreneurial**: Founded and scaled TekWinn to **$1.8M ARR before successful exit**. Understand what it takes to build, ship, and grow products from zero.

**Technical**: 8+ years as software engineer and product manager. Deployed AI tools at Fortune 500 companies, built ML systems at scale, and currently developing AI-powered products.

**Strategic**: VC Associate at Excelerate Health Ventures, investing in healthcare AI startups. Recent strategy work at EssilorLuxottica on Ray-Ban Meta smart glasses enterprise deployment.

**I'm seeking Senior Product Management roles** where I can combine technical execution with business strategy. This project demonstrates:
- **Build vs buy decisions**: When to use GenAI vs fine-tune specialists
- **Constraint optimization**: Cost, speed, accuracy tradeoffs with quantitative analysis  
- **Technical execution**: End-to-end ML pipelines from data labeling to deployment
- **Business impact**: $144k/year ROI analysis and break-even modeling

**Let's connect**: tejaswar.padala@gmail.com

---

## License

MIT License - feel free to use this for your own projects!

---

## Acknowledgments

- **CFPB** for public complaint data
- **HuggingFace** for transformers library
- **OpenAI, Anthropic, Google** for GenAI benchmarking APIs
- **UNC Kenan-Flagler** Data Science & AI course team

---

**⭐ If this project helped you think differently about GenAI vs traditional ML, give it a star!**
