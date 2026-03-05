# Notebooks

This folder contains Jupyter notebooks documenting the complete ML pipeline from data exploration to deployment.

## Notebook Flow

Execute notebooks in this order:

### 1. `01_data_exploration.ipynb`
**Purpose**: Explore CFPB complaint data and understand the classification problem

**What you'll learn**:
- Distribution of complaint categories
- Text length and complexity analysis  
- Class imbalance patterns
- Why "Bank Rage" is a challenging classification task

**Outputs**: 
- Class distribution plots
- Text statistics summary
- Sample complaints per category

---

### 2. `02_genai_labeling.ipynb`
**Purpose**: Use GPT-4o as "teacher model" to generate 15k training labels

**What you'll learn**:
- Prompt engineering for classification
- Cost-quality tradeoffs (GPT-4o vs GPT-4o-mini)
- How to validate synthetic labels against human ground truth

**Key Decisions**:
- Why 3-shot prompting over zero-shot
- Temperature=0 for consistency
- Cost: $180 for 15k labels

**Outputs**:
- `data/genai_labeled_15k.csv` (training data)
- Label quality analysis (87% agreement with humans)

---

### 3. `03_specialist_training.ipynb`
**Purpose**: Fine-tune RoBERTa on GenAI-generated training data

**What you'll learn**:
- Hyperparameter search results (learning rate, dropout)
- Why RoBERTa over DeBERTa/DistilBERT
- Training curves and convergence analysis

**Key Results**:
- Learning rate 2e-5 (vs BERT default 5e-5)
- Dropout 0.1
- 3 epochs
- Final F1: 0.849 on validation set

**Outputs**:
- Trained model weights (excluded from repo - too large)
- Training metrics and plots
- Confusion matrix

---

### 4. `04_genai_benchmark.ipynb`
**Purpose**: Benchmark 6+ GenAI models against specialist

**What you'll learn**:
- How to run consistent evaluations across multiple API providers
- Cost-speed-accuracy tradeoffs
- Where GenAI fails systematically

**Models Tested**:
- GPT-4o, GPT-4o-mini (OpenAI)
- Claude 3.5 Sonnet, Claude 3 Haiku (Anthropic)
- Gemini 1.5 Pro, Gemini Flash (Google)

**Key Finding**: Specialist beats GPT-4o by 5.9 F1 points at 16x speed and ~$0 cost

**Outputs**:
- `results/benchmarks/genai_comparison.csv`
- Performance comparison plots
- Cost analysis charts

---

### 5. `05_error_analysis.ipynb`
**Purpose**: Deep dive into the 73 cases where specialist differed from human labels

**What you'll learn**:
- Systematic vs random errors
- Where semantic overlap causes confusion
- How truncation affects long complaints
- Human baseline comparison (88% inter-annotator agreement)

**Key Patterns**:
- Semantic overlap (31 cases): Fraud vs Account Access
- Multi-issue confusion (18 cases): Prioritizing primary harm
- Context truncation (14 cases): Critical info at end of text

**Outputs**:
- Error distribution charts
- Failure case examples
- Recommendations for confidence thresholds

---

## Running the Notebooks

### Setup
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter
jupyter notebook
```

### Environment Variables
Create `.env` file for API keys (never commit this!):
```
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=...
```

### Data Requirements
- Notebooks 01-03 require full CFPB dataset (download separately, not in repo)
- Notebooks 04-05 use holdout set provided in `data/sample_complaints_100.csv`

---

## Notebook Style Guide

All notebooks follow this structure:

```markdown
# [Number]. [Title]
**Purpose**: [One sentence]
**Key Outcomes**: [What you'll learn/build]

---

## Executive Summary
[2-3 sentences: problem, approach, result]

## Context
[Why this step is necessary]

## Methodology
[Approach + code cells]

## Results
[Visualizations + interpretation]

## Key Takeaways
[What worked, what didn't, why it matters]
```

This makes notebooks scannable and professional for portfolio viewing.

---

## Tips for Reviewers

- **Skip to Results**: Each notebook has an "Executive Summary" section at the top
- **Focus on Decisions**: Look for "Why X over Y?" sections explaining tradeoffs
- **Check Visualizations**: All key findings have corresponding plots
- **Reproduce Easily**: All code is parameterized (change paths/models without editing cells)

---

For questions or issues, see main [README.md](../README.md) or open an issue.
