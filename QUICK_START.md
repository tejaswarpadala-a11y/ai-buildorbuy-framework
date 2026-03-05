# Quick Start Guide

Get the Bank Rage Classifier running in 15 minutes.

---

## Prerequisites

- Python 3.9+
- pip
- (Optional) GPU for training (CPU works for inference)

---

## Installation

### 1. Clone Repository
```bash
git clone https://github.com/tejaswarpadala/bank-rage-classifier.git
cd bank-rage-classifier
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## Option A: Explore Notebooks (Recommended)

**Best for**: Understanding the full pipeline and methodology

```bash
# Launch Jupyter
jupyter notebook

# Open notebooks/ folder
# Start with 01_data_exploration.ipynb
```

**What you'll see**:
- Data exploration and class distribution
- GenAI labeling approach (GPT-4o teacher model)
- RoBERTa training and hyperparameter search
- Multi-model benchmarking (6+ GenAI models)
- Error analysis and failure patterns

**Time**: 30-60 minutes to review all notebooks

---

## Option B: Run Inference (Fastest)

**Best for**: Testing the model quickly

### Download Pre-trained Model
```bash
# Model weights available on HuggingFace (500MB)
# Coming soon: Direct download link
```

### Classify a Complaint
```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

# Load model
model = AutoModelForSequenceClassification.from_pretrained(
    "tejaswar/bank-rage-roberta-specialist"
)
tokenizer = AutoTokenizer.from_pretrained("roberta-base")

# Example complaint
text = "They charged me a fee I never agreed to and won't refund it!"

# Tokenize
inputs = tokenizer(text, return_tensors="pt", max_length=256, truncation=True)

# Predict
with torch.no_grad():
    outputs = model(**inputs)
    prediction = torch.argmax(outputs.logits, dim=1).item()
    confidence = torch.softmax(outputs.logits, dim=1).max().item()

# Map to label
labels = [
    "Hidden Fees", "Fraud / Security", "Credit Reporting",
    "Loan Servicing", "Account Access", "Process Failure", "Other"
]

print(f"Prediction: {labels[prediction]}")
print(f"Confidence: {confidence:.2f}")
```

**Output**:
```
Prediction: Hidden Fees
Confidence: 0.94
```

---

## Option C: Run Benchmarking

**Best for**: Comparing GenAI models (requires API keys)

### 1. Set Up API Keys
Create `.env` file:
```bash
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=...
```

### 2. Run Benchmark Notebook
```bash
jupyter notebook notebooks/04_genai_benchmark.ipynb
```

**What you'll get**:
- Performance comparison across 6+ models
- Cost-speed-accuracy tradeoff charts
- Reproducibility analysis

**Cost**: ~$5-10 for 1,000 test predictions across all models

---

## File Structure

```
bank-rage-classifier/
├── README.md                    ← Start here
├── docs/
│   ├── tradeoffs.md            ← PM thinking (key differentiator!)
│   ├── architecture.md         ← Technical decisions
│   ├── methodology.md          ← Evaluation framework
│   └── results_analysis.md     ← Performance deep dive
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_genai_labeling.ipynb
│   ├── 03_specialist_training.ipynb
│   ├── 04_genai_benchmark.ipynb
│   └── 05_error_analysis.ipynb
├── data/
│   └── label_codebook.json     ← 7-category schema
├── results/
│   ├── benchmarks/             ← Model comparison data
│   ├── visualizations/         ← Charts and plots
│   └── model_cards/            ← Model documentation
└── requirements.txt
```

---

## Common Issues

### ImportError: transformers
```bash
pip install --upgrade transformers torch
```

### CUDA Out of Memory (Training)
```python
# Reduce batch size in training config
training_args = TrainingArguments(
    per_device_train_batch_size=8,  # was 16
    gradient_accumulation_steps=2
)
```

### API Rate Limits (GenAI Benchmarking)
```python
# Add delay between requests
import time
time.sleep(1)  # Wait 1 second between API calls
```

---

## Next Steps

After getting started:

1. **Read `docs/tradeoffs.md`** - Understand why RoBERTa beats GenAI
2. **Review `results/model_cards/roberta_specialist.md`** - Model details
3. **Explore error analysis** (`notebooks/05_error_analysis.ipynb`) - Learn from failures
4. **Check out Future Work** in main README - RAG, vector DB, active learning roadmap

---

## Getting Help

- **GitHub Issues**: Report bugs or ask questions
- **Email**: tejaswar.padala@gmail.com
- **LinkedIn**: https://www.linkedin.com/in/teja-padala/

---

## Contributing

Contributions welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

Ideas for contributions:
- Spanish language support (40% of CFPB complaints)
- Additional GenAI model benchmarks (Llama 3, Mistral)
- Active learning implementation
- FastAPI deployment template
- Docker containerization

---

Ready to dive in? Start with the notebooks! 🚀
