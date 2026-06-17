# Quick Start

This repository is a **portfolio write-up** of a completed applied-AI research project (the Bank Rage Analyzer), plus the supporting taxonomy. It is not a runnable end-to-end benchmark — the notebooks, raw CFPB data, API keys, and trained model weights are not included. See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for the full inclusion/exclusion list.

If you are reviewing this as a portfolio piece, start here:

1. **[README.md](README.md)** — the full write-up: the build-vs-buy question, method, GenAI leaderboard, specialist comparison, cost economics, and the deployment recommendation.
2. **[docs/jake-summary.md](docs/jake-summary.md)** — a one-page summary of the project and results.
3. **[data/label_codebook.json](data/label_codebook.json)** — the 7-category root-cause taxonomy and decision rules used in every prompt and for the specialist labels.
4. **[REPRODUCIBILITY.md](REPRODUCIBILITY.md)** — what would be required to rerun the full study from scratch.

---

## What the original pipeline looked like

The full project ran in Google Colab notebooks against the CFPB API and the OpenAI / Anthropic / Google APIs, then fine-tuned a RoBERTa specialist on Hugging Face weights. At a high level:

1. **Data pull & cleaning** — pull the banking-products subset of the CFPB Consumer Complaint Database, deduplicate, and split ~15,000 narratives into train/test (12,000 / 3,000, seed 42), plus a locked 1,000-row human-labeled holdout.
2. **GenAI benchmark** — run six foundation models (GPT-4o, GPT-4.1, Claude Sonnet 4, Claude Haiku 4.5, Gemini 2.5 Flash, Gemini 2.5 Flash-Lite) on the holdout with one identical fixed prompt; score accuracy, macro-F1, and MCC.
3. **Scale labeling** — use the benchmark winner (GPT-4o) to label the ~15k training pool.
4. **Fine-tune & evaluate** — fine-tune RoBERTa on those labels and evaluate on the same locked holdout.

The dependency set that pipeline used is captured in [requirements.txt](requirements.txt).

---

## Reproducing the environment

If you intend to rebuild the pipeline (you will need to supply your own data and API keys — see [REPRODUCIBILITY.md](REPRODUCIBILITY.md)):

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

API keys for the benchmark would be supplied via environment variables (e.g. an untracked `.env`):

```bash
OPENAI_API_KEY=...
ANTHROPIC_API_KEY=...
GOOGLE_API_KEY=...
```

---

## Getting Help

- **Email**: tejaswar.padala@gmail.com
- **LinkedIn**: https://www.linkedin.com/in/teja-padala/
