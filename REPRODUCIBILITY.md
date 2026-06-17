# Reproducibility

This document states plainly what this repository contains, what it deliberately does not, and what would be required to rerun the full study. The goal is credibility: the results in the README are real outputs from the submitted project, but this public repo is a **write-up plus representative supporting artifacts**, not a one-command reproduction.

## What is included

- The full project write-up ([README.md](README.md)) with the actual measured results.
- The 7-category root-cause taxonomy and decision rules ([data/label_codebook.json](data/label_codebook.json)) — the codebook used in every model prompt and to label the specialist's training data.
- A one-page forwardable summary ([docs/jake-summary.md](docs/jake-summary.md)).
- The dependency set the original pipeline used ([requirements.txt](requirements.txt)).

## What is not included (and why)

- **The Colab notebooks** (data pull, cleaning, GenAI benchmark, scaled labeling, specialist training/eval). These were authored in the team's Google Drive / Colab environment and are not vendored here.
- **The CFPB data** — the ~15,000-row training/test pool and the locked 1,000-row human-labeled holdout. The raw source is public (CFPB Consumer Complaint Database), but the cleaned, deduplicated, split, and human-adjudicated artifacts are not published here.
- **API keys** for OpenAI, Anthropic, and Google — required to reproduce the GenAI benchmark, never committed.
- **Trained model weights** — the fine-tuned RoBERTa specialist weights are not hosted in this repository, and there is no public hosted endpoint for them.

Because of the above, the code snippets and module paths that appeared in earlier drafts of this repo (e.g. `src/...`, hosted model downloads) have been removed: they implied a runnable package that this repo does not ship.

## What it would take to rerun the full study

1. **Data:** pull the banking-products subset of the CFPB Consumer Complaint Database (2025), deduplicate, and split ~15,000 narratives into train/test (12,000 / 3,000, seed 42). Assemble and human-label a 1,000-row holdout to consensus using the codebook in `data/label_codebook.json`.
2. **GenAI benchmark:** run the six models (GPT-4o, GPT-4.1, Claude Sonnet 4, Claude Haiku 4.5, Gemini 2.5 Flash, Gemini 2.5 Flash-Lite) on the holdout with one identical fixed prompt — a system role defining the classifier, the codebook definitions, and the complaint text, output constrained to exactly one label. Score accuracy, macro-F1, and MCC; include a reproducibility re-run.
3. **Scale labeling:** use the benchmark winner (GPT-4o) to label the ~15k training pool.
4. **Specialist:** fine-tune RoBERTa on the GenAI-labeled pool and evaluate on the same locked holdout.

## Result variance to expect

GenAI outputs vary run-to-run (temperature, model/version drift), so exact leaderboard numbers will shift somewhat on re-run; the relative ordering and the specialist's macro-F1 advantage are the durable findings. The specialist's inference is deterministic.
