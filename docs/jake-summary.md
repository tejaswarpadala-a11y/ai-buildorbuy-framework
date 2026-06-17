# The Bank Rage Analyzer — One-Page Summary

**Teja Padala** · UNC Kenan-Flagler MBA (Class of 2026) · Project Lead & codebase owner, Team 8 · Applied-AI research project (MBA742, Prof. Daniel M. Ringel)

## The question

For a high-volume text-classification task in financial services, should you call a general-purpose GenAI API for every prediction, or use GenAI once to bootstrap labels and then fine-tune a small specialist you own and run yourself? This project answers it empirically.

## The task

Single-label, 7-class root-cause classification of real CFPB consumer-banking complaint narratives (Hidden Fees, Fraud/Security, Credit Reporting Error, Loan/Mortgage Servicing, Account Access, Process Failure, Other). ~15,000-narrative training pool (12k/3k split, seed 42) plus a locked, human-consensus-labeled 1,000-row holdout. The holdout is heavily imbalanced — the top two classes are ~83% of traffic — so macro-F1 and MCC, not accuracy, are the load-bearing metrics.

## What was built

- A six-model GenAI benchmark across three providers (GPT-4o, GPT-4.1, Claude Sonnet 4, Claude Haiku 4.5, Gemini 2.5 Flash, Gemini 2.5 Flash-Lite), all run on the same holdout with one **identical fixed prompt** — no per-model tuning. Tracked accuracy, macro-F1, MCC, runtime, tokens, and estimated cost, with a reproducibility re-run.
- A **fine-tuned RoBERTa specialist**, trained on GPT-4o-labeled data and evaluated on the same locked holdout.
- Label reliability validated up front: a 200-item, 3-annotator pilot raised Krippendorff's α from 0.3974 to 0.5479 after a codebook refinement (honest, moderate agreement on a genuinely ambiguous taxonomy).

## The result

| | Best GenAI (GPT-4o) | RoBERTa Specialist | Difference |
|---|---|---|---|
| Macro-F1 | 0.7900 | **0.8492** | **+0.0592** |
| Accuracy | 0.9060 | **0.9275** | **+0.0215** |
| MCC | 0.8364 | **0.8634** | **+0.0270** |

The specialist won on every metric, at near-zero marginal inference cost (~15 min holdout inference vs ~4 hrs for the GenAI path) and fully deterministic, in-environment execution. Labeling the full ~15k training pool with GPT-4o cost ~$23.

## The recommendation

Deploy the RoBERTa specialist for production routing and trend monitoring; keep GPT-4o for periodic re-labeling and taxonomy refresh; run monthly human QA/drift checks on the high-risk categories (Fraud, Credit Reporting). No production system was deployed — this is a measured recommendation, not a shipped product.

## Honest scope

Teja was Project Lead and owned the build/codebase for a five-person team; human labeling and adjudication were shared. Prof. Ringel selected the project for an expanded follow-on phase (larger-scale benchmarking, expanded validation) and agreed to support next-phase API costs — a planned next phase, not a completed publication. The public repo is a write-up plus the taxonomy; notebooks, data, keys, and weights are not included (see REPRODUCIBILITY.md).
