# The Bank Rage Analyzer: Build-vs-Buy for Complaint Root-Cause Classification

**An applied-AI study comparing GenAI APIs against a fine-tuned specialist on a real financial-services classification task — with a deployment recommendation grounded in accuracy, cost, latency, and reproducibility.**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**Project Lead & codebase owner: Teja Padala** — UNC Kenan-Flagler MBA (Class of 2026), applied-AI research project (MBA742, Prof. Daniel M. Ringel). Team 8 (5 members); Teja led scope, owned the build/codebase, and ran the head-to-head model evaluation. Human labeling and adjudication were a shared team effort.

> **What this repository is:** a portfolio write-up plus the supporting taxonomy and methodology of a completed applied-AI research project. The notebooks, raw CFPB data, API keys, and trained model weights are **not** included (see [REPRODUCIBILITY.md](REPRODUCIBILITY.md)). All numbers below are the actual results from the submitted project, not illustrative placeholders.

---

## The Question

Financial institutions receive enormous volumes of consumer complaints. Buried in those free-text narratives is the operational root cause of each failure — fraud, a credit-reporting error, a servicing breakdown, a bureaucratic dead-end. Classifying that root cause at scale is the bottleneck.

The build-vs-buy question for any team facing this:

> **Should you call a general-purpose GenAI API for every classification, or use GenAI once to bootstrap labels and then fine-tune a small specialist you own and run yourself?**

This project answers it empirically on a single-label, 7-class root-cause classification task over real CFPB consumer-banking complaints — benchmarking six foundation models across three providers and a fine-tuned RoBERTa specialist on the same locked, human-labeled holdout.

---

## Headline Result

On a 1,000-narrative, human-consensus-labeled holdout (heavily imbalanced, reflecting real complaint traffic):

| | Best GenAI (GPT-4o) | RoBERTa Specialist | Difference |
|---|---|---|---|
| **Macro-F1** | 0.7900 | **0.8492** | **+0.0592** |
| **Accuracy** | 0.9060 | **0.9275** | **+0.0215** |
| **MCC** | 0.8364 | **0.8634** | **+0.0270** |

The fine-tuned specialist beat the best foundation model on every metric — most importantly on **macro-F1**, the metric that matters under heavy class imbalance. It did so at near-zero marginal inference cost and ~15-minute holdout inference time versus ~4 hours for the GenAI path.

**Recommendation:** deploy the RoBERTa specialist for production routing and trend monitoring; keep GPT-4o as a periodic re-labeling / taxonomy-refresh tool, with monthly human QA on the high-risk categories (Fraud, Credit Reporting). No production system was deployed — this is a recommendation backed by measured results.

---

## The Task

**Single-label, 7-class root-cause classification.** Each complaint narrative is assigned exactly one primary operational root cause, aligned to how a bank assigns internal ownership. Multi-issue narratives are resolved by deterministic precedence rules (a Fraud override and a primary-harm principle).

| Label | Operational meaning |
|---|---|
| Hidden Fees | Undisclosed, surprise, or confusing charges the customer didn't expect |
| Fraud / Security Issue | Unauthorized transactions, identity theft, account takeover, attempted fraud |
| Credit Reporting Error | Incorrect tradeline data, missing dispute markers, wrong balances |
| Loan / Mortgage Servicing Issue | Misapplied payments, escrow errors, modification/servicing disputes |
| Account Access / Administration | Lockouts, closures/freezes, ID-verification loops (non-fraud) |
| Process Failure / Red Tape | Endless transfers, bureaucratic loops, dropped tickets, unresolved disputes |
| Other | No clear operational root cause / ambiguous — used sparingly |

The full taxonomy with decision rules lives in [`data/label_codebook.json`](data/label_codebook.json).

---

## The Data

- **Source:** CFPB Consumer Complaint Database (2025), banking-products subset, pulled via the public API.
- **Training/test pool:** ~15,000 narratives, split 12,000 / 3,000 with a fixed seed (42).
- **Holdout (gold):** 1,000 narratives — locked, human-consensus-labeled, never used for tuning.
- **Class distribution is heavily imbalanced** (this is real complaint traffic, not a balanced benchmark):

| Label | Share of holdout |
|---|---|
| Credit Reporting Error | 59.6% |
| Fraud / Security | 23.7% |
| Process Failure | 5.0% |
| Other | 4.6% |
| Account Access | 2.9% |
| Loan / Mortgage Servicing | 2.8% |
| Hidden Fees | 1.4% |

Because the top two labels are ~83% of the holdout, accuracy alone is misleading — macro-F1 and MCC are the load-bearing metrics throughout.

---

## Label Reliability (Measured, Not Asserted)

Label quality was validated with a 200-item pilot labeled independently by three annotators, scored with **Krippendorff's alpha**. After a codebook refinement (v1 → v2), targeting the known ambiguity clusters (Fraud ↔ Credit Reporting, Credit Reporting ↔ Process Failure, Account Access ↔ Fraud):

| Metric | v1 | v2 |
|---|---|---|
| Krippendorff's α | 0.3974 | 0.5479 |
| Full three-way agreement | 50.0% | 63.0% |
| 2-vs-1 splits | 44.0% | 30.5% |

These are honest, moderate-agreement numbers for a genuinely ambiguous taxonomy. The point of reporting them is methodological transparency — the codebook refinement measurably improved consistency, and the residual disagreement is exactly why a locked, consensus-adjudicated holdout was used for evaluation.

---

## Method

A two-phase applied-AI pipeline:

```
GenAI benchmark (6 models, 3 providers)  ──>  pick winner (GPT-4o)
        │                                              │
        │ identical fixed prompt                       │ label ~15k training pool
        │ 1,000 locked holdout                         ▼
        ▼                                       Fine-tune RoBERTa specialist
   accuracy / macro-F1 / MCC                           │
   runtime / tokens / cost                             ▼
   reproducibility re-run                  Evaluate on the SAME locked holdout
                                                       │
                                                       ▼
                                    Head-to-head: buy GPT vs build a specialist
```

**The binding scientific constraint:** every model saw an *identical, fixed prompt* — a system role defining the classifier, the codebook definitions, and the complaint text, with output constrained to exactly one label and no per-model tuning. Per-model runs tracked predictions, runtime, tokens, estimated API cost, and a reproducibility re-run.

GPT-4o (the benchmark winner) was then used to label the ~15k training pool, and a RoBERTa classifier was fine-tuned on those labels and evaluated on the same locked 1,000-row human holdout — the apples-to-apples build-vs-buy test.

---

## GenAI Leaderboard (holdout, n=1,000)

Six models, three providers, identical prompt:

| Model | Provider | Accuracy | Macro-F1 | MCC |
|---|---|---|---|---|
| **GPT-4o** | OpenAI | **0.906** | **0.792** | **0.836** |
| GPT-4.1 | OpenAI | 0.826 | 0.673 | 0.695 |
| Claude Sonnet 4 | Anthropic | 0.796 | 0.632 | 0.640 |
| Gemini 2.5 Flash | Google | 0.780 | 0.662 | 0.432 |
| Claude Haiku 4.5 | Anthropic | 0.774 | 0.529 | 0.566 |
| Gemini 2.5 Flash-Lite | Google | 0.712 | 0.604 | 0.461 |

GPT-4o won on accuracy, macro-F1, and MCC, and was selected as the labeler for the training pool.

---

## Cost & Operational Economics

Cost to label the full ~15k training pool, by model (the "buy" side of the decision):

| Model | Holdout test cost | Cost / item | ~15k-row label cost |
|---|---|---|---|
| **GPT-4o** | **$7.26** | **$0.0087** | **$23.38** |
| Gemini 2.5 Flash-Lite | $0.52 | $0.0034 | $51.20 |
| GPT-4.1 | $3.61 | $0.0036 | $54.43 |
| Claude Haiku 4.5 | $1.53 | $0.0038 | $57.58 |
| Claude Sonnet 4 | $4.27 | $0.0049 | $73.49 |
| Gemini 2.5 Flash | $1.64 | $0.0109 | $164.00 |

Operationally:

- **GenAI path:** ~4 hours to run the holdout; cost recurs on every inference; subject to model/version drift.
- **Specialist path:** ~15 minutes for the holdout, near-zero marginal cost after a one-time training run, deterministic and reproducible, and runnable inside the bank's own environment (data sovereignty).

This is the core build-vs-buy economics: GenAI is cheap and fast to *bootstrap* labels, but a small owned specialist wins on recurring cost, latency, determinism, and control once the schema is stable.

---

## When to Build vs Buy (the PM takeaway)

**Use a fine-tuned specialist when:** volume is high, the schema is stable, consistency/auditability matters (regulatory), marginal cost and latency matter, and you can run it in your own environment.

**Use GenAI APIs when:** volume is low, the schema changes often, you need explanations or multi-task flexibility, or you are bootstrapping training data.

**The hybrid this project recommends:** use GPT-4o to generate silver-grade training labels, fine-tune and deploy the specialist for day-to-day routing and KPI dashboards, and bring GPT-4o back periodically for re-labeling and taxonomy refresh — with monthly human QA/drift checks on the high-risk categories.

---

## Repository Contents

This repo is a **portfolio write-up plus representative supporting artifacts**, not an end-to-end reproducible benchmark. See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for exactly what is and isn't included and what would be needed to rerun the full study.

```
.
├── README.md                 ← this write-up
├── REPRODUCIBILITY.md         ← what's included / excluded / how to rerun
├── docs/
│   └── jake-summary.md        ← one-page forwardable summary
├── data/
│   └── label_codebook.json    ← the 7-category taxonomy + decision rules
├── requirements.txt           ← dependencies the original pipeline used
├── CONTRIBUTING.md
└── LICENSE                    ← MIT
```

---

## Follow-On Research

Teja reports that Prof. Ringel selected this project for an expanded follow-on phase and agreed to support next-phase API costs. Planned (not yet executed) next steps include: expanded human validation, larger-scale benchmarking, and broader model coverage. This is a planned next phase, not a completed or published collaboration.

---

## About

**Teja Padala** — UNC Kenan-Flagler MBA (Class of 2026). Project Lead and codebase owner for this Team 8 applied-AI capstone.

- LinkedIn: https://www.linkedin.com/in/teja-padala/
- Email: tejaswar.padala@gmail.com

---

## Acknowledgments

- **CFPB** for the public Consumer Complaint Database
- **OpenAI, Anthropic, Google** for the benchmarked APIs
- **Hugging Face** for the RoBERTa weights used in the specialist
- **Prof. Daniel M. Ringel** and the UNC Kenan-Flagler MBA742 course
- **Team 8** for the shared human-labeling and adjudication effort

---

## License

[MIT](LICENSE) © 2026 Teja Padala
