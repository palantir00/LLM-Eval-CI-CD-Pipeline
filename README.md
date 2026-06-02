# LLM Eval CI/CD Pipeline

> An automated evaluation pipeline that tests LLM outputs whenever prompts, models, or the
> RAG knowledge base change — just like unit tests run on code changes.

<!-- TODO (Step 11): CI, coverage, license badges -->

## Table of contents

- [Problem](#problem)
- [Architecture](#architecture) <!-- TODO: mermaid diagram (Step 11) -->
- [Measured metrics](#measured-metrics)
- [Quick start](#quick-start)
- [Project structure](#project-structure)
- [Screenshots](#screenshots) <!-- TODO: dashboard screenshots (Step 11) -->
- [What I learned](#what-i-learned) <!-- TODO (Step 11) -->

## Problem

LLM outputs can silently degrade (regress) when you change a prompt, swap the model, or
update the RAG knowledge base. Classic unit tests do not catch this. This project treats
the quality of LLM answers as something you can **measure and gate in CI**.

The demonstration domain is a **generic digital bank / fintech app** (cards, transfers,
fees, KYC, security) — deliberately not tied to any specific brand, so the project stays
universal. The pipeline works the same way for any domain.

## Architecture

<!-- TODO (Step 11): mermaid diagram -->
Golden dataset → Pipeline (RAG + LLM) → Metrics → SLA gate → Storage (SQLite) → Dashboard.

## Measured metrics

- **Hallucination rate** — share of answers containing content unsupported by sources (LLM-as-judge).
- **Answer relevancy** — whether the answer actually addresses the question.
- **Faithfulness** — how faithful the answer is to the provided context.
- **Latency p50 / p95** — response time (percentiles, not the mean).
- **Cost per query** — number of tokens × model pricing.

## Quick start

```bash
# 1. Dependencies (requires uv installed: https://docs.astral.sh/uv/)
uv sync --extra dev

# 2. Environment configuration
cp .env.example .env   # the default "mock" mode works without an API key

# 3. Generate the sample golden dataset
uv run python -m scripts.seed_golden_dataset

# (more commands will appear in the sections below as the project grows)
```

## Project structure

```
config/    # SLA thresholds (thresholds.yaml) and model definitions (models.yaml)
data/      # golden_dataset.jsonl + RAG knowledge base documents
src/
  golden_dataset.py  # Pydantic schema + JSONL validation
  paths.py           # central project paths
  pipeline/          # RAG, LLM client, prompts
  eval/              # metrics, runner, SLA gate
  storage/           # metric storage in SQLite
  dashboard/         # Streamlit dashboard
scripts/   # data seeding, knowledge base ingest
tests/     # pytest tests
```

## Screenshots

<!-- TODO (Step 11) -->

## What I learned

<!-- TODO (Step 11) -->