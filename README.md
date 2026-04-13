# Agent Benchmarking Repository

This repository benchmarks three QA pipelines on a 1200-query JSONL dataset:

1. Baseline (no retrieval)
2. Simple RAG (single retrieval pass)
3. Agentic RAG (two-pass retrieval with LLM query rewrite)

The benchmark produces automatic metrics, LLM-as-a-judge metrics, retrieval quality metrics, failure analysis, and charts.

## Repository Structure

- `src/main.py`: end-to-end run pipeline (load data, build DB, run pipelines, write outputs)
- `src/evaluate.py`: EM/F1, retrieval metrics, and LLM judge scoring
- `configs/`: run presets
- `data/`: input datasets
- `evaluation/`: generated outputs
- `ARCHITECTURE.md`, `FLOWCHARTS.md`: design + flow documentation

## Pipeline Behavior (Current Implementation)

### Baseline
- Builds a QA prompt without retrieval context.
- If `answering.strict_context_only: true`, baseline is forced to abstain (`I don't know`).
- If `answering.strict_context_only: false`, baseline can still attempt an answer.

### Simple RAG
- Encodes each question.
- Retrieves `retrieval.simple_top_k` passages from Chroma.
- Builds QA prompt from retrieved context.
- Generates and post-processes final answer.

### Agentic RAG
- First retrieval pass with question embedding (`retrieval.agentic_first_top_k`).
- Prompts the LLM to produce a refined follow-up query from question + first-pass context.
- Second retrieval pass (`retrieval.agentic_second_top_k`) using the generated query.
- Generates and post-processes final answer from second-pass context.

## Dataset

- Format: JSONL entries with `question`, `context`, and `answers`.
- Main files:
  - `data/train_reference_1200.jsonl`
  - `data/val_benchmark_1200.jsonl`
  - `data/val_holdout_1200.jsonl`
- Default run dataset is config-driven.

## Metrics

### Automatic QA Metrics
- Exact Match
- F1 Score
- Latency and latency/request

### Retrieval Metrics
- Retrieval Hit@1
- Retrieval Hit@3
- Retrieval Hit@5
- Retrieval MRR@5
- Retrieval Context F1@5

### LLM Judge Metrics
- Correctness
- Completeness
- Reasoning
- Faithfulness

Judge model in code: `llama3.2:1b`.

## Setup

```bash
pip install -r requirements.txt
```

Pull Ollama models used by configs and judge:

```bash
ollama pull llama3.2:1b
ollama pull mistral:latest
```

## Run

Full run:

```bash
python src/main.py --config configs/experiment1.yaml
```

Low-budget smoke test:

```bash
python src/main.py --config configs/benchmark.low_budget.yaml
```

Override dataset / max queries from CLI:

```bash
python src/main.py --config configs/experiment2.yaml --dataset data/val_benchmark_1200.jsonl --max-queries 100
```

## Config Keys

Each YAML config can set:

- `dataset`
- `max_queries`
- `model_name`
- `embedding_model`
- `retrieval.simple_top_k`
- `retrieval.agentic_first_top_k`
- `retrieval.agentic_second_top_k`
- `answering.strict_context_only`
- `evaluation.run_llm_judge`
- `output.save_charts`
- `output.save_query_level_scores`

Code fallback defaults (if omitted in config):

- `model_name`: `mistral:latest`
- `embedding_model`: `all-MiniLM-L6-v2`

## Presets

- `configs/experiment1.yaml`: `llama3.2:1b`, strict context-only, BGE-large embedding
- `configs/experiment2.yaml`: `mistral:latest`, relaxed context policy, BGE-large embedding
- `configs/benchmark.low_budget.yaml`: 10 queries, `mistral:latest`, MiniLM embedding, relaxed context policy

## Outputs

Written under `evaluation/`:

- `baseline_responses.txt`
- `simple_rag_responses.txt`
- `agentic_rag_responses.txt`
- `evaluation_results.csv`
- `query_level_scores.csv` (optional)
- `analysis_report.md`
- `failure_mode_summary.csv`
- `failure_cases.csv`
- `charts/model_comparison.png`
- `charts/latency_comparison.png`
- `charts/idk_rate.png`
- `charts/f1_distribution.png`
- `charts/correctness_distribution.png`
