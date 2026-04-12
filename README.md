# Agent Benchmarking Repository

This repository benchmarks multiple QA agent architectures on a 1200-query dataset and produces quantitative metrics, LLM-as-a-judge metrics, failure analysis, and visualizations.

## Problem Alignment

This repo satisfies the assignment requirements:
- Compare 2 architectures: Simple RAG vs Agentic RAG (plus Baseline control).
- Use large dataset: 1200 queries (`data/val_benchmark_1200.jsonl`).
- Define inputs/outputs/tools in architecture docs.
- Include quantitative + LLM-as-a-judge evaluation.
- Run reproducible pipeline: Run -> Collect -> Evaluate -> Compare.
- Provide analysis artifacts: strengths, weaknesses, failure modes.
- Provide visualization charts (bonus).
- Support config-driven runs via YAML (`--config`) with CLI overrides.

## Repository Structure

- `src/main.py`: end-to-end benchmark pipeline.
- `src/evaluate.py`: evaluation metrics and LLM judge logic.
- `configs/`: benchmark config presets.
- `data/`: dataset docs.
- `evaluation/`: evaluation/reporting docs.
- `README.md`, `ARCHITECTURE.md`, `FLOWCHARTS.md`.

## Architectures Compared

1. **Baseline** (Control): No context → always returns "I don't know" (deterministic abstention).
2. **Simple RAG**: Retrieve top-k passages → generate single answer → post-process.
3. **Agentic RAG**: Retrieve top-k → draft answer → answer-guided second retrieval → merge contexts → generate answer → grounding verifier.

## Dataset

- **Source**: SQuAD-like JSONL format with `question`, `context`, and `answers`.
- **Files**: `data/train_reference_1200.jsonl`, `data/val_benchmark_1200.jsonl` (default), `data/val_holdout_1200.jsonl`.
- **Size**: 1200 queries (meets ≥1000 requirement).
- **Preprocessing**: `src/main.py` parses JSONL, builds a ChromaDB collection using dense embeddings (configurable, default: `BAAI/bge-large-en-v1.5`), and evaluates against the first gold reference answer.

## Evaluation Metrics

**Quantitative:** Exact Match, F1 Score, Retrieval Hit@1/3/5, Retrieval MRR@5, Retrieval Context F1@5, Latency per request.

**LLM-as-a-Judge:** Correctness, Completeness, Reasoning, Faithfulness (scored 0-10 by `llama3.2:1b`).

## Setup

1. Create a Python environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Ensure Ollama is running and the model is available:

```bash
ollama pull llama3.2:1b
```

## Run Benchmark

Full benchmark (1200 queries):

```bash
python src/main.py --config configs/benchmark.base.yaml
```

Quick benchmark (smoke test):

```bash
python src/main.py --config configs/benchmark.base.yaml --max-queries 20
```

Override dataset on top of config:

```bash
python src/main.py --config configs/benchmark.base.yaml --dataset data/val_benchmark_1200.jsonl
```

## Config-Driven Parameters

`configs/benchmark.base.yaml` controls:
- `dataset`, `max_queries`
- `model_name`
- `embedding_model` (default: `BAAI/bge-large-en-v1.5` — optimized for English retrieval quality)
- retrieval top-k values:
  - `retrieval.simple_top_k`
  - `retrieval.agentic_first_top_k`
  - `retrieval.agentic_second_top_k`
- answer policy:
  - `answering.strict_context_only`
    - `true`: only answer when the prompt context supports it
    - `false`: allow a best-effort draft answer before final verification
- evaluation toggle:
  - `evaluation.run_llm_judge`
- output toggles:
  - `output.save_charts`
  - `output.save_query_level_scores`

## Outputs

All outputs saved in `evaluation/` folder:

- Responses:
  - `evaluation/baseline_responses.txt`
  - `evaluation/simple_rag_responses.txt`
  - `evaluation/agentic_rag_responses.txt`
- Aggregate metrics:
  - `evaluation/evaluation_results.csv`
- Retrieval quality metrics in results:
  - `Retrieval Hit@1`, `Retrieval Hit@3`, `Retrieval Hit@5`
  - `Retrieval MRR@5`, `Retrieval Context F1@5`
- Agentic RAG now performs answer-guided second-pass retrieval and merges both retrieval passes before the final answer.
- Query-level distribution metrics:
  - `evaluation/query_level_scores.csv`
- Analysis:
  - `evaluation/analysis_report.md`
  - `evaluation/failure_mode_summary.csv`
  - `evaluation/failure_cases.csv`
- Charts:
  - `evaluation/charts/model_comparison.png`
  - `evaluation/charts/latency_comparison.png`
  - `evaluation/charts/idk_rate.png`
  - `evaluation/charts/f1_distribution.png`
  - `evaluation/charts/correctness_distribution.png`

## Submission Notes

- Public repo option: share GitHub URL.
- Private repo option: add `uptiq-chaitanya` as collaborator.
- Include your demo/video link in the report submission form or project notes.
