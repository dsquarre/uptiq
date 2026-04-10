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

## Repository Structure

- `src/main.py`: end-to-end benchmark pipeline.
- `src/evaluate.py`: evaluation metrics and LLM judge logic.
- `configs/`: benchmark config presets.
- `data/`: dataset docs.
- `evaluation/`: evaluation/reporting docs.
- `README.md`, `ARCHITECTURE.md`, `BENCHMARK_SPEC.md`, `DATASET.md`, `REPORT.md`, `ANY_OTHER_DIAGRAMS.md`, `DEMO_VIDEO.md`.

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
python src/main.py --dataset data/val_benchmark_1200.jsonl
```

Quick benchmark (smoke test):

```bash
python src/main.py --dataset data/val_benchmark_1200.jsonl --max-queries 20
```

## Outputs

All outputs saved in `evaluation/` folder:

- Responses:
  - `evaluation/baseline_responses.txt`
  - `evaluation/simple_rag_responses.txt`
  - `evaluation/agentic_rag_responses.txt`
- Aggregate metrics:
  - `evaluation/evaluation_results.csv`
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
- Include demo link in `DEMO_VIDEO.md`.
