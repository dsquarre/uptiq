# Benchmark Report

## 1. Experiment Setup

- Model: `llama3.2:1b` via Ollama
- Embedding model: `all-MiniLM-L6-v2`
- Retriever: ChromaDB
- Dataset: `data/val_benchmark_1200.jsonl` (1200 queries)

Architectures compared:
- Baseline (no context, deterministic abstain)
- Simple RAG (single retrieval + answer)
- Agentic RAG (retrieve -> sufficiency check -> optional expanded retrieval -> answer -> grounding verifier)

## 2. Metrics

Quantitative:
- Exact Match
- F1 Score
- Retrieval Hit Rate@2
- Latency

LLM-as-a-Judge:
- Correctness
- Completeness
- Reasoning
- Faithfulness

## 3. Results

All outputs saved in `evaluation/` folder:
- Pipeline-level metrics: `evaluation/evaluation_results.csv`
- Per-query scores: `evaluation/query_level_scores.csv`
- Analysis artifacts: `evaluation/analysis_report.md`, `evaluation/failure_mode_summary.csv`, `evaluation/failure_cases.csv`

## 5. Trade-offs

- Baseline gives strict abstention control but no answer coverage.
- Simple RAG is cheaper/faster than complex agents.
- Agentic RAG can improve grounding but may increase latency and can over-abstain if verifier is too strict.

## 4. Visualization

Charts saved to `evaluation/charts/`:
- Model comparison chart
- Latency comparison chart
- Abstention rate chart
- F1 score distribution
- Correctness distribution

## 5. Reproducibility

Run full benchmark:

```bash
python src/main.py --dataset data/val_benchmark_1200.jsonl
```
