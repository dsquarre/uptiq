# Demo Script (3-5 minutes)

## Goal

Show side-by-side benchmark behavior for Simple RAG and Agentic RAG.

## Pre-demo Commands

```bash
python -m src.data.prepare_squad --output data/processed/squad_v2_1200.jsonl --num-samples 1200 --seed 42
python -m src.pipeline.run_benchmark --config configs/benchmark.low_budget.yaml
python -m src.analysis.analyze_failures --predictions results/latest/predictions.jsonl --output results/latest/failures.json
python -m src.visualize.plots --metrics results/latest/summary.json --output-dir results/latest/plots
```

## Flow

1. Introduce objective and dataset.
2. Show one sample query output from each architecture.
3. Show aggregate metrics table (`results/latest/summary.json`).
4. Show one clear failure case category.
5. Show plot(s) for quick comparison.
6. Conclude with architecture trade-offs.

## Suggested Example Cases

- Easy factual answer (both should succeed).
- Context-heavy question (agentic may perform better).
- Ambiguous case (potential failure mode).
