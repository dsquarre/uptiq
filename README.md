# uptiq

Benchmarking suite for comparing **Simple RAG** vs **Agentic RAG** on 1000+ QA tasks with quantitative metrics and LLM-as-a-Judge evaluation.

## Scope

- Architectures: Simple RAG vs Agentic RAG
- Dataset: SQuAD v2 (deterministic subset generation, default 1200 queries)
- Quant metrics: Exact Match (EM), token-level F1, Recall@k, MRR
- LLM Judge: correctness, completeness, reasoning quality
- Optional metrics: latency and estimated cost

## Quickstart

1. Create environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Prepare dataset subset (1200 samples by default):

```bash
python -m src.data.prepare_squad --output data/processed/squad_v2_1200.jsonl --num-samples 1200 --seed 42
```

3. Run benchmark in low-budget mode:

```bash
python -m src.pipeline.run_benchmark --config configs/benchmark.low_budget.yaml
```

4. Analyze failures and generate plots:

```bash
python -m src.analysis.analyze_failures --predictions results/latest/predictions.jsonl --output results/latest/failures.json
python -m src.visualize.plots --metrics results/latest/summary.json --output-dir results/latest/plots
```

## Reproducibility

- All runs are config-driven through `configs/*.yaml`
- Dataset subset sampling is deterministic via seed
- Run artifacts are stored under `results/<run_name>/`
- Judge mode supports local-only or hybrid fallback

## Repository Layout

- `src/data/`: dataset acquisition and preprocessing
- `src/agents/`: architecture implementations
- `src/eval/`: metrics and LLM judge components
- `src/pipeline/`: run → collect → evaluate → compare orchestration
- `src/analysis/`: failure-mode analysis
- `src/visualize/`: plots and charts
- `data/`: prepared data artifacts and docs
- `evaluation/`: evaluation protocol docs

## Deliverables

- Benchmark pipeline and scripts
- Configs for base and low-budget runs
- Dataset documentation
- Benchmark report template
- Demo script

See:

- `ARCHITECTURE.md`
- `BENCHMARKS.md`
- `DATASET.md`
- `REPORT.md`
- `DEMO.md`