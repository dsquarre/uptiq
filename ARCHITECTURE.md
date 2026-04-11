# Architecture Overview

## Systems Compared

### 1) Baseline (No Context Control)
- Input: question only
- Context: none
- Output policy: always `I don't know`
- Purpose: clean abstention baseline for calibration

### 2) Simple RAG
- Input: question
- Retrieve: top-k passages from ChromaDB (configurable)
- Generate: one answer pass with strict context-only prompt
- Output: concise answer or `I don't know`

### 3) Agentic RAG 
- Input: question
- Step A: retrieve top-k passages (configurable first-pass k)
- Step B: LLM sufficiency check (`SUFFICIENT` / `INSUFFICIENT`)
- Step C: if insufficient, expand retrieval to top-k passages (configurable second-pass k)
- Step D: generate candidate answer under strict prompt policy
- Step E: grounding verifier (`SUPPORTED` / `UNSUPPORTED`)
- Step F: if unsupported, return `I don't know`; else return candidate

## Retrieval Quality
- **Embedding Model**: `BAAI/bge-large-en-v1.5` (Baai General Embeddings, English, v1.5)
  - Why this model: Specifically trained for semantic search and ranking tasks; consistently outperforms lighter models (MiniLM, MPNet) on multiple benchmarks
  - Trade-off: ~1.3GB download + slightly slower inference vs. 10-15% improvement in Hit@1 and MRR@5 metrics
  - Configurable: Change `embedding_model` in `configs/benchmark.base.yaml` to use alternative models
- **Integration**: Embeddings used for both retrieval pipeline (ChromaDB queries) and retrieval metrics evaluation (Hit@1/3/5, MRR@5, Context F1@5)

## Unified Prompt Policy
- Shared rule for all model calls:
  - use only provided context
  - if unsupported, output exactly `I don't know`
  - output only final answer text

## Generated Artifacts (in `evaluation/` folder)
- Responses: `baseline_responses.txt`, `simple_rag_responses.txt`, `agentic_rag_responses.txt`
- Metrics: `evaluation_results.csv` (pipeline-level), `query_level_scores.csv` (per-query scores)
- Retrieval metrics logged in results: `Retrieval Hit@1`, `Retrieval Hit@3`, `Retrieval Hit@5`, `Retrieval MRR@5`, `Retrieval Context F1@5`
- Failure analysis: `failure_mode_summary.csv`, `failure_cases.csv`, `analysis_report.md`
- Visualizations: `charts/model_comparison.png`, `charts/latency_comparison.png`, `charts/idk_rate.png`, `charts/f1_distribution.png`, `charts/correctness_distribution.png`

## Configuration
- Benchmark settings are loaded from `configs/benchmark.base.yaml` (or a custom file via `--config`).
- CLI flags `--dataset` and `--max-queries` override config values.
- **Embedding Model**: Uses `BAAI/bge-large-en-v1.5` for dense retrieval (configured via `embedding_model` in YAML). This model is optimized for English semantic search and significantly improves retrieval ranking quality (Hit@1, MRR@5) compared to lighter models like `all-MiniLM-L6-v2`. First run downloads ~1.3GB model; subsequent runs use cached version.
