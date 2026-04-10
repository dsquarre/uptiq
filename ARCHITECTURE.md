# Architecture Overview

## Systems Compared

### 1) Baseline (No Context Control)
- Input: question only
- Context: none
- Output policy: always `I don't know`
- Purpose: clean abstention baseline for calibration

### 2) Simple RAG
- Input: question
- Retrieve: top-2 passages from ChromaDB
- Generate: one answer pass with strict context-only prompt
- Output: concise answer or `I don't know`

### 3) Agentic RAG (Improved)
- Input: question
- Step A: retrieve top-2 passages
- Step B: LLM sufficiency check (`SUFFICIENT` / `INSUFFICIENT`)
- Step C: if insufficient, expand retrieval to top-5 passages
- Step D: generate candidate answer under strict prompt policy
- Step E: grounding verifier (`SUPPORTED` / `UNSUPPORTED`)
- Step F: if unsupported, return `I don't know`; else return candidate

## Why Agentic Should Outperform Old Version
- Removes brittle yes/no gating that previously blocked answer generation too early.
- Adds fallback expanded retrieval when initial context is weak.
- Adds grounding verification to reduce hallucinations.

## Unified Prompt Policy
- Shared rule for all model calls:
  - use only provided context
  - if unsupported, output exactly `I don't know`
  - output only final answer text

## Generated Artifacts (in `evaluation/` folder)
- Responses: `baseline_responses.txt`, `simple_rag_responses.txt`, `agentic_rag_responses.txt`
- Metrics: `evaluation_results.csv` (pipeline-level), `query_level_scores.csv` (per-query scores)
- Failure analysis: `failure_mode_summary.csv`, `failure_cases.csv`, `analysis_report.md`
- Visualizations: `charts/model_comparison.png`, `charts/latency_comparison.png`, `charts/idk_rate.png`, `charts/f1_distribution.png`, `charts/correctness_distribution.png`
