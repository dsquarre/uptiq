# Architecture Overview

## Compared Pipelines

### 1) Baseline
- Input: question only
- Retrieval: none
- Generation: QA prompt with empty context
- Policy control:
  - `answering.strict_context_only: true` forces abstention behavior (`I don't know`)
  - `answering.strict_context_only: false` allows best-effort answering
- Post-processing: normalization and lightweight answer cleanup

### 2) Simple RAG
- Input: question
- Retriever: SentenceTransformer embeddings + ChromaDB
- Retrieval: single pass with `retrieval.simple_top_k`
- Generation: one QA call using joined retrieved context
- Post-processing: same normalization and cleanup as baseline

### 3) Agentic RAG
- Input: question
- Step 1 retrieval: top-k using question embedding (`retrieval.agentic_first_top_k`)
- LLM planning step: generates a refined follow-up query from question + first-pass context
- Step 2 retrieval: second pass with `retrieval.agentic_second_top_k`
- Generation: final QA answer from second-pass context
- Post-processing: same normalization and cleanup

## Runtime Components

- Generator model: configured by `model_name` (global in `src/main.py`)
- Retriever embedding model: configured by `embedding_model`
- Vector store: persistent ChromaDB at `./chroma_db`
- Judge model for rubric scoring: `llama3.2:1b`

## Prompting and Output Policy

Shared QA prompt behavior:

- Uses context when present
- If strict mode is on and context is insufficient, expected output is `I don't know`
- If strict mode is off, allows grounded best-effort answering
- Returns only final answer text

All pipeline outputs are normalized through a post-processing chain:

1. Normalize abstentions to `I don't know`
2. Remove repeated question terms from answer span
3. Strip common grammatical stopwords to keep concise final output

## Evaluation Stack

### Automatic Metrics
- Exact Match
- F1 Score
- Latency and latency/request

### Retrieval Metrics
- Hit@1, Hit@3, Hit@5
- MRR@5
- Context F1@5

### LLM Judge Metrics
- Correctness
- Completeness
- Reasoning
- Faithfulness

## Configuration Model

- YAML config is loaded from `--config`
- CLI overrides supported for:
  - `--dataset`
  - `--max-queries`
- Core keys:
  - `dataset`, `max_queries`
  - `model_name`, `embedding_model`
  - retrieval top-k values
  - `answering.strict_context_only`
  - `evaluation.run_llm_judge`
  - output toggles for charts and query-level scores

## Generated Artifacts

Under `evaluation/`:

- Pipeline responses: baseline/simple/agentic text files
- Aggregate metrics: `evaluation_results.csv`
- Per-query metrics: `query_level_scores.csv` (optional)
- Failure analysis: summary CSV + failure case CSV
- Report: `analysis_report.md`
- Plots: quality, latency, abstention rate, score distributions
