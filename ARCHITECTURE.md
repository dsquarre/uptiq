# Architecture

## Goal

Provide a reproducible benchmark pipeline to compare **Simple RAG** and **Agentic RAG** across 1000+ QA tasks.

## System Components

1. **Data Preparation (`src/data/prepare_squad.py`)**
	- Loads SQuAD v2 from Hugging Face `datasets`.
	- Normalizes records into a canonical JSONL schema.
	- Deterministically samples N records using a fixed seed.

2. **Agents (`src/agents/`)**
	- `simple_rag.py`: single-pass retrieve + answer.
	- `agentic_rag.py`: iterative retrieve-refine loop with bounded steps.

3. **Evaluation (`src/eval/`)**
	- `metrics.py`: EM, F1, retrieval Recall@k, MRR.
	- `judge.py`: rubric-based local judge; optional API fallback.

4. **Pipeline (`src/pipeline/run_benchmark.py`)**
	- Executes `Run -> Collect -> Evaluate -> Compare`.
	- Produces run-level artifacts in `results/<run_name>/`.

5. **Analysis + Visualization**
	- `src/analysis/analyze_failures.py`: failure mode categorization.
	- `src/visualize/plots.py`: score distributions and architecture comparisons.

## Data Contracts

### Input Sample Schema

```json
{
  "id": "string",
  "question": "string",
  "context": "string",
  "answers": ["string", "..."],
  "is_impossible": false
}
```

### Prediction Schema

```json
{
  "id": "string",
  "architecture": "simple_rag|agentic_rag",
  "question": "string",
  "predicted_answer": "string",
  "gold_answers": ["string", "..."],
  "retrieved_contexts": ["string", "..."],
  "latency_ms": 0.0,
  "trace": {
	 "steps": ["retrieve", "answer"]
  }
}
```

### Evaluation Record Schema

```json
{
  "id": "string",
  "architecture": "simple_rag",
  "em": 0,
  "f1": 0.0,
  "recall_at_k": 0.0,
  "mrr": 0.0,
  "judge_correctness": 1,
  "judge_completeness": 1,
  "judge_reasoning": 1
}
```

## Execution Flow

1. **Prepare Data**
2. **Run Both Architectures on Same Samples**
3. **Compute Quantitative Metrics**
4. **Run Judge Evaluation**
5. **Aggregate + Compare + Save Artifacts**

## Reproducibility Guarantees

- Seeded sampling and run naming
- Config-driven parameters
- Versioned output artifacts
- Explicit judge mode (`local`, `hybrid`, `api`)

## Trade-offs

- Low-budget mode prioritizes deterministic local scoring and optional small judge sample.
- Agentic RAG is expected to improve completeness but may increase latency/cost.
