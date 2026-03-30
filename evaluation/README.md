# Evaluation Protocol

## Quantitative Metrics

- **EM**: normalized string exact match.
- **F1**: token overlap F1 against best gold answer.
- **Recall@k**: any top-k context contains gold answer text.
- **MRR**: rank quality for first relevant context.

## LLM-as-a-Judge Rubric

Score each output 1-5 on:

1. Correctness
2. Completeness
3. Reasoning quality

Judge modes:

- `local`: deterministic heuristic scorer.
- `hybrid`: local primary, API fallback on low-confidence cases.
- `api`: API judge only.

## Aggregation

- Per-sample scores are stored in JSONL.
- Run summaries include mean scores per architecture.
- Optional bootstrap confidence intervals can be added later.

## Reproducibility

- Fixed dataset sampling seed
- Config-driven execution
- Saved per-run artifacts under `results/<run_name>/`
