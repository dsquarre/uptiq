# Dataset Description

## Source

This benchmark uses JSONL QA datasets in SQuAD-like format with fields:
- `question`
- `context`
- `answers` (first answer used as reference)

Files:
- `data/train_reference_1200.jsonl`
- `data/val_benchmark_1200.jsonl` (default benchmark split)
- `data/val_holdout_1200.jsonl` (optional holdout)

## Size

- Benchmark split size: 1200 queries.
- Meets the assignment minimum requirement (>= 1000 queries).

## Preprocessing

Pipeline preprocessing in `src/main.py`:
1. Parse JSONL and extract question, context, answer.
2. Build ChromaDB collection from contexts.
3. Encode contexts and queries with `all-MiniLM-L6-v2`.
4. For evaluation, use first gold answer as reference.

## Notes

- `--max-queries` can be used for quick tests.
- Final benchmark should run without `--max-queries` for full 1200-query evaluation.
