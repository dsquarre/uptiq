# Dataset Description

## Source

Primary dataset: **SQuAD v2** via Hugging Face `datasets`.

## Size

- Raw: full SQuAD v2 train split
- Processed benchmark subset: default **1200** samples (configurable, minimum target 1000)

## Preprocessing

Pipeline script: `python -m src.data.prepare_squad`

Steps:

1. Load split from `datasets`.
2. Normalize to canonical fields:
   - `id`, `question`, `context`, `answers`, `is_impossible`
3. Filter malformed rows.
4. Deterministically sample `num_samples` with fixed seed.
5. Write JSONL to `data/processed/`.

## Format

Each JSONL line:

```json
{"id":"...","question":"...","context":"...","answers":["..."],"is_impossible":false}
```

## Notes and Limitations

- SQuAD v2 is English QA; results may not transfer to finance or multi-hop domains.
- Some samples are unanswerable (`is_impossible=true`) and are preserved.
- Retrieval quality depends on context chunking strategy.
