# Data Folder

## Expected Structure

- `raw/` (optional intermediate files)
- `processed/squad_v2_1200.jsonl` (canonical benchmark input)

## Build Process

Run:

```bash
python -m src.data.prepare_squad --output data/processed/squad_v2_1200.jsonl --num-samples 1200 --seed 42
```

## Validation Checklist

- Row count >= 1000
- Required fields exist (`id`, `question`, `context`, `answers`, `is_impossible`)
- IDs are unique
- File is deterministic under same seed
