# Benchmark Specification

## Goal
Compare 2 agent architectures on a QA benchmark (1200 queries) with rigorous quantitative and LLM-as-a-judge evaluation.

## Dataset
- **File**: `data/val_benchmark_1200.jsonl`
- **Size**: 1200 queries (meets ≥1000 requirement)
- **Format**: SQuAD-like JSONL with `question`, `context`, `answers` fields
- **Optional holdout**: `data/val_holdout_1200.jsonl`

## Architectures

1. **Baseline** (Control): No context → always returns "I don't know" (deterministic abstention)
2. **Simple RAG**: Retrieve top-2 passages → generate single answer → post-process
3. **Agentic RAG**: Retrieve top-2 → sufficiency check → (optional) expanded retrieval top-5 → generate answer → grounding verifier

### Inputs
- `question`: natural language QA query
- `context`: retrieved passages from ChromaDB

### Outputs
- concise answer string
- exact fallback: `I don't know` when unsupported

### Prompting and Testing Scheme
- All model prompts use one shared policy: answer only from context, otherwise output exactly `I don't know`.
- Baseline uses no context by design and is forced to abstain 100% (`I don't know`) to create a clean control condition.
- Simple RAG and Agentic RAG both follow identical abstention policy so comparisons are fair.
- Agentic adds a verifier stage that rejects unsupported answers and converts them to `I don't know`.

### Tools
- Vector store retrieval: ChromaDB
- Embeddings: sentence-transformers (`all-MiniLM-L6-v2`)
- LLM generation and judging: Ollama (`llama3.2:1b`)

## Metrics
Quantitative:
- Exact Match
- F1 Score
- Retrieval Hit Rate @2
- Latency per request

LLM-as-a-Judge:
- Correctness
- Completeness
- Reasoning
- Faithfulness

## Reproducible Pipeline
```
Run → Collect → Evaluate → Compare → Analyze → Visualize
```
1. **Run**: `python src/main.py --dataset data/val_benchmark_1200.jsonl`
2. **Collect**: response files saved to `evaluation/`
3. **Evaluate**: metrics computed in real-time
4. **Compare**: pipeline aggregates saved to `evaluation/evaluation_results.csv` and `evaluation/query_level_scores.csv`
5. **Analyze**: failure analysis in `evaluation/failure_*.csv` and `evaluation/analysis_report.md`
6. **Visualize**: charts in `evaluation/charts/`
