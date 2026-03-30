# Benchmarks

## Objective

Compare two architectures on the same 1000+ QA queries:

- **Simple RAG**
- **Agentic RAG**

## Experiment Matrix

| Experiment | Architecture | Top-k | Max Agent Steps | Judge Mode |
|---|---|---:|---:|---|
| baseline_simple | simple_rag | 3 | 1 | local |
| baseline_agentic | agentic_rag | 3 | 3 | local |
| hybrid_judge_compare | simple_rag + agentic_rag | 3 | 1/3 | hybrid |

## Metrics

### Quantitative

- **Exact Match (EM)**: strict normalized match against any gold answer.
- **Token F1**: overlap-based precision/recall F1.
- **Recall@k**: whether a context containing gold answer appears in retrieved top-k.
- **MRR**: reciprocal rank of first relevant retrieval.

### LLM-as-a-Judge

Rubric dimensions (1-5 each):

- Correctness
- Completeness
- Reasoning quality

## Protocol

1. Prepare deterministic dataset subset (`seed=42`).
2. Run both architectures on identical samples.
3. Compute quantitative metrics on all samples.
4. Run judge on full set (or sampled subset in low-budget mode).
5. Aggregate by architecture.

## Run Naming

Suggested format:

`<date>-<dataset>-<num_samples>-<judge_mode>-<profile>`

Example:

`2026-03-30-squadv2-1200-local-low_budget`
