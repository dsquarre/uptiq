# Benchmark Report

## 1) Experiment Setup

- Dataset: SQuAD v2 deterministic subset
- Query count: 1200 (target >=1000)
- Architectures compared: Simple RAG vs Agentic RAG
- Judge mode: local (or hybrid)
- Config files: `configs/benchmark.base.yaml`, `configs/benchmark.low_budget.yaml`

## 2) Metrics

### Quantitative
- Exact Match (EM)
- Token F1
- Recall@k
- MRR

### LLM-as-a-Judge
- Correctness (1-5)
- Completeness (1-5)
- Reasoning quality (1-5)

## 3) Results (Template)

| Architecture | EM | F1 | Recall@k | MRR | Judge Correctness | Judge Completeness | Judge Reasoning |
|---|---:|---:|---:|---:|---:|---:|---:|
| Simple RAG | - | - | - | - | - | - | - |
| Agentic RAG | - | - | - | - | - | - | - |

## 4) Insights

### Strengths
- Simple RAG:
- Agentic RAG:

### Weaknesses
- Simple RAG:
- Agentic RAG:

### Failure Modes
- Retrieval miss
- Partial answer
- Hallucination
- Judge disagreement

## 5) Trade-offs

- Quality vs latency
- Quality vs cost
- Simplicity vs controllability

## 6) Recommendation

- Suggested architecture for low-budget production-like QA:
- Suggested architecture for highest-quality QA:
