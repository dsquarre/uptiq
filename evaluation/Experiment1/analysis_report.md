# Benchmark Analysis

Note: Experiment1 used the earlier strict agentic flow. The current implementation now uses answer-guided second-pass retrieval with merged contexts for Agentic RAG.

Total queries evaluated: 1200
Database setup time: 827.99s

## Strengths
- Best Exact Match: Simple RAG (0.1433)
- Best F1 Score: Simple RAG (0.2382)
- Best Faithfulness: Baseline (0.9996)
- Fastest latency/request: Baseline (0.49s)

## Weaknesses
- Baseline: EM=0.0000, F1=0.0017, Reasoning=0.0012, Faithfulness=0.9996
- Simple RAG: EM=0.1433, F1=0.2382, Reasoning=0.2508, Faithfulness=0.8718
- Agentic RAG: EM=0.0508, F1=0.0864, Reasoning=0.0902, Faithfulness=0.9598

## Failure Modes
- Abstention overuse: high rates of 'I don't know' responses lower recall.
- Retrieval misses: relevant context not in top-k yields incorrect or empty answers.
- Over-compression in post-processing can remove useful answer tokens.
- Judge-model variance: LLM-as-a-judge introduces scoring noise.
