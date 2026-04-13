# Benchmark Analysis

Total queries evaluated: 10
Database setup time: 7.87s

## Strengths
- Best Exact Match: Simple RAG (0.6000)
- Best F1 Score: Simple RAG (0.3516)
- Best Faithfulness: Agentic RAG (0.7700)
- Fastest latency/request: Baseline (1.98s)

## Weaknesses
- Baseline: EM=0.1000, F1=0.0213, Reasoning=0.5100, Faithfulness=0.5300
- Simple RAG: EM=0.6000, F1=0.3516, Reasoning=0.6000, Faithfulness=0.7600
- Agentic RAG: EM=0.3000, F1=0.1994, Reasoning=0.5600, Faithfulness=0.7700

## Retrieval
- Retrieval Hit@1: 0.5000
- Retrieval Hit@3: 0.7000
- Retrieval Hit@5: 0.8000
- Retrieval MRR@5: 0.6083
- Retrieval Context F1@5: 0.8271

## Failure Modes
- Abstention overuse: high rates of 'I don't know' responses lower recall.
- Retrieval misses: relevant context not in top-k yields incorrect or empty answers.
- Over-compression in post-processing can remove useful answer tokens.
- Judge-model variance: LLM-as-a-judge introduces scoring noise.
