# Benchmark Analysis

Total queries evaluated: 10
Database setup time: 8.15s

## Strengths
- Best Exact Match: Simple RAG (0.6000)
- Best F1 Score: Simple RAG (0.4211)
- Best Faithfulness: Baseline (0.7300)
- Fastest latency/request: Baseline (12.01s)

## Weaknesses
- Baseline: EM=0.2000, F1=0.0284, Reasoning=0.5500, Faithfulness=0.7300
- Simple RAG: EM=0.6000, F1=0.4211, Reasoning=0.5200, Faithfulness=0.5700
- Agentic RAG: EM=0.6000, F1=0.2985, Reasoning=0.5900, Faithfulness=0.6300

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
