# Benchmark Analysis

Total queries evaluated: 20
Database setup time: 9.89s

## Strengths
- Best Exact Match: Simple RAG (0.7500)
- Best F1 Score: Simple RAG (0.6298)
- Best Faithfulness: Baseline (1.0000)
- Fastest latency/request: Baseline (0.67s)

## Weaknesses
- Baseline: EM=0.0000, F1=0.0000, Reasoning=0.0000, Faithfulness=1.0000
- Simple RAG: EM=0.7500, F1=0.6298, Reasoning=0.5550, Faithfulness=0.6950
- Agentic RAG: EM=0.7000, F1=0.5190, Reasoning=0.6800, Faithfulness=0.7350

## Retrieval
- Retrieval Hit@1: 0.6500
- Retrieval Hit@3: 0.8500
- Retrieval Hit@5: 0.9000
- Retrieval MRR@5: 0.7542
- Retrieval Context F1@5: 0.9136

## Failure Modes
- Abstention overuse: high rates of 'I don't know' responses lower recall.
- Retrieval misses: relevant context not in top-k yields incorrect or empty answers.
- Over-compression in post-processing can remove useful answer tokens.
- Judge-model variance: LLM-as-a-judge introduces scoring noise.
