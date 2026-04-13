# Benchmark Analysis

Total queries evaluated: 5
Database setup time: 7.92s

## Strengths
- Best Exact Match: Agentic RAG (0.8000)
- Best F1 Score: Simple RAG (0.5889)
- Best Faithfulness: Simple RAG (1.0000)
- Fastest latency/request: Baseline (1.39s)

## Weaknesses
- Baseline: EM=0.0000, F1=0.0000, Reasoning=0.2200, Faithfulness=0.5600
- Simple RAG: EM=0.6000, F1=0.5889, Reasoning=0.4200, Faithfulness=1.0000
- Agentic RAG: EM=0.8000, F1=0.4944, Reasoning=0.7400, Faithfulness=0.7000

## Retrieval
- Retrieval Hit@1: 0.6000
- Retrieval Hit@3: 1.0000
- Retrieval Hit@5: 1.0000
- Retrieval MRR@5: 0.7667
- Retrieval Context F1@5: 1.0000

## Failure Modes
- Abstention overuse: high rates of 'I don't know' responses lower recall.
- Retrieval misses: relevant context not in top-k yields incorrect or empty answers.
- Over-compression in post-processing can remove useful answer tokens.
- Judge-model variance: LLM-as-a-judge introduces scoring noise.
