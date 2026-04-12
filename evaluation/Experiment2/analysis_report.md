# Benchmark Analysis

Total queries evaluated: 1200
Database setup time: 809.65s

## Strengths
- Best Exact Match: Simple RAG (0.2150)
- Best F1 Score: Simple RAG (0.4044)
- Best Faithfulness: Baseline (0.9971)
- Fastest latency/request: Baseline (0.48s)

## Weaknesses
- Baseline: EM=0.0000, F1=0.0017, Reasoning=0.0043, Faithfulness=0.9971
- Simple RAG: EM=0.2150, F1=0.4044, Reasoning=0.4818, Faithfulness=0.7413
- Agentic RAG: EM=0.0758, F1=0.1422, Reasoning=0.1683, Faithfulness=0.9077

## Retrieval
- Retrieval Hit@1: 0.5442
- Retrieval Hit@3: 0.8508
- Retrieval Hit@5: 0.9200
- Retrieval MRR@5: 0.6997
- Retrieval Context F1@5: 0.9462

## Failure Modes
- Abstention overuse: high rates of 'I don't know' responses lower recall.
- Retrieval misses: relevant context not in top-k yields incorrect or empty answers.
- Over-compression in post-processing can remove useful answer tokens.
- Judge-model variance: LLM-as-a-judge introduces scoring noise.
