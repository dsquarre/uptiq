# Benchmark Analysis

Total queries evaluated: 3

## Strengths
- Best Exact Match: Baseline (0.0000)
- Best F1 Score: Baseline (0.0000)
- Best Faithfulness: Baseline (1.0000)
- Fastest latency/request: Baseline (0.00s)

## Weaknesses
- Baseline: EM=0.0000, F1=0.0000, Reasoning=0.0000, Faithfulness=1.0000
- Simple RAG: EM=0.0000, F1=0.0000, Reasoning=0.0000, Faithfulness=1.0000
- Agentic RAG: EM=0.0000, F1=0.0000, Reasoning=0.0000, Faithfulness=1.0000

## Failure Modes
- Abstention overuse: high rates of 'I don't know' responses lower recall.
- Retrieval misses: relevant context not in top-k yields incorrect or empty answers.
- Over-compression in post-processing can remove useful answer tokens.
- Judge-model variance: LLM-as-a-judge introduces scoring noise.
