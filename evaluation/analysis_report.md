# Benchmark Analysis

Total queries evaluated: 1200

## Strengths
- Best Exact Match: Simple RAG (0.1158)
- Best F1 Score: Simple RAG (0.1328)
- Best Faithfulness: Baseline (1.0000)
- Fastest latency/request: Baseline (0.48s)

## Weaknesses
- Baseline: EM=0.0000, F1=0.0007, Reasoning=0.0000, Faithfulness=1.0000
- Simple RAG: EM=0.1158, F1=0.1328, Reasoning=0.1792, Faithfulness=0.9093
- Agentic RAG: EM=0.0575, F1=0.0689, Reasoning=0.1087, Faithfulness=0.9425

## Failure Modes
- Abstention overuse: high rates of 'I don't know' responses lower recall.
- Retrieval misses: relevant context not in top-k yields incorrect or empty answers.
- Over-compression in post-processing can remove useful answer tokens.
- Judge-model variance: LLM-as-a-judge introduces scoring noise.
