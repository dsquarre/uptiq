# Flowcharts

## End-to-End Benchmark Flow

```mermaid
flowchart LR
    A[Parse CLI Args] --> B[Load YAML Config]
    B --> C[Resolve Dataset + Limits]
    C --> D[Load JSONL Questions Context Answers]
    D --> E[Build or Reuse Chroma Collection]
    E --> F[Compute Retrieval Metrics]
    F --> G[Run Baseline]
    G --> H[Run Simple RAG]
    H --> I[Run Agentic RAG]
    I --> J[Evaluate EM F1 Judge Scores]
    J --> K[Write CSV Reports and Response Files]
    K --> L[Write Failure Analysis]
    L --> M[Render Charts]
```

## Pipeline Comparison Flow

```mermaid
flowchart TD
    Q[Question]

    Q --> B1[Baseline: No Retrieval]
    B1 --> B2[QA Prompt with Empty Context]
    B2 --> B3[Generate + Post-process]

    Q --> S1[Simple RAG: Embed Question]
    S1 --> S2[Retrieve Top-K]
    S2 --> S3[QA Prompt from Retrieved Context]
    S3 --> S4[Generate + Post-process]

    Q --> A1[Agentic RAG: First Retrieval]
    A1 --> A2[LLM Query Rewrite]
    A2 --> A3[Second Retrieval]
    A3 --> A4[QA Prompt from Second-Pass Context]
    A4 --> A5[Generate + Post-process]
```

## Agentic RAG Internal Flow

```mermaid
flowchart LR
    Q[Input Question] --> E1[Encode Question]
    E1 --> R1[Retrieve First Top-K]
    R1 --> P1[Prompt LLM to Rewrite Query]
    P1 --> R2[Retrieve Second Top-K]
    R2 --> P2[Build Final QA Prompt]
    P2 --> G[Generate Final Answer]
    G --> PP[Post-process Output]
```

## Config Presets

- `configs/experiment1.yaml`: `llama3.2:1b`, BGE-large embedding, strict context-only.
- `configs/experiment2.yaml`: `mistral:latest`, BGE-large embedding, relaxed context policy.
- `configs/benchmark.low_budget.yaml`: 10-query smoke test, MiniLM embedding, relaxed context policy.
