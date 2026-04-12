# Additional Diagrams

## Agent Flow Diagram

```mermaid
flowchart TD
    A[Question] --> B[Retrieve top-k]
    B --> C{Agentic?}
    C -->|No| D[Simple RAG Answer]
    C -->|Yes| E[Draft Answer from First Context]
    E --> F[Answer-Guided Second Retrieval]
    F --> G[Merge Contexts]
    G --> H[Final Answer Generation]
    H --> I[Grounding Verifier]
    I -->|Supported| J[Final Answer]
    I -->|Unsupported| K[I don't know]
```

## Evaluation Flow Diagram

```mermaid
flowchart LR
    CFG[Load YAML Config] --> R[Run]
    R[Run] --> C[Collect Responses]
    C --> E[Evaluate Metrics: EM/F1 + LLM Judge + Retrieval Hit@k/MRR/ContextF1]
    E --> P[Compare Pipelines]
    P --> A[Analyze Failures]
    A --> V[Visualize]
```

## Experiment2 Note

Experiment2 enables `answering.strict_context_only: false` so the agentic pipeline can draft a best-effort answer before the verification pass.
