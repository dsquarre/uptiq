import json
import re
import argparse
import os
import yaml
from sentence_transformers import SentenceTransformer
import chromadb
import ollama
import time
import csv
import matplotlib.pyplot as plt
from evaluate import evaluate_pipeline, calculate_retrieval_metrics

MODEL_NAME = "llama3.2:1b"


def build_qa_prompt(context_text, question, force_idk=False, strict_context_only=True):
    if strict_context_only:
        rules = (
            "You are a strict QA evaluator. "
            "Use only the given Context. "
            "If the Context is missing or does not support the answer, output exactly: I don't know. "
            "Output only the final answer text."
        )
    else:
        rules = (
            "You are a helpful QA assistant. "
            "Use the Context when it is relevant, but do not refuse to answer by default. "
            "If the Context does not contain the answer, use your best grounded answer when possible; "
            "otherwise output exactly: I don't know. "
            "Output only the final answer text."
        )
    if force_idk:
        rules += " No context is available in this task. Only valid answer is 'I don't know.'"

    return (
        f"Instruction: {rules}\n\n"
        f"Context:\n{context_text if context_text else '[NO CONTEXT]'}\n\n"
        f"Question: {question}\n"
        "Answer:"
    )


def normalize_idk(response):
    text = response.strip().lower()
    idk_patterns = [
        r"^i\s*do\s*not\s*know[\.!]?$",
        r"^i\s*don't\s*know[\.!]?$",
        r"^unknown[\.!]?$",
        r"^not\s+enough\s+information[\.!]?$",
    ]
    for pattern in idk_patterns:
        if re.match(pattern, text):
            return "I don't know"
    return response.strip()


def llm_generate(prompt):
    response = ollama.chat(model=MODEL_NAME, messages=[{"role": "user", "content": prompt}])
    return response['message']['content'].replace('\n', ' ').strip()


def load_json(file_path, max_queries=None):
    questions,context,answers = [],[],[]
    with open(file_path,'r',encoding='utf-8') as f:
        for line in f:
            try:
                json_obj = json.loads(line)
                questions.append(json_obj['question'])
                context.append(json_obj['context'])
                answers.append(json_obj['answers'][0])
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e}")
    if max_queries is not None:
        return questions[:max_queries], context[:max_queries], answers[:max_queries]
    return questions, context, answers

def db(context, model):
    client = chromadb.PersistentClient(path="./chroma_db")
    try:
        collection = client.get_collection(name="vector")
        if collection is not None:
            print("Using existing ChromaDB collection.")
            return collection
    except Exception:
        pass
    print('Creating new ChromaDB collection.')
    collection = client.create_collection(name="vector")
    
    embeddings = model.encode(context, batch_size=256, show_progress_bar=True)
    ids = [f'id{i}' for i in range(len(context))]
    collection.add(ids=ids, embeddings=embeddings.tolist(), documents=context)
    return collection

def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark Baseline, Simple RAG, and Agentic RAG.")
    parser.add_argument(
        "--config",
        default="configs/benchmark.base.yaml",
        help="Path to YAML benchmark config file (default: configs/benchmark.base.yaml)",
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="Path to JSONL dataset file. Overrides config if provided.",
    )
    parser.add_argument(
        "--max-queries",
        type=int,
        default=None,
        help="Optional cap for number of queries to evaluate. Overrides config if provided.",
    )
    return parser.parse_args()

def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def resolve_dataset_path(dataset_path):
    if os.path.exists(dataset_path):
        return dataset_path
    candidate = os.path.join("data", dataset_path)
    if os.path.exists(candidate):
        return candidate
    return dataset_path

def write_failure_analysis(questions, answers, baseline_responses, simple_responses, agentic_responses):
    pipelines = {
        "Baseline": baseline_responses,
        "Simple RAG": simple_responses,
        "Agentic RAG": agentic_responses,
    }

    def normalize(text):
        return re.sub(r"\W+", "", text.lower().strip())

    summary_rows = []
    for name, responses in pipelines.items():
        em = 0
        idk = 0
        for pred, truth in zip(responses, answers):
            if pred.strip().lower() == "i don't know":
                idk += 1
            if normalize(pred) == normalize(truth):
                em += 1
        summary_rows.append({
            "Pipeline": name,
            "Exact Match Count": em,
            "I don't know Count": idk,
            "Total": len(responses),
        })

    with open("evaluation/failure_mode_summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
        writer.writeheader()
        writer.writerows(summary_rows)

    with open("evaluation/failure_cases.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "question",
            "ground_truth",
            "baseline",
            "simple_rag",
            "agentic_rag",
        ])
        for q, truth, b, s, a in zip(questions, answers, baseline_responses, simple_responses, agentic_responses):
            if normalize(b) != normalize(truth) or normalize(s) != normalize(truth) or normalize(a) != normalize(truth):
                writer.writerow([q, truth, b, s, a])

def write_analysis_report(results, total_queries, setup_time=None, retrieval_metrics=None):
    by_name = {row["Pipeline"]: row for row in results}
    lines = [
        "# Benchmark Analysis",
        "",
        f"Total queries evaluated: {total_queries}",
    ]
    
    if setup_time is not None:
        lines.append(f"Database setup time: {setup_time:.2f}s")
    
    lines.extend([
        "",
        "## Strengths",
    ])

    best_em = max(results, key=lambda r: r["Exact Match"])
    best_f1 = max(results, key=lambda r: r["F1 Score"])
    best_faithfulness = max(results, key=lambda r: r["LLM Faithfulness"])
    fastest = min(results, key=lambda r: r["Latency per request (s)"])

    lines.append(f"- Best Exact Match: {best_em['Pipeline']} ({best_em['Exact Match']:.4f})")
    lines.append(f"- Best F1 Score: {best_f1['Pipeline']} ({best_f1['F1 Score']:.4f})")
    lines.append(f"- Best Faithfulness: {best_faithfulness['Pipeline']} ({best_faithfulness['LLM Faithfulness']:.4f})")
    lines.append(f"- Fastest latency/request: {fastest['Pipeline']} ({fastest['Latency per request (s)']:.2f}s)")
    lines.append("")
    lines.append("## Weaknesses")

    for row in results:
        lines.append(
            "- "
            f"{row['Pipeline']}: EM={row['Exact Match']:.4f}, "
            f"F1={row['F1 Score']:.4f}, "
            f"Reasoning={row['LLM Reasoning']:.4f}, "
            f"Faithfulness={row['LLM Faithfulness']:.4f}"
        )

    lines.append("")
    if retrieval_metrics:
        lines.append("## Retrieval")
        lines.append(f"- Retrieval Hit@1: {retrieval_metrics.get('Retrieval Hit@1', 0.0):.4f}")
        lines.append(f"- Retrieval Hit@3: {retrieval_metrics.get('Retrieval Hit@3', 0.0):.4f}")
        lines.append(f"- Retrieval Hit@5: {retrieval_metrics.get('Retrieval Hit@5', 0.0):.4f}")
        lines.append(f"- Retrieval MRR@5: {retrieval_metrics.get('Retrieval MRR@5', 0.0):.4f}")
        lines.append(f"- Retrieval Context F1@5: {retrieval_metrics.get('Retrieval Context F1@5', 0.0):.4f}")
        lines.append("")
    lines.append("## Failure Modes")
    lines.append("- Abstention overuse: high rates of 'I don't know' responses lower recall.")
    lines.append("- Retrieval misses: relevant context not in top-k yields incorrect or empty answers.")
    lines.append("- Over-compression in post-processing can remove useful answer tokens.")
    lines.append("- Judge-model variance: LLM-as-a-judge introduces scoring noise.")

    with open("evaluation/analysis_report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def write_charts(results, save_query_level_scores=True):
    os.makedirs("evaluation/charts", exist_ok=True)

    pipelines = [row["Pipeline"] for row in results]
    em = [row["Exact Match"] for row in results]
    f1 = [row["F1 Score"] for row in results]
    faith = [row["LLM Faithfulness"] for row in results]
    latency = [row["Latency per request (s)"] for row in results]

    # Extract raw scores for distribution plots and remove them so they don't pollute the CSV
    raw_f1_data = []
    raw_c_data = []
    raw_em_data = []
    raw_comp_data = []
    raw_r_data = []
    raw_faith_data = []

    distribution_rows = []
    for row in results:
        em_raw = row.pop("Raw EM", [])
        f1_raw = row.pop("Raw F1", [])
        c_raw = row.pop("Raw Correctness", [])
        comp_raw = row.pop("Raw Completeness", [])
        r_raw = row.pop("Raw Reasoning", [])
        faith_raw = row.pop("Raw Faithfulness", [])

        raw_em_data.append(em_raw)
        raw_f1_data.append(f1_raw)
        raw_c_data.append(c_raw)
        raw_comp_data.append(comp_raw)
        raw_r_data.append(r_raw)
        raw_faith_data.append(faith_raw)

        for i in range(len(f1_raw)):
            distribution_rows.append({
                "Pipeline": row["Pipeline"],
                "Query Index": i,
                "Exact Match": em_raw[i],
                "F1 Score": f1_raw[i],
                "Correctness": c_raw[i],
                "Completeness": comp_raw[i],
                "Reasoning": r_raw[i],
                "Faithfulness": faith_raw[i],
            })

    if distribution_rows and save_query_level_scores:
        with open("evaluation/query_level_scores.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=distribution_rows[0].keys())
            writer.writeheader()
            writer.writerows(distribution_rows)

    # Comparison chart for core quality metrics.
    x = list(range(len(pipelines)))
    w = 0.22
    plt.figure(figsize=(10, 5))
    plt.bar([i - w for i in x], em, width=w, label="Exact Match")
    plt.bar(x, f1, width=w, label="F1 Score")
    plt.bar([i + w for i in x], faith, width=w, label="Faithfulness")
    plt.xticks(x, pipelines)
    plt.ylim(0, 1)
    plt.ylabel("Score")
    plt.title("Model Quality Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig("evaluation/charts/model_comparison.png", dpi=140)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.bar(pipelines, latency)
    plt.ylabel("Seconds per request")
    plt.title("Latency per Request")
    plt.tight_layout()
    plt.savefig("evaluation/charts/latency_comparison.png", dpi=140)
    plt.close()

    if os.path.exists("evaluation/failure_mode_summary.csv"):
        names = []
        idk_rates = []
        with open("evaluation/failure_mode_summary.csv", "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                total = int(row["Total"])
                idk = int(row["I don't know Count"])
                names.append(row["Pipeline"])
                idk_rates.append((idk / total) if total else 0.0)

        plt.figure(figsize=(8, 5))
        plt.bar(names, idk_rates)
        plt.ylim(0, 1)
        plt.ylabel("I don't know rate")
        plt.title("Abstention Rate by Pipeline")
        plt.tight_layout()
        plt.savefig("evaluation/charts/idk_rate.png", dpi=140)
        plt.close()
        
    # Score distributions (Boxplots)
    if raw_f1_data and raw_f1_data[0]:
        plt.figure(figsize=(8, 5))
        plt.boxplot(raw_f1_data, tick_labels=pipelines)
        plt.ylim(-0.1, 1.1)
        plt.ylabel("F1 Score")
        plt.title("F1 Score Distribution by Pipeline")
        plt.tight_layout()
        plt.savefig("evaluation/charts/f1_distribution.png", dpi=140)
        plt.close()

        plt.figure(figsize=(8, 5))
        plt.boxplot(raw_c_data, tick_labels=pipelines)
        plt.ylim(-0.1, 1.1)
        plt.ylabel("Correctness Score")
        plt.title("Correctness Score Distribution by Pipeline")
        plt.tight_layout()
        plt.savefig("evaluation/charts/correctness_distribution.png", dpi=140)
        plt.close()

# Custom Rule-Based NLP Post-Processing
def check_negative(response):
    return normalize_idk(response)

def extract_exact_answer(question, response):
    if response == "I don't know":
        return response
    q_words = set(re.sub(r'[^\w\s]', '', question.lower()).split())
    r_words = response.split()
    extracted = []
    for rw in r_words:
        clean_rw = re.sub(r'[^\w\s]', '', rw.lower())
        if clean_rw not in q_words:
            extracted.append(rw)
    return ' '.join(extracted) if extracted else response

def strip_grammar(text):
    if text == "I don't know":
        return text
    stopwords = {'is', 'am', 'are', 'was', 'were', 'be', 'being', 'been', 'has', 'have', 'had', 'do', 'does', 'did', 'will', 'would', 'shall', 'should', 'can', 'could', 'may', 'might', 'must', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'just', 'don', 'now', 'that', 'this', 'these', 'those', 'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'whose', 'based', 'provided', 'context', 'specified', 'according'}
    words = text.split()
    cleaned = [w for w in words if re.sub(r'[^\w\s]', '', w.lower()) not in stopwords]
    return ' '.join(cleaned) if cleaned else text

def post_process(question, response):
    resp = check_negative(response)
    if resp == "I don't know":
        return resp
    resp = extract_exact_answer(question, resp)
    resp = strip_grammar(resp)
    return resp

def baseline(questions,strict_context_only=True):
    baseline_responses = []
    for q in questions:
        _prompt = build_qa_prompt("", q, force_idk=strict_context_only, strict_context_only=strict_context_only)
        baseline_responses.append(post_process(q,llm_generate(_prompt)))
    return baseline_responses

#simple rag
def simple_rag(questions, model, collection, top_k, strict_context_only=True):
    simple = []
    for i, q in enumerate(questions):
        qe = model.encode(q)
        con = collection.query(query_embeddings=[qe.tolist()], n_results=top_k)
        
        # Join retrieved documents into a prompt context block.
        context_text = "\n".join(con['documents'][0])
        prompt = build_qa_prompt(context_text, q, strict_context_only=strict_context_only)
        raw_resp = llm_generate(prompt)
        simple.append(post_process(q, raw_resp))
        
        if (i + 1) % 100 == 0:
            print(f"Simple RAG: Processed {i + 1}/{len(questions)} requests...")
    return simple

def merge_retrieved_contexts(*context_lists):
    merged = []
    seen = set()
    for context_list in context_lists:
        for context in context_list:
            normalized = context.strip()
            if normalized and normalized not in seen:
                seen.add(normalized)
                merged.append(context)
    return merged

def build_agentic_retrieval_query(question, draft_answer):
    if draft_answer and draft_answer != "I don't know":
        return f"{question}\nLikely answer: {draft_answer}"
    return question

#agentic rag
def agentic_rag(questions, model, collection, first_top_k, second_top_k, strict_context_only=True):
    agentic = []
    for i, q in enumerate(questions):
        qe = model.encode(q)
        first_pass = collection.query(query_embeddings=[qe.tolist()], n_results=first_top_k)
        first_context = "\n".join(first_pass['documents'][0])

        draft_prompt = build_qa_prompt(first_context, q, strict_context_only=strict_context_only)
        draft_answer = post_process(q, llm_generate(draft_prompt))

        retrieval_query = build_agentic_retrieval_query(q, draft_answer)
        retrieval_embedding = model.encode(retrieval_query)
        second_pass = collection.query(query_embeddings=[retrieval_embedding.tolist()], n_results=second_top_k)

        merged_docs = merge_retrieved_contexts(first_pass['documents'][0], second_pass['documents'][0])
        merged_context = "\n".join(merged_docs)

        ans_prompt = build_qa_prompt(merged_context, q, strict_context_only=strict_context_only)
        candidate = post_process(q, llm_generate(ans_prompt))

        if candidate != "I don't know":
            verify_prompt = (
                f"Context:\n{merged_context}\n\n"
                f"Question: {q}\n"
                f"Proposed Answer: {candidate}\n"
                "Is the proposed answer fully supported by the context? "
                "Reply with only SUPPORTED or UNSUPPORTED."
            )
            verdict = llm_generate(verify_prompt).strip().upper()
            if "UNSUPPORTED" in verdict:
                candidate = "I don't know"

        agentic.append(candidate)
            
        if (i + 1) % 100 == 0:
            print(f"Agentic RAG: Processed {i + 1}/{len(questions)} requests...")
            
    return agentic

def main():
    global MODEL_NAME
    args = parse_args()
    cfg = load_config(args.config)

    dataset = resolve_dataset_path(args.dataset or cfg.get("dataset", "data/val_benchmark_1200.jsonl"))
    max_queries = args.max_queries if args.max_queries is not None else cfg.get("max_queries")
    MODEL_NAME = cfg.get("model_name", MODEL_NAME)

    retrieval_cfg = cfg.get("retrieval", {})
    simple_top_k = int(retrieval_cfg.get("simple_top_k", 5))
    agentic_first_top_k = int(retrieval_cfg.get("agentic_first_top_k", 5))
    agentic_second_top_k = int(retrieval_cfg.get("agentic_second_top_k", 5))

    evaluation_cfg = cfg.get("evaluation", {})
    run_llm_judge = bool(evaluation_cfg.get("run_llm_judge", True))

    answering_cfg = cfg.get("answering", {})
    strict_context_only = bool(answering_cfg.get("strict_context_only", True))

    output_cfg = cfg.get("output", {})
    save_charts = bool(output_cfg.get("save_charts", True))
    save_query_level_scores = bool(output_cfg.get("save_query_level_scores", True))

    embedding_model_name = cfg.get("embedding_model", "all-MiniLM-L6-v2")

    start_setup = time.time()
    questions,context,answers = load_json(dataset, max_queries)
    print(f"Loaded {len(questions)} queries from {dataset}")
    print(f"Using model={MODEL_NAME}, embedding={embedding_model_name}, simple_top_k={simple_top_k}, agentic_first_top_k={agentic_first_top_k}, agentic_second_top_k={agentic_second_top_k}, strict_context_only={strict_context_only}")
    model = SentenceTransformer(embedding_model_name)
    collection = db(context, model)
    end_setup = time.time()
    print(f"Data loading and DB setup took {end_setup - start_setup:.2f} seconds.")

    # Evaluate ChromaDB retrieval accuracy first
    retrieval_metrics = calculate_retrieval_metrics(questions, context, embedding_model_name=embedding_model_name)

    results = []

    start_baseline = time.time()
    print("Running Baseline evaluation...")
    baseline_responses = baseline(questions)
    with open('evaluation/baseline_responses.txt', 'w', encoding='utf-8') as f:
        for response in baseline_responses:
            f.write(response + '\n')
    end_baseline = time.time()
    baseline_latency = end_baseline - start_baseline
    print(f"Baseline evaluation took {baseline_latency:.2f} seconds.")
    results.append(evaluate_pipeline("Baseline", baseline_responses, questions, context, answers, baseline_latency, retrieval_metrics, run_llm_judge))
    
    start_simple = time.time()
    print("Running Simple RAG evaluation...")
    simple_responses = simple_rag(questions, model, collection, simple_top_k, strict_context_only=strict_context_only)
    with open('evaluation/simple_rag_responses.txt', 'w', encoding='utf-8') as f:
        for response in simple_responses:
            f.write(response + '\n')
    end_simple = time.time()
    simple_latency = end_simple - start_simple
    print(f"Simple RAG evaluation took {simple_latency:.2f} seconds.")
    results.append(evaluate_pipeline("Simple RAG", simple_responses, questions, context, answers, simple_latency, None, run_llm_judge))

    start_agentic = time.time()
    print("Running Agentic RAG evaluation...")
    agentic_responses = agentic_rag(questions, model, collection, agentic_first_top_k, agentic_second_top_k, strict_context_only=strict_context_only)
    with open('evaluation/agentic_rag_responses.txt', 'w', encoding='utf-8') as f:
        for response in agentic_responses:
            f.write(response + '\n')
    end_agentic = time.time()
    agentic_latency = end_agentic - start_agentic
    print(f"Agentic RAG evaluation took {agentic_latency:.2f} seconds.")
    results.append(evaluate_pipeline("Agentic RAG", agentic_responses, questions, context, answers, agentic_latency, None, run_llm_judge))

    write_failure_analysis(questions, answers, baseline_responses, simple_responses, agentic_responses)
    print("Saved failure analysis to failure_mode_summary.csv and failure_cases.csv")
    write_analysis_report(results, len(questions), setup_time=end_setup - start_setup, retrieval_metrics=retrieval_metrics)
    print("Saved analysis report to analysis_report.md")
    if save_charts:
        write_charts(results, save_query_level_scores=save_query_level_scores)
        print("Saved charts to evaluation/charts/")

    if results:
        csv_filename = "evaluation/evaluation_results.csv"
        print(f"Saving results to {csv_filename}...")
        with open(csv_filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        print("Done!")

if __name__ == "__main__":
    main()
