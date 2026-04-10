import json
import re
import collections
import csv
import ollama
from sentence_transformers import SentenceTransformer
import chromadb

def normalize_text(text):
    return re.sub(r'\W+', ' ', text.lower().strip()).strip()

def extract_answer_span(text):
    if not text:
        return ""

    lowered = text.lower().strip()
    if "i don't know" in lowered or "i do not know" in lowered:
        return "I don't know"

    marker_patterns = [
        r"(?:final\s+answer|answer|response)\s*:\s*",
        r"(?:final\s+answer|answer|response)\s*-\s*",
    ]
    for pattern in marker_patterns:
        match = re.search(pattern, lowered)
        if match:
            text = text[match.end():].strip()
            break

    for separator in ["\n", "。", "!", "?", ";"]:
        if separator in text:
            text = text.split(separator, 1)[0].strip()

    text = re.sub(r'^["\'`\-\s]+|["\'`\-\s]+$', '', text).strip()
    return text

def exact_match(prediction, truth):
    prediction_norm = normalize_text(extract_answer_span(prediction))
    truth_norm = normalize_text(extract_answer_span(truth))
    return int(prediction_norm == truth_norm)

def f1_score(prediction, truth):
    prediction_clean = extract_answer_span(prediction)
    truth_clean = extract_answer_span(truth)
    pred_tokens = normalize_text(prediction_clean).split()
    truth_tokens = normalize_text(truth_clean).split()
    common = collections.Counter(pred_tokens) & collections.Counter(truth_tokens)
    num_same = sum(common.values())
    if len(pred_tokens) == 0 or len(truth_tokens) == 0:
        return int(pred_tokens == truth_tokens)
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_tokens)
    recall = 1.0 * num_same / len(truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def calculate_retrieval_metrics(questions, contexts):
    print("Calculating Retrieval Metrics (Hit Rate @ 5)...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_collection(name="vector")
    
    hits = 0
    for i, q in enumerate(questions):
        qe = model.encode(q)
        res = collection.query(query_embeddings=[qe.tolist()], n_results=2)
        retrieved_docs = res['documents'][0]
        
        # Check if the ground truth context snippet is within the retrieved documents
        truth_snippet = contexts[i][:150] 
        hit = any(truth_snippet in doc for doc in retrieved_docs)
        if hit:
            hits += 1
            
        if (i + 1) % 100 == 0:
            print(f"Retrieval checked for {i+1}/{len(questions)} queries.")
            
    hit_rate = hits / len(questions)
    print(f"Retrieval Hit Rate @ 5: {hit_rate:.4f}\n")
    return hit_rate

def llm_judge(question, context, prediction, truth):
    # Optimization: Automatically score "I don't know" to save time and API calls
    if "i don't know" in prediction.lower() and len(prediction) < 20:
        return 0.0, 0.0, 0.0, 1.0

    prompt = f"""Evaluate the AI Answer based on the Ground Truth and Context.
Question: {question}
Context: {context}
Ground Truth: {truth}
AI Answer: {prediction}

Score these 4 metrics from 0 to 10 (where 0 is completely wrong/bad, 10 is perfect).
Format your response EXACTLY like this:
Correctness: [score]
Completeness: [score]
Reasoning: [score]
Faithfulness: [score]
"""
    try:
        res = ollama.chat(model='llama3.2:1b', messages=[{"role": "user", "content": prompt}])
        text = res['message']['content']
        
        def extract_score(metric, text):
            match = re.search(fr'{metric}:\s*(\d+)', text, re.IGNORECASE)
            if match:
                return min(int(match.group(1)), 10) / 10.0
            return 0.0
            
        c = extract_score('Correctness', text)
        comp = extract_score('Completeness', text)
        r = extract_score('Reasoning', text)
        f = extract_score('Faithfulness', text)
        return c, comp, r, f
    except Exception:
        return 0.0, 0.0, 0.0, 0.0

def evaluate_pipeline(name, responses, questions, contexts, truths, latency_seconds):
    print(f"=====================================")
    print(f"Evaluating {name}")
    print(f"=====================================")
    
    n = len(responses)
    em_total, f1_total = 0, 0
    c_total, comp_total, r_total, f_total = 0, 0, 0, 0
    
    em_raw, f1_raw = [], []
    c_raw, comp_raw, r_raw, f_raw = [], [], [], []
    
    print("Calculating Exact Match, F1, and running LLM-as-a-judge...")
    for i in range(n):
        em_val = exact_match(responses[i], truths[i])
        f1_val = f1_score(responses[i], truths[i])
        em_total += em_val
        f1_total += f1_val
        em_raw.append(em_val)
        f1_raw.append(f1_val)
        
        c, comp, r, f = llm_judge(questions[i], contexts[i], responses[i], truths[i])
        c_total += c
        comp_total += comp
        r_total += r
        f_total += f
        c_raw.append(c)
        comp_raw.append(comp)
        r_raw.append(r)
        f_raw.append(f)
        
        if (i + 1) % 100 == 0:
            print(f"Evaluated {i+1}/{n} responses for {name}...")

    metrics = {
        "Pipeline": name,
        "Latency (s)": latency_seconds,
        "Latency per request (s)": latency_seconds / n,
        "Exact Match": em_total / n,
        "F1 Score": f1_total / n,
        "LLM Correctness": c_total / n,
        "LLM Completeness": comp_total / n,
        "LLM Reasoning": r_total / n,
        "LLM Faithfulness": f_total / n,
        "Raw EM": em_raw,
        "Raw F1": f1_raw,
        "Raw Correctness": c_raw,
        "Raw Completeness": comp_raw,
        "Raw Reasoning": r_raw,
        "Raw Faithfulness": f_raw
    }

    print(f"\n--- {name} Results ---")
    print(f"Latency:         {latency_seconds}s ({(latency_seconds/n):.2f}s per request)")
    print(f"Exact Match:     {metrics['Exact Match']:.4f}")
    print(f"F1 Score:        {metrics['F1 Score']:.4f}")
    print(f"LLM Correctness: {metrics['LLM Correctness']:.4f}")
    print(f"LLM Completeness:{metrics['LLM Completeness']:.4f}")
    print(f"LLM Reasoning:   {metrics['LLM Reasoning']:.4f}")
    print(f"LLM Faithfulness:{metrics['LLM Faithfulness']:.4f}\n")
    
    return metrics
