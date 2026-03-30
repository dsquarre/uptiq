import argparse
import json
import random
import shutil
from pathlib import Path

import yaml
from tqdm import tqdm

from src.agents.agentic_rag import AgenticRAGAgent
from src.agents.simple_rag import SimpleRAGAgent
from src.common.schemas import EvalRecord, QASample
from src.eval.judge import Judge
from src.eval.metrics import exact_match_score, f1_score, max_over_ground_truths, mrr, recall_at_k


def read_jsonl(path: str) -> list[dict]:
    records: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_jsonl(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_agent(name: str, config: dict):
    top_k = int(config.get("retrieval", {}).get("top_k", 3))
    if name == "simple_rag":
        return SimpleRAGAgent(top_k=top_k)
    if name == "agentic_rag":
        steps = int(config.get("agentic", {}).get("max_steps", 3))
        return AgenticRAGAgent(top_k=top_k, max_steps=steps)
    raise ValueError(f"Unknown architecture: {name}")


def summarize(eval_rows: list[dict]) -> dict:
    grouped: dict[str, list[dict]] = {}
    for row in eval_rows:
        grouped.setdefault(row["architecture"], []).append(row)

    summary: dict[str, dict] = {}
    for arch, rows in grouped.items():
        n = len(rows)
        summary[arch] = {
            "count": n,
            "em": sum(r["em"] for r in rows) / n,
            "f1": sum(r["f1"] for r in rows) / n,
            "recall_at_k": sum(r["recall_at_k"] for r in rows) / n,
            "mrr": sum(r["mrr"] for r in rows) / n,
            "judge_correctness": sum(r["judge_correctness"] for r in rows) / n,
            "judge_completeness": sum(r["judge_completeness"] for r in rows) / n,
            "judge_reasoning": sum(r["judge_reasoning"] for r in rows) / n,
            "latency_ms": sum(r.get("latency_ms", 0.0) for r in rows) / n,
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    random.seed(int(config.get("seed", 42)))

    dataset_path = config["dataset_path"]
    source_rows = read_jsonl(dataset_path)
    samples = [QASample(**row) for row in source_rows]

    run_name = config.get("run_name", "run")
    results_dir = Path(config.get("results_dir", "results")) / run_name
    results_dir.mkdir(parents=True, exist_ok=True)

    judge_mode = config.get("judge", {}).get("mode", "local")
    judge_sample_size = config.get("judge", {}).get("sample_size")
    judge = Judge(mode=judge_mode)

    predictions: list[dict] = []
    eval_rows: list[dict] = []

    for architecture in config.get("architectures", ["simple_rag", "agentic_rag"]):
        agent = build_agent(architecture, config)
        for sample in tqdm(samples, desc=f"Running {architecture}"):
            prediction = agent.predict(sample)
            pred_dict = prediction.to_dict()
            predictions.append(pred_dict)

    judge_ids: set[str] | None = None
    if isinstance(judge_sample_size, int) and judge_sample_size > 0:
        ids = sorted({row["id"] for row in predictions})
        judge_ids = set(random.sample(ids, min(judge_sample_size, len(ids))))

    top_k = int(config.get("retrieval", {}).get("top_k", 3))
    for row in tqdm(predictions, desc="Evaluating"):
        gold = row.get("gold_answers", [])
        predicted = row.get("predicted_answer", "")

        em = int(max_over_ground_truths(exact_match_score, predicted, gold))
        f1 = float(max_over_ground_truths(f1_score, predicted, gold))
        r_at_k = float(recall_at_k(row.get("retrieved_contexts", []), gold, k=top_k))
        rr = float(mrr(row.get("retrieved_contexts", []), gold))

        run_judge = judge_ids is None or row["id"] in judge_ids
        if run_judge:
            judge_score = judge.score(row.get("question", ""), predicted, gold).to_dict()
        else:
            judge_score = {
                "judge_correctness": 0,
                "judge_completeness": 0,
                "judge_reasoning": 0,
            }

        eval_record = EvalRecord(
            id=row["id"],
            architecture=row["architecture"],
            em=em,
            f1=f1,
            recall_at_k=r_at_k,
            mrr=rr,
            judge_correctness=judge_score["judge_correctness"],
            judge_completeness=judge_score["judge_completeness"],
            judge_reasoning=judge_score["judge_reasoning"],
        ).to_dict()
        eval_record["latency_ms"] = row.get("latency_ms", 0.0)
        eval_rows.append(eval_record)

    summary = summarize(eval_rows)

    write_jsonl(predictions, results_dir / "predictions.jsonl")
    write_jsonl(eval_rows, results_dir / "evaluation.jsonl")
    (results_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    latest_dir = Path(config.get("results_dir", "results")) / "latest"
    latest_dir.mkdir(parents=True, exist_ok=True)
    for filename in ["predictions.jsonl", "evaluation.jsonl", "summary.json"]:
        shutil.copyfile(results_dir / filename, latest_dir / filename)

    print(json.dumps({
        "run_name": run_name,
        "results_dir": str(results_dir),
        "num_predictions": len(predictions),
        "num_eval_rows": len(eval_rows),
        "architectures": sorted(summary.keys()),
    }, indent=2))


if __name__ == "__main__":
    main()
