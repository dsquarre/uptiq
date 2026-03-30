import argparse
import json
from pathlib import Path

from src.eval.metrics import exact_match_score, f1_score, max_over_ground_truths


def read_jsonl(path: str) -> list[dict]:
    rows: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def categorize(row: dict) -> str:
    predicted = row.get("predicted_answer", "")
    gold = row.get("gold_answers", [])
    em = max_over_ground_truths(exact_match_score, predicted, gold)
    f1 = max_over_ground_truths(f1_score, predicted, gold)

    if em == 1:
        return "exact_match"
    if f1 >= 0.5:
        return "partial_answer"
    if not row.get("retrieved_contexts"):
        return "retrieval_miss"
    return "likely_hallucination"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    rows = read_jsonl(args.predictions)
    by_arch: dict[str, dict[str, int]] = {}

    for row in rows:
        arch = row.get("architecture", "unknown")
        cat = categorize(row)
        by_arch.setdefault(arch, {})
        by_arch[arch][cat] = by_arch[arch].get(cat, 0) + 1

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(by_arch, indent=2), encoding="utf-8")

    print(json.dumps({"output": str(output_path), "architectures": list(by_arch.keys())}, indent=2))


if __name__ == "__main__":
    main()
